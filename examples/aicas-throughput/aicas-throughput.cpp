#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
namespace fs = std::filesystem;

static constexpr int32_t kBenchmarkRuns = 10;
static constexpr int32_t kWarmupRuns = 10;
static constexpr int32_t kPrefillTokens = 1;
static constexpr int32_t kGenerationTokens = 50;

struct cmd_params {
    std::string model_path;
    std::string prompt_file;
    int32_t n_ctx = 0;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    int32_t n_gpu_layers = 0;
    bool flash_attn = false;
};

struct run_metrics {
    double prefill_seconds = 0.0;
    double generation_seconds = 0.0;
};

static double seconds_between(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("  %s -m model.gguf -f prompt.txt -t 4 -tb 4 -c 768 -b 563 -ub 512\n", argv[0]);
    printf("\noptions:\n");
    printf("  -m <path>          model gguf path\n");
    printf("  -f <path>          prompt file path\n");
    printf("  -c <n>             context size, 0 = auto\n");
    printf("  -b <n>             batch size, 0 = prompt token count\n");
    printf("  -ub <n>            physical ubatch size, 0 = llama.cpp default\n");
    printf("  -t <n>             generation threads, 0 = hardware default\n");
    printf("  -tb <n>            batch threads, 0 = same as -t\n");
    printf("  -ngl <n>           gpu layers (default: 0)\n");
    printf("  --flash-attn       enable Flash Attention\n");
    printf("\n");
    printf("This benchmark is fixed to %d measured runs.\n", kBenchmarkRuns);
    printf("Prefill phase warms up %d time and measures prompt -> first token using %d token.\n", kWarmupRuns, kPrefillTokens);
    printf("Generation phase warms up %d time and measures decode using %d tokens.\n", kWarmupRuns, kGenerationTokens);
    printf("\n");
}

static bool parse_int_arg(const char * arg, int32_t & out) {
    try {
        size_t pos = 0;
        const std::string value(arg);
        out = std::stoi(value, &pos);
        return pos == value.size();
    } catch (...) {
        return false;
    }
}

static bool parse_args(int argc, char ** argv, cmd_params & params) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 >= argc) {
                return false;
            }
            params.model_path = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--prompt-file") == 0) {
            if (i + 1 >= argc) {
                return false;
            }
            params.prompt_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_ctx)) {
                return false;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_batch)) {
                return false;
            }
        } else if (strcmp(argv[i], "-ub") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_ubatch)) {
                return false;
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_threads)) {
                return false;
            }
        } else if (strcmp(argv[i], "-tb") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_threads_batch)) {
                return false;
            }
        } else if (strcmp(argv[i], "-ngl") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_gpu_layers)) {
                return false;
            }
        } else if (strcmp(argv[i], "--flash-attn") == 0 || strcmp(argv[i], "-fa") == 0) {
            params.flash_attn = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argc, argv);
            std::exit(0);
        } else {
            return false;
        }
    }

    return !params.model_path.empty() && !params.prompt_file.empty();
}

static std::string read_text_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::string now_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_bos, true);
    if (n_prompt <= 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }

    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), add_bos, true) < 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }

    return tokens;
}

static void eval_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    auto * token_data = const_cast<llama_token *>(tokens.data());
    if (llama_decode(ctx, llama_batch_get_one(token_data, tokens.size()))) {
        throw std::runtime_error("failed to evaluate prompt");
    }
}

static void generate_tokens(
        llama_context * ctx,
        llama_sampler * smpl,
        int32_t n_predict) {
    for (int32_t i = 0; i < n_predict; ++i) {
        const llama_token token = llama_sampler_sample(smpl, ctx, -1);
        llama_token next_token = token;
        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1))) {
            throw std::runtime_error("failed to evaluate generated token");
        }
    }
}

static run_metrics run_prefill_once(
        llama_context * ctx,
        llama_sampler * smpl,
        const std::vector<llama_token> & prompt_tokens) {
    run_metrics metrics;

    llama_kv_self_clear(ctx);
    llama_sampler_reset(smpl);

    const auto prefill_begin = clock_type::now();
    eval_tokens(ctx, prompt_tokens);
    generate_tokens(ctx, smpl, kPrefillTokens);
    const auto prefill_end = clock_type::now();
    metrics.prefill_seconds = seconds_between(prefill_begin, prefill_end);

    return metrics;
}

static run_metrics run_generation_once(
        llama_context * ctx,
        llama_sampler * smpl,
        const std::vector<llama_token> & prompt_tokens) {
    run_metrics metrics;

    llama_kv_self_clear(ctx);
    llama_sampler_reset(smpl);

    eval_tokens(ctx, prompt_tokens);

    const auto generation_begin = clock_type::now();
    generate_tokens(ctx, smpl, kGenerationTokens);
    const auto generation_end = clock_type::now();
    metrics.generation_seconds = seconds_between(generation_begin, generation_end);

    return metrics;
}

int main(int argc, char ** argv) {
    cmd_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argc, argv);
        return 1;
    }

    try {
        const std::string prompt = read_text_file(params.prompt_file);

        llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);

        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers;

        llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }

        const std::vector<llama_token> prompt_tokens = tokenize_prompt(llama_model_get_vocab(model), prompt);

        const int32_t prompt_token_count = static_cast<int32_t>(prompt_tokens.size());
        const int32_t needed_ctx = prompt_token_count + kGenerationTokens + 8;
        const int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : std::max<int32_t>(needed_ctx, 1024);
        if (n_ctx < needed_ctx) {
            throw std::runtime_error("n_ctx is smaller than prompt_tokens + 50");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = params.n_batch > 0 ? params.n_batch : prompt_token_count;
        ctx_params.n_batch = std::max<int32_t>(1, std::min<int32_t>(ctx_params.n_batch, ctx_params.n_ctx));
        if (params.n_ubatch > 0) {
            ctx_params.n_ubatch = params.n_ubatch;
        }
        ctx_params.n_threads = params.n_threads > 0 ? params.n_threads : static_cast<int32_t>(std::thread::hardware_concurrency());
        ctx_params.n_threads_batch = params.n_threads_batch > 0 ? params.n_threads_batch : ctx_params.n_threads;
        ctx_params.flash_attn = params.flash_attn;
        ctx_params.no_perf = true;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("failed to create context");
        }

        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        for (int32_t i = 0; i < kWarmupRuns; ++i) {
            (void) run_prefill_once(ctx, smpl, prompt_tokens);
        }

        double prefill_total = 0.0;
        for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
            const run_metrics metrics = run_prefill_once(ctx, smpl, prompt_tokens);
            prefill_total += metrics.prefill_seconds;
        }

        for (int32_t i = 0; i < kWarmupRuns; ++i) {
            (void) run_generation_once(ctx, smpl, prompt_tokens);
        }

        double generation_total = 0.0;
        for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
            const run_metrics metrics = run_generation_once(ctx, smpl, prompt_tokens);
            generation_total += metrics.generation_seconds;
        }

        const double prefill_throughput = prefill_total > 0.0
                ? (prompt_token_count * kBenchmarkRuns) / prefill_total
                : 0.0;
        const double decode_throughput = generation_total > 0.0
                ? (kGenerationTokens * kBenchmarkRuns) / generation_total
                : 0.0;

        printf("prefill_throughput: %.6f\n", prefill_throughput);
        printf("decode_throughput: %.6f\n", decode_throughput);

        const fs::path model_path(params.model_path);
        const fs::path project_dir = fs::absolute(fs::path(params.prompt_file)).parent_path();
        const fs::path output_dir = project_dir / "results" / "perf";
        fs::create_directories(output_dir);
        const fs::path output_path = output_dir / (model_path.stem().string() + "-" + now_timestamp() + "-throughput.json");

        std::ofstream output_file(output_path);
        if (!output_file) {
            throw std::runtime_error("failed to create output json: " + output_path.string());
        }

        output_file << "{\n";
        output_file << "  \"prefill_throughput\": " << std::fixed << std::setprecision(6) << prefill_throughput << ",\n";
        output_file << "  \"decode_throughput\": " << std::fixed << std::setprecision(6) << decode_throughput << "\n";
        output_file << "}\n";
        output_file.close();

        printf("saved_json: %s\n", output_path.c_str());

        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: %s\n", __func__, err.what());
        return 1;
    }
}
