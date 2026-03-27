#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct cmd_params {
    std::string model_path;
    std::string prompt;
    std::string prompt_file;
    int32_t n_predict = 50;
    int32_t n_ctx = 0;
    int32_t n_batch = 0;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    int32_t n_gpu_layers = 0;
    bool flash_attn = false;
    bool no_mmap = false;
    bool print_prompt = false;
};

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("  %s -m model.gguf -f prompt.txt -n 50\n", argv[0]);
    printf("  %s -m model.gguf -p \"Hello\" -n 32 -t 8 --print-prompt\n", argv[0]);
    printf("\noptions:\n");
    printf("  -m <path>          model gguf path\n");
    printf("  -p <text>          prompt text\n");
    printf("  -f <path>          prompt file path\n");
    printf("  -n <n>             max new tokens to generate (default: 50)\n");
    printf("  -c <n>             context size, 0 = auto\n");
    printf("  -b <n>             batch size, 0 = prompt token count\n");
    printf("  -t <n>             generation threads, 0 = hardware default\n");
    printf("  -tb <n>            batch threads, 0 = same as -t\n");
    printf("  -ngl <n>           gpu layers (default: 0)\n");
    printf("  --flash-attn       enable Flash Attention\n");
    printf("  --no-mmap          disable mmap when loading model\n");
    printf("  --print-prompt     print prompt before generation\n");
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
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 >= argc) {
                return false;
            }
            params.prompt = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--prompt-file") == 0) {
            if (i + 1 >= argc) {
                return false;
            }
            params.prompt_file = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_predict)) {
                return false;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_ctx)) {
                return false;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], params.n_batch)) {
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
        } else if (strcmp(argv[i], "--no-mmap") == 0) {
            params.no_mmap = true;
        } else if (strcmp(argv[i], "--print-prompt") == 0) {
            params.print_prompt = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argc, argv);
            std::exit(0);
        } else {
            return false;
        }
    }

    if (params.model_path.empty()) {
        return false;
    }
    if (params.prompt.empty() == params.prompt_file.empty()) {
        return false;
    }
    if (params.n_predict < 0) {
        return false;
    }

    return true;
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

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    char piece[256];
    const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
    if (n < 0) {
        throw std::runtime_error("failed to convert token to piece");
    }
    return std::string(piece, n);
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

static int32_t generate_tokens(
        llama_context * ctx,
        llama_sampler * smpl,
        const llama_vocab * vocab,
        int32_t n_predict) {
    int32_t generated = 0;

    for (int32_t i = 0; i < n_predict; ++i) {
        const llama_token token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        const std::string piece = token_to_piece(vocab, token);
        printf("%s", piece.c_str());
        fflush(stdout);

        llama_token next_token = token;
        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1))) {
            throw std::runtime_error("failed to evaluate generated token");
        }

        ++generated;
    }

    return generated;
}

int main(int argc, char ** argv) {
    cmd_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argc, argv);
        return 1;
    }

    try {
        if (!params.prompt_file.empty()) {
            params.prompt = read_text_file(params.prompt_file);
        }

        llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);

        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers;
        model_params.use_mmap = !params.no_mmap;

        llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab, params.prompt);

        const int32_t prompt_token_count = static_cast<int32_t>(prompt_tokens.size());
        const int32_t needed_ctx = prompt_token_count + std::max<int32_t>(params.n_predict, 1) + 8;
        const int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : std::max<int32_t>(needed_ctx, 1024);
        if (n_ctx < needed_ctx) {
            throw std::runtime_error("n_ctx is smaller than prompt_tokens + n_predict");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = params.n_batch > 0 ? params.n_batch : prompt_token_count;
        ctx_params.n_batch = std::max<int32_t>(1, std::min<int32_t>(ctx_params.n_batch, ctx_params.n_ctx));
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

        printf("prompt_tokens: %d\n", prompt_token_count);
        printf("generating_tokens: %d\n", params.n_predict);

        if (params.print_prompt) {
            printf("\nprompt:\n%s\n", params.prompt.c_str());
        }

        printf("\noutput:\n");
        llama_kv_self_clear(ctx);
        llama_sampler_reset(smpl);
        eval_tokens(ctx, prompt_tokens);
        const int32_t generated = generate_tokens(ctx, smpl, vocab, params.n_predict);
        printf("\n");
        printf("\ngenerated_tokens: %d\n", generated);

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
