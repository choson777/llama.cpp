#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct ggml_ctx_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ggml_threadpool_deleter {
    void operator()(ggml_threadpool * threadpool) const {
        if (threadpool != nullptr) {
            ggml_threadpool_free(threadpool);
        }
    }
};

struct aligned_buffer {
    uint8_t * data = nullptr;
    size_t size = 0;

    ~aligned_buffer() {
        if (data != nullptr) {
            ggml_aligned_free(data, size);
        }
    }

    aligned_buffer() = default;
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    aligned_buffer(aligned_buffer && other) noexcept {
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
    }

    aligned_buffer & operator=(aligned_buffer && other) noexcept {
        if (this != &other) {
            if (data != nullptr) {
                ggml_aligned_free(data, size);
            }
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

using ggml_ctx_ptr = std::unique_ptr<ggml_context, ggml_ctx_deleter>;
using ggml_threadpool_ptr = std::unique_ptr<ggml_threadpool, ggml_threadpool_deleter>;

struct bench_options {
    int threads = 4;
    int warmup = 5;
    int iterations = 20;
};

struct attn_shape {
    const char * name;
    int64_t head_size;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t n_tokens;
    int64_t n_kv;
    bool causal_mask;
};

struct attn_inputs {
    std::vector<float> q_f32;
    std::vector<ggml_fp16_t> q_f16;
    std::vector<ggml_fp16_t> k_f16;
    std::vector<ggml_fp16_t> v_flash_f16;
    std::vector<ggml_fp16_t> v_mha_f16;
    std::vector<ggml_fp16_t> mask_f16;
};

struct bench_graph {
    ggml_ctx_ptr ctx;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_cplan plan = {};
    aligned_buffer work;
    int n_nodes = 0;
};

struct bench_result {
    double avg_us = 0.0;
    std::vector<float> output;
    int n_nodes = 0;
    ggml_type out_type = GGML_TYPE_COUNT;
};

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());

    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        const double da = a[i];
        const double db = b[i];
        mse_a_b += (da - db) * (da - db);
        mse_a_0 += da * da;
    }

    return mse_a_0 > 0.0 ? mse_a_b / mse_a_0 : mse_a_b;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, size_t * worst_idx = nullptr) {
    assert(a.size() == b.size());

    float worst = 0.0f;
    size_t idx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > worst) {
            worst = diff;
            idx = i;
        }
    }

    if (worst_idx != nullptr) {
        *worst_idx = idx;
    }
    return worst;
}

static bool parse_int_arg(const char * value, int * out) {
    if (value == nullptr || out == nullptr) {
        return false;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        return false;
    }

    *out = (int) parsed;
    return true;
}

static bool parse_options(int argc, char ** argv, bench_options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &opts.threads)) {
                return false;
            }
        } else if ((arg == "--warmup" || arg == "-w") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &opts.warmup)) {
                return false;
            }
        } else if ((arg == "--iterations" || arg == "--iters" || arg == "-n") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &opts.iterations)) {
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return true;
}

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s [--threads N] [--warmup N] [--iterations N]\n",
            argv0);
}

static void fp32_to_fp16(const std::vector<float> & src, std::vector<ggml_fp16_t> & dst) {
    dst.resize(src.size());
    ggml_fp32_to_fp16_row(src.data(), dst.data(), (int64_t) src.size());
}

static attn_inputs make_attn_inputs(const attn_shape & shape, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    attn_inputs inputs;

    const size_t q_elems = (size_t) shape.head_size * shape.n_tokens * shape.n_head;
    const size_t kv_elems = (size_t) shape.head_size * shape.n_kv * shape.n_head_kv;

    inputs.q_f32.resize(q_elems);
    std::vector<float> k_f32(kv_elems);
    std::vector<float> v_sem_f32(kv_elems);

    for (float & v : inputs.q_f32) {
        v = dist(rng);
    }
    for (float & v : k_f32) {
        v = dist(rng);
    }
    for (float & v : v_sem_f32) {
        v = dist(rng);
    }

    fp32_to_fp16(inputs.q_f32, inputs.q_f16);
    fp32_to_fp16(k_f32, inputs.k_f16);
    fp32_to_fp16(v_sem_f32, inputs.v_flash_f16);

    std::vector<float> v_mha_f32((size_t) shape.n_kv * shape.head_size * shape.n_head_kv);
    for (int64_t h = 0; h < shape.n_head_kv; ++h) {
        for (int64_t d = 0; d < shape.head_size; ++d) {
            for (int64_t kv = 0; kv < shape.n_kv; ++kv) {
                const size_t src = ((size_t) h * shape.n_kv + kv) * shape.head_size + d;
                const size_t dst = ((size_t) h * shape.head_size + d) * shape.n_kv + kv;
                v_mha_f32[dst] = v_sem_f32[src];
            }
        }
    }
    fp32_to_fp16(v_mha_f32, inputs.v_mha_f16);

    if (shape.causal_mask) {
        const int64_t n_mask_rows = GGML_PAD(shape.n_tokens, GGML_KQ_MASK_PAD);
        inputs.mask_f16.resize((size_t) shape.n_kv * n_mask_rows);

        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

        for (int64_t iq = 0; iq < n_mask_rows; ++iq) {
            const int64_t q_pos = shape.n_kv - shape.n_tokens + iq;
            for (int64_t ik = 0; ik < shape.n_kv; ++ik) {
                const bool valid_row = iq < shape.n_tokens;
                const bool visible = valid_row && ik <= q_pos;
                inputs.mask_f16[(size_t) iq * shape.n_kv + ik] = visible ? zero : neg_inf;
            }
        }
    }

    return inputs;
}

static void copy_q_tensor(ggml_tensor * q, const attn_inputs & inputs) {
    switch (q->type) {
        case GGML_TYPE_F32:
            std::memcpy(q->data, inputs.q_f32.data(), inputs.q_f32.size() * sizeof(float));
            break;
        case GGML_TYPE_F16:
            std::memcpy(q->data, inputs.q_f16.data(), inputs.q_f16.size() * sizeof(ggml_fp16_t));
            break;
        default:
            GGML_ABORT("unsupported q type in benchmark");
    }
}

static std::vector<float> tensor_to_fp32(const ggml_tensor * tensor) {
    std::vector<float> out((size_t) ggml_nelements(tensor));

    switch (tensor->type) {
        case GGML_TYPE_F32:
            std::memcpy(out.data(), tensor->data, out.size() * sizeof(float));
            break;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) tensor->data, out.data(), (int64_t) out.size());
            break;
        default:
            GGML_ABORT("unsupported output type in benchmark");
    }

    return out;
}

static size_t estimate_ctx_size(const attn_shape & shape) {
    const size_t q_bytes = sizeof(float) * (size_t) shape.head_size * shape.n_tokens * shape.n_head;
    const size_t k_bytes = sizeof(ggml_fp16_t) * (size_t) shape.head_size * shape.n_kv * shape.n_head_kv;
    const size_t v_bytes = sizeof(ggml_fp16_t) * (size_t) shape.head_size * shape.n_kv * shape.n_head_kv;
    const size_t mask_bytes = shape.causal_mask ? sizeof(ggml_fp16_t) * (size_t) shape.n_kv * GGML_PAD(shape.n_tokens, GGML_KQ_MASK_PAD) : 0;
    const size_t kq_bytes = sizeof(float) * (size_t) shape.n_kv * shape.n_tokens * shape.n_head;
    const size_t kqv_bytes = sizeof(float) * (size_t) shape.head_size * shape.n_tokens * shape.n_head;

    // The direct MHA path materializes KQ and then makes several view/permute/cont nodes on top of KQV.
    // Be generous here so large-token benchmarks don't fail due to context exhaustion.
    return 64 * 1024 * 1024 + q_bytes + k_bytes + v_bytes + mask_bytes + 3 * kq_bytes + 2 * kqv_bytes;
}

static bool prepare_plan(bench_graph & bench, int n_threads, ggml_threadpool * threadpool) {
    bench.gf = ggml_new_graph(bench.ctx.get());
    ggml_build_forward_expand(bench.gf, bench.out);
    bench.n_nodes = bench.gf->n_nodes;

    bench.plan = ggml_graph_plan(bench.gf, n_threads, threadpool);
    if (bench.plan.work_size > 0) {
        bench.work.size = bench.plan.work_size;
        bench.work.data = static_cast<uint8_t *>(ggml_aligned_malloc(bench.work.size));
        if (bench.work.data == nullptr) {
            std::fprintf(stderr, "ggml_aligned_malloc failed for %zu bytes\n", bench.work.size);
            return false;
        }
        bench.plan.work_data = bench.work.data;
    }

    return true;
}

static bench_graph make_flash_graph(
        const attn_shape & shape,
        const attn_inputs & inputs,
        ggml_type q_type,
        int n_threads,
        ggml_threadpool * threadpool) {
    bench_graph bench;

    const size_t ctx_size = estimate_ctx_size(shape);
    const ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    bench.ctx.reset(ggml_init(params));
    if (bench.ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed for flash graph\n");
        return bench;
    }

    ggml_tensor * q = ggml_new_tensor_3d(bench.ctx.get(), q_type, shape.head_size, shape.n_tokens, shape.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(bench.ctx.get(), GGML_TYPE_F16, shape.head_size, shape.n_kv, shape.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(bench.ctx.get(), GGML_TYPE_F16, shape.head_size, shape.n_kv, shape.n_head_kv);
    ggml_tensor * m = nullptr;

    if (shape.causal_mask) {
        m = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F16, shape.n_kv, GGML_PAD(shape.n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    }

    copy_q_tensor(q, inputs);
    std::memcpy(k->data, inputs.k_f16.data(), inputs.k_f16.size() * sizeof(ggml_fp16_t));
    std::memcpy(v->data, inputs.v_flash_f16.data(), inputs.v_flash_f16.size() * sizeof(ggml_fp16_t));
    if (m != nullptr) {
        std::memcpy(m->data, inputs.mask_f16.data(), inputs.mask_f16.size() * sizeof(ggml_fp16_t));
    }

    ggml_tensor * out = ggml_flash_attn_ext(
            bench.ctx.get(),
            q,
            k,
            v,
            m,
            1.0f / std::sqrt((float) shape.head_size),
            0.0f,
            0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    bench.out = ggml_reshape_2d(bench.ctx.get(), out, shape.head_size * shape.n_head, shape.n_tokens);

    if (!prepare_plan(bench, n_threads, threadpool)) {
        bench.ctx.reset();
    }

    return bench;
}

static bench_graph make_mha_graph(
        const attn_shape & shape,
        const attn_inputs & inputs,
        ggml_type q_type,
        int n_threads,
        ggml_threadpool * threadpool) {
    bench_graph bench;

    const size_t ctx_size = estimate_ctx_size(shape);
    const ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    bench.ctx.reset(ggml_init(params));
    if (bench.ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed for mha graph\n");
        return bench;
    }

    ggml_tensor * q = ggml_new_tensor_3d(bench.ctx.get(), q_type, shape.head_size, shape.n_tokens, shape.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(bench.ctx.get(), GGML_TYPE_F16, shape.head_size, shape.n_kv, shape.n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(bench.ctx.get(), GGML_TYPE_F16, shape.n_kv, shape.head_size, shape.n_head_kv);
    ggml_tensor * m = nullptr;

    if (shape.causal_mask) {
        m = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F16, shape.n_kv, GGML_PAD(shape.n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    }

    copy_q_tensor(q, inputs);
    std::memcpy(k->data, inputs.k_f16.data(), inputs.k_f16.size() * sizeof(ggml_fp16_t));
    std::memcpy(v->data, inputs.v_mha_f16.data(), inputs.v_mha_f16.size() * sizeof(ggml_fp16_t));
    if (m != nullptr) {
        std::memcpy(m->data, inputs.mask_f16.data(), inputs.mask_f16.size() * sizeof(ggml_fp16_t));
    }

    ggml_tensor * kq = ggml_mul_mat(bench.ctx.get(), k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(bench.ctx.get(), kq, m, 1.0f / std::sqrt((float) shape.head_size), 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(bench.ctx.get(), v, kq);
    ggml_tensor * kqv_merged = ggml_permute(bench.ctx.get(), kqv, 0, 2, 1, 3);
    bench.out = ggml_cont_2d(bench.ctx.get(), kqv_merged, shape.head_size * shape.n_head, shape.n_tokens);

    if (!prepare_plan(bench, n_threads, threadpool)) {
        bench.ctx.reset();
    }

    return bench;
}

static bool run_graph(bench_graph & bench) {
    const enum ggml_status status = ggml_graph_compute(bench.gf, &bench.plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_graph_compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }
    return true;
}

static bench_result benchmark_graph(bench_graph & bench, const bench_options & opts) {
    bench_result result;
    result.n_nodes = bench.n_nodes;

    for (int i = 0; i < opts.warmup; ++i) {
        if (!run_graph(bench)) {
            return result;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opts.iterations; ++i) {
        if (!run_graph(bench)) {
            return result;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    result.avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / opts.iterations;
    result.out_type = bench.out->type;
    result.output = tensor_to_fp32(bench.out);

    return result;
}

static bool compute_graph_output(bench_graph & bench, std::vector<float> & output, ggml_type * out_type = nullptr) {
    if (!run_graph(bench)) {
        return false;
    }

    output = tensor_to_fp32(bench.out);
    if (out_type != nullptr) {
        *out_type = bench.out->type;
    }

    return true;
}

static bool benchmark_case(
        const attn_shape & shape,
        const bench_options & opts,
        ggml_threadpool * threadpool) {
    std::mt19937 rng(20260405 + (uint32_t) shape.n_tokens + (uint32_t) shape.n_kv);
    const attn_inputs inputs = make_attn_inputs(shape, rng);

    bench_graph flash = make_flash_graph(shape, inputs, GGML_TYPE_F32, opts.threads, threadpool);
    bench_graph mha   = make_mha_graph(shape, inputs, GGML_TYPE_F32, opts.threads, threadpool);
    bench_graph flash_ref = make_flash_graph(shape, inputs, GGML_TYPE_F32, opts.threads, threadpool);
    bench_graph mha_ref   = make_mha_graph(shape, inputs, GGML_TYPE_F32, opts.threads, threadpool);

    if (flash.ctx == nullptr || mha.ctx == nullptr || flash_ref.ctx == nullptr || mha_ref.ctx == nullptr ||
            flash.out == nullptr || mha.out == nullptr || flash_ref.out == nullptr || mha_ref.out == nullptr) {
        return false;
    }

    const bench_result flash_result = benchmark_graph(flash, opts);
    const bench_result mha_result   = benchmark_graph(mha, opts);

    if (flash_result.output.empty() || mha_result.output.empty()) {
        return false;
    }

    std::vector<float> flash_ref_output;
    std::vector<float> mha_ref_output;
    ggml_type flash_ref_type = GGML_TYPE_COUNT;
    ggml_type mha_ref_type = GGML_TYPE_COUNT;

    if (!compute_graph_output(flash_ref, flash_ref_output, &flash_ref_type) ||
            !compute_graph_output(mha_ref, mha_ref_output, &mha_ref_type)) {
        return false;
    }

    size_t worst_flash_ref_idx = 0;
    size_t worst_mha_ref_idx = 0;
    size_t worst_cross_idx = 0;
    const float flash_ref_diff = max_abs_diff(flash_result.output, flash_ref_output, &worst_flash_ref_idx);
    const float mha_ref_diff = max_abs_diff(mha_result.output, mha_ref_output, &worst_mha_ref_idx);
    const double flash_ref_nmse = nmse(flash_result.output, flash_ref_output);
    const double mha_ref_nmse = nmse(mha_result.output, mha_ref_output);

    const float diff = max_abs_diff(flash_result.output, mha_result.output, &worst_cross_idx);
    const double diff_nmse = nmse(flash_result.output, mha_result.output);
    const double speedup = mha_result.avg_us / flash_result.avg_us;

    std::printf(
            "case=%s hs=%lld n_head=%lld n_head_kv=%lld n_tokens=%lld n_kv=%lld mask=%s threads=%d\n",
            shape.name,
            (long long) shape.head_size,
            (long long) shape.n_head,
            (long long) shape.n_head_kv,
            (long long) shape.n_tokens,
            (long long) shape.n_kv,
            shape.causal_mask ? "causal" : "none",
            opts.threads);
    std::printf("  flash_attn_f32q_f16kv: avg=%.2f us nodes=%d out=%s\n", flash_result.avg_us, flash_result.n_nodes, ggml_type_name(flash_result.out_type));
    std::printf("  direct_mha_f32q_f16kv: avg=%.2f us nodes=%d out=%s\n", mha_result.avg_us, mha_result.n_nodes, ggml_type_name(mha_result.out_type));
    std::printf("  flash_ref_f32: out=%s\n", ggml_type_name(flash_ref_type));
    std::printf("  direct_ref_f32: out=%s\n", ggml_type_name(mha_ref_type));
    std::printf("  speedup(mha/flash)=%.3fx\n", speedup);
    std::printf("  flash_vs_ref: max_abs_diff=%.8f nmse=%.8e worst_idx=%zu\n", flash_ref_diff, flash_ref_nmse, worst_flash_ref_idx);
    std::printf("  direct_vs_ref: max_abs_diff=%.8f nmse=%.8e worst_idx=%zu\n", mha_ref_diff, mha_ref_nmse, worst_mha_ref_idx);
    std::printf("  flash_vs_direct: max_abs_diff=%.8f nmse=%.8e worst_idx=%zu\n", diff, diff_nmse, worst_cross_idx);

    return true;
}

int main(int argc, char ** argv) {
    bench_options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return argc == 1 ? 0 : 1;
    }

    ggml_cpu_init();

    std::printf(
            "cpu_features: neon=%d fp16_va=%d dotprod=%d sve=%d\n",
            ggml_cpu_has_neon(),
            ggml_cpu_has_fp16_va(),
            ggml_cpu_has_dotprod(),
            ggml_cpu_has_sve());

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(opts.threads);
    ggml_threadpool_ptr threadpool(ggml_threadpool_new(&tpp));
    if (threadpool == nullptr) {
        std::fprintf(stderr, "ggml_threadpool_new failed\n");
        return 1;
    }

    const std::vector<attn_shape> cases = {
        { "qwen25_prefill_512", 64, 14, 2, 512, 512, true },
        { "qwen25_prefill_51",  64, 14, 2,  51, 563, true },
        { "qwen25_decode_1",    64, 14, 2,   1, 563, true },
    };

    bool ok = true;
    for (const attn_shape & shape : cases) {
        ok = benchmark_case(shape, opts, threadpool.get()) && ok;
    }

    return ok ? 0 : 1;
}
