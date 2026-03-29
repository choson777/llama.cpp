#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void dequantize_row_q4_0_f16_scalar_ref(
        const block_q4_0 * x,
        ggml_fp16_t * y,
        int64_t k) {
    const int qk = QK4_0;
    const int nb = (int) (k / qk);

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = ggml_fp32_to_fp16(x0 * d);
            y[i*qk + j + qk/2] = ggml_fp32_to_fp16(x1 * d);
        }
    }
}

static bool benchmark_dequantize_q4_0_f16(std::mt19937 & rng) {
    constexpr int64_t n_embd = 896;
    constexpr int64_t n_rows = 4096;
    constexpr int iters = 200;

    std::uniform_real_distribution<float> weight_dist(-8.0f, 8.0f);

    std::vector<float> src_f32(n_embd * n_rows);
    for (float & v : src_f32) {
        v = weight_dist(rng);
    }

    const size_t row_size_q4_0 = ggml_row_size(GGML_TYPE_Q4_0, n_embd);
    std::vector<uint8_t> src_q4_0(row_size_q4_0 * n_rows);
    for (int64_t row = 0; row < n_rows; ++row) {
        quantize_row_q4_0_ref(
                src_f32.data() + row * n_embd,
                reinterpret_cast<block_q4_0 *>(src_q4_0.data() + row * row_size_q4_0),
                n_embd);
    }

    std::vector<ggml_fp16_t> dst_ref(n_embd * n_rows);
    std::vector<ggml_fp16_t> dst_opt(n_embd * n_rows);

    const auto * src_blocks = reinterpret_cast<const block_q4_0 *>(src_q4_0.data());

    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        dequantize_row_q4_0_f16_scalar_ref(src_blocks, dst_ref.data(), n_embd * n_rows);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        dequantize_row_q4_0_f16(src_blocks, dst_opt.data(), n_embd * n_rows);
    }
    const auto t3 = std::chrono::steady_clock::now();

    if (std::memcmp(dst_ref.data(), dst_opt.data(), dst_ref.size() * sizeof(ggml_fp16_t)) != 0) {
        std::fprintf(stderr, "benchmark mismatch between scalar and optimized dequantize\n");
        return false;
    }

    const double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double opt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double speedup = ref_ms / opt_ms;

    std::printf("benchmark q4_0->f16: scalar=%.3f ms opt=%.3f ms speedup=%.2fx\n", ref_ms, opt_ms, speedup);
    return true;
}

static bool run_case(
        const int64_t n_embd,
        const int64_t n_vocab,
        const int64_t n_tokens,
        std::mt19937 & rng) {
    assert(n_embd % QK4_0 == 0);

    std::uniform_real_distribution<float> weight_dist(-8.0f, 8.0f);
    std::uniform_int_distribution<int32_t> token_dist(0, n_vocab - 1);

    std::vector<float> embd_f32(n_embd * n_vocab);
    for (float & v : embd_f32) {
        v = weight_dist(rng);
    }

    const size_t row_size_q4_0 = ggml_row_size(GGML_TYPE_Q4_0, n_embd);
    std::vector<uint8_t> embd_q4_0(row_size_q4_0 * n_vocab);
    for (int64_t row = 0; row < n_vocab; ++row) {
        quantize_row_q4_0_ref(
                embd_f32.data() + row * n_embd,
                reinterpret_cast<block_q4_0 *>(embd_q4_0.data() + row * row_size_q4_0),
                n_embd);
    }

    std::vector<int32_t> tokens(n_tokens);
    for (int32_t & token : tokens) {
        token = token_dist(rng);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    struct ggml_tensor * embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_vocab);
    struct ggml_tensor * inp  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * out  = ggml_get_rows_ext(ctx, embd, inp, GGML_TYPE_F16);

    std::memcpy(embd->data, embd_q4_0.data(), embd_q4_0.size());
    std::memcpy(inp->data, tokens.data(), tokens.size() * sizeof(int32_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    if (out->type != GGML_TYPE_F16) {
        std::fprintf(stderr, "unexpected output type: %s\n", ggml_type_name(out->type));
        ggml_free(ctx);
        return false;
    }

    std::vector<float> expected_f32(n_embd);
    std::vector<ggml_fp16_t> expected_f16(n_embd);

    for (int64_t row = 0; row < n_tokens; ++row) {
        const int32_t token = tokens[row];
        dequantize_row_q4_0(
                reinterpret_cast<const block_q4_0 *>(embd_q4_0.data() + token * row_size_q4_0),
                expected_f32.data(),
                n_embd);
        ggml_fp32_to_fp16_row(expected_f32.data(), expected_f16.data(), n_embd);

        const auto * actual = reinterpret_cast<const ggml_fp16_t *>(
                static_cast<const char *>(out->data) + row * out->nb[1]);

        for (int64_t col = 0; col < n_embd; ++col) {
            if (actual[col] != expected_f16[col]) {
                std::fprintf(stderr,
                        "mismatch at token_row=%lld token=%d col=%lld: got=%f expected=%f\n",
                        (long long) row,
                        token,
                        (long long) col,
                        ggml_fp16_to_fp32(actual[col]),
                        ggml_fp16_to_fp32(expected_f16[col]));
                ggml_free(ctx);
                return false;
            }
        }
    }

    ggml_free(ctx);
    return true;
}

int main() {
    ggml_cpu_init();
    ggml_quantize_init(GGML_TYPE_Q4_0);

    std::mt19937 rng(12345);

    const bool ok =
            run_case(32,  17, 11, rng) &&
            run_case(64,  53, 19, rng) &&
            run_case(128, 97, 23, rng) &&
            benchmark_dequantize_q4_0_f16(rng);

    if (!ok) {
        return 1;
    }

    std::printf("test-get-rows-q4-f16: ok\n");
    return 0;
}
