#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static uint16_t fp16_bits(const ggml_fp16_t x) {
    uint16_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static bool fp16_close_ulp(const ggml_fp16_t a, const ggml_fp16_t b, const uint16_t max_ulp_diff = 1) {
    if (a == b) {
        return true;
    }

    const float af = ggml_fp16_to_fp32(a);
    const float bf = ggml_fp16_to_fp32(b);

    if (std::isnan(af) || std::isnan(bf)) {
        return false;
    }

    if (std::signbit(af) != std::signbit(bf)) {
        return false;
    }

    const uint16_t ua = fp16_bits(a);
    const uint16_t ub = fp16_bits(b);
    const uint16_t diff = ua > ub ? ua - ub : ub - ua;
    return diff <= max_ulp_diff;
}

static bool run_case(
        const int64_t n_cols,
        const int64_t n_rows,
        const float eps,
        std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

    std::vector<float> input_f32(n_cols * n_rows);
    for (float & v : input_f32) {
        v = dist(rng);
    }

    std::vector<ggml_fp16_t> input_f16(input_f32.size());
    ggml_fp32_to_fp16_row(input_f32.data(), input_f16.data(), (int64_t) input_f32.size());

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

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cols, n_rows);
    struct ggml_tensor * out = ggml_rms_norm(ctx, inp, eps);

    std::memcpy(inp->data, input_f16.data(), input_f16.size() * sizeof(ggml_fp16_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, 4);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_graph_compute_with_ctx failed: %d\n", (int) status);
        ggml_free(ctx);
        return false;
    }

    if (out->type != GGML_TYPE_F16) {
        std::fprintf(stderr, "unexpected output type: %s\n", ggml_type_name(out->type));
        ggml_free(ctx);
        return false;
    }

    std::vector<float> expected_f32(n_cols);
    std::vector<ggml_fp16_t> expected_f16(n_cols);

    for (int64_t row = 0; row < n_rows; ++row) {
        const ggml_fp16_t * x = input_f16.data() + row * n_cols;
        float sum = 0.0f;
        for (int64_t col = 0; col < n_cols; ++col) {
            const float xv = ggml_fp16_to_fp32(x[col]);
            sum += xv * xv;
        }

        const float mean = sum / n_cols;
        const float scale = 1.0f / std::sqrt(mean + eps);

        for (int64_t col = 0; col < n_cols; ++col) {
            expected_f32[col] = ggml_fp16_to_fp32(x[col]) * scale;
        }
        ggml_fp32_to_fp16_row(expected_f32.data(), expected_f16.data(), n_cols);

        const ggml_fp16_t * actual = reinterpret_cast<const ggml_fp16_t *>(
                static_cast<const char *>(out->data) + row * out->nb[1]);

        for (int64_t col = 0; col < n_cols; ++col) {
            if (!fp16_close_ulp(actual[col], expected_f16[col])) {
                std::fprintf(stderr,
                        "mismatch at row=%lld col=%lld: got=%f expected=%f bits=(0x%04x vs 0x%04x)\n",
                        (long long) row,
                        (long long) col,
                        ggml_fp16_to_fp32(actual[col]),
                        ggml_fp16_to_fp32(expected_f16[col]),
                        fp16_bits(actual[col]),
                        fp16_bits(expected_f16[col]));
                ggml_free(ctx);
                return false;
            }
        }
    }

    ggml_free(ctx);
    return true;
}

static void rms_norm_f16_scalar_ref(
        const ggml_fp16_t * x,
        ggml_fp16_t * y,
        const int64_t n_cols,
        const int64_t n_rows,
        const float eps) {
    for (int64_t row = 0; row < n_rows; ++row) {
        const ggml_fp16_t * xr = x + row * n_cols;
        ggml_fp16_t * yr = y + row * n_cols;

        float sum = 0.0f;
        for (int64_t col = 0; col < n_cols; ++col) {
            const float xv = ggml_fp16_to_fp32(xr[col]);
            sum += xv * xv;
        }

        const float mean = sum / n_cols;
        const float scale = 1.0f / std::sqrt(mean + eps);

        for (int64_t col = 0; col < n_cols; ++col) {
            yr[col] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(xr[col]) * scale);
        }
    }
}

static bool benchmark_rms_norm_f16(std::mt19937 & rng) {
    constexpr int64_t n_cols = 896;
    constexpr int64_t n_rows = 4096;
    constexpr int iters = 200;
    constexpr float eps = 1e-5f;

    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

    std::vector<float> input_f32(n_cols * n_rows);
    for (float & v : input_f32) {
        v = dist(rng);
    }

    std::vector<ggml_fp16_t> input_f16(input_f32.size());
    ggml_fp32_to_fp16_row(input_f32.data(), input_f16.data(), (int64_t) input_f32.size());

    std::vector<ggml_fp16_t> ref_out(input_f16.size());

    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        rms_norm_f16_scalar_ref(input_f16.data(), ref_out.data(), n_cols, n_rows, eps);
    }
    const auto t1 = std::chrono::steady_clock::now();

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed in benchmark\n");
        return false;
    }

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cols, n_rows);
    struct ggml_tensor * out = ggml_rms_norm(ctx, inp, eps);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::memcpy(inp->data, input_f16.data(), input_f16.size() * sizeof(ggml_fp16_t));

    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        std::memcpy(inp->data, input_f16.data(), input_f16.size() * sizeof(ggml_fp16_t));
        const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, 4);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "ggml_graph_compute_with_ctx failed in benchmark: %d\n", (int) status);
            ggml_free(ctx);
            return false;
        }
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto * actual = reinterpret_cast<const ggml_fp16_t *>(out->data);
    for (size_t i = 0; i < ref_out.size(); ++i) {
        if (!fp16_close_ulp(actual[i], ref_out[i])) {
            std::fprintf(stderr, "benchmark mismatch at idx=%zu: got=%f expected=%f\n",
                    i,
                    ggml_fp16_to_fp32(actual[i]),
                    ggml_fp16_to_fp32(ref_out[i]));
            ggml_free(ctx);
            return false;
        }
    }

    ggml_free(ctx);

    const double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double opt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double speedup = ref_ms / opt_ms;

    std::printf("benchmark rms_norm_f16: scalar=%.3f ms neon=%.3f ms speedup=%.2fx\n", ref_ms, opt_ms, speedup);
    return true;
}

int main() {
    ggml_cpu_init();

    std::mt19937 rng(20260329);

    const bool ok =
            run_case(32,  7, 1e-5f, rng) &&
            run_case(64, 11, 1e-5f, rng) &&
            run_case(128, 5, 1e-6f, rng) &&
            benchmark_rms_norm_f16(rng);

    if (!ok) {
        return 1;
    }

    std::printf("test-rms-norm-f16: ok\n");
    return 0;
}
