#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

struct ggml_ctx_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct ggml_backend_deleter {
    void operator()(ggml_backend * backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct ggml_buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using ggml_ctx_ptr = std::unique_ptr<ggml_context, ggml_ctx_deleter>;
using ggml_backend_ptr = std::unique_ptr<ggml_backend, ggml_backend_deleter>;
using ggml_buffer_ptr = std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter>;

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

    if (worst_idx) {
        *worst_idx = idx;
    }
    return worst;
}

static void fp32_to_fp16_vector(const std::vector<float> & src, std::vector<ggml_fp16_t> & dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }
}

static std::vector<float> fp16_to_fp32_vector(const std::vector<ggml_fp16_t> & src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = ggml_fp16_to_fp32(src[i]);
    }
    return dst;
}

static ggml_buffer_ptr alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (buffer == nullptr) {
        return ggml_buffer_ptr(nullptr);
    }

    const enum ggml_status status = ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        return ggml_buffer_ptr(nullptr);
    }

    return ggml_buffer_ptr(buffer);
}

static ggml_backend_buffer_type_t get_cpu_aarch64_buft(ggml_backend_t backend) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));

    if (get_extra_bufts == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_type_t * bufts = get_extra_bufts(device);
    if (bufts == nullptr) {
        return nullptr;
    }

    for (int i = 0; bufts[i] != nullptr; ++i) {
        if (std::strcmp(ggml_backend_buft_name(bufts[i]), "CPU_AARCH64") == 0) {
            return bufts[i];
        }
    }

    return nullptr;
}

static std::vector<uint8_t> quantize_q4_0_weights(const std::vector<float> & weights_f32, int64_t k, int64_t m) {
    const size_t row_size_q4_0 = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<uint8_t> weights_q4_0(row_size_q4_0 * m);

    for (int64_t row = 0; row < m; ++row) {
        quantize_row_q4_0_ref(
                weights_f32.data() + row * k,
                reinterpret_cast<block_q4_0 *>(weights_q4_0.data() + row * row_size_q4_0),
                k);
    }

    return weights_q4_0;
}

static std::vector<float> dequantize_q4_0_weights(const std::vector<uint8_t> & weights_q4_0, int64_t k, int64_t m) {
    const size_t row_size_q4_0 = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<float> weights_f32(k * m);

    for (int64_t row = 0; row < m; ++row) {
        dequantize_row_q4_0(
                reinterpret_cast<const block_q4_0 *>(weights_q4_0.data() + row * row_size_q4_0),
                weights_f32.data() + row * k,
                k);
    }

    return weights_f32;
}

static std::vector<float> matmul_ref_q4_f16_input(
        const std::vector<uint8_t> & weights_q4_0,
        const std::vector<ggml_fp16_t> & input_f16,
        int64_t k,
        int64_t m,
        int64_t n) {
    const std::vector<float> weights_f32 = dequantize_q4_0_weights(weights_q4_0, k, m);
    const std::vector<float> input_f32 = fp16_to_fp32_vector(input_f16);

    std::vector<float> out_f32(m * n, 0.0f);
    for (int64_t row_in = 0; row_in < n; ++row_in) {
        const float * x = input_f32.data() + row_in * k;
        float * y = out_f32.data() + row_in * m;

        for (int64_t row_w = 0; row_w < m; ++row_w) {
            const float * w = weights_f32.data() + row_w * k;
            float acc = 0.0f;
            for (int64_t col = 0; col < k; ++col) {
                acc += w[col] * x[col];
            }
            y[row_w] = acc;
        }
    }

    return out_f32;
}

template<typename T>
static bool run_mul_mat_case(
        const std::vector<uint8_t> & weights_q4_0,
        const std::vector<T> & input,
        const ggml_type input_type,
        const ggml_type output_type,
        const int64_t k,
        const int64_t m,
        const int64_t n,
        std::vector<T> & out,
        bool & supports_op) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_ctx_ptr ctx(ggml_init(params));
    if (ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, k, m);
    ggml_tensor * inp    = ggml_new_tensor_2d(ctx.get(), input_type, k, n);
    ggml_tensor * out_t  = ggml_mul_mat_ext(ctx.get(), weight, inp, output_type);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    if (backend == nullptr) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return false;
    }

    ggml_backend_buffer_type_t weight_buft = get_cpu_aarch64_buft(backend.get());
    if (weight_buft == nullptr) {
        std::fprintf(stderr, "CPU_AARCH64 buffer type unavailable\n");
        return false;
    }

    ggml_buffer_ptr weight_buf = alloc_tensor_buffer(weight_buft, weight);
    ggml_buffer_ptr input_buf  = alloc_tensor_buffer(ggml_backend_cpu_buffer_type(), inp);
    ggml_buffer_ptr out_buf    = alloc_tensor_buffer(ggml_backend_cpu_buffer_type(), out_t);

    if (weight_buf == nullptr || input_buf == nullptr || out_buf == nullptr) {
        std::fprintf(stderr, "tensor buffer allocation failed\n");
        return false;
    }

    ggml_backend_tensor_set(weight, weights_q4_0.data(), 0, weights_q4_0.size());
    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(T));

    if (ggml_backend_buffer_get_type(weight->buffer) != weight_buft) {
        std::fprintf(stderr, "weight tensor is not allocated in CPU_AARCH64 buffer\n");
        return false;
    }

    supports_op = ggml_backend_supports_op(backend.get(), out_t);
    if (!supports_op) {
        std::fprintf(stderr,
                "backend does not support op: src0=%s src1=%s dst=%s\n",
                ggml_type_name(weight->type),
                ggml_type_name(inp->type),
                ggml_type_name(out_t->type));
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out_t);

    const enum ggml_status status = ggml_backend_graph_compute(backend.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    out.resize(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * sizeof(T));
    return true;
}

static bool run_case(int64_t k, int64_t m, int64_t n, std::mt19937 & rng) {
#if !defined(__aarch64__)
    GGML_UNUSED(k);
    GGML_UNUSED(m);
    GGML_UNUSED(n);
    GGML_UNUSED(rng);
    std::printf("test-mul-mat-q4-f16: skipped (not aarch64)\n");
    return true;
#else
    std::uniform_real_distribution<float> dist_w(-8.0f, 8.0f);
    std::uniform_real_distribution<float> dist_x(-6.0f, 6.0f);

    std::vector<float> weights_f32(k * m);
    for (float & v : weights_f32) {
        v = dist_w(rng);
    }
    const std::vector<uint8_t> weights_q4_0 = quantize_q4_0_weights(weights_f32, k, m);

    std::vector<float> input_src_f32(k * n);
    for (float & v : input_src_f32) {
        v = dist_x(rng);
    }

    std::vector<ggml_fp16_t> input_f16;
    fp32_to_fp16_vector(input_src_f32, input_f16);

    const std::vector<float> input_f32_same_values = fp16_to_fp32_vector(input_f16);

    std::vector<ggml_fp16_t> out_f16;
    bool supports_f16 = false;
    if (!run_mul_mat_case(weights_q4_0, input_f16, GGML_TYPE_F16, GGML_TYPE_F16, k, m, n, out_f16, supports_f16)) {
        return false;
    }

    std::vector<float> out_f32;
    bool supports_f32 = false;
    if (!run_mul_mat_case(weights_q4_0, input_f32_same_values, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n, out_f32, supports_f32)) {
        return false;
    }

    if (!supports_f16 || !supports_f32) {
        std::fprintf(stderr, "AArch64 mul_mat path was not accepted for this case\n");
        return false;
    }

    const std::vector<float> out_f16_as_f32 = fp16_to_fp32_vector(out_f16);
    const std::vector<float> out_ref_math = matmul_ref_q4_f16_input(weights_q4_0, input_f16, k, m, n);

    const double ref_nmse = nmse(out_ref_math, out_f32);
    const double f16_nmse = nmse(out_f32, out_f16_as_f32);

    size_t worst_ref_idx = 0;
    size_t worst_f16_idx = 0;
    const float ref_max_abs = max_abs_diff(out_ref_math, out_f32, &worst_ref_idx);
    const float f16_max_abs = max_abs_diff(out_f32, out_f16_as_f32, &worst_f16_idx);

    if (ref_nmse > 5e-4) {
        std::fprintf(stderr,
                "mul_mat q4->f32 mismatch: k=%lld m=%lld n=%lld nmse=%g max_abs=%g worst_idx=%zu ref=%f got=%f\n",
                (long long) k,
                (long long) m,
                (long long) n,
                ref_nmse,
                ref_max_abs,
                worst_ref_idx,
                out_ref_math[worst_ref_idx],
                out_f32[worst_ref_idx]);
        return false;
    }

    if (f16_nmse > 5e-5) {
        std::fprintf(stderr,
                "mul_mat f16 path mismatch: k=%lld m=%lld n=%lld nmse=%g max_abs=%g worst_idx=%zu ref=%f got=%f\n",
                (long long) k,
                (long long) m,
                (long long) n,
                f16_nmse,
                f16_max_abs,
                worst_f16_idx,
                out_f32[worst_f16_idx],
                out_f16_as_f32[worst_f16_idx]);
        return false;
    }

    return true;
#endif
}

int main() {
    ggml_cpu_init();
    ggml_quantize_init(GGML_TYPE_Q4_0);

    std::mt19937 rng(20260331);

    const bool ok =
            run_case(128, 8, 1, rng) &&
            run_case(128, 8, 5, rng) &&
            run_case(256, 12, 4, rng);

    if (!ok) {
        return 1;
    }

    std::printf("test-mul-mat-q4-f16: ok\n");
    return 0;
}
