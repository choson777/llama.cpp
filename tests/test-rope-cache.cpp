#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());

    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());

    double mse = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = a[i];
        const double db = b[i];
        mse += (da - db) * (da - db);
        ref += da * da;
    }

    return ref > 0.0 ? mse / ref : mse;
}

template<typename T>
static std::vector<float> to_f32_vector(const std::vector<T> & src);

template<>
std::vector<float> to_f32_vector<float>(const std::vector<float> & src) {
    return src;
}

template<>
std::vector<float> to_f32_vector<ggml_fp16_t>(const std::vector<ggml_fp16_t> & src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = ggml_fp16_to_fp32(src[i]);
    }
    return dst;
}

template<typename T>
static void from_f32_vector(const std::vector<float> & src, std::vector<T> & dst);

template<>
void from_f32_vector<float>(const std::vector<float> & src, std::vector<float> & dst) {
    dst = src;
}

template<>
void from_f32_vector<ggml_fp16_t>(const std::vector<float> & src, std::vector<ggml_fp16_t> & dst) {
    dst.resize(src.size());
    ggml_fp32_to_fp16_row(src.data(), dst.data(), (int64_t) src.size());
}

template<typename T>
static std::vector<T> run_rope_once(
        const std::vector<T> & input,
        const std::vector<int32_t> & pos,
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t n_dims,
        int mode,
        int n_ctx,
        bool use_cache) {
    const int n_ctx_orig = 4096;
    const float freq_base = 10000.0f;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;

    if (use_cache) {
        ggml_rope_cache_prepare_normal(
                n_ctx,
                (int32_t) n_dims,
                freq_base,
                freq_scale,
                n_ctx_orig,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
    } else {
        ggml_rope_cache_clear_normal();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * x = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne2);
    ggml_tensor * r = ggml_rope_ext(
            ctx, x, p, nullptr,
            (int) n_dims, mode, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    std::memcpy(x->data, input.data(), input.size() * sizeof(T));
    std::memcpy(p->data, pos.data(), pos.size() * sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    const ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, 4);
    assert(status == GGML_STATUS_SUCCESS);

    std::vector<T> out(input.size());
    std::memcpy(out.data(), r->data, out.size() * sizeof(T));

    ggml_free(ctx);
    ggml_rope_cache_clear_normal();

    return out;
}

template<typename T>
static bool run_case(
        const char * label,
        ggml_type type,
        int mode,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t n_dims,
        int n_ctx,
        std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> input_f32((size_t) ne0 * (size_t) ne1 * (size_t) ne2);
    for (float & v : input_f32) {
        v = dist(rng);
    }

    std::vector<T> input;
    from_f32_vector(input_f32, input);

    std::vector<int32_t> pos((size_t) ne2);
    const int32_t values[] = { 0, 1, 7, 31, 127, 255 };
    for (int64_t i = 0; i < ne2; ++i) {
        pos[(size_t) i] = values[i % (int64_t) (sizeof(values) / sizeof(values[0]))] % n_ctx;
    }

    const std::vector<T> out_ref = run_rope_once(input, pos, type, ne0, ne1, ne2, n_dims, mode, n_ctx, false);
    const std::vector<T> out_new = run_rope_once(input, pos, type, ne0, ne1, ne2, n_dims, mode, n_ctx, true);

    const bool bytes_equal = std::memcmp(out_ref.data(), out_new.data(), out_ref.size() * sizeof(T)) == 0;
    const std::vector<float> out_ref_f32 = to_f32_vector(out_ref);
    const std::vector<float> out_new_f32 = to_f32_vector(out_new);
    const float worst = max_abs_diff(out_ref_f32, out_new_f32);
    const double err = nmse(out_ref_f32, out_new_f32);

    std::printf(
            "rope cache test %-18s type=%-3s mode=%d ne0=%lld ne1=%lld ne2=%lld n_dims=%lld bytes_equal=%d max_abs_diff=%.9g nmse=%.9g\n",
            label,
            ggml_type_name(type),
            mode,
            (long long) ne0,
            (long long) ne1,
            (long long) ne2,
            (long long) n_dims,
            bytes_equal ? 1 : 0,
            worst,
            err);

    return bytes_equal && worst == 0.0f && err == 0.0;
}

int main() {
    ggml_cpu_init();

    std::mt19937 rng(20260331);

    const bool ok =
            run_case<float>("plain-f32", GGML_TYPE_F32, 0, 12, 3, 5,  8, 256, rng) &&
            run_case<float>("neox-f32",  GGML_TYPE_F32, GGML_ROPE_TYPE_NEOX, 32, 2, 6, 32, 256, rng) &&
            run_case<ggml_fp16_t>("plain-f16", GGML_TYPE_F16, 0, 12, 3, 5,  8, 256, rng) &&
            run_case<ggml_fp16_t>("neox-f16",  GGML_TYPE_F16, GGML_ROPE_TYPE_NEOX, 32, 2, 6, 32, 256, rng);

    if (!ok) {
        std::fprintf(stderr, "rope cache test failed\n");
        return 1;
    }

    std::printf("rope cache test passed\n");
    return 0;
}
