#include <metal_stdlib>
using namespace metal;

// ============================================================================
// vision.metal — Ephemeral GPU kernels for Qwen 3.6 Vision Transformer.
//
// Loaded lazily via MetalContext.moduleLibrary(device:module:"vision") only
// when encoding an image or PDF, adding zero startup cost to text sessions.
// ============================================================================

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr int kVisTileM = 64;
constant constexpr int kVisTileN = 32;
constant constexpr int kVisTileK = 64;

constant constexpr uint kVisFlagBias        = 1u << 0;
constant constexpr uint kVisFlagGeluTanh    = 1u << 1;
constant constexpr uint kVisFlagGeluExact   = 1u << 2;
constant constexpr uint kVisFlagResidualAdd = 1u << 3;

static inline float vis_gelu_tanh(float v) {
    const float c0 = 0.7978845608028654f; // sqrt(2 / pi)
    const float c1 = 0.044715f;
    const float inner = clamp(c0 * (v + c1 * v * v * v), -20.0f, 20.0f);
    return 0.5f * v * (1.0f + tanh(inner));
}

// Abramowitz & Stegun 7.1.26 rational approximation (max abs error < 1.5e-7).
static inline float vis_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly =
        ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
          - 0.284496736f) * t + 0.254829592f) * t;
    const float y = 1.0f - poly * exp(-ax * ax);
    return sign * y;
}

static inline float vis_gelu_exact(float v) {
    return 0.5f * v * (1.0f + vis_erf(v * 0.7071067811865475f));
}

// Dense FP32-activation × BF16-weight GEMM on Apple10 matmul2d tensor ops
// with FP32 accumulation and fused optional bias / GELU / residual add.
// Both M, N, and K may be arbitrary positive integers (e.g. mlp K = 4304):
// out-of-bounds elements are zero-padded inside the threadgroup tiles.
kernel void mpp_vision_bf16_gemm_f32(
    device const float*  X     [[buffer(0)]],
    device const bfloat* W     [[buffer(1)]],
    device const bfloat* bias  [[buffer(2)]],
    device float*        Y     [[buffer(3)]],
    constant uint&       M     [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    constant uint&       K     [[buffer(6)]],
    constant uint&       flags [[buffer(7)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint3 lid3                 [[thread_position_in_threadgroup]],
    uint3 threads3             [[threads_per_threadgroup]]
) {
    constexpr auto descriptor = matmul2d_descriptor(
        kVisTileM, kVisTileN, kVisTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using threadgroup_half_tensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half actTile[kVisTileM * kVisTileK];
    threadgroup half weightTile[kVisTileN * kVisTileK];

    threadgroup_half_tensor tileA(
        actTile,
        dextents<int32_t, 2>(kVisTileK, kVisTileM),
        array<int32_t, 2>({1, kVisTileK}));
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kVisTileK, kVisTileN),
        array<int32_t, 2>({1, kVisTileK}));

    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(tileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(tileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint kBase = 0; kBase < K; kBase += uint(kVisTileK)) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kVisTileM * kVisTileK);
             linear += threads) {
            const uint localM = linear / uint(kVisTileK);
            const uint localK = linear % uint(kVisTileK);
            const uint globalM = tgid.y * uint(kVisTileM) + localM;
            const uint globalK = kBase + localK;
            if (globalM < M && globalK < K) {
                actTile[linear] = half(clamp(X[globalM * K + globalK], -65000.0f, 65000.0f));
            } else {
                actTile[linear] = half(0.0f);
            }
        }
        for (uint linear = lid;
             linear < uint(kVisTileN * kVisTileK);
             linear += threads) {
            const uint localN = linear / uint(kVisTileK);
            const uint localK = linear % uint(kVisTileK);
            const uint globalN = tgid.x * uint(kVisTileN) + localN;
            const uint globalK = kBase + localK;
            if (globalN < N && globalK < K) {
                weightTile[linear] = half(float(W[globalN * K + globalK]));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const bool hasBias = (flags & kVisFlagBias) != 0u;
    const bool geluTanh = (flags & kVisFlagGeluTanh) != 0u;
    const bool geluExact = (flags & kVisFlagGeluExact) != 0u;
    const bool residualAdd = (flags & kVisFlagResidualAdd) != 0u;

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kVisTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kVisTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            float v = accumulator[element];
            if (hasBias) {
                v += float(bias[globalN]);
            }
            if (geluTanh) {
                v = vis_gelu_tanh(v);
            } else if (geluExact) {
                v = vis_gelu_exact(v);
            }
            const uint outIdx = globalM * N + globalN;
            if (residualAdd) {
                Y[outIdx] += v;
            } else {
                Y[outIdx] = v;
            }
        }
    }
}
#endif // defined(__HAVE_TENSOR__)

// LayerNorm over FP32 [rows, dim] with BF16 weight/bias, eps = 1e-6.
// One 256-threadgroup per row.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void vision_layernorm_f32(
    device const float*  x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       rows   [[buffer(4)]],
    constant uint&       dim    [[buffer(5)]],
    constant float&      eps    [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    if (row >= rows) return;

    threadgroup float partial[8];
    const uint numSimd = (threads_per_tg + 31u) / 32u;
    device const float* rowIn = x + row * dim;
    device float* rowOut = out + row * dim;

    float sum = 0.0f;
    for (uint i = tid; i < dim; i += threads_per_tg) {
        sum += rowIn[i];
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float total = 0.0f;
        for (uint s = 0u; s < numSimd; ++s) total += partial[s];
        partial[0] = total / float(dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0];

    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += threads_per_tg) {
        const float d = rowIn[i] - mean;
        sumSq = fma(d, d, sumSq);
    }
    sumSq = simd_sum(sumSq);
    if (lane == 0u) partial[simd_idx] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float totalSq = 0.0f;
        for (uint s = 0u; s < numSimd; ++s) totalSq += partial[s];
        partial[0] = rsqrt(totalSq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invStd = partial[0];

    for (uint i = tid; i < dim; i += threads_per_tg) {
        const float normed = (rowIn[i] - mean) * invStd;
        rowOut[i] = fma(normed, float(weight[i]), float(bias[i]));
    }
}

// In-place 2D Rotary Embedding on Q and K slices of qkv [seqLen, 3 * numHeads * headDim].
kernel void vision_rope_2d_f32(
    device float*        qkv      [[buffer(0)]],
    device const float*  cosTable [[buffer(1)]],
    device const float*  sinTable [[buffer(2)]],
    constant uint&       seqLen   [[buffer(3)]],
    constant uint&       numHeads [[buffer(4)]],
    constant uint&       headDim  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint halfD = headDim / 2u;
    const uint totalCols = numHeads * halfD;
    const uint idx = gid.x;
    const uint s = gid.y;
    if (idx >= totalCols || s >= seqLen) return;

    const uint h = idx / halfD;
    const uint j = idx % halfD;
    const uint d = numHeads * headDim;
    const uint qkvStride = 3u * d;

    const float cs = cosTable[s * halfD + j];
    const float sn = sinTable[s * halfD + j];

    device float* qh = qkv + s * qkvStride + h * headDim;
    const float q0 = qh[j];
    const float q1 = qh[j + halfD];
    qh[j] = q0 * cs - q1 * sn;
    qh[j + halfD] = q1 * cs + q0 * sn;

    device float* kh = qh + d;
    const float k0 = kh[j];
    const float k1 = kh[j + halfD];
    kh[j] = k0 * cs - k1 * sn;
    kh[j + halfD] = k1 * cs + k0 * sn;
}

// Bidirectional multi-head FP32 attention over qkv [seqLen, 3 * d] -> out [seqLen, d].
// Uses online softmax per simdgroup (4 simdgroups / 128 threads per (qRow, head))
// to avoid allocating an O(seqLen^2) score matrix. Supports headDim <= 128.
[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void vision_attention_f32(
    device const float* qkv      [[buffer(0)]],
    device float*       out      [[buffer(1)]],
    constant uint&      seqLen   [[buffer(2)]],
    constant uint&      numHeads [[buffer(3)]],
    constant uint&      headDim  [[buffer(4)]],
    constant float&     scale    [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint qRow = tg.x;
    const uint h = tg.y;
    if (qRow >= seqLen || h >= numHeads || headDim > 128u) return;

    threadgroup float tg_m[4];
    threadgroup float tg_l[4];
    threadgroup float tg_acc[4 * 128];

    const uint d = numHeads * headDim;
    const uint qkvStride = 3u * d;
    const uint qBase = qRow * qkvStride + h * headDim;

    float q_reg[4];
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma clang loop unroll(full)
    for (uint e = 0u; e < 4u; ++e) {
        const uint c = e * 32u + lane;
        q_reg[e] = (c < headDim) ? qkv[qBase + c] : 0.0f;
    }

    float m_sg = -INFINITY;
    float l_sg = 0.0f;

    for (uint kRow = simd_group; kRow < seqLen; kRow += 4u) {
        const uint kvBase = kRow * qkvStride + d + h * headDim;
        float dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint e = 0u; e < 4u; ++e) {
            const uint c = e * 32u + lane;
            const float kv = (c < headDim) ? qkv[kvBase + c] : 0.0f;
            dot = fma(q_reg[e], kv, dot);
        }
        const float score = simd_sum(dot) * scale;
        const float m_new = fmax(m_sg, score);
        const float alpha = (l_sg > 0.0f) ? exp(m_sg - m_new) : 0.0f;
        const float p = exp(score - m_new);
        m_sg = m_new;
        l_sg = fma(l_sg, alpha, p);

        const uint vBase = kvBase + d;
        #pragma clang loop unroll(full)
        for (uint e = 0u; e < 4u; ++e) {
            const uint c = e * 32u + lane;
            const float vv = (c < headDim) ? qkv[vBase + c] : 0.0f;
            acc[e] = fma(acc[e], alpha, p * vv);
        }
    }

    if (lane == 0u) {
        tg_m[simd_group] = m_sg;
        tg_l[simd_group] = l_sg;
    }
    #pragma clang loop unroll(full)
    for (uint e = 0u; e < 4u; ++e) {
        const uint c = e * 32u + lane;
        tg_acc[simd_group * 128u + c] = acc[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < headDim) {
        float m_all = -INFINITY;
        for (uint sg = 0u; sg < 4u; ++sg) {
            if (tg_l[sg] > 0.0f) {
                m_all = fmax(m_all, tg_m[sg]);
            }
        }
        float l_all = 0.0f;
        float outVal = 0.0f;
        for (uint sg = 0u; sg < 4u; ++sg) {
            if (tg_l[sg] > 0.0f) {
                const float w = exp(tg_m[sg] - m_all);
                l_all += tg_l[sg] * w;
                outVal += tg_acc[sg * 128u + tid] * w;
            }
        }
        out[qRow * d + h * headDim + tid] = (l_all > 0.0f) ? (outVal / l_all) : 0.0f;
    }
}
