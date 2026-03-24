// moe_fused_kernel.cpp
// Composable Kernel extension for Stage 1+2 fused MoE
//
// This file implements the CK kernel for fused MoE with LDS intermediate caching.
// It requires AMD ROCm and Composable Kernel to build.
//
// Build instructions:
//   1. Ensure ROCm and CK are installed
//   2. Build with:
//      hipcc -I/path/to/ck/include -c moe_fused_kernel.cpp -o moe_fused_kernel.o
//   3. Link into AITER library

#include <ck/ck.hpp>
#include <ck/tensor_operation/gemm/gemm.hpp>
#include <ck/tensor_operation/element_wise/element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/get_default_device.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

namespace aiter {
namespace moe_fused {

using namespace ck;
using namespace ck::tensor_operation;
using namespace ck::tensor_operation::element_wise;

// ============================================================================
// Configuration Constants
// ============================================================================

// MI355X (CDNA4) LDS configuration
constexpr index_t LDS_SIZE_BYTES = 128 * 1024;  // 128 KB per CU
constexpr index_t TOKENS_PER_BLOCK = 32;         // Tokens processed per block
constexpr index_t D_EXPERT_MAX = 2048;           // Max expert dimension
constexpr index_t D_HIDDEN_TILE = 256;           // Hidden dimension tile size
constexpr index_t D_EXPERT_TILE = 256;           // Expert dimension tile size

// MXFP4 constants
constexpr index_t MXFP4_BLOCK_SIZE = 32;  // Elements per scale block
constexpr index_t FP4_E2M1_MAX = 6;       // Max FP4 E2M1 value

// ============================================================================
// Type Definitions
// ============================================================================

using BF16 = ck::bfloat16_t;
using FP32 = float;
using FP8_E8M0 = ck::f8_e8m0_t;

// Packed FP4x2 type (2 FP4 values per byte)
struct FP4x2 {
    uint8_t data;

    __host__ __device__ static constexpr FP4x2 from_float(FP32 x, FP32 scale) {
        // Simplified FP4 quantization
        float scaled = x / scale;
        int quant = static_cast<int>(scaled * 2.0f);
        quant = quant < 0 ? 0 : (quant > 7 ? 7 : quant);
        return FP4x2{static_cast<uint8_t>(quant)};
    }

    __host__ __device__ constexpr FP32 to_float(FP32 scale) const {
        // FP4 E2M1 lookup table: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        static constexpr FP32 fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        return fp4_table[data & 0x7] * scale;
    }
};

// ============================================================================
// LDS Intermediate Buffer
// ============================================================================

/**
 * LDS buffer layout for intermediate storage:
 *
 * [TOKENS_PER_BLOCK][D_EXPERT_PAD]
 *
 * For D_EXPERT=2048, BF16 (2 bytes):
 *   32 * 2048 * 2 = 131072 bytes = 128 KB (full LDS)
 */
__device__ extern __shared__ BF16 lds_intermediate[];

// ============================================================================
// Kernel: Fused MoE Stage 1+2
// ============================================================================

/**
 * Fused MoE Kernel with LDS intermediate caching.
 *
 * Data flow per block:
 *   1. Load hidden_states for assigned tokens
 *   2. Stage 1 GEMM: gate = hidden @ W_gate.T, up = hidden @ W_up.T
 *   3. SwiGLU: intermediate = silu(gate) * up -> store to LDS
 *   4. Stage 2 GEMM: output = intermediate @ W_down.T
 *   5. Atomic add to global output with routing weight
 *
 * @param hidden_states       [M, d_hidden] input activations
 * @param gate_up_weight      [E, 2*d_expert, d_hidden//2] MXFP4 weights
 * @param down_weight         [E, d_hidden, d_expert//2] MXFP4 weights
 * @param w1_scale            [E, 2*d_expert, d_hidden//32] MXFP4 scales
 * @param w2_scale            [E, d_hidden, d_expert//32] MXFP4 scales
 * @param topk_weights        [M, top_k] routing weights
 * @param topk_ids            [M, top_k] expert assignments
 * @param expert_schedule     [num_blocks] (expert_id, token_start, token_count)
 * @param output              [M, d_hidden] output accumulator
 * @param M                   number of tokens
 * @param E                   number of experts
 * @param d_hidden            hidden dimension
 * @param d_expert            expert dimension
 * @param top_k               experts per token
 * @param block_ptr           block index in schedule
 */
__global__ void fused_moe_stage12_kernel(
    const BF16* __restrict__ hidden_states,
    const FP4x2* __restrict__ gate_up_weight,
    const FP4x2* __restrict__ down_weight,
    const FP8_E8M0* __restrict__ w1_scale,
    const FP8_E8M0* __restrict__ w2_scale,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_schedule,  // [num_blocks, 3] packed
    BF16* __restrict__ output,

    index_t M,
    index_t E,
    index_t d_hidden,
    index_t d_expert,
    index_t top_k,

    index_t d_hidden_pad,
    index_t d_expert_pad
) {
    const index_t block_id = blockIdx.x;
    const index_t tid = threadIdx.x;
    const index_t warp_id = tid / 64;
    const index_t lane_id = tid % 64;

    // Decode schedule entry
    const int32_t* schedule_entry = &expert_schedule[block_id * 3];
    const index_t expert_id = schedule_entry[0];
    const index_t token_start = schedule_entry[1];
    const index_t token_count = schedule_entry[2];

    // Early exit for empty blocks
    if (token_count == 0) return;

    // Get pointer to expert weights
    const FP4x2* expert_gate_up = &gate_up_weight[expert_id * (2 * d_expert_pad * d_hidden_pad / 2)];
    const FP4x2* expert_down = &down_weight[expert_id * (d_hidden_pad * d_expert_pad / 2)];
    const FP8_E8M0* expert_w1_scale = &w1_scale[expert_id * (2 * d_expert_pad * d_hidden_pad / MXFP4_BLOCK_SIZE)];
    const FP8_E8M0* expert_w2_scale = &w2_scale[expert_id * (d_hidden_pad * d_expert_pad / MXFP4_BLOCK_SIZE)];

    // Process tokens in this block
    for (index_t token_offset = warp_id; token_offset < token_count; token_offset += 4) {
        const index_t token_idx = token_start + token_offset;
        if (token_idx >= M) continue;

        // Get routing weight for this token->expert assignment
        float routing_weight = 1.0f;
        for (index_t k = 0; k < top_k; ++k) {
            if (topk_ids[token_idx * top_k + k] == expert_id) {
                routing_weight = topk_weights[token_idx * top_k + k];
                break;
            }
        }

        // Pointer to this token's hidden state
        const BF16* hidden = &hidden_states[token_idx * d_hidden_pad];

        // ====================================================================
        // Stage 1: GEMM + SwiGLU
        // ====================================================================

        // Accumulators for gate and up projections
        FP32 gate_acc[D_EXPERT_TILE] = {0.0f};
        FP32 up_acc[D_EXPERT_TILE] = {0.0f};

        // Stage 1 GEMM: hidden @ [gate; up].T
        // Loop over hidden dimension tiles
        for (index_t d_h_tile = 0; d_h_tile < d_hidden; d_h_tile += D_HIDDEN_TILE) {
            // Load hidden tile
            BF16 hidden_tile[D_HIDDEN_TILE];
            for (index_t i = 0; i < D_HIDDEN_TILE; ++i) {
                hidden_tile[i] = (d_h_tile + i < d_hidden) ? hidden[d_h_tile + i] : BF16{0};
            }

            // Load gate and up weights, accumulate
            for (index_t d_e = lane_id; d_e < d_expert; d_e += 64) {
                for (index_t i = 0; i < D_HIDDEN_TILE; ++i) {
                    // Dequantize gate weight
                    FP4x2 w_gate = expert_gate_up[(d_e * d_hidden_pad + d_h_tile + i) / 2];
                    FP8_E8M0 s_gate = expert_w1_scale[(d_e * d_hidden_pad + d_h_tile + i) / MXFP4_BLOCK_SIZE];
                    FP32 w_gate_f32 = w_gate.to_float(__exp2f(static_cast<FP32>(s_gate.data)));

                    gate_acc[d_e] += static_cast<FP32>(hidden_tile[i]) * w_gate_f32;

                    // Dequantize up weight (offset by d_expert)
                    FP4x2 w_up = expert_gate_up[((d_expert + d_e) * d_hidden_pad + d_h_tile + i) / 2];
                    FP8_E8M0 s_up = expert_w1_scale[((d_expert + d_e) * d_hidden_pad + d_h_tile + i) / MXFP4_BLOCK_SIZE];
                    FP32 w_up_f32 = w_up.to_float(__exp2f(static_cast<FP32>(s_up.data)));

                    up_acc[d_e] += static_cast<FP32>(hidden_tile[i]) * w_up_f32;
                }
            }
        }

        // Apply SwiGLU: intermediate = silu(gate) * up
        BF16 intermediate_tile[D_EXPERT_TILE];
        for (index_t d_e = 0; d_e < d_expert; ++d_e) {
            FP32 g = gate_acc[d_e];
            FP32 u = up_acc[d_e];
            FP32 silu_g = g / (1.0f + __expf(-g));  // SiLU approximation
            intermediate_tile[d_e] = BF16(silu_g * u);
        }

        // Store intermediate to LDS
        // Layout: [token_offset][d_e]
        BF16* lds_token = &lds_intermediate[token_offset * d_expert_pad];
        for (index_t d_e = lane_id; d_e < d_expert; d_e += 64) {
            lds_token[d_e] = intermediate_tile[d_e];
        }

        __syncwarp();

        // ====================================================================
        // Stage 2: GEMM (down projection)
        // ====================================================================

        FP32 output_acc[D_HIDDEN_TILE] = {0.0f};

        // Stage 2 GEMM: intermediate @ down.T
        for (index_t d_e_tile = 0; d_e_tile < d_expert; d_e_tile += D_EXPERT_TILE) {
            // Load intermediate from LDS
            BF16 inter_tile[D_EXPERT_TILE];
            for (index_t i = lane_id; i < D_EXPERT_TILE; i += 64) {
                inter_tile[i] = (d_e_tile + i < d_expert) ? lds_token[d_e_tile + i] : BF16{0};
            }

            // Load down weights, accumulate
            for (index_t d_h = 0; d_h < d_hidden; ++d_h) {
                FP32 acc = 0.0f;
                for (index_t i = 0; i < D_EXPERT_TILE; ++i) {
                    if (d_e_tile + i < d_expert) {
                        FP4x2 w_down = expert_down[(d_h * d_expert_pad + d_e_tile + i) / 2];
                        FP8_E8M0 s_down = expert_w2_scale[(d_h * d_expert_pad + d_e_tile + i) / MXFP4_BLOCK_SIZE];
                        FP32 w_down_f32 = w_down.to_float(__exp2f(static_cast<FP32>(s_down.data)));
                        acc += static_cast<FP32>(inter_tile[i]) * w_down_f32;
                    }
                }
                output_acc[d_h] += acc;
            }
        }

        // Apply routing weight and atomic add to output
        for (index_t d_h = 0; d_h < d_hidden; ++d_h) {
            FP32 out_val = routing_weight * output_acc[d_h];
            atomicAdd(&output[token_idx * d_hidden_pad + d_h], static_cast<BF16>(out_val));
        }
    }
}

// ============================================================================
// Host: Launch Configuration
// ============================================================================

/**
 * Launch the fused MoE kernel.
 *
 * @param stream HIP stream for async execution
 * @param hidden_states [M, d_hidden] input
 * @param gate_up_weight [E, 2*d_expert, d_hidden//2] MXFP4 weights
 * @param down_weight [E, d_hidden, d_expert//2] MXFP4 weights
 * @param w1_scale, w2_scale MXFP4 scales
 * @param topk_weights, topk_ids routing info
 * @param expert_schedule [num_blocks, 3] schedule
 * @param output [M, d_hidden] output
 * @param M, E, d_hidden, d_expert, top_k dimensions
 * @param d_hidden_pad, d_expert_pad padded dimensions
 */
void launch_fused_moe_stage12(
    hipStream_t stream,
    const BF16* hidden_states,
    const FP4x2* gate_up_weight,
    const FP4x2* down_weight,
    const FP8_E8M0* w1_scale,
    const FP8_E8M0* w2_scale,
    const float* topk_weights,
    const int32_t* topk_ids,
    const int32_t* expert_schedule,
    BF16* output,

    index_t M,
    index_t E,
    index_t d_hidden,
    index_t d_expert,
    index_t top_k,

    index_t d_hidden_pad,
    index_t d_expert_pad,

    index_t num_blocks
) {
    // Calculate LDS size
    size_t lds_size = TOKENS_PER_BLOCK * d_expert_pad * sizeof(BF16);

    // Configure kernel launch
    const index_t block_size = 256;  // 4 warps per block
    const dim3 grid_dim(num_blocks);
    const dim3 block_dim(block_size);

    // Set dynamic shared memory
    hipFuncSetSharedMemConfig(fused_moe_stage12_kernel, hipSharedMemBankSizeEightByte);

    // Launch kernel
    fused_moe_stage12_kernel<<<grid_dim, block_dim, lds_size, stream>>>(
        hidden_states,
        gate_up_weight,
        down_weight,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        expert_schedule,
        output,
        M, E, d_hidden, d_expert, top_k,
        d_hidden_pad, d_expert_pad
    );

    // Check for launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch error: " << hipGetErrorString(err) << std::endl;
    }
}

// ============================================================================
// Python Bindings (pybind11)
// ============================================================================

#ifdef PYBIND11_MODULE

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(moe_fused_kernel, m) {
    m.doc() = "CK Fused MoE Kernel - Stage 1+2 fusion with LDS caching";

    m.def("launch_fused_moe",
          &launch_fused_moe_stage12,
          "Launch fused MoE Stage 1+2 kernel",
          py::arg("stream"),
          py::arg("hidden_states"),
          py::arg("gate_up_weight"),
          py::arg("down_weight"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("expert_schedule"),
          py::arg("output"),
          py::arg("M"),
          py::arg("E"),
          py::arg("d_hidden"),
          py::arg("d_expert"),
          py::arg("top_k"),
          py::arg("d_hidden_pad"),
          py::arg("d_expert_pad"),
          py::arg("num_blocks"));
}

#endif  // PYBIND11_MODULE

}  // namespace moe_fused
}  // namespace aiter
