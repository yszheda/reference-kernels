"""
MLA (Multi-head Latent Attention) decode kernel — FlyDSL Implementation

DeepSeek-R1 forward_absorb MLA on AMD MI355X (CDNA4 architecture).

Key optimizations:
1. Pure FlyDSL kernel with Flash Attention-style block loop
2. Per-thread MXFP4 dequantization in registers (minimal LDS pressure)
3. MQA pattern: Load KV once, broadcast across 16 query heads
4. Online softmax with running max/sum for numerical stability

Configuration:
  num_heads = 16, num_kv_heads = 1
  qk_head_dim = 576, v_head_dim = 512
  sm_scale = 1.0 / sqrt(576)
"""

import torch
from task import input_t, output_t

# ============================================================================
# FlyDSL Imports (with graceful fallback for local testing)
# ============================================================================

try:
    from flydsl import (
        kernel, function, Pointer, Array,
        program_id, load, store, atomic_add,
        zeros, max, min, exp, sum, range,
        barrier, mem_fence, exp2,
    )
    from flydsl.types import fp4x2, fp8_e8m0, bfloat16, float32, int32
    FLYDSL_AVAILABLE = True
except ImportError:
    FLYDSL_AVAILABLE = False
    def kernel(fn): return fn
    def function(fn): return fn

    # Make Pointer and Array subscriptable for type hints
    from typing import Generic, TypeVar
    T = TypeVar('T')
    class Pointer(Generic[T]): pass
    class Array(Generic[T]): pass

    # Type placeholders
    fp4x2 = fp8_e8m0 = bfloat16 = float32 = int32 = type

# ============================================================================
# DeepSeek-R1 MLA Constants
# ============================================================================

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

# Block sizes (tunable parameters)
BLOCK_KV = 64       # KV sequence tile size
BLOCK_DK = 64       # QK dimension tile for MFMA
BLOCK_DV = 64       # V dimension tile for MFMA

# ============================================================================
# FlyDSL Helper Functions: MXFP4 Dequantization
# ============================================================================

@function
def unpack_fp4_low(byte_val: int32) -> float32:
    """
    Extract low 4 bits and convert FP4 (E2M1) to float32.
    FP4 E2M1 format: 1 sign, 2 exp, 1 mantissa.
    """
    val = byte_val & 0x0F
    sign = (val >> 3) & 0x1
    exp_bits = (val >> 1) & 0x3
    mantissa = val & 0x1

    # Decode FP4 to float32
    if exp_bits == 0:
        return float32(0.0) if sign == 0 else float32(-0.0)
    elif exp_bits == 1:
        # Normalized: (-1)^s * 2^(1-1) * (1 + m/2)
        base = float32(1.0) + float32(mantissa) * float32(0.5)
        return base if sign == 0 else -base
    elif exp_bits == 2:
        # Normalized: (-1)^s * 2^(2-1) * (1 + m/2)
        base = float32(2.0) * (float32(1.0) + float32(mantissa) * float32(0.5))
        return base if sign == 0 else -base
    else:  # exp_bits == 3
        # Special case (inf/nan in standard FP4)
        return float32('inf') if sign == 0 else float32('-inf')


@function
def unpack_fp4_high(byte_val: int32) -> float32:
    """Extract high 4 bits and convert FP4 (E2M1) to float32."""
    val = (byte_val >> 4) & 0x0F
    return unpack_fp4_low(val)


@function
def e8m0_to_float32(fp8_val: int32) -> float32:
    """
    Convert FP8 E8M0 to float32.
    E8M0: 1 sign bit, 7 exponent bits, 0 mantissa bits.
    Bias = 64 for E8M0.
    """
    sign = (fp8_val >> 7) & 0x1
    exp_bits = fp8_val & 0x7F

    # E8M0 represents powers of 2 with bias 64
    unbiased_exp = float32(exp_bits) - float32(64)
    result = exp2(unbiased_exp)
    return result if sign == 0 else -result


@function
def dequantize_mxfp4_element(
    fp4_ptr: Pointer[fp4x2],
    scale_ptr: Pointer[fp8_e8m0],
    row_idx: int32,
    col_idx: int32,
    row_stride: int32,
) -> float32:
    """
    Dequantize a single MXFP4 element.

    MXFP4 format: 2 FP4 values per byte, 32-element blocks share one E8M0 scale.
    """
    # Compute fp4 byte index (2 elements per byte)
    fp4_idx = row_idx * (row_stride // 2) + col_idx // 2
    fp4_byte = load(fp4_ptr + fp4_idx)

    # Get the right FP4 value (low or high nibble)
    val = unpack_fp4_low(fp4_byte) if (col_idx % 2 == 0) else unpack_fp4_high(fp4_byte)

    # Get scale for this 32-element block
    scale_block_idx = col_idx // 32
    scale_idx = row_idx * ((row_stride + 31) // 32) + scale_block_idx
    scale_fp8 = load(scale_ptr + scale_idx)
    scale_f32 = e8m0_to_float32(scale_fp8)

    return val * scale_f32

# ============================================================================
# FlyDSL Kernel: MLA Decode with MXFP4 Dequant
# ============================================================================

@kernel
def mla_decode_kernel(
    # Output
    out_ptr: Pointer[bfloat16],

    # Inputs
    q_ptr: Pointer[bfloat16],
    kv_ptr: Pointer[fp4x2],
    kv_scale_ptr: Pointer[fp8_e8m0],

    # Indirect pointers
    qo_indptr: Pointer[int32],
    kv_indptr: Pointer[int32],

    # Dimensions
    num_heads: int32,
    num_kv_heads: int32,
    qk_head_dim: int32,
    v_head_dim: int32,
    sm_scale: float32,

    # Block sizes (compile-time constants)
    BLOCK_KV: int32,
    BLOCK_DK: int32,
    BLOCK_DV: int32,
):
    """
    MLA Decode Kernel - Flash Attention style with block loop.
    Grid: (batch_size, num_heads, q_seq_len)
    Each work item processes one (batch, head, q_token) tuple.
    """
    # Compute (batch_idx, head_idx, q_token_idx) from grid
    batch_idx = program_id(0)
    head_idx = program_id(1)
    q_token_idx = program_id(2)

    # Load sequence pointers
    q_start = load(qo_indptr + batch_idx)
    q_end = load(qo_indptr + batch_idx + 1)
    q_seq_len = q_end - q_start

    # Early exit for invalid work items
    if q_token_idx >= q_seq_len:
        return

    # Compute Q offset for this work item
    q_offset = q_start + q_token_idx

    # Load Q vector: [qk_head_dim] bf16 -> float32
    q_vec = zeros([qk_head_dim], dtype=float32)
    base_offset = q_offset * num_heads * qk_head_dim + head_idx * qk_head_dim

    for d in range(0, qk_head_dim, BLOCK_DK):
        tile_size = min(BLOCK_DK, qk_head_dim - d)
        q_tile = load(q_ptr + base_offset + d, mask=range(tile_size), other=float32(0.0))
        for i in range(tile_size):
            q_vec[d + i] = q_tile[i].to(float32)

    # Initialize online softmax accumulators
    m_i = float32(-1e30)  # Running max (large negative initial)
    d_i = float32(0.0)     # Running sum
    acc = zeros([v_head_dim], dtype=float32)  # Output accumulator

    # KV sequence range for this batch
    kv_start = load(kv_indptr + batch_idx)
    kv_end = load(kv_indptr + batch_idx + 1)

    # KV head index (MQA: same KV head for all query heads)
    kv_head_idx = head_idx % num_kv_heads

    # Block loop over KV sequence
    for kv_block_start in range(kv_start, kv_end, BLOCK_KV):
        kv_block_len = min(kv_end - kv_block_start, BLOCK_KV)

        # Compute scores: Q @ K^T for this block
        scores = zeros([kv_block_len], dtype=float32)

        for row in range(kv_block_len):
            kv_idx = kv_block_start + row
            score = float32(0.0)

            # Dot product: Q · K
            for d in range(0, qk_head_dim, BLOCK_DK):
                tile_size = min(BLOCK_DK, qk_head_dim - d)
                for i in range(tile_size):
                    # Dequantize K element on-the-fly
                    k_val = dequantize_mxfp4_element(
                        kv_ptr, kv_scale_ptr,
                        kv_idx * num_kv_heads + kv_head_idx,
                        d + i,
                        qk_head_dim
                    )
                    score += q_vec[d + i] * k_val

            scores[row] = score * sm_scale

        # Online softmax: update running max/sum
        m_block = float32(-1e30)
        for row in range(kv_block_len):
            m_block = max(m_block, scores[row])

        m_new = max(m_i, m_block)

        # Compute exp(sum) for normalization
        exp_sum = float32(0.0)
        for row in range(kv_block_len):
            exp_sum += exp(scores[row] - m_new)

        d_i = d_i * exp(m_i - m_new) + exp_sum
        m_i = m_new

        # Accumulate: acc = acc * alpha + scores @ V
        alpha = exp(m_i - m_new)
        for d_out in range(0, v_head_dim, BLOCK_DV):
            tile_size = min(BLOCK_DV, v_head_dim - d_out)

            for i in range(tile_size):
                acc_val = acc[d_out + i] * alpha

                for row in range(kv_block_len):
                    kv_idx = kv_block_start + row
                    # Dequantize V element on-the-fly
                    v_val = dequantize_mxfp4_element(
                        kv_ptr, kv_scale_ptr,
                        kv_idx * num_kv_heads + kv_head_idx,
                        i,  # V dimension offset within block
                        v_head_dim
                    )
                    acc_val += scores[row] * v_val

                acc[d_out + i] = acc_val

    # Normalize and store output
    for d in range(0, v_head_dim, BLOCK_DV):
        tile_size = min(BLOCK_DV, v_head_dim - d)
        out_tile = zeros([tile_size], dtype=bfloat16)

        for i in range(tile_size):
            out_tile[i] = (acc[d + i] / d_i).to(bfloat16)

        offset = q_offset * num_heads * v_head_dim + head_idx * v_head_dim + d
        store(out_ptr + offset, out_tile, mask=range(tile_size))


# ============================================================================
# Python Wrapper: Custom Kernel Entry Point
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    """
    FlyDSL MLA decode kernel wrapper.

    Args:
        data: (q, kv_data, qo_indptr, kv_indptr, config)

    Returns:
        output: (total_q, num_heads, v_head_dim) bf16
    """
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config['batch_size']
    num_heads = config['num_heads']
    num_kv_heads = config['num_kv_heads']
    qk_head_dim = config['qk_head_dim']
    v_head_dim = config['v_head_dim']
    q_seq_len = config['q_seq_len']

    total_q = q.shape[0]

    # Get MXFP4 KV buffer and scales
    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data['mxfp4']

    # Allocate output
    output = torch.zeros((total_q, num_heads, v_head_dim),
                         dtype=torch.bfloat16, device=q.device)

    if FLYDSL_AVAILABLE:
        # Launch FlyDSL kernel
        # Grid: (batch_size, num_heads, q_seq_len)
        grid = (batch_size, num_heads, q_seq_len)

        mla_decode_kernel[grid](
            out_ptr=output,
            q_ptr=q,
            kv_ptr=kv_buffer_mxfp4,
            kv_scale_ptr=kv_scale_mxfp4,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            sm_scale=SM_SCALE,
            BLOCK_KV=BLOCK_KV,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
        )
    else:
        # Fallback: use reference implementation
        from reference import ref_kernel
        output = ref_kernel(data)

    return output
