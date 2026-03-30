#!/usr/bin/env python3
"""
MXFP4 GEMM for AMD MI355X.

Uses aiter's optimized gemm_a4w4 kernel with per-1x32 MXFP4 quantization.
"""
import torch
from task import input_t, output_t
from aiter import dtypes
import aiter
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 GEMM for AMD MI355X.

    Args:
        data: Tuple of (A, B, B_q, B_shuffle, B_scale_sh)
              A: [M, K] bf16 activation
              B: [N, K] bf16 weight (unused)
              B_q: [N, K/2] fp4x2 raw quantized (unused)
              B_shuffle: [N, K/2] fp4x2 shuffled weight
              B_scale_sh: [N, K/32] e8m0 shuffled scale

    Returns:
        output: [M, N] bf16 result
    """
    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    m, k = A.shape
    n, _ = B.shape

    # Quantize activation with shuffle
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_scale = e8m0_shuffle(A_scale)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = A_scale.view(dtypes.fp8_e8m0)

    # Execute GEMM
    return aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
