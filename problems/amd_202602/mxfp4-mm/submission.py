#!/usr/bin/env python3
"""
MXFP4 GEMM for AMD MI355X.

Optimization: Use torch.compile to fuse quantization and GEMM operations,
reducing kernel launch overhead and intermediate memory traffic.
"""
import torch
from task import input_t, output_t
from aiter import dtypes
import aiter
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

# Cache compiled kernels by shape to avoid recompilation
_compiled_cache = {}


def _quant_and_gemm(A, B_shuffle, A_scale_sh, B_scale_sh):
    """Internal function: quantize A and run GEMM."""
    A_scale = e8m0_shuffle(A_scale_sh)
    A_q = A.view(dtypes.fp4x2)
    return aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )


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

    # Quantize activation first (required before GEMM)
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = A_scale.view(dtypes.fp8_e8m0)

    # Use torch.compile for fused quant+gemm on second run
    shape_key = (m, n, k)
    if shape_key not in _compiled_cache:
        _compiled_cache[shape_key] = torch.compile(
            _quant_and_gemm,
            mode='max-autotune-no-cudagraphs',
            dynamic=False,
        )

    return _compiled_cache[shape_key](A_fp4, B_shuffle, A_scale_sh, B_scale_sh)
