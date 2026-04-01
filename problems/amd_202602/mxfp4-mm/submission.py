"""
MXFP4 GEMM for AMD MI355X — Pure FlyDSL Implementation

C [M, N] = A [M, K] × B [N, K].T
A: bf16, B: MXFP4 (fp4x2 + e8m0 scale), C: bf16
"""

import torch
from task import input_t, output_t

import flydsl.compiler as flyc
import flydsl.expr as fx

# Block sizes
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64


@flyc.kernel(known_block_size=[256, 1, 1])
def mxfp4_gemm_kernel(
    A: fx.Tensor,  # [M, K] bf16
    B_shuffle: fx.Tensor,  # [N, K//2] fp4x2 shuffled
    B_scale: fx.Tensor,  # [N, K//32] e8m0 shuffled
    C: fx.Tensor,  # [M, N] bf16 output
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Int32,
):
    """
    MXFP4 GEMM kernel using FlyDSL.
    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N), 1)
    Block: 256 threads
    """
    tid = fx.thread_idx.x
    bid_m = fx.block_idx.x
    bid_n = fx.block_idx.y

    m_start = fx.arith.muli(bid_m, fx.Int32(BLOCK_M))
    n_start = fx.arith.muli(bid_n, fx.Int32(BLOCK_N))

    m_end = fx.min(m_start + fx.Int32(BLOCK_M), M)
    n_end = fx.min(n_start + fx.Int32(BLOCK_N), N)

    m_tile = fx.arith.subi(m_end, m_start)
    n_tile = fx.arith.subi(n_end, n_start)

    # Buffer resources
    A_rsrc = fx.rocdl.make_buffer_tensor(A)
    B_rsrc = fx.rocdl.make_buffer_tensor(B_shuffle)
    C_rsrc = fx.rocdl.make_buffer_tensor(C)

    # Each thread computes multiple output elements
    # Total outputs per tile: m_tile * n_tile
    # Threads: 256

    # Loop through output tile
    for m_local in range(fx.Int32(0), m_tile, fx.Int32(1)):
        m_idx = fx.arith.addi(m_start, m_local)

        if m_idx >= M:
            continue

        # Each thread handles a subset of N outputs for this M
        for n_local in range(fx.Int32(0), n_tile, fx.Int32(1)):
            if fx.arith.addi(n_local, tid) >= n_tile:
                continue

            n_idx = fx.arith.addi(n_start, fx.arith.addi(n_local, tid))

            if n_idx >= N:
                continue

            # Compute dot product: C[m, n] = sum_k A[m, k] * B[n, k]
            acc = fx.Float32(0.0)

            # Accumulate over K dimension
            for k in range(fx.Int32(0), K, fx.Int32(1)):
                # Load A[m, k]
                A_offset = fx.arith.addi(fx.arith.muli(m_idx, K), k)
                a_val = fx.rocdl.buffer_load(
                    A_rsrc, A_offset, idx_size=1, value_type=fx.T.bf16
                )
                a_f32 = fx.arith.bitcast(a_val, fx.T.f32)

                # Load B[n, k] (MXFP4 - simplified)
                # B is packed: 2 FP4 values per byte
                B_offset = fx.arith.addi(
                    fx.arith.muli(n_idx, fx.arith.divui(K, fx.Int32(2))),
                    fx.arith.divui(k, fx.Int32(2))
                )
                b_val = fx.rocdl.buffer_load(
                    B_rsrc, B_offset, idx_size=1, value_type=fx.T.i8
                )
                # Simple conversion (placeholder for proper MXFP4 dequant)
                b_f32 = fx.arith.bitcast(b_val, fx.T.f32)

                # Accumulate
                acc = fx.arith.addf(acc, fx.arith.mulf(a_f32, b_f32))

            # Store result
            C_offset = fx.arith.addi(fx.arith.muli(m_idx, N), n_idx)
            c_val = fx.arith.bitcast(acc, fx.T.bf16)
            fx.rocdl.buffer_store(c_val, C_rsrc, C_offset, idx_size=1)


@flyc.jit
def launch_mxfp4_gemm(
    A: fx.Tensor,
    B_shuffle: fx.Tensor,
    B_scale: fx.Tensor,
    C: fx.Tensor,
    M: int,
    N: int,
    K: int,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch MXFP4 GEMM kernel."""
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    mxfp4_gemm_kernel(A, B_shuffle, B_scale, C, fx.Int32(M), fx.Int32(N), fx.Int32(K)).launch(
        grid=(grid_m, grid_n, 1),
        block=(256, 1, 1),
        stream=stream
    )


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 GEMM using pure FlyDSL.
    """
    A, B, B_q, B_shuffle, B_scale_sh = data

    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape

    # Allocate output
    C = torch.zeros((m, n), dtype=torch.bfloat16, device=A.device)

    # Launch FlyDSL kernel
    launch_mxfp4_gemm(A, B_shuffle, B_scale_sh, C, m, n, k)

    return C
