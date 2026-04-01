"""
MLA (Multi-head Latent Attention) decode kernel — Pure FlyDSL Implementation

DeepSeek-R1 forward_absorb MLA on AMD MI355X.
Q: [total_q, num_heads, qk_head_dim=576]
KV: [total_kv, num_kv_heads=1, qk_head_dim=576]
Out: [total_q, num_heads, v_head_dim=512]
"""

import torch
from task import input_t, output_t

import flydsl.compiler as flyc
import flydsl.expr as fx

# DeepSeek-R1 Constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576  # 512 + 64
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (576 ** 0.5)

# Tile sizes
BLOCK_Q = 1  # Decode mode: q_seq_len = 1 per batch
BLOCK_KV = 64  # KV sequence tile
BLOCK_D = 32  # Dimension tile for QK
BLOCK_DV = 64  # V dimension tile


@flyc.kernel(known_block_size=[256, 1, 1])
def mla_decode_kernel(
    q: fx.Tensor,  # [total_q, NUM_HEADS, QK_HEAD_DIM] bf16
    kv: fx.Tensor,  # [total_kv, NUM_KV_HEADS, QK_HEAD_DIM] bf16
    out: fx.Tensor,  # [total_q, NUM_HEADS, V_HEAD_DIM] bf16
    qo_indptr: fx.Tensor,  # [batch_size + 1] int32
    kv_indptr: fx.Tensor,  # [batch_size + 1] int32
    batch_size: fx.Constexpr[int],
    max_q_len: fx.Constexpr[int],
):
    """
    MLA Decode Kernel using FlyDSL.
    Grid: (batch_size, NUM_HEADS, max_q_len)
    Block: 256 threads
    """
    tid = fx.thread_idx.x
    bid = fx.block_idx.x  # batch index
    hid = fx.block_idx.y  # head index
    qid = fx.block_idx.z  # query token index

    # Create buffer resources for global memory access
    q_rsrc = fx.rocdl.make_buffer_tensor(q)
    kv_rsrc = fx.rocdl.make_buffer_tensor(kv)
    out_rsrc = fx.rocdl.make_buffer_tensor(out)
    qo_rsrc = fx.rocdl.make_buffer_tensor(qo_indptr)
    kv_rsrc_indptr = fx.rocdl.make_buffer_tensor(kv_indptr)

    # Load sequence boundaries for this batch
    q_start = fx.rocdl.buffer_load(qo_rsrc, bid, idx_size=1, value_type=fx.T.i32)
    q_end = fx.rocdl.buffer_load(qo_rsrc, bid + 1, idx_size=1, value_type=fx.T.i32)
    q_len = fx.arith.subi(q_end, q_start)

    kv_start = fx.rocdl.buffer_load(kv_rsrc_indptr, bid, idx_size=1, value_type=fx.T.i32)
    kv_end = fx.rocdl.buffer_load(kv_rsrc_indptr, bid + 1, idx_size=1, value_type=fx.T.i32)
    kv_len = fx.arith.subi(kv_end, kv_start)

    # Early exit if this query token doesn't exist
    if qid >= q_len:
        return

    # Allocate shared memory for Q tile and KV scores
    # Q: [QK_HEAD_DIM] bf16 - 576 * 2 bytes = 1152 bytes
    # Scores: [BLOCK_KV] f32 - 64 * 4 bytes = 256 bytes
    smem_size = 2048  # Total shared memory

    # Compute Q offset: (q_start + qid) * NUM_HEADS * QK_HEAD_DIM + hid * QK_HEAD_DIM
    q_global_idx = fx.arith.addi(q_start, qid)
    q_stride_head = fx.Int32(NUM_HEADS * QK_HEAD_DIM)
    q_offset = fx.arith.addi(
        fx.arith.muli(q_global_idx, q_stride_head),
        fx.arith.muli(hid, fx.Int32(QK_HEAD_DIM))
    )

    # Load Q vector [QK_HEAD_DIM] into registers
    # Each thread loads QK_HEAD_DIM / 256 elements
    q_frag = fx.make_fragment((fx.Int32(QK_HEAD_DIM),), fx.T.bf16)

    for d_base in range(fx.Int32(0), fx.Int32(QK_HEAD_DIM), fx.Int32(256)):
        d_idx = fx.arith.addi(d_base, tid)
        if d_idx < fx.Int32(QK_HEAD_DIM):
            q_val = fx.rocdl.buffer_load(
                q_rsrc,
                fx.arith.addi(q_offset, d_idx),
                idx_size=1,
                value_type=fx.T.bf16
            )
            fx.vector.store(q_val, q_frag, [d_idx])

    # Synchronize after loading Q
    fx.gpu.barrier()

    # Online softmax accumulators
    m_i = fx.Float32(-1e30)  # Running max
    d_i = fx.Float32(0.0)  # Running sum

    # Output accumulator [V_HEAD_DIM]
    acc = fx.make_fragment((fx.Int32(V_HEAD_DIM),), fx.T.f32)
    for i in range(fx.Int32(V_HEAD_DIM)):
        fx.vector.store(fx.Float32(0.0), acc, [i])

    # Loop over KV tiles
    num_kv_tiles = fx.arith.divui(
        fx.arith.addi(kv_len, fx.Int32(BLOCK_KV - 1)),
        fx.Int32(BLOCK_KV)
    )

    for kv_tile in range(num_kv_tiles):
        kv_tile_start = fx.arith.muli(kv_tile, fx.Int32(BLOCK_KV))
        kv_tile_end = fx.min(kv_tile_start + fx.Int32(BLOCK_KV), kv_len)
        kv_tile_len = fx.arith.subi(kv_tile_end, kv_tile_start)

        # Compute Q @ K^T for this KV tile
        # scores[kv_local] = sum_d Q[d] * K[kv_idx, d]
        scores = fx.make_fragment((fx.Int32(BLOCK_KV),), fx.T.f32)

        # Each thread computes scores for a subset of KV positions
        for kv_local in range(kv_tile_len):
            kv_idx = fx.arith.addi(kv_start, fx.arith.addi(kv_tile_start, kv_local))

            # KV offset: kv_idx * NUM_KV_HEADS * QK_HEAD_DIM
            kv_offset = fx.arith.muli(kv_idx, fx.Int32(NUM_KV_HEADS * QK_HEAD_DIM))

            # Dot product Q · K - parallel reduction across threads
            # Each thread computes partial dot product
            dot_acc = fx.Float32(0.0)

            # Unroll loop for QK_HEAD_DIM
            for d in range(fx.Int32(0), fx.Int32(QK_HEAD_DIM), fx.Int32(4)):
                d0 = fx.arith.addi(d, tid)

                # Load 4 consecutive elements from Q
                if d0 < fx.Int32(QK_HEAD_DIM):
                    q_val = fx.vector.load(fx.T.bf16, q_frag, [d0])
                    # Convert bf16 to f32
                    q_f32 = fx.arith.bitcast(q_val, fx.T.f32)

                    # Load K from global memory
                    k_val = fx.rocdl.buffer_load(
                        kv_rsrc,
                        fx.arith.addi(kv_offset, d0),
                        idx_size=1,
                        value_type=fx.T.bf16
                    )
                    k_f32 = fx.arith.bitcast(k_val, fx.T.f32)

                    # Accumulate
                    dot_acc = fx.arith.addf(dot_acc, fx.arith.mulf(q_f32, k_f32))

            # Store partial score (will need reduction across warp/block)
            # For simplicity, thread 0 handles the final reduction
            if tid == fx.Int32(0):
                # Apply scale
                score_scaled = fx.arith.mulf(dot_acc, fx.Float32(SM_SCALE))
                fx.vector.store(score_scaled, scores, [kv_local])

        fx.gpu.barrier()

        # Online softmax update (only thread 0 for simplicity)
        if tid == fx.Int32(0):
            # Find max in this tile
            m_tile = fx.Float32(-1e30)
            for i in range(kv_tile_len):
                s = fx.vector.load(fx.T.f32, scores, [i])
                m_tile = fx.max(m_tile, s)

            # Update running max
            m_new = fx.max(m_i, m_tile)

            # Rescale previous accumulator
            if m_i > fx.Float32(-1e29):  # Not first iteration
                scale_prev = fx.math.exp2(fx.arith.mulf(fx.arith.subf(m_i, m_new), fx.Float32(1.442695)))
                d_i = fx.arith.mulf(d_i, scale_prev)
                # Rescale output accumulator
                for d_out in range(fx.Int32(V_HEAD_DIM)):
                    acc_val = fx.vector.load(fx.T.f32, acc, [d_out])
                    acc_val = fx.arith.mulf(acc_val, scale_prev)
                    fx.vector.store(acc_val, acc, [d_out])

            # Compute exp and sum for this tile
            exp_sum = fx.Float32(0.0)
            for i in range(kv_tile_len):
                s = fx.vector.load(fx.T.f32, scores, [i])
                exp_val = fx.math.exp2(fx.arith.mulf(fx.arith.subf(s, m_new), fx.Float32(1.442695)))
                fx.vector.store(exp_val, scores, [i])
                exp_sum = fx.arith.addf(exp_sum, exp_val)

            d_i = fx.arith.addf(d_i, exp_sum)
            m_i = m_new

            # Accumulate: acc += scores @ V
            for kv_local in range(kv_tile_len):
                kv_idx = fx.arith.addi(kv_start, fx.arith.addi(kv_tile_start, kv_local))
                # V offset: kv_idx * NUM_KV_HEADS * QK_HEAD_DIM (first V_HEAD_DIM dims)
                kv_base = fx.arith.muli(kv_idx, fx.Int32(NUM_KV_HEADS * QK_HEAD_DIM))

                prob = fx.vector.load(fx.T.f32, scores, [kv_local])

                for d_out in range(fx.Int32(V_HEAD_DIM)):
                    v_val = fx.rocdl.buffer_load(
                        kv_rsrc,
                        fx.arith.addi(kv_base, d_out),
                        idx_size=1,
                        value_type=fx.T.bf16
                    )
                    v_f32 = fx.arith.bitcast(v_val, fx.T.f32)

                    acc_val = fx.vector.load(fx.T.f32, acc, [d_out])
                    acc_val = fx.arith.addf(acc_val, fx.arith.mulf(prob, v_f32))
                    fx.vector.store(acc_val, acc, [d_out])

        fx.gpu.barrier()

    # Write output (normalize and store)
    out_offset = fx.arith.addi(
        fx.arith.muli(q_global_idx, fx.Int32(NUM_HEADS * V_HEAD_DIM)),
        fx.arith.muli(hid, fx.Int32(V_HEAD_DIM))
    )

    # Each thread writes V_HEAD_DIM / 256 elements
    for d_base in range(fx.Int32(0), fx.Int32(V_HEAD_DIM), fx.Int32(256)):
        d_idx = fx.arith.addi(d_base, tid)
        if d_idx < fx.Int32(V_HEAD_DIM):
            acc_val = fx.vector.load(fx.T.f32, acc, [d_idx])
            # Normalize
            out_val = fx.arith.divf(acc_val, d_i)
            # Convert to bf16
            out_bf16 = fx.arith.bitcast(out_val, fx.T.bf16)
            fx.rocdl.buffer_store(
                out_bf16,
                out_rsrc,
                fx.arith.addi(out_offset, d_idx),
                idx_size=1
            )


@flyc.jit
def launch_mla_decode(
    q: fx.Tensor,
    kv: fx.Tensor,
    out: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    batch_size: int,
    max_q_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch MLA decode kernel."""
    grid = (batch_size, NUM_HEADS, max_q_len)
    mla_decode_kernel(
        q, kv, out, qo_indptr, kv_indptr,
        batch_size, max_q_len
    ).launch(
        grid=grid,
        block=(256, 1, 1),
        smem=2048,
        stream=stream
    )


def custom_kernel(data: input_t) -> output_t:
    """
    MLA decode kernel wrapper using pure FlyDSL.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config['batch_size']
    num_heads = config['num_heads']
    v_head_dim = config['v_head_dim']
    q_seq_len = config['q_seq_len']

    total_q = q.shape[0]
    device = q.device

    # Get bf16 KV buffer
    kv_buffer = kv_data['bf16']

    # Allocate output
    output = torch.zeros((total_q, num_heads, v_head_dim), dtype=torch.bfloat16, device=device)

    # Launch FlyDSL kernel
    launch_mla_decode(
        q, kv_buffer, output,
        qo_indptr, kv_indptr,
        batch_size, q_seq_len
    )

    return output
