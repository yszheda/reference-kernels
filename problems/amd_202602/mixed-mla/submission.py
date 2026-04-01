"""
MLA (Multi-head Latent Attention) decode kernel — FlyDSL MFMA Implementation

DeepSeek-R1 forward_absorb MLA on AMD MI355X.
Q: [total_q, num_heads, qk_head_dim=576]
KV: [total_kv, num_kv_heads=1, qk_head_dim=576]
Out: [total_q, num_heads, v_head_dim=512]

Uses FlyDSL's MFMA32 instructions (mfma_f32_32x32x16_bf16) for efficient attention.
"""

import torch
import math
from task import input_t, output_t

# FlyDSL imports
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

# DeepSeek-R1 Constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576  # 512 + 64
V_HEAD_DIM = 512
SM_SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)
LOG2E = 1.4426950408889634

_kernel_cache = {}


def compile_mla_decode_kernel():
    """Compile MLA decode kernel using FlyDSL MFMA32."""
    key = "mla_decode_mfma"
    if key in _kernel_cache:
        return _kernel_cache[key]

    gpu_arch = get_hip_arch()

    # Flash attention parameters
    BLOCK_M = 32   # Q rows per block (1 decode position per head)
    BLOCK_N = 64   # KV positions per tile
    WARP_SIZE = 64
    BLOCK_SIZE = 64  # 1 wave for decode

    # MFMA parameters for gfx950
    USE_K16 = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = QK_HEAD_DIM // K_STEP_QK

    # LDS allocation
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="mla_decode_smem")

    K_STRIDE = QK_HEAD_DIM
    V_STRIDE = V_HEAD_DIM
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_KV_TOTAL_SIZE = LDS_K_TILE_SIZE + LDS_V_TILE_SIZE

    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2  # bf16 = 2 bytes

    # Vector types
    elem_type = T.bf16
    compute_type = T.f32
    v4bf16_type = T.vec(4, elem_type)
    v8bf16_type = T.vec(8, elem_type)
    v16f32_type = T.vec(16, compute_type)

    mfma_pack_type = v8bf16_type if USE_K16 else v4bf16_type
    MFMA_LANE_K = 8 if USE_K16 else 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def mla_decode_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        kv_len: fx.Int32,
    ):
        """MLA decode kernel using MFMA32 for attention."""

        fm_fast = arith.FastMathFlags.fast

        # Constants
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(float('-inf'), type=compute_type)
        c_sm_scale_log2e = arith.constant(SM_SCALE * LOG2E, type=compute_type)
        c_log2e = arith.constant(LOG2E, type=compute_type)

        # MFMA helper
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        def _mfma(ods_fn, a, b, c):
            return ods_fn(v16f32_type, a, b, c, _mfma_zero, _mfma_zero, _mfma_zero).result

        def mfma_acc(a, b, c):
            """MFMA accumulate: 32x32x16 bf16"""
            if USE_K16:
                return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
            # K=8 path
            a_i16 = vector.bitcast(T.i16x4, a)
            b_i16 = vector.bitcast(T.i16x4, b)
            return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a_i16, b_i16, c)

        # Thread/block IDs
        block_id = gpu.block_id("x")
        tid = gpu.thread_id("x")

        # Grid: (num_q * num_heads,)
        head_idx = block_id % NUM_HEADS
        q_idx = block_id // NUM_HEADS

        # Lane decomposition for MFMA32
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        # LDS view
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(base_ptr, lds_kv_offset, elem_type, shape=(LDS_KV_TOTAL_SIZE,)).get()

        # Load Q into registers (B-operand for MFMA: transposed)
        # Q layout: [total_q, num_heads, head_dim]
        q_stride = NUM_HEADS * QK_HEAD_DIM
        q_base = q_idx * q_stride + head_idx * QK_HEAD_DIM

        # Load Q as MFMA B-operand packs
        # B-operand uses j = lane_mod_32, k-subblock = lane_div_32 * MFMA_LANE_K
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = ks * K_STEP_QK + lane_div_32 * MFMA_LANE_K
            q_offset = q_base + q_col
            q_vec = buffer_ops.buffer_load(Q, q_offset, vec_width=8 if USE_K16 else 4, dtype=mfma_pack_type)
            q_b_packs.append(q_vec)

        # Initialize accumulators for output (V_HEAD_DIM = 512 = 16 * 32)
        D_CHUNKS = V_HEAD_DIM // 32
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        o_accs = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        # Running softmax state
        m_running = c_neg_inf
        l_running = c_zero_f

        # Process KV in tiles
        kv_len_v = arith.index_cast(T.index, kv_len)
        num_kv_tiles = (kv_len_v + BLOCK_N - 1) // BLOCK_N

        for kv_tile in range(num_kv_tiles):
            kv_start = kv_tile * BLOCK_N
            kv_end = arith.minui(kv_start + BLOCK_N, kv_len_v)
            tile_len = kv_end - kv_start

            # === Cooperative load K tile to LDS ===
            # Each thread loads VEC_WIDTH elements
            VEC_WIDTH = 8
            THREADS_PER_ROW = QK_HEAD_DIM // VEC_WIDTH
            for load_row in range_constexpr(BLOCK_N // (BLOCK_SIZE // THREADS_PER_ROW)):
                row_in_tile = load_row * (BLOCK_SIZE // THREADS_PER_ROW) + tid // THREADS_PER_ROW
                col_base = (tid % THREADS_PER_ROW) * VEC_WIDTH

                row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_in_tile, tile_len)
                _if_k = scf.IfOp(row_valid)
                with ir.InsertionPoint(_if_k.then_block):
                    global_row = kv_start + row_in_tile
                    k_offset = global_row * QK_HEAD_DIM + col_base
                    k_vec = buffer_ops.buffer_load(K, k_offset, vec_width=VEC_WIDTH, dtype=v8bf16_type)
                    lds_idx = row_in_tile * K_STRIDE + col_base
                    vector.store(k_vec, lds_kv, [lds_idx])
                    scf.YieldOp([])

            # === Cooperative load V tile to LDS ===
            V_THREADS_PER_ROW = V_HEAD_DIM // VEC_WIDTH
            for load_row in range_constexpr(BLOCK_N // (BLOCK_SIZE // V_THREADS_PER_ROW)):
                row_in_tile = load_row * (BLOCK_SIZE // V_THREADS_PER_ROW) + tid // V_THREADS_PER_ROW
                col_base = (tid % V_THREADS_PER_ROW) * VEC_WIDTH

                row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_in_tile, tile_len)
                _if_v = scf.IfOp(row_valid)
                with ir.InsertionPoint(_if_v.then_block):
                    global_row = kv_start + row_in_tile
                    v_offset = global_row * V_HEAD_DIM + col_base
                    v_vec = buffer_ops.buffer_load(V, v_offset, vec_width=VEC_WIDTH, dtype=v8bf16_type)
                    lds_idx = LDS_K_TILE_SIZE + row_in_tile * V_STRIDE + col_base
                    vector.store(v_vec, lds_kv, [lds_idx])
                    scf.YieldOp([])

            gpu.barrier()

            # === Stage 1: Q @ K^T using MFMA ===
            # Each lane processes K rows: lane_mod_32 selects which of 32 K rows
            # Accumulate scores across K_STEPS_QK MFMA operations
            s_acc = c_zero_v16f32

            for k_row_block in range_constexpr(2):  # Process 32 K rows at a time (MFMA32)
                k_row_base = k_row_block * 32
                k_row = k_row_base + lane_mod_32

                k_row_valid = arith.cmpi(arith.CmpIPredicate.ult, k_row, tile_len)
                k_row_safe = arith.select(k_row_valid, k_row, arith.index(0))

                for ks in range_constexpr(K_STEPS_QK):
                    # Load K A-operand from LDS
                    k_col = ks * K_STEP_QK + lane_div_32 * MFMA_LANE_K
                    lds_k_idx = k_row_safe * K_STRIDE + k_col
                    k_pack = vector.load_op(mfma_pack_type, lds_kv, [lds_k_idx])

                    # MFMA: K (A) @ Q^T (B) -> scores
                    # Use zero pack for invalid rows to avoid NaN
                    k_pack_safe = arith.select(k_row_valid, k_pack, arith.constant_vector(0.0, mfma_pack_type))
                    s_acc = mfma_acc(k_pack_safe, q_b_packs[ks], s_acc)

            # Extract scores from MFMA accumulator (16 values per lane)
            s_raw = []
            for r in range_constexpr(16):
                s_raw.append(vector.extract(s_acc, static_position=[r], dynamic_position=[]))

            # === Online softmax ===
            # Find local max across 16 scores
            local_max = s_raw[0]
            for r in range_constexpr(15):
                local_max = arith.maximumf(local_max, s_raw[r + 1], fastmath=fm_fast)

            # Wave reduce max across 64 lanes
            # Use shuffle_xor for wave reduction
            width_i32 = arith.constant(WARP_SIZE, type=T.i32)
            for _ in range_constexpr(6):  # log2(64) = 6
                peer = arith.ArithValue(local_max).shuffle_xor(arith.constant(1 << _, type=T.i32), width_i32)
                local_max = arith.maximumf(local_max, peer, fastmath=fm_fast)

            m_new = arith.maximumf(m_running, local_max, fastmath=fm_fast)

            # Correction factor for previous accumulators
            diff_m = arith.subf(m_running, m_new, fastmath=fm_fast)
            corr = arith.exp2(arith.mulf(diff_m, c_log2e, fastmath=fm_fast), fastmath=fm_fast)

            # Compute exp(scores - max) with scaling
            neg_scaled_max = arith.subf(c_zero_f, arith.mulf(m_new, c_sm_scale_log2e, fastmath=fm_fast), fastmath=fm_fast)

            p_vals = []
            local_sum = c_zero_f

            for r in range_constexpr(16):
                scaled_s = arith.mulf(s_raw[r], c_sm_scale_log2e, fastmath=fm_fast)
                diff = arith.addf(scaled_s, neg_scaled_max, fastmath=fm_fast)
                p = arith.exp2(diff, fastmath=fm_fast)
                p_vals.append(p)
                local_sum = arith.addf(local_sum, p, fastmath=fm_fast)

            # Wave reduce sum
            for _ in range_constexpr(6):
                peer = arith.ArithValue(local_sum).shuffle_xor(arith.constant(1 << _, type=T.i32), width_i32)
                local_sum = arith.addf(local_sum, peer, fastmath=fm_fast)

            l_new = arith.addf(arith.mulf(corr, l_running, fastmath=fm_fast), local_sum, fastmath=fm_fast)

            # Rescale O accumulators
            corr_v16 = vector.broadcast(v16f32_type, corr)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = arith.mulf(o_accs[dc], corr_v16, fastmath=fm_fast)

            # === Stage 2: P @ V using MFMA ===
            # Pack P values into bf16 for MFMA B-operand
            p_bf16_packs = []
            for pr in range_constexpr(16 // 8 if USE_K16 else 16 // 4):
                start_idx = pr * (8 if USE_K16 else 4)
                p_f32_vals = [p_vals[start_idx + i] for i in range_constexpr(8 if USE_K16 else 4)]
                # Convert f32 to bf16 via truncation
                p_i32_vals = []
                c16 = arith.constant(16, type=T.i32)
                cmask = arith.constant(0xFFFF0000, type=T.i32)
                for i in range_constexpr(len(p_f32_vals)):
                    p_i32 = arith.ArithValue(p_f32_vals[i]).bitcast(T.i32)
                    p_i32_vals.append(arith.shrui(p_i32, c16))
                # Pack into bf16 vector
                if USE_K16:
                    p_pack = vector.bitcast(v8bf16_type, vector.from_elements(T.vec(4, T.i32), [
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[1]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[0]).bitcast(T.i32), c16)).result,
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[3]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[2]).bitcast(T.i32), c16)).result,
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[5]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[4]).bitcast(T.i32), c16)).result,
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[7]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[6]).bitcast(T.i32), c16)).result,
                    ]))
                else:
                    p_pack = vector.bitcast(v4bf16_type, vector.from_elements(T.vec(2, T.i32), [
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[1]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[0]).bitcast(T.i32), c16)).result,
                        arith.OrIOp(arith.shrui(arith.ArithValue(p_f32_vals[3]).bitcast(T.i32), c16),
                                   arith.shli(arith.ArithValue(p_f32_vals[2]).bitcast(T.i32), c16)).result,
                    ]))
                p_bf16_packs.append(p_pack)

            # V @ P^T MFMA for each D chunk
            for dc in range_constexpr(D_CHUNKS):
                v_col_base = dc * 32

                for v_k_row_block in range_constexpr(2):  # 32 K rows per MFMA
                    v_k_row = v_k_row_block * 32 + lane_mod_32
                    v_k_row_valid = arith.cmpi(arith.CmpIPredicate.ult, v_k_row, tile_len)
                    v_k_row_safe = arith.select(v_k_row_valid, v_k_row, arith.index(0))

                    for ks in range_constexpr(32 // K_STEP_QK):
                        v_col = v_col_base + ks * K_STEP_QK + lane_div_32 * MFMA_LANE_K
                        lds_v_idx = LDS_K_TILE_SIZE + v_k_row_safe * V_STRIDE + v_col
                        v_pack = vector.load_op(mfma_pack_type, lds_kv, [lds_v_idx])
                        v_pack_safe = arith.select(v_k_row_valid, v_pack, arith.constant_vector(0.0, mfma_pack_type))

                        # P is B-operand, V is A-operand
                        p_idx = v_k_row_block * (32 // K_STEP_QK) + ks
                        if p_idx < len(p_bf16_packs) if isinstance(len(p_bf16_packs), int) else True:
                            o_accs[dc] = mfma_acc(v_pack_safe, p_bf16_packs[p_idx % (16 // (8 if USE_K16 else 4))], o_accs[dc])

            # Update running state
            m_running = m_new
            l_running = l_new

            gpu.barrier()

        # === Final normalization ===
        inv_l = arith.divf(arith.constant(1.0, type=compute_type), l_running, fastmath=fm_fast)
        inv_l_v16 = vector.broadcast(v16f32_type, inv_l)

        # === Store output ===
        o_stride = NUM_HEADS * V_HEAD_DIM
        o_base = q_idx * o_stride + head_idx * V_HEAD_DIM

        for dc in range_constexpr(D_CHUNKS):
            o_acc_scaled = arith.mulf(o_accs[dc], inv_l_v16, fastmath=fm_fast)

            # Convert f32 to bf16 and store
            o_offset = o_base + dc * 32
            # Extract 16 f32 values, convert to 16 bf16, pack into 2 x v8bf16
            for half in range_constexpr(2):
                o_vec_f32 = []
                for i in range_constexpr(8):
                    idx = half * 8 + i
                    val = vector.extract(o_acc_scaled, static_position=[idx], dynamic_position=[])
                    o_vec_f32.append(val)

                # Pack 8 f32 -> v8bf16
                c16 = arith.constant(16, type=T.i32)
                packed = []
                for i in range_constexpr(4):
                    hi = arith.ArithValue(o_vec_f32[i * 2 + 1]).bitcast(T.i32)
                    lo = arith.ArithValue(o_vec_f32[i * 2]).bitcast(T.i32)
                    packed.append(arith.OrIOp(arith.shrui(hi, c16), arith.shli(lo, c16)).result)
                o_v8bf16 = vector.bitcast(v8bf16_type, vector.from_elements(T.vec(4, T.i32), packed))

                buffer_ops.buffer_store(o_v8bf16, O, o_offset + half * 8)

    @flyc.jit
    def launch_mla_decode(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        kv_len: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = flyc.CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid: one block per (q_pos, head)
        total_q = Q.shape[0]
        grid_size = total_q * NUM_HEADS

        launcher = mla_decode_kernel(Q, K, V, O, kv_len)
        launcher.launch(
            grid=(grid_size, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _kernel_cache[key] = launch_mla_decode
    return launch_mla_decode


def custom_kernel(data: input_t) -> output_t:
    """MLA decode kernel using FlyDSL MFMA."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config['batchsize']
    num_heads = config.get('num_heads', NUM_HEADS)
    v_head_dim = config.get('v_head_dim', V_HEAD_DIM)

    total_q = q.shape[0]
    device = q.device

    kv_buffer = kv_data['bf16']
    output = torch.zeros((total_q, num_heads, v_head_dim), dtype=torch.bfloat16, device=device)

    # K cache: full 576 dims
    # V cache: first 512 dims
    k_cache = kv_buffer[:, 0, :]  # [total_kv, 576]
    v_cache = kv_buffer[:, 0, :v_head_dim]  # [total_kv, 512]

    # Get KV length from indptr
    kv_len = kv_indptr[1] - kv_indptr[0]

    # Compile and launch kernel
    kernel_fn = compile_mla_decode_kernel()

    kernel_fn(
        q,
        k_cache,
        v_cache,
        output,
        kv_len,
        None  # stream
    )

    return output