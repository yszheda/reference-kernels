"""
MXFP4 MoE Fused Kernel — FlyDSL MFMA Implementation

DeepSeek-R1 style Mixture-of-Experts (MoE) on AMD MI355X.

Uses FlyDSL's MFMA_SCALE for FP4 GEMM in two stages:
1. Stage 1: gate and up projections with SiLU fusion
2. Stage 2: down projection with weighted reduction
"""

import torch
from task import input_t, output_t

import aiter
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle, mxfp4_to_f32, e8m0_to_f32
from aiter import dtypes

# FlyDSL imports
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr
from flydsl._mlir import ir
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

# FP4 MFMA constants
_FP4_CBSZ = 4
_FP4_BLGP = 4
_FP4_PACK_M = 2
_FP4_PACK_N = 2
_FP4_PACK_K = 2

_kernel_cache = {}


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def dequantize_mxfp4_weight(w_fp4, w_scale, out_dim):
    """Dequantize MXFP4 weight to bf16 for computation."""
    float_vals = mxfp4_to_f32(w_fp4)
    scale_f32 = e8m0_to_f32(w_scale)
    n_rows, n_blocks = float_vals.shape[0], scale_f32.shape[1]
    float_vals = float_vals.view(n_rows, n_blocks, 32)
    dequant = float_vals * scale_f32.unsqueeze(-1)
    return dequant.view(n_rows, -1).bfloat16()[:, :out_dim]


def compile_moe_fused_kernel(
    d_hidden: int,
    d_expert: int,
    K: int,  # K dimension (d_hidden for stage1, d_expert for stage2)
    tile_m: int = 64,
    tile_n: int = 64,
    tile_k: int = 128,
):
    """Compile fused MoE kernel using FlyDSL with MFMA_SCALE."""
    key = ("moe_fused", d_hidden, d_expert, K, tile_m, tile_n, tile_k)
    if key in _kernel_cache:
        return _kernel_cache[key]

    BLOCK_SIZE = 256
    WARP_SIZE = 64
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    elem_bytes = 1  # FP4x2 packed
    a_elem_vec_pack = 2

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_smem")

    lds_stride_bytes = tile_k // a_elem_vec_pack
    lds_k_dim = tile_k // a_elem_vec_pack
    lds_tile_bytes = tile_m * lds_stride_bytes

    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + lds_tile_bytes

    m_repeat = tile_m // 16
    n_per_wave = tile_n // NUM_WARPS
    num_acc_n = n_per_wave // 16

    def _elem_type():
        return T.i8

    def _vec16_type():
        return T.i8x16

    @flyc.kernel
    def moe_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        """MXFP4 GEMM kernel using MFMA_SCALE for FP4."""
        c_m = arith.index_cast(T.index, i32_m)
        c_n = arith.index_cast(T.index, i32_n)

        acc_init = arith.constant_vector(0.0, T.f32x4)

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        bx_m = bx * tile_m
        by_n = by * tile_n

        # Wave/lane decomposition
        layout_wave_lane = fx.make_layout((NUM_WARPS, WARP_SIZE), (WARP_SIZE, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        n_tile_base = wave_id * n_per_wave

        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_offset, _elem_type(), shape=(tile_m * tile_k,)).get()

        _lds_k_dim = fx.Index(lds_k_dim)
        k_blocks16 = lds_k_dim // 16

        n_accs = m_repeat * num_acc_n
        accs = [acc_init] * n_accs

        num_tiles = K // tile_k

        # Main compute loop
        for tile_idx in range_constexpr(num_tiles):
            k_base = tile_idx * tile_k

            # Load A tile to LDS cooperatively
            bytes_per_thread_a = (tile_m * tile_k // a_elem_vec_pack) // BLOCK_SIZE
            for load_idx in range_constexpr(max(1, bytes_per_thread_a // 16)):
                chunk_offset = load_idx * BLOCK_SIZE * 4
                tile_idx_i32 = tx * 4 + chunk_offset

                layout_tile = fx.make_layout((tile_m, tile_k // 4 // a_elem_vec_pack), (tile_k // 4 // a_elem_vec_pack, 1))
                coord_local = fx.idx2crd(tile_idx_i32, layout_tile)
                local_row = fx.get(coord_local, 0)
                local_col_i32 = fx.get(coord_local, 1)

                global_row = bx_m + local_row
                global_col = k_base // 4 // a_elem_vec_pack + local_col_i32

                a_offset = global_row * (K // 2) // 4 + global_col
                a_data = buffer_ops.buffer_load(a_rsrc, a_offset, vec_width=4, dtype=T.i32x4)

                swizzled_col = local_col_i32 * 4 ^ ((local_row % k_blocks16) * 16)
                lds_idx = local_row * _lds_k_dim + swizzled_col
                vector.store(vector.bitcast(_vec16_type(), a_data), lds_a, [lds_idx])

            gpu.barrier()

            # MFMA computation
            for mi in range_constexpr(m_repeat):
                curr_row = lane_mod_16 + (mi * 16)
                col_base = lane_div_16 * 16

                swizzled_col = col_base ^ ((curr_row % k_blocks16) * 16)
                lds_idx = curr_row * _lds_k_dim + swizzled_col
                a_vec = vector.load(_vec16_type(), lds_a, [lds_idx])

                a_i64x2 = vector.bitcast(T.i64x2, a_vec)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                c0_i64 = arith.constant(0, type=T.i64)
                a128 = vector.bitcast(T.i32x8, vector.from_elements(T.vec(4, T.i64), [a0, a1, c0_i64, c0_i64]))

                for ni in range_constexpr(num_acc_n):
                    n_global = by_n + n_tile_base + ni * 16 + lane_mod_16
                    n_blk = n_global // 16
                    n_intra = n_global % 16
                    k_blk = k_base // 64

                    # Preshuffle B layout strides
                    stride_n0 = (K // 64) * 64
                    stride_k0 = 64
                    stride_klane = 16
                    stride_nlane = 16

                    b_offset = n_blk * stride_n0 + k_blk * stride_k0 + lane_div_16 * stride_klane + n_intra * stride_nlane

                    b_vec = buffer_ops.buffer_load(b_rsrc, b_offset, vec_width=4, dtype=T.i32x4)
                    b_i64x2 = vector.bitcast(T.i64x2, b_vec)
                    b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                    b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                    b128 = vector.bitcast(T.i32x8, vector.from_elements(T.vec(4, T.i64), [b0, b1, c0_i64, c0_i64]))

                    scale_k_idx = k_base // 128
                    scale_a_row = bx_m // _FP4_PACK_M // 16
                    scale_b_row = (by_n + n_tile_base + ni * 16) // _FP4_PACK_N // 16

                    a_scale_offset = scale_a_row * (K // 128) * 64 + scale_k_idx * 64 + lane_id
                    b_scale_offset = scale_b_row * (K // 128) * 64 + scale_k_idx * 64 + lane_id

                    a_scale = buffer_ops.buffer_load(scale_a_rsrc, a_scale_offset, vec_width=1, dtype=T.i32)
                    b_scale = buffer_ops.buffer_load(scale_b_rsrc, b_scale_offset, vec_width=1, dtype=T.i32)

                    acc_idx = mi * num_acc_n + ni
                    rocdl.sched_barrier(0)
                    accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [a128, b128, accs[acc_idx],
                         _FP4_CBSZ, _FP4_BLGP,
                         0, a_scale,
                         0, b_scale],
                    )

            gpu.barrier()

        # Store output
        for mi in range_constexpr(m_repeat):
            row_base = bx_m + wave_id * 16 + mi * 16 * NUM_WARPS
            for ii in range_constexpr(4):
                row = row_base + lane_div_16 * 4 + ii
                row_guard = row < c_m
                for ni in range_constexpr(num_acc_n):
                    col = by_n + n_tile_base + ni * 16 + lane_mod_16
                    col_guard = col < c_n

                    acc_idx = mi * num_acc_n + ni
                    val = vector.extract(accs[acc_idx], static_position=[ii], dynamic_position=[])
                    val_bf16 = arith.trunc_f(T.bf16(), val)

                    out_idx = row * c_n + col
                    buffer_ops.buffer_store(val_bf16, c_rsrc, out_idx)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = flyc.CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        launcher = moe_gemm_kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _kernel_cache[key] = launch_gemm
    return launch_gemm


def custom_kernel(data: input_t) -> output_t:
    """
    MoE MXFP4 kernel using FlyDSL MFMA implementation.
    """
    (hidden_states,
     gate_up_weight,
     down_weight,
     gate_up_weight_scale,
     down_weight_scale,
     gate_up_weight_shuffled,
     down_weight_shuffled,
     gate_up_weight_scale_shuffled,
     down_weight_scale_shuffled,
     topk_weights,
     topk_ids,
     config) = data

    M, d_hidden = hidden_states.shape
    d_expert = config.get('dexpert', 256)
    top_k = config.get('nexpertspertoken', 8) + config.get('nsharedexperts', 1)

    device = hidden_states.device
    output = torch.zeros((M, d_hidden), dtype=torch.bfloat16, device=device)

    # Quantize hidden states to MXFP4
    hidden_fp4, hidden_scale = dynamic_mxfp4_quant(hidden_states)
    hidden_scale_sh = e8m0_shuffle(hidden_scale)
    hidden_q = hidden_fp4.view(dtypes.fp4x2)
    hidden_scale_sh = hidden_scale_sh.view(dtypes.fp8_e8m0)

    # Compile kernels for this configuration
    stage1_kernel = compile_moe_fused_kernel(d_hidden, d_expert, d_hidden)
    stage2_kernel = compile_moe_fused_kernel(d_expert, d_hidden, d_expert)

    # Process each position in top-k
    for k_idx in range(top_k):
        expert_ids = topk_ids[:, k_idx]
        weights_k = topk_weights[:, k_idx].float()

        unique_experts = expert_ids.unique()

        for expert_id in unique_experts:
            expert_id_val = expert_id.item()
            mask = (expert_ids == expert_id)
            token_indices = mask.nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            num_tokens = len(token_indices)

            # Get hidden states for this expert's tokens
            x = hidden_q[token_indices]  # [num_tokens, d_hidden//2] fp4x2
            x_scale = hidden_scale_sh[token_indices]  # [num_tokens, d_hidden//32]

            # Get expert weights
            w_gate_up = gate_up_weight_shuffled[expert_id_val]
            w_gate_up_sc = gate_up_weight_scale_shuffled[expert_id_val]
            w_down = down_weight_shuffled[expert_id_val]
            w_down_sc = down_weight_scale_shuffled[expert_id_val]

            # Allocate intermediate buffer
            intermediate = torch.zeros((num_tokens, 2 * d_expert), dtype=torch.bfloat16, device=device)

            # Stage 1: fused gate + up projection using FlyDSL kernel
            stage1_kernel(
                intermediate, x, w_gate_up, x_scale, w_gate_up_sc,
                num_tokens, 2 * d_expert, None
            )

            # Split and apply SiLU
            gate = intermediate[:, :d_expert].float()
            up = intermediate[:, d_expert:].float()
            intermediate_fused = silu(gate) * up

            # Allocate expert output
            expert_out = torch.zeros((num_tokens, d_hidden), dtype=torch.bfloat16, device=device)

            # Stage 2: down projection using FlyDSL kernel
            # Need to quantize intermediate for stage2
            int_fp4, int_scale = dynamic_mxfp4_quant(intermediate_fused.bfloat16())
            int_scale_sh = e8m0_shuffle(int_scale)
            int_q = int_fp4.view(dtypes.fp4x2)
            int_scale_sh = int_scale_sh.view(dtypes.fp8_e8m0)

            stage2_kernel(
                expert_out, int_q, w_down, int_scale_sh, w_down_sc,
                num_tokens, d_hidden, None
            )

            # Accumulate with weights
            for i, token_idx in enumerate(token_indices):
                w = weights_k[token_idx.item()].item()
                output[token_idx] += w * expert_out[i]

    return output