"""
MXFP4 GEMM for AMD MI355X — Complete FlyDSL Implementation with MFMA_SCALE

C [M, N] = A [M, K] × B [N, K].T
A: bf16 (quantized to MXFP4), B: MXFP4 (fp4x2 + e8m0 scale), C: bf16

Uses FlyDSL's rocdl.mfma_scale_f32_16x16x128_f8f6f4 for hardware-accelerated FP4 GEMM.
"""

import torch
from task import input_t, output_t

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


def compile_mxfp4_gemm_kernel(K: int, tile_m: int = 64, tile_n: int = 64, tile_k: int = 128):
    """Compile MXFP4 GEMM kernel using FlyDSL with MFMA_SCALE."""
    key = (K, tile_m, tile_n, tile_k)
    if key in _kernel_cache:
        return _kernel_cache[key]

    BLOCK_SIZE = 256
    WARP_SIZE = 64
    elem_bytes = 1  # FP4x2 packed
    a_elem_vec_pack = 2

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_a")

    total_threads = BLOCK_SIZE
    bytes_a_per_tile = tile_m * tile_k // a_elem_vec_pack
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16
    num_a_loads = bytes_per_thread_a // a_load_bytes

    lds_stride_bytes = tile_k // a_elem_vec_pack
    lds_k_dim = tile_k // a_elem_vec_pack
    lds_tile_bytes = tile_m * lds_stride_bytes

    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + lds_tile_bytes

    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16

    def _elem_type():
        return T.i8

    def _vec16_type():
        return T.i8x16

    # =========================================================================
    # GPU Kernel
    # =========================================================================
    @flyc.kernel
    def kernel_gemm(
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

        # Accumulator initialization
        acc_init = arith.constant_vector(0.0, T.f32x4)

        # Buffer resources
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        # Thread and block IDs
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        bx_m = bx * tile_m
        by_n = by * tile_n

        # Wave / lane decomposition
        layout_wave_lane = fx.make_layout((num_waves, WARP_SIZE), (WARP_SIZE, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        n_tile_base = wave_id * n_per_wave

        # LDS buffer
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(
            base_ptr, lds_offset, _elem_type(), shape=(tile_m * tile_k,)
        ).get()

        _lds_k_dim = fx.Index(lds_k_dim)

        # Initialize accumulators
        n_accs = m_repeat * num_acc_n
        accs = [acc_init] * n_accs

        # Main compute loop over K tiles
        num_tiles = K // tile_k

        for tile_idx in range_constexpr(num_tiles):
            k_base = tile_idx * tile_k

            # Load A tile from global to LDS
            for load_idx in range_constexpr(num_a_loads):
                chunk_offset = load_idx * total_threads * 4
                tile_idx_i32 = tx * 4 + chunk_offset

                # Map to local coordinates
                layout_tile = fx.make_layout((tile_m, tile_k // 4 // a_elem_vec_pack), (tile_k // 4 // a_elem_vec_pack, 1))
                coord_local = fx.idx2crd(tile_idx_i32, layout_tile)
                local_row = fx.get(coord_local, 0)
                local_col_i32 = fx.get(coord_local, 1)

                global_row = bx_m + local_row
                global_col = k_base // 4 // a_elem_vec_pack + local_col_i32

                # Load 16 bytes (4 dwords) from A
                a_offset = global_row * (K // 2) // 4 + global_col
                a_data = buffer_ops.buffer_load(a_rsrc, a_offset, vec_width=4, dtype=T.i32x4)

                # Store to LDS with XOR swizzle for bank conflict avoidance
                k_blocks16 = lds_k_dim // 16
                swizzled_col = local_col_i32 * 4 ^ ((local_row % k_blocks16) * 16)
                lds_idx = local_row * _lds_k_dim + swizzled_col
                vector.store(vector.bitcast(_vec16_type(), a_data), lds_a, [lds_idx])

            gpu.barrier()

            # Compute MFMA for this K tile
            for mi in range_constexpr(m_repeat):
                curr_row = lane_mod_16 + (mi * 16)
                col_base = lane_div_16 * 16

                # Load A from LDS with swizzle
                k_blocks16 = lds_k_dim // 16
                swizzled_col = col_base ^ ((curr_row % k_blocks16) * 16)
                lds_idx = curr_row * _lds_k_dim + swizzled_col
                a_vec = vector.load(_vec16_type(), lds_a, [lds_idx])

                # Bitcast to i64x2 and extract
                a_i64x2 = vector.bitcast(T.i64x2, a_vec)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                # Pack for MFMA (128 FP4 elements)
                c0_i64 = arith.constant(0, type=T.i64)
                a128 = vector.bitcast(T.i32x8, vector.from_elements(T.vec(4, T.i64), [a0, a1, c0_i64, c0_i64]))

                for ni in range_constexpr(num_acc_n):
                    # Compute B global position (preshuffled layout)
                    # Layout: (n0, k0, klane, nlane, kpack) where n0 = N//16, k0 = K//64
                    n_global = by_n + n_tile_base + ni * 16 + lane_mod_16
                    n_blk = n_global // 16
                    n_intra = n_global % 16
                    k_blk = k_base // 64

                    # Strides for preshuffle layout
                    stride_n0 = (K // 64) * 64
                    stride_k0 = 64
                    stride_klane = 16
                    stride_nlane = 16

                    b_offset = n_blk * stride_n0 + k_blk * stride_k0 + lane_div_16 * stride_klane + n_intra * stride_nlane

                    # Load B
                    b_vec = buffer_ops.buffer_load(b_rsrc, b_offset, vec_width=4, dtype=T.i32x4)
                    b_i64x2 = vector.bitcast(T.i64x2, b_vec)
                    b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                    b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                    b128 = vector.bitcast(T.i32x8, vector.from_elements(T.vec(4, T.i64), [b0, b1, c0_i64, c0_i64]))

                    # Load scales
                    # Scale layout: [M//pack_M//16, K//128, pack_M*16] for A
                    #               [N//pack_N//16, K//128, pack_N*16] for B
                    scale_k_idx = k_base // 128
                    scale_a_row = bx_m // _FP4_PACK_M // 16
                    scale_b_row = (by_n + n_tile_base + ni * 16) // _FP4_PACK_N // 16

                    a_scale_offset = scale_a_row * (K // 128) * 64 + scale_k_idx * 64 + lane_id
                    b_scale_offset = scale_b_row * (K // 128) * 64 + scale_k_idx * 64 + lane_id

                    a_scale = buffer_ops.buffer_load(scale_a_rsrc, a_scale_offset, vec_width=1, dtype=T.i32)
                    b_scale = buffer_ops.buffer_load(scale_b_rsrc, b_scale_offset, vec_width=1, dtype=T.i32)

                    # MFMA_SCALE
                    acc_idx = mi * num_acc_n + ni
                    rocdl.sched_barrier(0)
                    accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [a128, b128, accs[acc_idx],
                         _FP4_CBSZ, _FP4_BLGP,
                         0, a_scale,  # opselA=0
                         0, b_scale],  # opselB=0
                    )

            gpu.barrier()

        # Store output
        for mi in range_constexpr(m_repeat):
            row_base = bx_m + wave_id * 16 + mi * 16 * num_waves
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

    # =========================================================================
    # Host Launcher
    # =========================================================================
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

        launcher = kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _kernel_cache[key] = launch_gemm
    return launch_gemm


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 GEMM using FlyDSL with MFMA_SCALE for FP4.

    Input:
        A: [M, K] bf16 - will be quantized to MXFP4
        B: [N, K] bf16 - reference weights (not used directly)
        B_q: [N, K//2] fp4x2 - quantized B (raw, not shuffled)
        B_shuffle: [N, K//2] fp4x2 - shuffled quantized B
        B_scale_sh: [N, K//32] e8m0 - shuffled scales for B

    Output:
        C: [M, N] bf16
    """
    A, B, B_q, B_shuffle, B_scale_sh = data

    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape

    # Quantize A to MXFP4
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = A_scale_sh.view(dtypes.fp8_e8m0)

    # Ensure B_shuffle and B_scale_sh are in correct format
    B_shuffle = B_shuffle.view(dtypes.fp4x2) if B_shuffle.dtype != torch.int8 else B_shuffle
    B_scale_sh = B_scale_sh.view(dtypes.fp8_e8m0) if B_scale_sh.dtype != torch.int8 else B_scale_sh

    # Allocate output
    C = torch.zeros((m, n), dtype=torch.bfloat16, device=A.device)

    # Compile and launch FlyDSL kernel
    tile_m, tile_n, tile_k = 64, 64, 128
    launch_fn = compile_mxfp4_gemm_kernel(k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    # Launch with stream=None (default stream)
    launch_fn(C, A_q, B_shuffle, A_scale_sh, B_scale_sh, m, n, None)

    return C
