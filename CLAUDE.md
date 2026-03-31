# CLAUDE.md – GPU Computing Project (CUDA / ROCm)

## Project Overview
- **Name**: reference-kernels
- **Description**: Optimized kernel implementations for AMD MI355X GPU, focusing on MXFP4 quantized GEMM and MoE operations for LLM inference.
- **GPU Target**: AMD MI355X (CDNA4 architecture, ROCm 7.1+)

## Testing & Submission

### Popcorn CLI
```bash
cd problems/amd_202602/<problem-dir>

# Correctness test
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode test submission.py

# Benchmark
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode benchmark submission.py

# Profile (generates ROCm profile data)
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode profile submission.py

# Leaderboard ranking
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode leaderboard submission.py
```

### Rate Limits
- 6 submissions per hour per leaderboard
- If rate limited, wait and retry: "Try again in Xs"

### Common Submission Errors
- **"Device not configured (os error 6)"**: GPU service temporarily unavailable, retry after 30-60s
- **"another stream"**: Multiple submission.py files modified across directories, run:
  ```bash
  git checkout -- problems/amd_202602/<other-dirs>/submission.py
  ```

---

## Problem: moe-mxfp4 (MXFP4 MoE Fused Kernel)

### Problem Description
DeepSeek-R1 style MXFP4 Mixture-of-Experts (MoE) fused kernel:
- **Input**: `hidden_states [M, d_hidden]` bf16, MXFP4 weights (per-1x32 block scales)
- **Operation**: For each token, route to top-k experts, execute:
  - Stage 1: `gate = x @ W_gate.T`, `up = x @ W_up.T`, `intermediate = SiLU(gate) * up`
  - Stage 2: `output = intermediate @ W_down.T`
  - Weighted reduction across experts
- **Output**: `[M, d_hidden]` bf16

### DeepSeek-R1 MoE Specs
- hidden_size = 7168, moe_intermediate_size = 2048
- 256 routed experts + 1 shared expert (total 257)
- Top-8 routed + 1 shared = 9 experts per token
- 58 MoE layers (layer 3-60)

### AITER Reference Performance
| Batch | E | d_expert | Time (μs) |
|-------|---|----------|-----------|
| 16 | 257 | 256 | 152.7 |
| 128 | 257 | 256 | 239.0 |
| 512 | 257 | 256 | 336.5 |
| 16 | 33 | 512 | 106.2 |
| 128 | 33 | 512 | 141.1 |
| 512 | 33 | 512 | 225.0 |
| 512 | 33 | 2048 | 380.4 |

**Leaderboard best**: 109.793μs (geom mean) | **Target**: <150μs

### Implementation Details

```python
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

output = fused_moe(
    hidden_states,
    gate_up_weight_shuffled,
    down_weight_shuffled,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,  # MXFP4 per-1x32 block scaling
    doweight_stage1=False,
    w1_scale=gate_up_weight_scale_shuffled,
    w2_scale=down_weight_scale_shuffled,
    a1_scale=None,
    a2_scale=None,
    hidden_pad=hidden_pad,
    intermediate_pad=intermediate_pad,
)
```

### Key Insights
1. **AITER auto-tuning**: Selects optimal `block_m` based on batch size:
   - bs=16 → block_m=64 (kernel: `moe_ck2stages_gemm1_64x32x32x128_1x1`)
   - bs=128 → block_m=256 (kernel: `moe_ck2stages_gemm1_256x32x128x128_1x4`)
   - bs=512 → block_m=64
2. **JIT compilation**: First run has ~100s CK kernel build time
3. **2-stage pipeline**: Stage 1 (GEMM+SwiGLU) → LDS → Stage 2 (GEMM)

### Profile Analysis
From successful profile run (#673615):
- `ck_moe_stage1` dominates (~62% of kernel time)
- `ck_moe_stage2` is fast (<1% of kernel time)
- Sorting overhead is minimal

### File Structure
```
problems/amd_202602/moe-mxfp4/
├── submission.py              # Optimized implementation
├── reference.py               # AITER reference
├── task.py                    # Type definitions
├── task.yml                   # Test/benchmark configs
├── eval.py                    # Evaluation framework
└── moe_fused/                 # Custom CK kernel (not compiled on remote)
    ├── __init__.py
    ├── scheduler.py           # Expert load balancer
    ├── ck_fused.py            # Python interface
    └── cpp/moe_fused_kernel.cpp  # C++/HIP source
```

---

## Problem: mixed-mla (MLA Decode Kernel)

### Problem Description
DeepSeek-R1 forward_absorb MLA (Multi-head Latent Attention) decode kernel:
- **Input**: `q [total_q, 16, 576]` bf16 (absorbed query), MXFP4/FP8/BF16 KV cache
- **Operation**: Multi-head latent attention with MQA (1 KV head shared across 16 query heads):
  - QK computation: `scores = Q @ K.T * sm_scale` where `sm_scale = 1/sqrt(576)`
  - Attention: `output = softmax(scores) @ V`
  - KV buffer: 576 dims for keys, first 512 dims (kv_lora_rank) for values
- **Output**: `[total_q, 16, 512]` bf16

### DeepSeek-R1 MLA Config
```python
num_heads        = 16      # Query heads (after TP split)
num_kv_heads     = 1       # Shared latent KV head (MQA)
kv_lora_rank     = 512     # Latent dimension
qk_rope_head_dim = 64      # RoPE dimension
qk_head_dim      = 576     # Absorbed Q/K dimension (512 + 64)
v_head_dim       = 512     # Output dimension
sm_scale         = 1.0 / sqrt(576)
```

### KV Cache Formats
```python
kv_data = {
    "bf16":   Tensor(total_kv, 1, 576) bf16,                    # Highest precision
    "fp8":    (Tensor, Tensor) fp8 + scalar scale,              # Per-tensor quantized
    "mxfp4":  (Tensor, Tensor) fp4x2 + fp8_e8m0 scale,          # Block-32 quantized
}
```

### AITER Reference Implementation
```python
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# Persistent mode metadata setup
info = get_mla_metadata_info_v1(batch_size, max_q_len, nhead, q_dtype, kv_dtype, ...)
work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
(work_metadata, work_indptr, work_info_set,
 reduce_indptr, reduce_final_map, reduce_partial_map) = work

get_mla_metadata_v1(qo_indptr, kv_indptr, kv_last_page_len, ...)

# MLA decode forward
mla_decode_fwd(
    q.view(-1, nq, dq),
    kv_buffer_4d,
    o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
    max_q_len, page_size=PAGE_SIZE, nhead_kv=nkv,
    sm_scale=SM_SCALE, logit_cap=0.0,
    num_kv_splits=NUM_KV_SPLITS,
    q_scale=q_scale, kv_scale=kv_scale,
    intra_batch_mode=True,
    **meta,
)
```

### Implementation Strategies

#### 1. MXFP4 Path (4-bit quantization, 4x bandwidth savings)
```python
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

# Dequantize KV buffer once upfront
fp4_data_2d = kv_buffer_mxfp4.reshape(total_kv, qk_head_dim // 2)
float_vals = mxfp4_to_f32(fp4_data_2d)           # Unpack FP4 to float32
scale_f32 = e8m0_to_f32(scale_e8m0)              # Convert E8M0 scales
scale_f32 = scale_f32[:num_rows, :num_blocks]    # Trim padded dimensions
kv_dequant = (float_vals.view(...) * scale_f32.unsqueeze(-1)).view(...)

# Standard attention: Q @ K^T -> softmax -> @ V
scores = torch.matmul(qi_t, ki_t) * sm_scale
probs = F.softmax(scores, dim=-1, dtype=torch.float32)
output = torch.matmul(probs, vi_t)
```

#### 2. FP8 Path (a8w8, torch._scaled_mm)
```python
# Quantize Q to FP8
q_fp8, q_scale = quantize_fp8(q)

# FP8 GEMM using torch._scaled_mm
raw_scores = torch._scaled_mm(
    qi_fp8, ki_fp8.t(),
    scale_a=q_scale, scale_b=kv_scale_fp8,
    out_dtype=torch.float32,
)
```

### Performance Considerations
1. **Persistent mode overhead**: `get_mla_metadata_v1` adds ~15-20% overhead for small batches
2. **NUM_KV_SPLITS**: More splits = better parallelism but higher metadata overhead
   - Decode mode (seq_q=1): Use fewer splits (1-8)
   - Reference uses 32 splits
3. **MQA broadcast**: Load KV once, broadcast to all 16 query heads
4. **Dequant strategy**: For decode mode, dequant KV once upfront is optimal

### Test Tolerance
```python
check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-01)
# Relaxed tolerance due to quantization + aiter kernel differences
```

### Leaderboard Target
- **Leaderboard best**: ~11μs (aiter native kernel with optimal configuration)
- **Target**: <50μs for competitive ranking
- **Key to performance**: Use aiter's `mla_decode_fwd` directly (reference implementation)

---

## Problem: mxfp4-mm (MXFP4 Matrix Multiplication)

### Problem
MXFP4 GEMM: `bf16 A[M,K] × MXFP4 B[N,K] → bf16 C[M,N]`

### Performance (as of latest commit)

| Shape (M,N,K) | Time (µs) | Target <20µs |
|---------------|-----------|-------------|
| 4, 2880, 512 | 19.4 | ✅ |
| 16, 2112, 7168 | 33.3 | ❌ |
| 32, 4096, 512 | 19.6 | ✅ |
| 32, 2880, 512 | 19.8 | ✅ |
| 64, 7168, 2048 | 24.2 | ❌ |
| 256, 3072, 1536 | 23.1 | ❌ |

**Geometric mean: ~22.4µs** | **Target: <20µs** | **Leaderboard best: 1.0µs**

### Key Insights
1. **aiter baseline (~11.3µs)** measures pure GEMM with pre-quantized inputs
2. **Our implementation (~22.4µs)** includes `dynamic_mxfp4_quant(A)` adding ~8-10µs overhead
3. **To beat aiter**: Need fused quant+GEMM kernel to avoid HBM write of intermediate quantized activation
4. **Large K bottleneck**: Quantization overhead scales linearly with K dimension

---

## aiter Library Usage

### MXFP4 Quantization
```python
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter import dtypes

# Quantize bf16 -> MXFP4 per-1x32
A_fp4, A_scale = dynamic_mxfp4_quant(A)  # A: [M,K] bf16 -> A_fp4: [M,K/2] fp4
A_scale = e8m0_shuffle(A_scale)          # Rearrange scales for coalesced access
A_q = A_fp4.view(dtypes.fp4x2)           # View as packed FP4x2
A_scale_sh = A_scale.view(dtypes.fp8_e8m0)
```

### GEMM
```python
import aiter

C = aiter.gemm_a4w4(
    A_q,           # [M, K/2] fp4x2
    B_shuffle,     # [N, K/2] fp4x2 (shuffled with layout=(16,16))
    A_scale_sh,    # [M, K/32] fp8_e8m0
    B_scale_sh,    # [N, K/32] fp8_e8m0
    dtype=dtypes.bf16,
    bpreshuffle=True,
)
```

### Fused MoE
```python
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

output = fused_moe(
    hidden_states,     # [M, d_hidden] bf16
    w1_shuffled,       # [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
    w2_shuffled,       # [E, d_hidden_pad, d_expert_pad//2] fp4x2
    topk_weights,      # [M, top_k] float32
    topk_ids,          # [M, top_k] int32
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    w1_scale=w1_scale_shuffled,  # E8M0 scales
    w2_scale=w2_scale_shuffled,
    hidden_pad=...,
    intermediate_pad=...,
)
```

### MLA Decode
```python
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1, dtypes as aiter_dtypes

# FP8 quantization
finfo = torch.finfo(aiter_dtypes.fp8)
amax = tensor.abs().amax().clamp(min=1e-12)
scale = amax / finfo.max
fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(aiter_dtypes.fp8)

# Persistent mode metadata (required for mla_decode_fwd)
info = get_mla_metadata_info_v1(
    batch_size, max_q_len, nhead, q_dtype, kv_dtype,
    is_sparse=False, fast_mode=False,
    num_kv_splits=32, intra_batch_mode=True,
)
work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]

get_mla_metadata_v1(
    qo_indptr, kv_indptr, kv_last_page_len,
    nhead // nhead_kv, nhead_kv, True,  # is_causal
    work_metadata, work_info_set, work_indptr,
    reduce_indptr, reduce_final_map, reduce_partial_map,
    page_size=1, kv_granularity=16,
    max_seqlen_qo=max_q_len, uni_seqlen_qo=max_q_len,
    fast_mode=False, max_split_per_batch=32,
    intra_batch_mode=True, dtype_q=q_dtype, dtype_kv=kv_dtype,
)

# MLA decode forward
o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
mla_decode_fwd(
    q_fp8.view(-1, nq, dq),
    kv_buffer_4d,  # (total_kv, page_size, nhead_kv, dim)
    o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
    max_q_len, page_size=1, nhead_kv=nkv,
    sm_scale=SM_SCALE, logit_cap=0.0,
    num_kv_splits=32,
    q_scale=q_scale, kv_scale=kv_scale_fp8,
    intra_batch_mode=True,
    work_meta_data=work_metadata, work_indptr=work_indptr,
    work_info_set=work_info_set, reduce_indptr=reduce_indptr,
    reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
)
```

### MXFP4 Utilities
```python
from aiter.utility.fp4_utils import (
    dynamic_mxfp4_quant,  # bf16 -> fp4x2 + fp8_e8m0 scale
    mxfp4_to_f32,         # Unpack fp4x2 -> float32
    e8m0_to_f32,          # Convert fp8_e8m0 -> float32
)

# Quantize: bf16 -> MXFP4 (block-32, E8M0 scales)
fp4_data, scale_e8m0 = dynamic_mxfp4_quant(tensor_2d)  # Input: (rows, N)

# Dequantize: MXFP4 -> float32
float_vals = mxfp4_to_f32(fp4_data_2d)         # (num_rows, N)
scale_f32 = e8m0_to_f32(scale_e8m0)            # (padded_rows, padded_blocks)
scale_f32 = scale_f32[:num_rows, :num_blocks]  # Trim to actual dimensions
scaled = float_vals.view(num_rows, num_blocks, 32) * scale_f32.unsqueeze(-1)
```

---

## File Structure
```
problems/amd_202602/
├── mxfp4-mm/
│   ├── submission.py      # MXFP4 GEMM optimized kernel
│   ├── reference.py       # Reference implementation
│   ├── task.py            # Type definitions
│   ├── task.yml           # Benchmark shapes & aiter baseline
│   └── run_test.py        # Local testing (requires GPU)
├── moe-mxfp4/
│   ├── submission.py      # MoE fused kernel
│   ├── reference.py       # AITER fused_moe reference
│   ├── task.py            # Type definitions
│   ├── task.yml           # Test/benchmark configs
│   ├── eval.py            # Evaluation framework
│   └── moe_fused/         # Custom CK kernel module
├── mixed-mla/             # MLA (Multi-head Latent Attention) decode kernel
│   ├── submission.py      # Optimized MLA decode with MXFP4/FP8/BF16 paths
│   ├── reference.py       # AITER mla_decode_fwd reference
│   ├── task.py            # Type definitions
│   └── task.yml           # Test/benchmark configs
└── mxfp4/                 # Legacy MXFP4 problems
```

## Reference
- [AITER GitHub](https://github.com/ROCm/aiter) - AMD's kernel library
- [FlyDSL](https://github.com/ROCm/FlyDSL) - MLIR-based kernel DSL (for advanced fusion)
