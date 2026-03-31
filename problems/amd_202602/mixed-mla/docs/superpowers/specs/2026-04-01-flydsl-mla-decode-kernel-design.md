# FlyDSL MLA Decode Kernel Design Specification

**Date:** 2026-04-01
**Author:** Claude (via superpowers:brainstorming skill)
**Status:** Approved

---

## Overview

This document specifies the design of a pure FlyDSL implementation of the DeepSeek-R1 forward_absorb MLA (Multi-head Latent Attention) decode kernel, optimized for AMD MI355X (CDNA4 architecture).

### Goals

1. **Performance**: Achieve competitive latency (<50μs geometric mean across benchmark configs)
2. **Correctness**: Match reference implementation within tolerance (rtol=1e-01, atol=1e-01)
3. **Self-contained**: All code in single `submission.py` file for Popcorn CLI submission

### Non-Goals

1. Training support — decode only (q_seq_len=1)
2. FP8/BF16 KV paths — MXFP4 is the primary optimization target
3. Custom HIP/CK kernels — FlyDSL only, no external compilation

---

## Architecture

### Kernel Strategy

| Aspect | Design Choice |
|--------|---------------|
| **Program model** | One workgroup per (batch_idx, head_idx, q_token) |
| **KV tiling** | Block loop over KV sequence with on-the-fly dequant |
| **MXFP4 dequant** | Per-thread dequant in registers (Option A) |
| **GEMM** | Blocked dot-product using MFMA-style accumulation |
| **Softmax** | Online softmax with running max/sum |
| **MQA** | Load KV once per workgroup, broadcast across 16 query heads |

### Memory Hierarchy

```
HBM (Global Memory)
├── q_ptr: [total_q, 16, 576] bf16
├── kv_ptr: [total_kv, 1, 288] fp4x2
├── kv_scale_ptr: [total_kv, 18] fp8_e8m0
└── out_ptr: [total_q, 16, 512] bf16

VGPR (Per-Thread Registers)
├── q_vec: [576] float32 — loaded once
├── acc: [512] float32 — output accumulator
├── k_block: [BLOCK_KV, 576] float32 — dequantized per block
└── v_block: [BLOCK_KV, 512] float32 — dequantized per block

LDS (Shared Memory, optional)
└── Not used in initial design (per-thread dequant avoids LDS)
```

---

## Kernel Interface

### FlyDSL Kernel Signature

```python
@kernel
def mla_decode_kernel(
    # Output
    out_ptr: Pointer[bfloat16],     # [total_q, num_heads, v_head_dim]

    # Inputs
    q_ptr: Pointer[bfloat16],       # [total_q, num_heads, qk_head_dim]
    kv_ptr: Pointer[fp4x2],         # [total_kv, num_kv_heads, qk_head_dim//2]
    kv_scale_ptr: Pointer[fp8_e8m0],# [total_kv, qk_head_dim//32]

    # Indirect pointers
    qo_indptr: Pointer[int32],      # [batch_size+1]
    kv_indptr: Pointer[int32],      # [batch_size+1]

    # Dimensions
    num_heads: int32,
    num_kv_heads: int32,
    qk_head_dim: int32,
    v_head_dim: int32,
    sm_scale: float32,

    # Block sizes (compile-time constants)
    BLOCK_KV: int32 = 64,
    BLOCK_DK: int32 = 64,
    BLOCK_DV: int32 = 64,
):
```

### Grid Configuration

```python
# Grid: (batch_size, num_heads, q_seq_len)
# For decode mode: q_seq_len = 1, so effectively (batch_size, num_heads)
grid = (config['batch_size'], config['num_heads'], config['q_seq_len'])
```

---

## Algorithm

### High-Level Flow

```
1. Compute (batch_idx, head_idx, q_token_idx) from program_id
2. Load Q vector from HBM → VGPR (576 dims)
3. Initialize online softmax state: m_i = -inf, d_i = 0, acc = 0
4. Block loop over KV sequence:
   a. Load MXFP4 K block from HBM
   b. Per-thread dequantize K (fp4x2 → float32, per-32 block scales)
   c. Q @ K^T → scores (blocked dot-product)
   d. Apply sm_scale, update online softmax (m_i, d_i)
   e. Load MXFP4 V block from HBM
   f. Per-thread dequantize V
   g. acc = acc * exp(m_i - m_new) + scores @ V
5. Finalize: output = acc / d_i → bf16, store to HBM
```

### Online Softmax

Following the Flash Attention approach:

```python
# Per block
m_block = max(scores)
m_new = max(m_i, m_block)
d_i = d_i * exp(m_i - m_new) + sum(exp(scores - m_new))
m_i = m_new

# After all blocks
output = acc / d_i
```

### MXFP4 Dequantization (Per-Thread)

```python
@function
def dequantize_mxfp4(fp4_byte: int32, scale_fp8: int32) -> tuple[float32, float32]:
    """Dequantize one fp4x2 byte with E8M0 scale."""
    val0 = fp4_byte & 0x0F           # Low 4 bits
    val1 = (fp4_byte >> 4) & 0x0F    # High 4 bits
    scale_f32 = e8m0_to_float32(scale_fp8)
    return fp4_to_float32(val0) * scale_f32, fp4_to_float32(val1) * scale_f32
```

**MXFP4 Format:**
- 2 FP4 values per byte (E2M1 format: 1 sign, 2 exp, 1 mantissa)
- Block size: 32 elements share one E8M0 scale
- Layout: `fp4_data[row, col//2]`, `scale[row, col//32]`

---

## File Structure

```
problems/amd_202602/mixed-mla/
├── submission.py          # FlyDSL kernel + wrapper (ALL implementation)
├── reference.py           # Unchanged (aiter reference)
├── task.py                # Unchanged (type definitions)
├── task.yml               # Unchanged (test/benchmark configs)
└── eval.py                # Unchanged (evaluation framework)
```

### `submission.py` Sections

1. **Imports & Constants** — FlyDSL imports, MLA configuration
2. **Helper Functions** — `@function` decorated dequant helpers
3. **Main Kernel** — `mla_decode_kernel` with block loop
4. **Python Wrapper** — `custom_kernel()` entry point
5. **Fallback** — Reference implementation for local testing

---

## Testing Strategy

### Correctness Validation

```bash
# Run all test cases
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode test submission.py

# Expected: All tests pass with rtol=1e-01, atol=1e-01
```

### Benchmark Targets

| Config | Target Latency |
|--------|----------------|
| bs=4, kv=1024 | <20μs |
| bs=4, kv=8192 | <40μs |
| bs=32, kv=1024 | <25μs |
| bs=32, kv=8192 | <50μs |
| bs=64, kv=1024 | <30μs |
| bs=64, kv=8192 | <60μs |
| bs=256, kv=1024 | <40μs |
| bs=256, kv=8192 | <80μs |

**Geometric mean target:** <50μs

### Profiling

```bash
# Generate ROCm profile data
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode profile submission.py
```

Use profile data to identify bottlenecks (memory vs. compute bound).

---

## Tuning Parameters

### Block Sizes

| Parameter | Default | Tuning Range |
|-----------|---------|--------------|
| BLOCK_KV | 64 | 32, 64, 128 |
| BLOCK_DK | 64 | 32, 64, 128 |
| BLOCK_DV | 64 | 32, 64, 128 |

### Optimization Phases

1. **Phase 1:** Correct kernel with default blocks (prove functionality)
2. **Phase 2:** Tune block sizes per benchmark config
3. **Phase 3:** Hybrid dequant (Option C) if memory-bound
4. **Phase 4:** MFMA intrinsics if compute-bound

---

## Error Handling

### FlyDSL Unavailable (Local Testing)

```python
if not FLYDSL_AVAILABLE:
    # Fallback to reference implementation
    return ref_kernel_fallback(data)
```

### Kernel Launch Failures

- Verify GPU device is available
- Check tensor shapes match expected layout
- Ensure block sizes divide dimensions evenly (or handle boundary)

---

## Success Criteria

- [ ] All test cases pass correctness validation
- [ ] Geometric mean <50μs across benchmark configs
- [ ] Profile data shows efficient memory utilization
- [ ] Code fits in single `submission.py` file
- [ ] No external dependencies beyond FlyDSL

---

## References

- [DeepSeek-R1 Model Configuration](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/config.json)
- [AMD CDNA4 Architecture](https://www.amd.com/en/products/accelerators/instinct/mi355x.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [MXFP4 Format (OCP Microscaling)](https://opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [FlyDSL GitHub](https://github.com/ROCm/FlyDSL)
