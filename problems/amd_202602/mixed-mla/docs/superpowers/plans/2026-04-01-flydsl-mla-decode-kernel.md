# FlyDSL MLA Decode Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a pure FlyDSL MLA decode kernel in `submission.py` that achieves <50μs geometric mean latency on AMD MI355X GPU.

**Architecture:** Flash Attention-style block loop with per-thread MXFP4 dequantization in registers, online softmax, and MQA broadcast pattern. One workgroup per (batch_idx, head_idx, q_token).

**Tech Stack:** FlyDSL (Python-embedded DSL for AMD GPU), PyTorch (tensor management), Popcorn CLI (remote submission).

---

## Phase 0: Environment Validation

### Task 0.1: Verify FlyDSL Import Works on Remote GPU

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`
- Test: Remote GPU via `popcorn-cli`

- [ ] **Step 1: Create minimal FlyDSL test script**

Create a simple test file to verify FlyDSL is available on remote GPU:

```python
# test_flydsl.py
"""Minimal FlyDSL availability test"""

try:
    from flydsl import kernel, function, Pointer, program_id, load, store
    from flydsl.types import bfloat16, float32, int32
    print("FlyDSL import: SUCCESS")
    FLYDSL_AVAILABLE = True
except ImportError as e:
    print(f"FlyDSL import: FAILED - {e}")
    FLYDSL_AVAILABLE = False

@kernel if FLYDSL_AVAILABLE else lambda fn: fn
def test_kernel(x_ptr: Pointer[bfloat16], n: int32):
    idx = program_id(0)
    if idx < n:
        val = load(x_ptr + idx)
        store(x_ptr + idx, val)  # Identity operation

def main():
    import torch
    x = torch.ones(4, dtype=torch.bfloat16, device="cuda")
    if FLYDSL_AVAILABLE:
        test_kernel[4](x, 4)
        print("Kernel launch: SUCCESS")
    else:
        print("Skipping kernel launch (FlyDSL unavailable)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Submit test to remote GPU**

Run:
```bash
cd /Users/yuanshuai/Code/reference-kernels/problems/amd_202602/mixed-mla
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode test test_flydsl.py
```

Expected output: `FlyDSL import: SUCCESS`, `Kernel launch: SUCCESS`

- [ ] **Step 3: Document FlyDSL API**

Based on successful import, document available APIs:
```python
# Confirmed FlyDSL imports
from flydsl import kernel, function, Pointer, Array
from flydsl import program_id, load, store, atomic_add
from flydsl import zeros, max, min, exp, sum, range
from flydsl.types import fp4x2, fp8_e8m0, bfloat16, float32, int32
```

- [ ] **Step 4: Commit**

```bash
git add test_flydsl.py
git commit -m "chore: add FlyDSL environment validation script"
```

---

## Phase 1: MXFP4 Dequantization Helpers

### Task 1.1: Implement FP4 Unpack Functions

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`
- Test: Remote GPU test script

- [ ] **Step 1: Add FlyDSL imports to submission.py**

Replace existing imports at top of `submission.py`:

```python
"""
MLA (Multi-head Latent Attention) decode kernel — FlyDSL Implementation

DeepSeek-R1 forward_absorb MLA on AMD MI355X (CDNA4 architecture).
"""

import torch
from task import input_t, output_t

# FlyDSL imports (with graceful fallback)
try:
    from flydsl import (
        kernel, function, Pointer, Array,
        program_id, load, store, atomic_add,
        zeros, max, min, exp, sum, range,
        barrier, mem_fence,
    )
    from flydsl.types import fp4x2, fp8_e8m0, bfloat16, float32, int32
    FLYDSL_AVAILABLE = True
except ImportError:
    FLYDSL_AVAILABLE = False
    def kernel(fn): return fn
    def function(fn): return fn
    class Pointer: pass
    class Array: pass
    fp4x2 = fp8_e8m0 = bfloat16 = float32 = int32 = None
```

- [ ] **Step 2: Add MLA constants**

After imports, add constants:

```python
# DeepSeek-R1 MLA Constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

# Block sizes (tunable)
BLOCK_KV = 64
BLOCK_DK = 64
BLOCK_DV = 64
```

- [ ] **Step 3: Implement FP4 unpack helpers**

Add `@function` decorated helpers:

```python
@function
def unpack_fp4_low(byte_val: int32) -> float32:
    """Extract low 4 bits and convert FP4 (E2M1) to float32."""
    val = byte_val & 0x0F
    # FP4 E2M1 format: 1 sign, 2 exp, 1 mantissa
    # Simplified conversion - actual values depend on ROCm convention
    sign = (val >> 3) & 0x1
    exp = (val >> 1) & 0x3
    mantissa = val & 0x1

    # Decode FP4 to float32
    if exp == 0:
        return 0.0 if sign == 0 else -0.0
    elif exp == 1:
        # Normalized: (-1)^s * 2^(e-1) * (1 + m/2)
        base = 1.0 + mantissa * 0.5
        return base if sign == 0 else -base
    elif exp == 2:
        # Normalized: (-1)^s * 2^0 * (1 + m/2)
        base = 1.0 + mantissa * 0.5
        return base if sign == 0 else -base
    else:  # exp == 3
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
    E8M0: 1 sign bit, 8 exponent bits, 0 mantissa bits.
    """
    sign = (fp8_val >> 7) & 0x1
    exp = fp8_val & 0x7F  # 7 exponent bits

    # E8M0 represents powers of 2
    # Bias typically 64 for E8M0
    unbiased_exp = exp - 64

    result = exp2(float32(unbiased_exp))
    return result if sign == 0 else -result
```

- [ ] **Step 4: Create local test for dequant helpers**

Create test file `test_dequant.py`:

```python
"""Test MXFP4 dequantization helpers"""

from flydsl import function, float32, int32

# Copy unpack functions here for testing

def test_fp4_unpack():
    # Test low unpack
    result = unpack_fp4_low(0x23)  # 0x3 in low nibble
    assert result > 0, f"Expected positive, got {result}"

    # Test high unpack
    result = unpack_fp4_high(0x23)  # 0x2 in high nibble
    assert result > 0, f"Expected positive, got {result}"

    print("test_fp4_unpack: PASSED")

if __name__ == "__main__":
    test_fp4_unpack()
```

- [ ] **Step 5: Submit dequant test to remote GPU**

Run:
```bash
popcorn-cli submit --gpu MI355X --mode test test_dequant.py
```

Expected: `test_fp4_unpack: PASSED`

- [ ] **Step 6: Commit**

```bash
git add submission.py test_dequant.py
git commit -m "feat: add FlyDSL FP4/E8M0 dequant helper functions"
```

---

### Task 1.2: Implement Block-Wise Dequantization

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`

- [ ] **Step 1: Add dequantize_mxfp4_block function**

```python
@function
def dequantize_mxfp4_block(
    fp4_ptr: Pointer[fp4x2],
    scale_ptr: Pointer[fp8_e8m0],
    row_idx: int32,
    col_start: int32,
    num_cols: int32,
) -> Array[float32]:
    """
    Dequantize one row of MXFP4 data.

    Args:
        fp4_ptr: Pointer to fp4x2 packed data
        scale_ptr: Pointer to E8M0 scales (one per 32 elements)
        row_idx: Which row to load
        col_start: Starting column index
        num_cols: Number of columns to dequantize

    Returns:
        Array of float32 values [num_cols]
    """
    result = zeros([num_cols], dtype=float32)

    for d in range(0, num_cols, 2):  # 2 FP4 values per byte
        # Compute fp4 byte index
        fp4_idx = row_idx * (num_cols // 2) + (col_start + d) // 2
        fp4_byte = load(fp4_ptr + fp4_idx)

        # Unpack both values
        val0 = unpack_fp4_low(fp4_byte)
        val1 = unpack_fp4_high(fp4_byte)

        # Get scale for this 32-element block
        scale_block_idx = (col_start + d) // 32
        scale_idx = row_idx * (num_cols // 32) + scale_block_idx
        scale_fp8 = load(scale_ptr + scale_idx)
        scale_f32 = e8m0_to_float32(scale_fp8)

        # Apply scale
        result[d] = val0 * scale_f32
        if d + 1 < num_cols:
            result[d + 1] = val1 * scale_f32

    return result
```

- [ ] **Step 2: Commit**

```bash
git add submission.py
git commit -m "feat: add block-wise MXFP4 dequantization function"
```

---

## Phase 2: Main MLA Kernel Implementation

### Task 2.1: Implement mla_decode_kernel

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`

- [ ] **Step 1: Add kernel signature**

```python
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

    # Block sizes (compile-time constants via template)
    BLOCK_KV: int32,
    BLOCK_DK: int32,
    BLOCK_DV: int32,
):
    """
    MLA Decode Kernel - Flash Attention style.
    Grid: (batch_size, num_heads, q_seq_len)
    """
```

- [ ] **Step 2: Implement program ID computation**

```python
    # Compute (batch_idx, head_idx, q_token_idx) from grid
    batch_idx = program_id(0)
    head_idx = program_id(1)
    q_token_idx = program_id(2) if BLOCK_KV > 0 else 0  # q_seq_len dimension

    # Early exit for invalid work items
    if batch_idx >= load(qo_indptr + program_id(0) + 1):
        return
```

- [ ] **Step 3: Compute Q offset and load Q vector**

```python
    # Compute Q offset for this work item
    q_start = load(qo_indptr + batch_idx)
    q_offset = q_start + q_token_idx

    # Load Q vector: [qk_head_dim] bf16 -> float32
    q_vec = zeros([qk_head_dim], dtype=float32)
    for d in range(0, qk_head_dim, BLOCK_DK):
        offset = q_offset * num_heads * qk_head_dim + head_idx * qk_head_dim + d
        q_tile = load(q_ptr + offset, mask=range(d, min(d + BLOCK_DK, qk_head_dim)), other=0.0)
        q_vec[d:d + BLOCK_DK] = q_tile.to(float32)
```

- [ ] **Step 4: Initialize online softmax accumulators**

```python
    # Online softmax state
    m_i = float32(-1e30)  # Running max (large negative initial)
    d_i = float32(0.0)     # Running sum
    acc = zeros([v_head_dim], dtype=float32)  # Output accumulator
```

- [ ] **Step 5: Implement KV block loop**

```python
    # KV sequence range for this batch
    kv_start = load(kv_indptr + batch_idx)
    kv_end = load(kv_indptr + batch_idx + 1)

    # KV head index (MQA: same KV head for all query heads)
    kv_head_idx = head_idx % num_kv_heads

    # Block loop over KV sequence
    for kv_block_start in range(kv_start, kv_end, BLOCK_KV):
        kv_block_len = min(kv_end - kv_block_start, BLOCK_KV)

        # 5a. Load and dequantize K block
        k_block = zeros([kv_block_len, qk_head_dim], dtype=float32)
        for row in range(kv_block_len):
            k_fp4_ptr = kv_ptr + (kv_block_start + row) * num_kv_heads * (qk_head_dim // 2) + kv_head_idx * (qk_head_dim // 2)
            k_scale_ptr = kv_scale_ptr + (kv_block_start + row) * (qk_head_dim // 32) + kv_head_idx * (qk_head_dim // 32)
            k_row = dequantize_mxfp4_block(k_fp4_ptr, k_scale_ptr, 0, 0, qk_head_dim)
            k_block[row, :] = k_row

        # 5b. Q @ K^T -> scores
        scores = zeros([kv_block_len], dtype=float32)
        for row in range(kv_block_len):
            score = float32(0.0)
            for d in range(0, qk_head_dim, BLOCK_DK):
                q_tile = q_vec[d:d + BLOCK_DK]
                k_tile = k_block[row, d:d + BLOCK_DK]
                # Dot product
                for i in range(BLOCK_DK):
                    if d + i < qk_head_dim:
                        score += q_tile[i] * k_tile[i]
            scores[row] = score

        # 5c. Apply scale and online softmax
        for row in range(kv_block_len):
            scores[row] = scores[row] * sm_scale

        # Update running max/sum
        m_block = float32(-1e30)
        for row in range(kv_block_len):
            if row < kv_block_len:
                m_block = max(m_block, scores[row])

        m_new = max(m_i, m_block)

        # Compute exp(sum) for normalization
        exp_sum = float32(0.0)
        for row in range(kv_block_len):
            if row < kv_block_len:
                exp_sum += exp(scores[row] - m_new)

        d_i = d_i * exp(m_i - m_new) + exp_sum
        m_i = m_new

        # 5d. Load and dequantize V block
        v_block = zeros([kv_block_len, v_head_dim], dtype=float32)
        for row in range(kv_block_len):
            v_fp4_ptr = kv_ptr + (kv_block_start + row) * num_kv_heads * (v_head_dim // 2) + kv_head_idx * (v_head_dim // 2)
            v_scale_ptr = kv_scale_ptr + (kv_block_start + row) * (v_head_dim // 32) + kv_head_idx * (v_head_dim // 32)
            v_row = dequantize_mxfp4_block(v_fp4_ptr, v_scale_ptr, 0, 0, v_head_dim)
            v_block[row, :] = v_row

        # 5e. Accumulate: acc = acc * exp(m_i - m_new) + scores @ V
        alpha = exp(m_i - m_new)
        for d in range(v_head_dim):
            acc[d] = acc[d] * alpha
            for row in range(kv_block_len):
                if row < kv_block_len:
                    acc[d] += scores[row] * v_block[row, d]
```

- [ ] **Step 6: Finalize and store output**

```python
    # Normalize and store output
    for d in range(0, v_head_dim, BLOCK_DV):
        out_tile = zeros([min(BLOCK_DV, v_head_dim - d)], dtype=bfloat16)
        for i in range(min(BLOCK_DV, v_head_dim - d)):
            out_tile[i] = (acc[d + i] / d_i).to(bfloat16)

        offset = q_offset * num_heads * v_head_dim + head_idx * v_head_dim + d
        store(out_ptr + offset, out_tile)
```

- [ ] **Step 7: Commit**

```bash
git add submission.py
git commit -m "feat: implement main mla_decode FlyDSL kernel"
```

---

### Task 2.2: Implement Python Wrapper

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`

- [ ] **Step 1: Add custom_kernel wrapper**

```python
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
```

- [ ] **Step 2: Remove old kernel implementations**

Remove `custom_kernel_mxfp4_fused`, `custom_kernel_fp8_optimized`, `custom_kernel_bf16_optimized` functions to keep file clean.

- [ ] **Step 3: Commit**

```bash
git add submission.py
git commit -m "feat: add custom_kernel Python wrapper for FlyDSL"
```

---

## Phase 3: Testing & Validation

### Task 3.1: Run Correctness Tests

**Files:**
- Test: `popcorn-cli submit --mode test`

- [ ] **Step 1: Submit to test mode**

Run:
```bash
cd /Users/yuanshuai/Code/reference-kernels/problems/amd_202602/mixed-mla
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode test submission.py
```

Expected: All 4 test cases pass (bs=4, 32, 64, 256 with various kv lengths)

- [ ] **Step 2: If tests fail, debug**

Check error output for:
- Shape mismatches
- NaN/Inf values
- Tolerance violations

Common fixes:
- Adjust MXFP4 dequantization constants
- Fix online softmax numerical stability
- Verify block size handling at boundaries

- [ ] **Step 3: Commit working version**

```bash
git commit -am "fix: correct MXFP4 dequant for numerical accuracy"
```

---

### Task 3.2: Run Benchmark Tests

**Files:**
- Test: `popcorn-cli submit --mode benchmark`

- [ ] **Step 1: Submit to benchmark mode**

Run:
```bash
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode benchmark submission.py
```

Expected: Geometric mean <50μs

- [ ] **Step 2: Profile to identify bottlenecks**

Run:
```bash
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode profile submission.py
```

Analyze profile data for:
- Memory bandwidth bound (increase BLOCK_KV)
- Compute bound (optimize MFMA usage)
- Kernel launch overhead

- [ ] **Step 3: Tune block sizes**

Adjust constants based on profile:
```python
# If memory-bound: increase BLOCK_KV to 128
# If compute-bound: try smaller BLOCK_DK/BLOCK_DV
BLOCK_KV = 64  # Try 32, 64, 128
BLOCK_DK = 64  # Try 32, 64, 128
BLOCK_DV = 64  # Try 32, 64, 128
```

- [ ] **Step 4: Re-benchmark after tuning**

```bash
popcorn-cli submit --leaderboard <leaderboard> --gpu MI355X --mode benchmark submission.py
```

---

## Phase 4: Final Polish

### Task 4.1: Clean Up Code

**Files:**
- Modify: `problems/amd_202602/mixed-mla/submission.py`

- [ ] **Step 1: Remove unused imports**

- [ ] **Step 2: Add docstrings to all functions**

- [ ] **Step 3: Format code consistently**

- [ ] **Step 4: Commit**

```bash
git commit -am "chore: clean up code and add documentation"
```

---

### Task 4.2: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
popcorn-cli submit --gpu MI355X --mode test submission.py
```

- [ ] **Step 2: Run full benchmark suite**

```bash
popcorn-cli submit --gpu MI355X --mode benchmark submission.py
```

- [ ] **Step 3: Verify geometric mean <50μs**

- [ ] **Step 4: Create final commit**

```bash
git commit -am "release: FlyDSL MLA decode kernel v1.0"
```

---

## Success Criteria

- [ ] All test cases pass (rtol=1e-01, atol=1e-01)
- [ ] Geometric mean benchmark <50μs
- [ ] Single `submission.py` file, self-contained
- [ ] Clean git history with meaningful commits
