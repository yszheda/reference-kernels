# CLAUDE.md – GPU Computing Project (CUDA / ROCm)

## Project Overview
- **Name**: reference-kernels
- **Description**: Optimized kernel implementations for AMD MI355X GPU, focusing on MXFP4 quantized GEMM operations for LLM inference.
- **GPU Target**: AMD MI355X (CDNA4 architecture, ROCm 7.1+)

## Testing & Submission

### Popcorn CLI
```bash
# Correctness test
popcorn-cli submit --leaderboard amd-mxfp4-mm --gpu MI355X --mode test submission.py

# Benchmark
popcorn-cli submit --leaderboard amd-mxfp4-mm --gpu MI355X --mode benchmark submission.py

# Profile (generates Nsight/ROCm profile data)
popcorn-cli submit --leaderboard amd-mxfp4-mm --gpu MI355X --mode profile submission.py
```

### Rate Limits
- 6 submissions per hour per leaderboard
- If rate limited, wait and retry: "Try again in Xs"

## Current Optimization Status: mxfp4-mm

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

### Optimization Attempts & Lessons

#### ✅ What Worked
1. **Fixed L2 cache bug**: Removed buggy persistent cache that caused tensor size mismatch across tests
2. **Simplified imports**: Direct `dynamic_mxfp4_quant` + `gemm_a4w4` calls
3. **Proper data layout**: `e8m0_shuffle` for scale rearrangement, `bpreshuffle=True` for GEMM

#### ❌ What Didn't Work
1. **torch.compile**: Causes `TypeError: cannot pickle 'module' object` in multiprocessing benchmark
2. **gemm_afp4wfp4**: Not available in remote aiter version (silently falls back)
3. **L2 cache residency**: Overhead exceeded benefits for variable shapes

### Key Insights
1. **aiter baseline (~11.3µs)** measures pure GEMM with pre-quantized inputs
2. **Our implementation (~22.4µs)** includes `dynamic_mxfp4_quant(A)` adding ~8-10µs overhead
3. **To beat aiter**: Need fused quant+GEMM kernel to avoid HBM write of intermediate quantized activation
4. **Large K bottleneck**: Quantization overhead scales linearly with K dimension

### Next Optimization Directions
1. Explore aiter's Triton-based fused kernels (if available)
2. Investigate custom quantization kernel with better memory access patterns
3. Consider using pre-quantized B information to optimize A quantization path

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

## File Structure
```
problems/amd_202602/
├── mxfp4-mm/
│   ├── submission.py      # Your optimized kernel
│   ├── reference.py       # Reference implementation
│   ├── task.py            # Type definitions
│   ├── task.yml           # Benchmark shapes & aiter baseline
│   └── run_test.py        # Local testing (requires GPU)
```

## Reference
- [aiter GitHub](https://github.com/ROCm/aiter) - AMD's kernel library
- [FlyDSL](https://github.com/ROCm/FlyDSL) - MLIR-based kernel DSL (for advanced fusion)
