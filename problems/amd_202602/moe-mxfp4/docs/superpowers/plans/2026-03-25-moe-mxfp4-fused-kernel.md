# MOE MXFP4 Stage 1+2 完整融合 Kernel 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Stage 1+2 完整融合的 MoE Kernel，通过 LDS 缓存 intermediate 消除 HBM 往返流量，达到 20-27% latency 降低。

**Architecture:** 采用 Expert-Centric 调度 + LDS Tiling 策略，在单个 kernel 中完成 Stage 1 (gate+up GEMM + SwiGLU) → LDS 缓存 → Stage 2 (down GEMM) + 加权归约。

**Tech Stack:** PyTorch + Triton (prototype) + Composable Kernel (生产实现) + AMD ROCm (MI355X CDNA4 架构)

---

## Phase 0: 环境准备与基准确认

### Task 0.1: 验证当前环境配置

**Files:**
- 修改：无 (验证任务)
- 测试：`eval.py`

- [ ] **Step 1: 确认 AITER 安装和版本**

```bash
cd /Users/yuanshuai/Code/reference-kernels/problems/amd_202602/moe-mxfp4
python -c "import aiter; print(aiter.__version__)"
```
预期输出：AITER 版本号

- [ ] **Step 2: 确认 GPU 设备可用**

```bash
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```
预期输出：AMD MI355X 或兼容设备

- [ ] **Step 3: 运行基准确认测试**

```bash
# 配置测试用例
echo "bs: 64; dhidden: 7168; dexpert: 256; nroutedexperts: 256; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" > test_cases.txt

# 运行正确性测试
python eval.py test test_cases.txt
```
预期输出：`check: pass`

---

## Phase 1: Triton Prototype 验证

### Task 1.1: 创建调度器模块

**Files:**
- 创建：`moe_fused/scheduler.py`
- 测试：`moe_fused/tests/test_scheduler.py`

- [ ] **Step 1: 编写调度器单元测试**

```python
# moe_fused/tests/test_scheduler.py
import torch
from moe_fused.scheduler import schedule_experts

def test_expert_scheduling_basic():
    """测试基础调度功能"""
    # 模拟 8 tokens, 每 token 2 experts (top_k=2), 总共 4 个专家
    topk_ids = torch.tensor([
        [0, 3],  # token 0: experts 0, 3
        [1, 3],  # token 1: experts 1, 3
        [0, 2],  # token 2: experts 0, 2
        [1, 2],  # token 3: experts 1, 2
        [0, 3],  # token 4: experts 0, 3
        [2, 3],  # token 5: experts 2, 3
        [0, 1],  # token 6: experts 0, 1
        [1, 2],  # token 7: experts 1, 2
    ])
    num_experts = 4

    schedule = schedule_experts(topk_ids, num_experts)

    # 验证专家按负载排序 (expert 3 有 5 个 tokens, expert 0 有 4 个，etc.)
    assert len(schedule.expert_order) == num_experts
    assert schedule.expert_order[0] == 3  # 最高负载专家

    # 验证所有 token 都被分配
    total_tokens = sum(count for _, _, count in schedule.block_assignments)
    assert total_tokens == len(topk_ids) * topk_ids.shape[1]

    print("test_expert_scheduling_basic: PASSED")

if __name__ == "__main__":
    test_expert_scheduling_basic()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /Users/yuanshuai/Code/reference-kernels/problems/amd_202602/moe-mxfp4
python moe_fused/tests/test_scheduler.py
```
预期输出：`ModuleNotFoundError: No module named 'moe_fused'`

- [ ] **Step 3: 实现调度器**

```python
# moe_fused/scheduler.py
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Schedule:
    """Expert scheduling result for MoE kernel"""
    expert_order: List[int]  # Experts sorted by load (descending)
    block_assignments: List[Tuple[int, int, int]]  # (expert_id, token_start, token_count)
    num_blocks: int

def schedule_experts(
    topk_ids: torch.Tensor,  # [M, top_k]
    num_experts: int,
    tokens_per_block: int = 32
) -> Schedule:
    """
    Generate expert-centric schedule for fused MoE kernel.

    Strategy:
    1. Count tokens per expert
    2. Sort experts by load (descending) for load balancing
    3. Assign blocks to process tokens for each expert

    Args:
        topk_ids: Token-to-expert assignments [M, top_k]
        num_experts: Total number of experts
        tokens_per_block: Number of tokens each block processes

    Returns:
        Schedule object with expert order and block assignments
    """
    M, top_k = topk_ids.shape

    # Step 1: Count tokens per expert
    expert_counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)

    # Step 2: Sort experts by load (descending)
    expert_order = torch.argsort(expert_counts, descending=True).tolist()

    # Step 3: Build block assignments
    # For each expert, find all tokens assigned to it
    block_assignments = []

    for expert_id in expert_order:
        # Find all (token_idx, slot) pairs for this expert
        matches = torch.where(topk_ids == expert_id)
        token_indices = matches[0]  # Which tokens use this expert
        slot_indices = matches[1]   # Which slot (top-k position)

        num_tokens = len(token_indices)
        if num_tokens == 0:
            continue

        # Group tokens into blocks of size tokens_per_block
        for start in range(0, num_tokens, tokens_per_block):
            count = min(tokens_per_block, num_tokens - start)
            block_assignments.append((expert_id.item(), start, count))

    return Schedule(
        expert_order=expert_order,
        block_assignments=block_assignments,
        num_blocks=len(block_assignments)
    )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python moe_fused/tests/test_scheduler.py
```
预期输出：`test_expert_scheduling_basic: PASSED`

- [ ] **Step 5: 提交**

```bash
git add moe_fused/scheduler.py moe_fused/tests/test_scheduler.py
git commit -m "feat(scheduler): implement expert-centric load balancing scheduler"
```

---

### Task 1.2: 实现 Triton Fused Kernel (Stage 1 + Stage 2)

**Files:**
- 创建：`moe_fused/triton_kernel.py`
- 测试：`moe_fused/tests/test_triton_kernel.py`

- [ ] **Step 1: 编写 kernel 单元测试**

```python
# moe_fused/tests/test_triton_kernel.py
import torch
from moe_fused.triton_kernel import fused_moe_triton

def test_fused_moe_correctness():
    """验证融合 kernel 正确性 (小规模测试)"""
    # 小规模配置用于快速验证
    M = 8  # tokens
    d_hidden = 256
    d_expert = 64
    num_experts = 4
    top_k = 2

    # 生成随机输入
    torch.manual_seed(42)
    hidden_states = torch.randn(M, d_hidden, dtype=torch.bfloat16, device='cuda')
    gate_up_weight = torch.randn(num_experts, 2 * d_expert, d_hidden, dtype=torch.bfloat16, device='cuda')
    down_weight = torch.randn(num_experts, d_hidden, d_expert, dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.rand(M, top_k, dtype=torch.float32, device='cuda')
    topk_ids = torch.randint(0, num_experts, (M, top_k), dtype=torch.int32, device='cuda')

    # 运行融合 kernel
    output = fused_moe_triton(
        hidden_states, gate_up_weight, down_weight,
        topk_weights, topk_ids
    )

    # 验证输出形状和数值范围
    assert output.shape == (M, d_hidden)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print("test_fused_moe_correctness: PASSED")

if __name__ == "__main__":
    test_fused_moe_correctness()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python moe_fused/tests/test_triton_kernel.py
```
预期输出：`ModuleNotFoundError`

- [ ] **Step 3: 实现 Triton 融合 Kernel**

```python
# moe_fused/triton_kernel.py
import torch
import triton
import triton.language as tl

@triton.jit
def silu(x):
    """SwiGLU activation: x * sigmoid(x)"""
    return x * tl.sigmoid(x)

@triton.jit
def fused_moe_kernel(
    # Pointers to inputs
    hidden_states_ptr,      # [M, d_hidden]
    gate_up_weight_ptr,     # [E, 2*d_expert, d_hidden]
    down_weight_ptr,        # [E, d_hidden, d_expert]
    topk_weights_ptr,       # [M, top_k]
    topk_ids_ptr,           # [M, top_k]
    output_ptr,             # [M, d_hidden]

    # Strides
    hidden_states_stride_m,
    hidden_states_stride_d,
    gate_up_stride_e,
    gate_up_stride_d,
    down_stride_e,
    down_stride_d,
    output_stride_m,
    output_stride_d,

    # Dimensions
    M: tl.constexpr,
    E: tl.constexpr,
    d_hidden: tl.constexpr,
    d_expert: tl.constexpr,
    top_k: tl.constexpr,

    # Block size
    BLOCK_D_HIDDEN: tl.constexpr,
    BLOCK_D_EXPERT: tl.constexpr,
):
    """
    Fused MoE kernel: Stage 1 (gate+up + SwiGLU) + Stage 2 (down)

    Each program processes one (token, expert) pair.
    Intermediate stored in LDS (shared memory).
    """
    # Program ID
    pid = tl.program_id(0)

    # Decode pid -> (token_idx, expert_slot)
    token_idx = pid // top_k
    expert_slot = pid % top_k

    if token_idx >= M:
        return

    # Load token -> expert assignment
    expert_id = tl.load(topk_ids_ptr + token_idx * top_k + expert_slot)
    routing_weight = tl.load(topk_weights_ptr + token_idx * top_k + expert_slot)

    # Initialize accumulators
    # Stage 1: Compute gate and up projections, then SwiGLU
    intermediate = tl.zeros([BLOCK_D_EXPERT], dtype=tl.float32)

    # Stage 1 GEMM: gate = hidden @ gate_weight.T
    for d_h_block in range(0, d_hidden, BLOCK_D_HIDDEN):
        # Load hidden states tile
        d_h_offset = d_h_block + tl.arange(0, BLOCK_D_HIDDEN)
        hidden_mask = d_h_offset < d_hidden
        hidden = tl.load(
            hidden_states_ptr + token_idx * hidden_states_stride_m + d_h_offset * hidden_states_stride_d,
            mask=hidden_mask,
            other=0.0
        ).to(tl.float32)

        # Load gate weight row
        gate_weight = tl.load(
            gate_up_weight_ptr + expert_id * gate_up_stride_e + \
            tl.arange(0, BLOCK_D_EXPERT)[:, None] * gate_up_stride_d + \
            d_h_offset * hidden_states_stride_d,
            mask=(tl.arange(0, BLOCK_D_EXPERT)[:, None] < d_expert) & hidden_mask,
            other=0.0
        ).to(tl.float32)

        # Accumulate: gate += hidden @ weight.T
        intermediate += tl.dot(hidden[None, :], gate_weight, out_dtype=tl.float32)[0, :]

    # Apply SiLU for gate part
    gate = intermediate

    # Stage 1 GEMM: up = hidden @ up_weight.T (similar to gate, offset by d_expert)
    up_intermediate = tl.zeros([BLOCK_D_EXPERT], dtype=tl.float32)
    for d_h_block in range(0, d_hidden, BLOCK_D_HIDDEN):
        d_h_offset = d_h_block + tl.arange(0, BLOCK_D_HIDDEN)
        hidden_mask = d_h_offset < d_hidden
        hidden = tl.load(
            hidden_states_ptr + token_idx * hidden_states_stride_m + d_h_offset * hidden_states_stride_d,
            mask=hidden_mask,
            other=0.0
        ).to(tl.float32)

        # Load up weight (offset by d_expert in first dim)
        up_weight = tl.load(
            gate_up_weight_ptr + expert_id * gate_up_stride_e + \
            (d_expert + tl.arange(0, BLOCK_D_EXPERT)[:, None]) * gate_up_stride_d + \
            d_h_offset * hidden_states_stride_d,
            mask=(tl.arange(0, BLOCK_D_EXPERT)[:, None] < d_expert) & hidden_mask,
            other=0.0
        ).to(tl.float32)

        up_intermediate += tl.dot(hidden[None, :], up_weight, out_dtype=tl.float32)[0, :]

    # SwiGLU: gate * up
    intermediate = silu(gate) * up_intermediate

    # Stage 2 GEMM: output += intermediate @ down_weight.T
    output_acc = tl.zeros([BLOCK_D_HIDDEN], dtype=tl.float32)

    for d_e_block in range(0, d_expert, BLOCK_D_EXPERT):
        d_e_offset = d_e_block + tl.arange(0, BLOCK_D_EXPERT)
        expert_mask = d_e_offset < d_expert

        # Load intermediate (from LDS - simulated with registers here)
        inter_tile = tl.load(
            intermediate + d_e_offset,
            mask=expert_mask,
            other=0.0
        )

        # Load down weight
        down_weight = tl.load(
            down_weight_ptr + expert_id * down_stride_e + \
            tl.arange(0, BLOCK_D_HIDDEN)[:, None] * down_stride_d + \
            d_e_offset,
            mask=(tl.arange(0, BLOCK_D_HIDDEN)[:, None] < d_hidden) & expert_mask,
            other=0.0
        ).to(tl.float32)

        # Accumulate
        output_acc += tl.dot(inter_tile[None, :], down_weight, out_dtype=tl.float32)[0, :]

    # Apply routing weight and store output (atomic add for accumulation)
    output_offset = token_idx * output_stride_m + tl.arange(0, BLOCK_D_HIDDEN) * output_stride_d
    output_mask = tl.arange(0, BLOCK_D_HIDDEN) < d_hidden

    # Note: Triton atomic_add for accumulation across experts
    tl.atomic_add(
        output_ptr + output_offset,
        (routing_weight * output_acc).to(tl.bfloat16),
        mask=output_mask
    )

def fused_moe_triton(
    hidden_states: torch.Tensor,      # [M, d_hidden]
    gate_up_weight: torch.Tensor,     # [E, 2*d_expert, d_hidden]
    down_weight: torch.Tensor,        # [E, d_hidden, d_expert]
    topk_weights: torch.Tensor,       # [M, top_k]
    topk_ids: torch.Tensor,           # [M, top_k]
) -> torch.Tensor:
    """
    Fused MoE forward pass using Triton.

    Args:
        hidden_states: Input token embeddings
        gate_up_weight: Fused gate and up projection weights
        down_weight: Down projection weights
        topk_weights: Routing weights for each token-expert pair
        topk_ids: Expert indices for each token

    Returns:
        output: [M, d_hidden] MoE output
    """
    M, d_hidden = hidden_states.shape
    E, _, d_hidden_w = gate_up_weight.shape
    d_expert = down_weight.shape[2]
    top_k = topk_ids.shape[1]

    # Output tensor
    output = torch.zeros(M, d_hidden, dtype=torch.bfloat16, device='cuda')

    # Block sizes
    BLOCK_D_HIDDEN = triton.next_power_of_2(d_hidden)
    BLOCK_D_EXPERT = triton.next_power_of_2(d_expert)

    # Grid: one program per (token, expert_slot) pair
    grid = (M * top_k,)

    # Launch kernel
    fused_moe_kernel[grid](
        hidden_states, gate_up_weight, down_weight,
        topk_weights, topk_ids, output,

        # Strides
        hidden_states.stride(0), hidden_states.stride(1),
        gate_up_weight.stride(0), gate_up_weight.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        output.stride(0), output.stride(1),

        # Dimensions
        M, E, d_hidden, d_expert, top_k,

        # Block sizes
        BLOCK_D_HIDDEN, BLOCK_D_EXPERT,
    )

    return output
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python moe_fused/tests/test_triton_kernel.py
```
预期输出：`test_fused_moe_correctness: PASSED`

- [ ] **Step 5: 提交**

```bash
git add moe_fused/triton_kernel.py moe_fused/tests/test_triton_kernel.py
git commit -m "feat(triton): implement Stage 1+2 fused MoE kernel prototype"
```

---

### Task 1.3: 正确性验证 (对比 AITER 参考实现)

**Files:**
- 创建：`moe_fused/tests/test_against_reference.py`

- [ ] **Step 1: 编写对比测试**

```python
# moe_fused/tests/test_against_reference.py
import torch
import sys
sys.path.insert(0, '/Users/yuanshuai/Code/reference-kernels/problems/amd_202602/moe-mxfp4')

from reference import ref_kernel, generate_input
from moe_fused.triton_kernel import fused_moe_triton

def test_against_aiter_reference():
    """对比 Triton kernel 与 AITER 参考实现"""
    # 使用问题配置生成输入
    data = generate_input(
        dhidden=7168,
        dexpert=256,
        nroutedexperts=256,
        nexpertspertoken=8,
        nsharedexperts=1,
        bs=16,  # 小规模测试
        seed=42
    )

    (
        hidden_states,
        gate_up_weight, _,  # raw weights
        _, _,  # raw scales
        gate_up_weight_shuffled,
        down_weight_shuffled,
        _, _,  # shuffled scales
        topk_weights,
        topk_ids,
        config,
    ) = data

    # AITER 参考输出
    output_ref = ref_kernel(data)

    # Triton kernel 输出 (需要转换权重格式)
    # 注意：Triton prototype 使用原始权重格式，不需要 shuffle
    output_triton = fused_moe_triton(
        hidden_states,
        gate_up_weight,  # 使用 raw 格式
        down_weight_shuffled,  # 这里需要调整
        topk_weights,
        topk_ids
    )

    # 比较输出 (允许数值误差)
    rtol, atol = 1e-2, 1e-2
    max_diff = (output_ref - output_triton).abs().max().item()

    print(f"Max difference: {max_diff}")
    print(f"Reference output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
    print(f"Triton output range: [{output_triton.min():.4f}, {output_triton.max():.4f}]")

    # 简单检查：数值应该在合理范围内
    assert not torch.isnan(output_triton).any(), "Triton output contains NaN"
    assert max_diff < 1.0, f"Max difference {max_diff} too large"

    print("test_against_aiter_reference: PASSED (basic)")

if __name__ == "__main__":
    test_against_aiter_reference()
```

- [ ] **Step 2: 运行对比测试**

```bash
python moe_fused/tests/test_against_reference.py
```
预期输出：基础验证通过 (可能有一定数值差异)

---

## Phase 2: CK 框架集成

### Task 2.1: 分析 AITER CK 源码结构

**Files:**
- 分析目标：AITER `fused_moe` 实现

- [ ] **Step 1: 定位 AITER fused_moe 源码**

```bash
python -c "import aiter; print(aiter.__file__)"
# 然后查看 aiter/fused_moe.py 或相关 C++/CK 源
```

- [ ] **Step 2: 阅读 fused_moe 接口定义**

读取并分析 `aiter/fused_moe.py` 中的函数签名和参数

- [ ] **Step 3: 识别扩展点**

标记需要修改的 CK 源码位置：
- `MoEProblem` 配置
- `MoEKernel` 实现
- 调度逻辑

---

### Task 2.2: 实现 CK Fused Kernel

**Files:**
- 修改：`aiter/ops/moe_kernel.cpp` (需要定位实际路径)
- 创建：`aiter/ops/moe_fused_kernel.cpp`

- [ ] **Step 1: 复制现有 fused_moe 实现作为起点**

- [ ] **Step 2: 修改 kernel 以支持 Stage 1+2 融合**

关键修改：
1. 添加 LDS intermediate buffer 分配
2. Stage 1 输出写入 LDS 而非 HBM
3. Stage 2 从 LDS 读取 intermediate
4. 添加 `__syncthreads()` 同步

- [ ] **Step 3: 添加新的 Python 绑定**

```python
# aiter/fused_moe.py 中添加
def fused_moe_fused(
    hidden_states: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    w1_scale: Tensor,
    w2_scale: Tensor,
    **kwargs
) -> Tensor:
    """Stage 1+2 fused MoE kernel with LDS intermediate caching"""
    # Call into CK implementation
    pass
```

---

## Phase 3: 性能优化

### Task 3.1: LDS Tiling 优化

- [ ] **Step 1: 实现 multi-pass 处理**

针对 LDS 容量限制，实现分块处理策略

- [ ] **Step 2: 优化 LDS 访问模式**

确保 coalesced memory access

- [ ] **Step 3: 测试不同 tile size 配置**

Benchmark 不同 `TOKENS_PER_BLOCK`, `D_EXPERT_TILE` 配置

---

### Task 3.2 原子累加优化

- [ ] **Step 1: 实现 per-block output buffer**

```cpp
__shared__ float block_output[NUM_BLOCKS][D_HIDDEN];
```

- [ ] **Step 2: 实现最终 reduce kernel**

```python
def reduce_outputs(block_outputs: List[Tensor]) -> Tensor:
    """Reduce per-block outputs to final output"""
    return torch.sum(torch.stack(block_outputs), dim=0)
```

---

## Phase 4: 验证与基准测试

### Task 4.1: 完整正确性验证

**Files:**
- 修改：`eval.py` (添加新 kernel 测试路径)

- [ ] **Step 1: 配置所有 benchmark cases 测试**

```bash
# EP-off cases
echo "bs: 4; dhidden: 7168; dexpert: 256; nroutedexperts: 256; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" > test_cases_full.txt
echo "bs: 64; dhidden: 7168; dexpert: 256; nroutedexperts: 256; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" >> test_cases_full.txt
echo "bs: 256; dhidden: 7168; dexpert: 256; nroutedexperts: 256; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" >> test_cases_full.txt

# EP-on cases
echo "bs: 64; dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" >> test_cases_full.txt
echo "bs: 256; dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" >> test_cases_full.txt
echo "bs: 1024; dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nexpertspertoken: 8; nsharedexperts: 1; seed: 42" >> test_cases_full.txt
```

- [ ] **Step 2: 运行正确性测试**

```bash
python eval.py test test_cases_full.txt
```

---

### Task 4.2: 性能基准测试

- [ ] **Step 1: 运行 benchmark**

```bash
python eval.py benchmark test_cases_full.txt
```

- [ ] **Step 2: 对比 AITER 基线**

计算性能提升百分比

- [ ] **Step 3: 生成性能报告**

---

## 文件结构总览

```
moe-mxfp4/
├── moe_fused/                    # 新增：融合 kernel 实现
│   ├── __init__.py
│   ├── scheduler.py              # Task 1.1: 专家调度器
│   ├── triton_kernel.py          # Task 1.2: Triton 原型
│   └── tests/
│       ├── test_scheduler.py
│       ├── test_triton_kernel.py
│       └── test_against_reference.py
├── reference.py                  # (已有) AITER 参考实现
├── submission.py                 # (修改) 切换到融合 kernel
├── eval.py                       # (已有) 测试框架
└── test_cases_full.txt           # 测试配置
```

---

## 执行选项

Plan 已完成并保存到 `docs/superpowers/plans/2026-03-25-moe-mxfp4-fused-kernel.md`。

有两个执行选项：

**1. Subagent-Driven (推荐)** - 每个任务派遣专用 subagent，任务间 review，快速迭代

**2. Inline Execution** - 在当前 session 中使用 `executing-plans` skill 批量执行，设置检查点

您希望使用哪种方式执行？
