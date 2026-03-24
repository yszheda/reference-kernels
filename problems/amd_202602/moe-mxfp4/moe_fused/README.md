# moe_fused - MoE Stage 1+2 融合 Kernel

## 概述

`moe_fused` 是一个针对 AMD MI355X (CDNA4) GPU 优化的 DeepSeek-R1 风格 MoE 融合内核实现。

**核心优化**: 通过 LDS 缓存 intermediate 数据，消除 Stage 1 (gate+up + SwiGLU) 和 Stage 2 (down) 之间的 HBM 往返流量。

## 性能目标

| 指标 | 当前 AITER | moe_fused 目标 | 改善 |
|------|-----------|---------------|------|
| HBM 流量 (bs=256) | ~50 MB | ~31 MB | -38% |
| Kernel Launch | 2 次 | 1 次 | -50% |
| Latency (bs=256) | 276 μs | 200-220 μs | -20~27% |

## 安装

### 要求

- AMD ROCm 5.7+
- AMD Instinct MI300X/MI355X GPU
- Python 3.8+
- PyTorch with ROCm 支持

### 安装步骤

```bash
# 1. 确保 ROCm 环境配置正确
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export CK_INCLUDE_PATH=/opt/rocm/include/ck

# 2. 安装 moe_fused
cd moe-mxfp4/moe_fused
pip install -e .

# 3. 验证安装
python3 -c "from moe_fused import fused_moe_ck; print('OK')"
```

## 使用示例

### 基本用法

```python
import torch
from moe_fused import fused_moe_ck

# 准备输入 (与 AITER fused_moe 兼容)
hidden_states = torch.randn(256, 7168, dtype=torch.bfloat16, device='cuda')
gate_up_weight = ...  # [E, 2*d_expert, d_hidden//2] fp4x2
down_weight = ...     # [E, d_hidden, d_expert//2] fp4x2
topk_weights = ...    # [M, top_k] float32
topk_ids = ...        # [M, top_k] int32
w1_scale = ...        # MXFP4 scales
w2_scale = ...        # MXFP4 scales

# 运行融合 kernel
output = fused_moe_ck(
    hidden_states,
    gate_up_weight,
    down_weight,
    topk_weights,
    topk_ids,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    fuse_stage12=True,        # 启用 Stage 1+2 融合
    schedule_mode="balanced", # 负载均衡调度
    tokens_per_block=32,      # 每 block token 数
)
```

### 使用调度器

```python
from moe_fused import schedule_experts

# 生成专家调度
topk_ids = ...  # [M, top_k]
num_experts = 257

schedule = schedule_experts(
    topk_ids,
    num_experts,
    tokens_per_block=32,
    schedule_mode="balanced"  # 或 "compact", "interleaved"
)

print(f"Blocks: {schedule.num_blocks}")
print(f"Top expert by load: {schedule.expert_order[0]}")
```

## API 参考

### `fused_moe_ck(...)`

```python
def fused_moe_ck(
    hidden_states: torch.Tensor,           # [M, d_hidden] bf16
    gate_up_weight: torch.Tensor,          # [E, 2*d_expert, d_hidden//2] fp4x2
    down_weight: torch.Tensor,             # [E, d_hidden, d_expert//2] fp4x2
    topk_weights: torch.Tensor,            # [M, top_k] float32
    topk_ids: torch.Tensor,                # [M, top_k] int32
    w1_scale: Optional[torch.Tensor] = None,  # MXFP4 scales
    w2_scale: Optional[torch.Tensor] = None,
    fuse_stage12: bool = True,             # 启用融合
    schedule_mode: str = "balanced",       # 调度模式
    tokens_per_block: int = 32,            # 每 block token 数
) -> torch.Tensor:
    """
    Fused MoE kernel with Stage 1+2 fusion.

    当 fuse_stage12=True:
    - 使用 LDS 缓存 intermediate (零 HBM 流量)
    - 单次 kernel launch 完成两个 stage
    - 专家中心负载均衡调度
    """
```

### `schedule_experts(...)`

```python
def schedule_experts(
    topk_ids: torch.Tensor,      # [M, top_k]
    num_experts: int,             # 专家总数
    tokens_per_block: int = 32,   # 每 block token 数
    schedule_mode: str = "balanced",  # 调度模式
) -> Schedule:
    """
    生成专家中心调度表。

    调度模式:
    - "balanced": 按负载排序专家，均匀分配 (推荐)
    - "compact": 最小化 block 数量
    - "interleaved": 轮转分配，更好的延迟隐藏
    """
```

## 架构设计

### 数据流

```
Stage 1 (GEMM + SwiGLU)          Stage 2 (GEMM)
       │                               │
       ▼                               │
  hidden_states ──→ LDS ───────────────┘
       │            │
       │            ▼
       │      intermediate
       │            │
       │            ▼
       │      down @ inter.T
       │            │
       ▼            ▼
    gate/up  →  output (atomic add)
```

### LDS 布局

```
LDS (128 KB):
┌─────────────────────────────────┐
│ Token 0: [d0, d1, ..., d2047]   │  4 KB
│ Token 1: [d0, d1, ..., d2047]   │  4 KB
│ ...                             │
│ Token 31: [d0, d1, ..., d2047]  │  4 KB
└─────────────────────────────────┘
总计：32 × 4 KB = 128 KB (满载)
```

## 测试

```bash
# 运行测试套件
python3 test_against_reference.py

# 运行基准测试
python3 eval.py benchmark test_cases.txt
```

## 文件结构

```
moe-mxfp4/
├── moe_fused/                    # 融合 kernel 实现
│   ├── __init__.py
│   ├── scheduler.py              # 专家调度器
│   ├── ck_fused.py               # CK 融合接口
│   ├── setup.py                  # 构建脚本
│   ├── cpp/
│   │   └── moe_fused_kernel.cpp  # CK kernel 源码
│   └── tests/
│       └── test_scheduler.py     # 调度器测试
├── submission.py                 # 提交入口 (已更新)
├── test_against_reference.py     # 对比测试
└── docs/
    ├── superpowers/specs/        # 设计文档
    └── superpowers/plans/        # 实现计划
```

## 优化技巧

### 1. 调整 block size

```python
# 小 batch (bs < 64): 减少 tokens_per_block
output = fused_moe_ck(..., tokens_per_block=16)

# 大 batch (bs > 256): 增加 tokens_per_block
output = fused_moe_ck(..., tokens_per_block=64)
```

### 2. 选择调度模式

```python
# 均匀负载：balanced
output = fused_moe_ck(..., schedule_mode="balanced")

# 最小 block 数：compact
output = fused_moe_ck(..., schedule_mode="compact")

# 延迟敏感：interleaved
output = fused_moe_ck(..., schedule_mode="interleaved")
```

## 故障排除

### "LDS size exceeds limit"

确保 `tokens_per_block * d_expert * 2 <= 128 KB`:
```python
# 对于 d_expert=2048: tokens_per_block <= 32
# 对于 d_expert=4096: tokens_per_block <= 16
```

### "Kernel launch failed"

检查 ROCm 环境：
```bash
rocminfo | grep "Name:"
hipcc --version
```

## 参考资源

- [设计文档](docs/superpowers/specs/2026-03-25-moe-mxfp4-fused-kernel-design.md)
- [实现计划](docs/superpowers/plans/2026-03-25-moe-mxfp4-fused-kernel.md)
- [AMD CDNA 架构](https://www.amd.com/en/products/accelerators/instinct/mi355x.html)
- [Composable Kernel](https://github.com/ROCm/composable_kernel)

## 许可证

MIT License
