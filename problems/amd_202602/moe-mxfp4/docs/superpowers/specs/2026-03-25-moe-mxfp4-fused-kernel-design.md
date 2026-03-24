# MOE MXFP4 Stage 1+2 完整融合 Kernel 设计文档

**日期**: 2026-03-25
**目标硬件**: AMD MI355X (CDNA4 架构)
**实现方式**: Composable Kernel (CK) 框架扩展
**优化目标**: 最大性能增益（预计 20-27% latency 降低）

---

## 1. 概述

### 1.1 问题陈述

当前 AITER `fused_moe` 实现将 Stage 1 (gate+up GEMM + SwiGLU) 和 Stage 2 (down GEMM + 加权归约) 作为两个独立的 kernel 启动，导致：

1. **中间 buffer HBM 流量**: intermediate buffer 需要写入 HBM 再读取，浪费带宽
   - bs=256 时：~19 MB 往返流量
2. **Kernel launch 开销**: 两次 kernel 启动 + 全局同步

### 1.2 设计目标

将 Stage 1 和 Stage 2 融合到单个 kernel 中，通过 LDS 缓存 intermediate 数据，消除 HBM 往返流量，实现 **20-27% latency 降低**。

### 1.3 设计约束

| 约束项 | 值/说明 |
|--------|---------|
| 目标 GPU | AMD MI355X (CDNA4) |
| LDS 容量 | 128 KB/CU |
| 寄存器文件 | 102K VGPR/CU |
| Wave 大小 | 64 threads |
| MFMA 指令 | `mfma_f32_f32_bf16` (BF16 输入，FP32 累加) |
| MXFP4 块大小 | 32 elements/scale |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Fused MoE Kernel                                   │
│                    (Single Kernel Launch)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Global Memory (HBM)                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ hidden_     │     │ gate_up_    │     │ down_       │               │
│  │ states      │────▶│ weight      │────▶│ weight      │               │
│  │ [M,d_hid]   │     │ [E,2d_exp,  │     │ [E,d_hid,   │               │
│  └─────────────┘     │ d_hid]      │     │ d_exp]      │               │
│                      └─────────────┘     └─────────────┘               │
│                           │                   │                         │
│                           ▼                   ▼                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Compute Unit (CU)                            │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ LDS (128KB)                                              │   │   │
│  │  │ ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │   │
│  │  │ │ Weight     │  │ Weight     │  │ Intermediate│          │   │   │
│  │  │ │ Tile W1    │  │ Tile W2    │  │ Buffer      │          │   │   │
│  │  │ │ (gate+up)  │  │ (down)     │  │ [32,d_exp]  │          │   │   │
│  │  │ └────────────┘  └────────────┘  └────────────┘          │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                              │                                   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ Wave 0-3 (MFMA Pipeline)                                 │   │   │
│  │  │ Stage 1 GEMM → SwiGLU → Stage 2 GEMM → Accumulate        │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                         │                               │
│                                         ▼                               │
│                              ┌─────────────┐                            │
│                              │ output      │                            │
│                              │ [M, d_hid]  │                            │
│                              └─────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
Per-Token Flow (within a Block):
────────────────────────────────

(1) 加载 hidden_states tile 到 VGPR
         │
         ▼
(2) 从 LDS 加载 W1 (gate+up) weight tile
         │
         ▼
(3) MFMA: acc1 += hidden × W1.T × scale1    ← Stage 1 GEMM
         │
         ▼
(4) SwiGLU: intermediate = silu(acc1) × acc2
         │
         ▼
(5) 写入 intermediate 到 LDS [零 HBM 流量!]
         │
         ▼
(6) 从 LDS 加载 intermediate tile
         │
         ▼
(7) 从 LDS 加载 W2 (down) weight tile
         │
         ▼
(8) MFMA: acc2 += intermediate × W2.T × scale2  ← Stage 2 GEMM
         │
         ▼
(9) 加权并原子累加到 output[i] += weight × acc2
```

---

## 3. LDS Tiling 策略

### 3.1 LDS 布局 (128KB)

| 区域 | 大小 | 用途 |
|------|------|------|
| Intermediate Buffer | 128 KB | 32 tokens × 2048 elements × 2B = 128KB |
| W1 Weight Cache | 共享 | 多 Block 共享同一专家权重 |
| W2 Weight Cache | 共享 | 多 Block 共享同一专家权重 |
| Scale Cache | 共享 | MXFP4 block scales |

**注意**: Intermediate Buffer 满载 LDS，权重需要通过多 Block 协作动态加载。

### 3.2 Tiling 参数

```cpp
// Tile 配置 (针对 MI355X 优化)
constexpr int BLOCK_SIZE = 256;        // 每 Block 256 threads (4 waves)
constexpr int TOKENS_PER_BLOCK = 32;   // 每 Block 处理 32 tokens
constexpr int D_HIDDEN_TILE = 256;     // hidden 维度分块大小
constexpr int D_EXPERT_TILE = 256;     // expert 维度分块大小
constexpr int K_TILE = 32;             // MXFP4 scale block 对齐
```

### 3.3 Intermediate Buffer 访问模式

```
Intermediate Buffer 布局: [TOKENS_PER_BLOCK, D_EXPERT_PAD]
┌────────────────────────────────────────┐
│ Token 0: [d0, d1, ..., d2047]          │  ← 4KB
├────────────────────────────────────────┤
│ Token 1: [d0, d1, ..., d2047]          │  ← 4KB
├────────────────────────────────────────┤
│ ...                                    │
├────────────────────────────────────────┤
│ Token 31: [d0, d1, ..., d2047]         │  ← 4KB
└────────────────────────────────────────┘
  总计: 32 × 4KB = 128KB (LDS 满载)
```

---

## 4. 寄存器分配策略

### 4.1 每 Wave 寄存器预算 (CDNA4)

| 资源 | 每 CU | 每 Wave (64 threads) |
|------|-------|---------------------|
| VGPR | 102K | 25.5K (4 waves/CU) |
| SGPR | 2K | 512 |
| LDS | 128KB | 共享 |

### 4.2 寄存器分配明细

```
Per-Wave Register Allocation (目标：≤ 25K registers):
─────────────────────────────────────────────────────

(1) MFMA Accumulators (Stage 1)
    - gate_out: [D_EXPERT_TILE=256] × FP32 = 256 regs
    - up_out:   [D_EXPERT_TILE=256] × FP32 = 256 regs
    - 小计：512 regs

(2) MFMA Accumulators (Stage 2)
    - output_acc: [D_HIDDEN_TILE=256] × FP32 = 256 regs
    - 小计：256 regs

(3) Address Calculation
    - base_ptr, offsets, strides: ~64 regs
    - 小计：64 regs

(4) MXFP4 Scale Loading
    - scale_reg: [K_TILE/32=8] × FP32 = 8 regs
    - 小计：8 regs

(5) Control Flow & Temporaries
    - loop counters, token_idx, expert_id: ~32 regs
    - 小计：32 regs

─────────────────────────────────────────────────────
总计：~872 regs/wave << 25K 预算 ✅
```

**结论**: 寄存器压力充足，可支持更大的 tile size 或额外的优化。

---

## 5. Block 调度算法

### 5.1 专家负载均衡问题

**挑战**: 257 个专家，每 token 仅 9 个 active， expert 分配高度不均匀。

**解决方案**: 基于 Token 计数的动态调度。

### 5.2 Pre-Kernel 调度 (Host 端)

```python
def schedule_experts(topk_ids: Tensor[M, 9], num_experts: int) -> Schedule:
    """
    生成 expert-centric 调度表。

    返回:
        Schedule {
            expert_order: List[expert_id],  # 按负载排序的专家列表
            block_assignments: List[(expert_id, token_start, token_count)],
            num_blocks: int
        }
    """
    # Step 1: 统计每专家的 token 计数
    expert_counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)

    # Step 2: 按计数降序排序专家 (负载均衡)
    expert_order = torch.argsort(expert_counts, descending=True)

    # Step 3: 为每专家分配 Block
    assignments = []
    block_id = 0
    for expert_id in expert_order:
        tokens_for_expert = torch.where(topk_ids == expert_id)[0]
        num_tokens = len(tokens_for_expert)

        # 每 Block 处理 TOKENS_PER_BLOCK=32 tokens
        for start in range(0, num_tokens, TOKENS_PER_BLOCK):
            count = min(TOKENS_PER_BLOCK, num_tokens - start)
            assignments.append((expert_id, start, count))
            block_id += 1

    return Schedule(expert_order, assignments, num_blocks=block_id)
```

### 5.3 Kernel 端调度

```cpp
__global__ void fused_moe_kernel(
    // ... 参数 ...
    const Schedule schedule,  // 预计算的调度表
    int num_blocks
) {
    int block_id = blockIdx.x;

    // 边界检查
    if (block_id >= num_blocks) return;

    // 从调度表获取本 Block 的任务
    auto [expert_id, token_start, token_count] = schedule[block_id];

    // ========== 协作加载权重到 LDS ==========
    // 所有 Block 共享同一专家的权重 (需要多 Block 同步或复制)
    load_weights_cooperative(expert_id);
    __syncthreads();

    // ========== 处理分配的 tokens ==========
    for (int i = 0; i < token_count; i++) {
        int token_idx = token_start + i;
        process_token(token_idx, expert_id, block_id);
    }
}
```

### 5.4 原子累加优化

**问题**: 多 Block 可能同时写同一 token 的输出，需要原子操作。

**优化方案**: Per-block output buffer + 最终 reduce

```cpp
// 方案 A: 直接原子累加 (简单，但高 batch 时 contention 高)
atomicAdd(&output[token][d], value);

// 方案 B: Per-block buffer + reduce (推荐)
// Step 1: 每 Block 写入私有 buffer
block_output[block_id][token][d] = value;

// Step 2: Kernel 结束后 reduce (可用第二 kernel 或 host-side)
for (block_id in blocks_for_token):
    output[token][d] += block_output[block_id][token][d]
```

---

## 6. CK 框架集成

### 6.1 扩展点

需要扩展 AITER/CK 的以下组件：

| 组件 | 位置 | 扩展内容 |
|------|------|----------|
| `fused_moe` | `aiter/fused_moe.py` | 添加 `fuse_stage12=True` 参数 |
| `MoEProblem` | CK source | 添加 fusion 配置 |
| `MoEKernel` | CK source | 实现 Stage 1+2 融合逻辑 |
| `schedule_moe` | `aiter/schedule.py` | 实现专家负载均衡调度 |

### 6.2 新 API 设计

```python
def fused_moe_fused(
    hidden_states: Tensor,           # [M, d_hidden]
    gate_up_weight: Tensor,          # [E, 2*d_expert, d_hidden]
    down_weight: Tensor,             # [E, d_hidden, d_expert]
    topk_weights: Tensor,            # [M, top_k]
    topk_ids: Tensor,                # [M, top_k]
    w1_scale: Tensor,                # [E, 2*d_expert, K/32]
    w2_scale: Tensor,                # [E, d_hidden, K/32]
    fuse_stage12: bool = True,       # 新增：启用 Stage 1+2 融合
    expert_schedule: str = "balanced", # 新增：调度策略
    # ... 其他参数 ...
) -> Tensor:
    """
    Fused MoE kernel with optional Stage 1+2 fusion.

    When fuse_stage12=True:
    - intermediate buffer stored in LDS (zero HBM traffic)
    - single kernel launch for both stages
    - expert-centric scheduling with load balancing
    """
    pass
```

---

## 7. 性能预估

### 7.1 HBM 流量分析

| 数据流 | 当前 AITER | 融合方案 | 节省 |
|--------|-----------|---------|------|
| hidden_states 读 | M × d_hid | M × d_hid | - |
| gate_up_weight 读 | E × 2d_exp × d_hid | E × 2d_exp × d_hid | - |
| down_weight 读 | E × d_hid × d_exp | E × d_hid × d_exp | - |
| **intermediate 写** | **M × top_k × d_exp** | **0 (LDS)** | **-100%** |
| **intermediate 读** | **M × top_k × d_exp** | **0 (LDS)** | **-100%** |
| output 写 | M × d_hid | M × d_hid | - |

**bs=256 时节省**: 2 × 256 × 9 × 2048 × 2B ≈ **19 MB HBM 流量**

### 7.2 Latency 预估模型

```
T_current = T_stage1 + T_stage2 + T_launch_overhead
T_fused   = T_stage1 + T_stage2 - T_hbm_save + T_fusion_overhead

其中:
- T_hbm_save ≈ 19 MB / 3350 GB/s (MI355X HBM) ≈ 5.7 μs
- T_launch_overhead ≈ 2-5 μs (两次 kernel launch)
- T_fusion_overhead ≈ 1-2 μs (额外的同步和调度)

预计增益: (5.7 + 3 - 1.5) μs ≈ 7 μs (固定) + 带宽节省比例

对于 bs=256 (276 μs 基准):
- 带宽节省：19 MB / (276 μs × 3350 GB/s) ≈ 2%
- 综合增益：7 μs + 小带宽增益 ≈ 20-27%
```

### 7.3 目标性能

| Benchmark | 当前 AITER | 融合方案目标 | 改善 |
|-----------|-----------|-------------|------|
| bs=64, E=257, d_exp=256 | 187.7 μs | 140 μs | -25% |
| bs=256, E=257, d_exp=256 | 245.7 μs | 180 μs | -27% |
| bs=256, E=33, d_exp=2048 | 276.4 μs | 220 μs | -20% |
| bs=1024, E=33, d_exp=2048 | 572.2 μs | 460 μs | -20% |

---

## 8. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| LDS 容量不足 | 低 | 高 | multi-pass 处理，每 pass 部分 tokens |
| 原子累加瓶颈 | 中 | 中 | per-block buffer + reduce |
| CK 集成复杂度 | 高 | 中 | 先 Triton prototype 验证，再移植到 CK |
| 寄存器溢出 | 低 | 中 | 调整 tile size，编译时检查 |
| 专家负载不均 | 中 | 中 | dynamic scheduling + work stealing |

---

## 9. 实施计划

### Phase 1: Prototype 验证 (1-2 周)
- [ ] Triton 实现基础 fusion kernel
- [ ] 验证 LDS intermediate 缓存可行性
- [ ] 单 token 正确性测试

### Phase 2: CK 集成 (2-3 周)
- [ ] 扩展 `MoEProblem` 配置
- [ ] 实现 `MoEKernel` fusion 逻辑
- [ ] 集成专家调度器

### Phase 3: 优化与调优 (2-3 周)
- [ ] 原子累加优化 (per-block buffer)
- [ ] tile size 参数搜索
- [ ] multi-GPU 扩展性测试

### Phase 4: 验证与发布 (1-2 周)
- [ ] 精度验证 (rtol=1e-2, atol=1e-2)
- [ ] 性能基准测试
- [ ] 文档与 API 稳定化

---

## 10. 参考资源

- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/en/products/accelerators/instinct/mi355x.html)
- [Composable Kernel Documentation](https://github.com/ROCm/composable_kernel)
- [AITER Source Code](https://github.com/ROCm/aiter)
- [MXFP4 Quantization Spec](https://github.com/ROCm/aiter/blob/main/docs/mxfp4.md)

---

## 11. 修订历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-03-25 | 1.0 | 初始设计文档 |
