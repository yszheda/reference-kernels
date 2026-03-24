# moe_fused - MoE Stage 1+2 融合 Kernel 实现
"""
MoE Fused Kernel implementation with Stage 1+2 fusion.

This module provides:
- scheduler: Expert-centric load balancing scheduler
- ck_fused: Composable Kernel extension for Stage 1+2 fusion
- tests: Unit tests and benchmark utilities

Usage:
    from moe_fused import fused_moe_ck, schedule_experts

    # Schedule experts
    schedule = schedule_experts(topk_ids, num_experts=E, tokens_per_block=32)

    # Launch fused kernel
    output = fused_moe_ck(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_weights,
        topk_ids,
        w1_scale=gate_up_weight_scale,
        w2_scale=down_weight_scale,
        fuse_stage12=True,
    )
"""

__version__ = "0.1.0"

from moe_fused.scheduler import (
    Schedule,
    schedule_experts,
    create_expert_mask,
    create_block_offsets,
)

from moe_fused.ck_fused import (
    fused_moe_ck,
    fused_moe_fused_reference,
)

__all__ = [
    # Scheduler
    "Schedule",
    "schedule_experts",
    "create_expert_mask",
    "create_block_offsets",
    # CK Fused
    "fused_moe_ck",
    "fused_moe_fused_reference",
]
