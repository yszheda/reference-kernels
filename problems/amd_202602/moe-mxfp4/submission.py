import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# 导入融合 kernel 实现
try:
    from moe_fused import fused_moe_ck
    MOE_FUSED_AVAILABLE = True
except ImportError:
    MOE_FUSED_AVAILABLE = False
    print("Warning: moe_fused not available, falling back to standard fused_moe")


def custom_kernel(data: input_t) -> output_t:
    """
    Submission for DeepSeek-R1 MXFP4 MoE kernel with Stage 1+2 fusion.

    This implementation uses LDS intermediate caching to eliminate HBM
    traffic between Stage 1 (gate+up + SwiGLU) and Stage 2 (down).

    Features:
    - Expert-centric load balancing scheduler
    - LDS intermediate caching (zero HBM traffic between stages)
    - Single kernel launch for both stages
    - Atomic accumulation for cross-expert reduction

    Input data tuple:
        hidden_states:                [M, d_hidden]                           bf16
        gate_up_weight:               [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (raw)
        down_weight:                  [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (raw)
        gate_up_weight_scale:         [E, 2*d_expert_pad, scale_K]            e8m0   (raw)
        down_weight_scale:            [E, d_hidden_pad, scale_K]              e8m0   (raw)
        gate_up_weight_shuffled:      [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (shuffled)
        down_weight_shuffled:         [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (shuffled)
        gate_up_weight_scale_shuffled:[padded, flat]                          e8m0   (shuffled)
        down_weight_scale_shuffled:   [padded, flat]                          e8m0   (shuffled)
        topk_weights:                 [M, total_top_k]                        float32
        topk_ids:                     [M, total_top_k]                        int32
        config:                       dict

    Returns:
        output: [M, d_hidden] bf16
    """
    (
        hidden_states,
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
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if MOE_FUSED_AVAILABLE:
        # 使用融合 kernel (Stage 1+2 fused)
        output = fused_moe_ck(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            topk_weights,
            topk_ids,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            fuse_stage12=True,           # 启用融合
            schedule_mode="balanced",    # 负载均衡调度
            tokens_per_block=32,         # 每 block 32 tokens
        )
    else:
        # 回退到标准 fused_moe
        output = fused_moe(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            topk_weights,
            topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None,
            a2_scale=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )

    return output
