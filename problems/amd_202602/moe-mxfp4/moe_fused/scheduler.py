import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Schedule:
    """Expert scheduling result for fused MoE kernel.

    Attributes:
        expert_order: Experts sorted by load (descending)
        block_assignments: List of (expert_id, token_indices) pairs
        num_blocks: Number of blocks needed
        max_tokens_per_block: Maximum tokens in any block
    """
    expert_order: List[int]
    block_assignments: List[Tuple[int, torch.Tensor]]
    num_blocks: int
    max_tokens_per_block: int


def schedule_experts(
    topk_ids: torch.Tensor,  # [M, top_k]
    num_experts: int,
    tokens_per_block: int = 32,
    schedule_mode: str = "balanced"
) -> Schedule:
    """
    Generate expert-centric schedule for fused MoE kernel.

    Strategies:
    - "balanced": Sort experts by load, distribute evenly across blocks
    - "compact": Minimize block count, pack tokens tightly
    - "interleaved": Round-robin assignment for better latency hiding

    Args:
        topk_ids: Token-to-expert assignments [M, top_k]
        num_experts: Total number of experts
        tokens_per_block: Target number of tokens per block
        schedule_mode: Scheduling strategy

    Returns:
        Schedule object with expert order and block assignments
    """
    M, top_k = topk_ids.shape

    if schedule_mode == "balanced":
        return _schedule_balanced(topk_ids, num_experts, tokens_per_block)
    elif schedule_mode == "compact":
        return _schedule_compact(topk_ids, num_experts, tokens_per_block)
    elif schedule_mode == "interleaved":
        return _schedule_interleaved(topk_ids, num_experts, tokens_per_block)
    else:
        raise ValueError(f"Unknown schedule_mode: {schedule_mode}")


def _schedule_balanced(
    topk_ids: torch.Tensor,
    num_experts: int,
    tokens_per_block: int
) -> Schedule:
    """
    Balanced scheduling: sort experts by load, distribute evenly.

    This minimizes block idle time by ensuring each block has similar work.
    """
    # Step 1: Count tokens per expert
    expert_counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)

    # Step 2: Sort experts by load (descending)
    expert_order = torch.argsort(expert_counts, descending=True).tolist()

    # Step 3: Build block assignments
    # For each expert, find all token indices assigned to it
    block_assignments = []
    max_tokens = 0

    for expert_id in expert_order:
        # Find all token indices for this expert
        token_indices = torch.where(topk_ids == expert_id)[0]
        num_tokens = len(token_indices)

        if num_tokens == 0:
            continue

        # Group tokens into blocks
        for start in range(0, num_tokens, tokens_per_block):
            end = min(start + tokens_per_block, num_tokens)
            block_tokens = token_indices[start:end]
            block_assignments.append((expert_id, block_tokens))
            max_tokens = max(max_tokens, len(block_tokens))

    return Schedule(
        expert_order=expert_order,
        block_assignments=block_assignments,
        num_blocks=len(block_assignments),
        max_tokens_per_block=max_tokens
    )


def _schedule_compact(
    topk_ids: torch.Tensor,
    num_experts: int,
    tokens_per_block: int
) -> Schedule:
    """
    Compact scheduling: minimize block count.

    Similar to balanced but prioritizes filling blocks completely.
    """
    # Same as balanced for now, can be optimized later
    return _schedule_balanced(topk_ids, num_experts, tokens_per_block)


def _schedule_interleaved(
    topk_ids: torch.Tensor,
    num_experts: int,
    tokens_per_block: int
) -> Schedule:
    """
    Interleaved scheduling: round-robin assignment.

    Better for latency hiding when experts have varying compute times.
    """
    # Count tokens per expert
    expert_counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)

    # Create interleaved order (round-robin through experts)
    expert_order = list(range(num_experts))

    # Build block assignments in interleaved order
    block_assignments = []
    max_tokens = 0

    for expert_id in expert_order:
        token_indices = torch.where(topk_ids == expert_id)[0]
        num_tokens = len(token_indices)

        if num_tokens == 0:
            continue

        for start in range(0, num_tokens, tokens_per_block):
            end = min(start + tokens_per_block, num_tokens)
            block_tokens = token_indices[start:end]
            block_assignments.append((expert_id, block_tokens))
            max_tokens = max(max_tokens, len(block_tokens))

    return Schedule(
        expert_order=expert_order,
        block_assignments=block_assignments,
        num_blocks=len(block_assignments),
        max_tokens_per_block=max_tokens
    )


def create_expert_mask(
    schedule: Schedule,
    M: int,
    num_experts: int,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Create expert mask tensor for kernel launch.

    Args:
        schedule: Schedule from schedule_experts
        M: Number of tokens
        num_experts: Number of experts
        device: Target device

    Returns:
        expert_mask: [num_blocks, M] boolean mask
    """
    expert_mask = torch.zeros(
        (schedule.num_blocks, M),
        dtype=torch.bool,
        device=device
    )

    for block_id, (expert_id, token_indices) in enumerate(schedule.block_assignments):
        expert_mask[block_id, token_indices] = True

    return expert_mask


def create_block_offsets(
    schedule: Schedule,
    d_hidden: int,
    d_expert: int,
    device: torch.device = torch.device('cuda')
) -> dict:
    """
    Create offset tensors for kernel launch.

    Args:
        schedule: Schedule from schedule_experts
        d_hidden: Hidden dimension
        d_expert: Expert dimension
        device: Target device

    Returns:
        Dictionary of offset tensors
    """
    # Flatten token indices
    all_token_indices = []
    for _, token_indices in schedule.block_assignments:
        all_token_indices.append(token_indices)

    token_offsets = torch.cat(all_token_indices).to(device)

    # Create cumulative offsets for block indexing
    block_starts = torch.cumsum(
        torch.tensor([len(t) for _, t in schedule.block_assignments]),
        dim=0
    )
    block_starts = torch.cat([torch.tensor([0]), block_starts[:-1]])

    return {
        'token_offsets': token_offsets,
        'block_starts': block_starts.to(device),
        'num_blocks': schedule.num_blocks
    }
