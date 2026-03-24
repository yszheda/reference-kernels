"""
Unit tests for moe_fused.scheduler module.

Run with: python3 moe_fused/tests/test_scheduler.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from moe_fused.scheduler import (
    Schedule,
    schedule_experts,
    create_expert_mask,
    create_block_offsets
)


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

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=32)

    # 验证专家按负载排序 (expert 3 有 5 个 tokens, expert 0 有 4 个，etc.)
    # 统计每个专家的出现次数
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    expected_order = torch.argsort(counts, descending=True).tolist()

    assert len(schedule.expert_order) == num_experts, f"Expected {num_experts} experts, got {len(schedule.expert_order)}"
    assert schedule.expert_order == expected_order, f"Expected order {expected_order}, got {schedule.expert_order}"

    # 验证所有 token 都被分配
    total_tokens = sum(len(token_indices) for _, token_indices in schedule.block_assignments)
    expected_total = len(topk_ids) * topk_ids.shape[1]  # M * top_k
    assert total_tokens == expected_total, f"Expected {expected_total} token assignments, got {total_tokens}"

    print("✓ test_expert_scheduling_basic: PASSED")


def test_expert_scheduling_load_balance():
    """测试负载均衡效果"""
    # 创建不均匀的专家分配
    # expert 0: 100 tokens, expert 1: 50 tokens, expert 2: 25 tokens
    M = 100
    top_k = 1
    num_experts = 3

    # 手动构造 topk_ids
    topk_ids = torch.zeros(M, top_k, dtype=torch.int32)
    topk_ids[:50, 0] = 0   # tokens 0-49 -> expert 0
    topk_ids[50:75, 0] = 0 # tokens 50-74 -> expert 0 (total 50)
    topk_ids[75:85, 0] = 1 # tokens 75-84 -> expert 1
    topk_ids[85:95, 0] = 1 # tokens 85-94 -> expert 1 (total 20)
    topk_ids[95:100, 0] = 2 # tokens 95-99 -> expert 2 (total 5)

    # 重新构造：让分布更明显
    topk_ids = torch.cat([
        torch.zeros(50, 1, dtype=torch.int32),  # expert 0
        torch.ones(30, 1, dtype=torch.int32),   # expert 1
        torch.full((20, 1), 2, dtype=torch.int32),  # expert 2
    ], dim=0)

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=16)

    # expert 0 应该有最多的 blocks (50/16 = 4 blocks)
    # expert 1 其次 (30/16 = 2 blocks)
    # expert 2 最少 (20/16 = 2 blocks)

    # 验证 expert order 按负载排序
    assert schedule.expert_order[0] == 0, "Expert 0 should have highest load"

    print(f"  Schedule: {schedule.num_blocks} blocks, max_tokens={schedule.max_tokens_per_block}")
    print("✓ test_expert_scheduling_load_balance: PASSED")


def test_expert_scheduling_small_block():
    """测试小 block size 的调度"""
    topk_ids = torch.tensor([
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
    ])
    num_experts = 4
    tokens_per_block = 2  # 小 block size

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=tokens_per_block)

    # 验证每个 block 不超过 tokens_per_block
    for expert_id, token_indices in schedule.block_assignments:
        assert len(token_indices) <= tokens_per_block, \
            f"Block has {len(token_indices)} tokens, exceeds {tokens_per_block}"

    print(f"  Schedule: {schedule.num_blocks} blocks for small block test")
    print("✓ test_expert_scheduling_small_block: PASSED")


def test_create_expert_mask():
    """测试专家掩码创建"""
    topk_ids = torch.tensor([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
    ])
    num_experts = 4
    M = 4

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=32)
    expert_mask = create_expert_mask(schedule, M, num_experts)

    # 验证掩码形状
    assert expert_mask.shape[0] == schedule.num_blocks
    assert expert_mask.shape[1] == M

    # 验证每个 block 至少有一个 True
    for block_id in range(schedule.num_blocks):
        assert expert_mask[block_id].any(), f"Block {block_id} has no active tokens"

    print(f"  Expert mask shape: {expert_mask.shape}")
    print("✓ test_create_expert_mask: PASSED")


def test_create_block_offsets():
    """测试块偏移量创建"""
    topk_ids = torch.tensor([
        [0, 1],
        [0, 2],
        [1, 3],
    ])
    num_experts = 4
    d_hidden = 7168
    d_expert = 256

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=32)
    offsets = create_block_offsets(schedule, d_hidden, d_expert)

    assert 'token_offsets' in offsets
    assert 'block_starts' in offsets
    assert 'num_blocks' in offsets

    assert offsets['num_blocks'] == schedule.num_blocks
    assert len(offsets['token_offsets']) == len(topk_ids) * topk_ids.shape[1]

    print(f"  Block offsets: {offsets['num_blocks']} blocks")
    print("✓ test_create_block_offsets: PASSED")


def test_schedule_modes():
    """测试不同调度模式"""
    topk_ids = torch.randint(0, 8, (32, 4))
    num_experts = 8

    modes = ['balanced', 'compact', 'interleaved']

    results = {}
    for mode in modes:
        schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=16, schedule_mode=mode)
        results[mode] = schedule.num_blocks
        print(f"  Mode '{mode}': {schedule.num_blocks} blocks")

    # 所有模式应该产生有效的调度
    for mode, num_blocks in results.items():
        assert num_blocks > 0, f"Mode {mode} produced no blocks"

    print("✓ test_schedule_modes: PASSED")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running moe_fused.scheduler tests")
    print("=" * 60)

    tests = [
        test_expert_scheduling_basic,
        test_expert_scheduling_load_balance,
        test_expert_scheduling_small_block,
        test_create_expert_mask,
        test_create_block_offsets,
        test_schedule_modes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__}: ERROR - {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
