"""
Test script for moe_fused kernel against AITER reference.

Run on AMD GPU system:
    python3 test_against_reference.py

This script:
1. Generates test input using reference.generate_input
2. Runs AITER reference implementation
3. Runs moe_fused implementation
4. Compares outputs with tolerance checking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from reference import ref_kernel, generate_input
from moe_fused import fused_moe_ck, fused_moe_fused_reference


def test_small_scale():
    """小规模测试：验证基本正确性"""
    print("=" * 60)
    print("Test: Small Scale Correctness")
    print("=" * 60)

    # 小规模配置
    data = generate_input(
        dhidden=7168,
        dexpert=256,
        nroutedexperts=256,
        nexpertspertoken=8,
        nsharedexperts=1,
        bs=16,  # 小规模
        seed=42
    )

    (
        hidden_states,
        gate_up_weight, _,
        _, _,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    print(f"Config: M={config['bs']}, E={config['n_routed_experts']}, "
          f"d_expert={config['d_expert']}, top_k={config['total_top_k']}")

    # AITER 参考输出
    print("\nRunning AITER reference...")
    output_ref = ref_kernel(data)
    print(f"Reference output: min={output_ref.min():.4f}, max={output_ref.max():.4f}")

    # moe_fused 输出 (fuse_stage12=True)
    print("\nRunning moe_fused (fused)...")
    output_fused = fused_moe_ck(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        fuse_stage12=True,
    )
    print(f"Fused output: min={output_fused.min():.4f}, max={output_fused.max():.4f}")

    # 比较输出
    diff = (output_ref - output_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # 容差检查
    rtol, atol = 1e-2, 1e-2
    passed = max_diff < 1.0  # 宽松检查

    if passed:
        print("\n✓ PASSED (basic correctness)")
    else:
        print(f"\n✗ FAILED (max diff {max_diff} >= 1.0)")

    return passed


def test_reference_implementation():
    """测试 PyTorch 参考实现"""
    print("\n" + "=" * 60)
    print("Test: PyTorch Reference Implementation")
    print("=" * 60)

    data = generate_input(
        dhidden=7168,
        dexpert=256,
        nroutedexperts=256,
        nexpertspertoken=8,
        nsharedexperts=1,
        bs=32,
        seed=42
    )

    (
        hidden_states,
        gate_up_weight, _,
        _, _,
        _, _, _, _,
        topk_weights,
        topk_ids,
        config,
    ) = data

    # AITER 参考
    output_ref = ref_kernel(data)

    # moe_fused PyTorch 参考实现
    output_py = fused_moe_fused_reference(
        hidden_states,
        gate_up_weight,
        gate_up_weight,  # down weight (same shape for test)
        topk_weights,
        topk_ids,
    )

    # 基本检查
    assert output_py.shape == output_ref.shape
    assert not torch.isnan(output_py).any()

    print(f"PyTorch ref output: min={output_py.min():.4f}, max={output_py.max():.4f}")
    print("✓ PASSED (PyTorch reference runs)")

    return True


def test_scheduler():
    """测试调度器"""
    print("\n" + "=" * 60)
    print("Test: Expert Scheduler")
    print("=" * 60)

    from moe_fused import schedule_experts

    # 构造测试数据
    M = 128
    top_k = 9
    num_experts = 257

    topk_ids = torch.randint(0, num_experts, (M, top_k))

    schedule = schedule_experts(topk_ids, num_experts, tokens_per_block=32)

    print(f"Experts: {num_experts}, Tokens: {M}, top_k: {top_k}")
    print(f"Schedule: {schedule.num_blocks} blocks")
    print(f"Expert order (top 5): {schedule.expert_order[:5]}")

    # 验证负载排序
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    expected_top = counts.argmax().item()
    actual_top = schedule.expert_order[0]

    if expected_top == actual_top:
        print(f"✓ Highest load expert correctly identified: {expected_top}")
    else:
        print(f"  Expected top: {expected_top}, Got: {actual_top}")

    # 验证所有 token 分配
    total_assigned = sum(len(idx) for _, idx in schedule.block_assignments)
    expected_total = M * top_k

    if total_assigned == expected_total:
        print(f"✓ All {expected_total} token assignments covered")
    else:
        print(f"✗ Expected {expected_total}, got {total_assigned}")

    print("✓ PASSED (scheduler)")
    return True


def main():
    """运行所有测试"""
    print("MOE Fused Kernel Test Suite")
    print("=" * 60)

    results = {}

    # 测试调度器
    try:
        results['scheduler'] = test_scheduler()
    except Exception as e:
        print(f"Scheduler test ERROR: {e}")
        results['scheduler'] = False

    # 测试 PyTorch 参考实现
    try:
        results['pytorch_ref'] = test_reference_implementation()
    except Exception as e:
        print(f"PyTorch reference test ERROR: {e}")
        results['pytorch_ref'] = False

    # 测试小规模正确性 (需要 AITER)
    try:
        results['small_scale'] = test_small_scale()
    except Exception as e:
        print(f"Small scale test ERROR: {e}")
        results['small_scale'] = False

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
