#!/usr/bin/env python3
"""
Local test runner for mxfp4-mm problem.
Usage: python run_test.py
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from task import TestSpec
from reference import check_implementation, generate_input
from submission import custom_kernel

def run_single_test(m: int, n: int, k: int, seed: int):
    """Run a single correctness test."""
    print(f"\nTest: m={m}, n={n}, k={k}, seed={seed}")

    # Generate input
    data = generate_input(m, n, k, seed)
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Run submission
    try:
        output = custom_kernel(data)
    except Exception as e:
        print(f"  FAIL: Kernel execution error: {e}")
        return False

    # Check against reference
    good, message = check_implementation(data, output)

    if good:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL: {message}")
        return False


def run_benchmark(m: int, n: int, k: int, seed: int, warmup: int = 10, repeats: int = 100):
    """Run benchmark and return timing stats."""
    print(f"\nBenchmark: m={m}, n={n}, k={k}, seed={seed}")

    # Generate input
    data = generate_input(m, n, k, seed)

    # Warmup
    for _ in range(warmup):
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Mean: {mean_time:.3f} ms, Min: {min_time:.3f} ms, Max: {max_time:.3f} ms")

    return mean_time, min_time, max_time


def main():
    # Test cases from task.yml
    tests = [
        {"m": 8, "n": 2112, "k": 7168, "seed": 124},
        {"m": 16, "n": 3072, "k": 1536, "seed": 6635},
        {"m": 64, "n": 3072, "k": 1536, "seed": 45},
        {"m": 256, "n": 2880, "k": 512, "seed": 78},
    ]

    benchmarks = [
        {"m": 4, "n": 2880, "k": 512, "seed": 4565},
        {"m": 16, "n": 2112, "k": 7168, "seed": 15},
        {"m": 32, "n": 4096, "k": 512, "seed": 457},
        {"m": 32, "n": 2880, "k": 512, "seed": 54},
        {"m": 64, "n": 7168, "k": 2048, "seed": 687},
        {"m": 256, "n": 3072, "k": 1536, "seed": 7856},
    ]

    print("=" * 60)
    print("MXFP4-MM Correctness Tests")
    print("=" * 60)

    all_passed = True
    for test in tests:
        if not run_single_test(**test):
            all_passed = False

    if all_passed:
        print("\n✓ All correctness tests passed!")
    else:
        print("\n✗ Some tests FAILED")
        return 1

    print("\n" + "=" * 60)
    print("MXFP4-MM Benchmarks")
    print("=" * 60)

    for bench in benchmarks:
        run_benchmark(**bench)

    return 0


if __name__ == "__main__":
    sys.exit(main())
