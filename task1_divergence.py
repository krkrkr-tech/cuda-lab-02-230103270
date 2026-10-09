"""CUDA warp-divergence microbenchmark."""

from __future__ import annotations

import time

import numpy as np
from numba import cuda

ITERATIONS = 1_000
THREADS_PER_BLOCK = 256


@cuda.jit
def uniform_path_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]
        for _ in range(ITERATIONS):
            value = value * 1.000001 + 0.000001
        data[idx] = value


@cuda.jit
def interleaved_divergence_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]
        if idx % 2 == 0:
            for _ in range(ITERATIONS):
                value = value * 1.000001 + 0.000001
        else:
            for _ in range(ITERATIONS):
                value = (value - 0.000001) / 1.000001
        data[idx] = value


@cuda.jit
def warp_aligned_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]
        if (idx // 32) % 2 == 0:
            for _ in range(ITERATIONS):
                value = value * 1.000001 + 0.000001
        else:
            for _ in range(ITERATIONS):
                value = (value - 0.000001) / 1.000001
        data[idx] = value


def _time_kernel(kernel, device_data, blocks: int) -> float:
    kernel[blocks, THREADS_PER_BLOCK](device_data)
    cuda.synchronize()
    measurements = []
    for _ in range(10):
        start = time.perf_counter()
        kernel[blocks, THREADS_PER_BLOCK](device_data)
        cuda.synchronize()
        measurements.append(time.perf_counter() - start)
    return float(np.mean(measurements))


def benchmark_divergence(n: int = 256 * 1_024) -> dict[str, float]:
    """Benchmark kernels using device execution time only."""
    if n <= 0:
        raise ValueError("n must be positive")
    host_data = np.ones(n, dtype=np.float32)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    results = {}
    for name, kernel in (
        ("uniform", uniform_path_kernel),
        ("interleaved", interleaved_divergence_kernel),
        ("warp_aligned", warp_aligned_kernel),
    ):
        device_data = cuda.to_device(host_data)
        results[name] = _time_kernel(kernel, device_data, blocks)
    print("TASK 1 BENCHMARK (seconds)")
    for name, seconds in results.items():
        print(f"{name:>12}: {seconds:.6f}")
    return results


if __name__ == "__main__":
    benchmark_divergence()
