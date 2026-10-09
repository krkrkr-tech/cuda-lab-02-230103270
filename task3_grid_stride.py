"""Arbitrary-size vector scaling using a grid-stride loop."""

from __future__ import annotations

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, n):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, n, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr: np.ndarray, factor: float) -> np.ndarray:
    """Scale every element with exactly 64 blocks of 256 threads."""
    values = np.asarray(h_arr, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("h_arr must be a non-empty one-dimensional array")
    d_arr = cuda.to_device(values)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        d_arr, np.float32(factor), values.size
    )
    cuda.synchronize()
    return d_arr.copy_to_host()


if __name__ == "__main__":
    input_values = np.ones(100_000, dtype=np.float32)
    output = run_grid_stride(input_values, 4.25)
    print(f"TASK 3 PASSED: {np.allclose(output, 4.25)}")
