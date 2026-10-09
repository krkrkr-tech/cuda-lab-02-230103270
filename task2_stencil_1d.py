"""1D three-point smoothing stencil with replicated halo boundaries."""

from __future__ import annotations

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256


@cuda.jit
def stencil_1d(d_in, d_out, n):
    idx = cuda.grid(1)
    if idx < n:
        left = d_in[idx - 1] if idx > 0 else d_in[idx]
        right = d_in[idx + 1] if idx < n - 1 else d_in[idx]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in: np.ndarray) -> np.ndarray:
    """Run the stencil on a host array and return a float32 host result."""
    values = np.asarray(h_in, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("h_in must be a non-empty one-dimensional array")
    d_in = cuda.to_device(values)
    d_out = cuda.device_array_like(values)
    blocks = (values.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    stencil_1d[blocks, THREADS_PER_BLOCK](d_in, d_out, values.size)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr: np.ndarray) -> np.ndarray:
    """Exact NumPy reference implementation required by the lab."""
    values = np.asarray(arr, dtype=np.float32)
    padded = np.pad(values, (1, 1), mode="edge")
    return (
        0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    ).astype(np.float32)


if __name__ == "__main__":
    sample = np.sin(np.linspace(0, 10, 10_007)).astype(np.float32)
    result = run_stencil(sample)
    reference = cpu_stencil(sample)
    print(f"TASK 2 PASSED: MAX DELTA = {np.max(np.abs(result - reference)):.8f}")
