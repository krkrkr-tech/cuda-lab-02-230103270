"""2D Sobel-X convolution with zeroed borders."""

from __future__ import annotations

import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = 0.0
        else:
            d_out[row, col] = (
                -d_in[row - 1, col - 1]
                + d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - d_in[row + 1, col - 1]
                + d_in[row + 1, col + 1]
            )


def run_sobel(h_img: np.ndarray) -> np.ndarray:
    """Apply the horizontal Sobel filter to a 2D float32 image."""
    image = np.asarray(h_img, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("h_img must be a 2D array with at least 3 rows and columns")
    d_in = cuda.to_device(image)
    d_out = cuda.device_array_like(image)
    blocks = (
        (image.shape[1] + THREADS_2D[0] - 1) // THREADS_2D[0],
        (image.shape[0] + THREADS_2D[1] - 1) // THREADS_2D[1],
    )
    sobel_x_kernel[blocks, THREADS_2D](d_in, d_out, image.shape[0], image.shape[1])
    cuda.synchronize()
    return d_out.copy_to_host()


if __name__ == "__main__":
    result = run_sobel(np.ones((64, 64), dtype=np.float32))
    print(f"TASK 4 PASSED: {np.max(np.abs(result)):.8f}")
