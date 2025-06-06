"""
Benchmarking Hadamard Transform Implementations

This script compares the runtime and correctness of different implementations of the Hadamard Transform.
It benchmarks:
- A CUDA-accelerated Hadamard (from `fast_hadamard_transform`)
- A recursive PyTorch-based Hadamard
- A matrix-multiplication-based Walsh-Hadamard Transform using SciPy
- A dense Gaussian matrix multiplication as an exact baseline

Usage:
-----
The script initializes random input data, runs each method over multiple trials with GPU synchronization,
and reports the average runtime with standard error. It also compares equality of the outputs.

Adjust `dim`, `batch_size`, `num_trials`, and `num_warmup` to configure the benchmarking.

Run:
----
$ python test/benchmarks/hadamard_test.py
"""


import torch
import numpy as np
import time
from fast_hadamard_transform import hadamard_transform as cuda_hadamard
import scipy
import torch.nn.functional as F

# Recursive version
def pytorch_hadamard(u, normalize=False):
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

# Set up test data
dim = 8_192
batch_size = 8_192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
num_warmup = 5
num_trials = 100

# Create random test data
data = torch.randn(batch_size, dim, device=device, dtype=dtype)
# Hadamard matrix
had = torch.tensor(scipy.linalg.hadamard(dim), device=device, dtype=dtype)
# Random Gaussian Matrix
gaus = torch.randn((dim, dim), device=device, dtype=dtype)

# Warm up the GPU
for _ in range(num_warmup):
    _ = cuda_hadamard(data)
    _ = pytorch_hadamard(data)
    _ = F.linear(data, had)
    _ = F.linear(data, gaus)

# Test CUDA Hadamard
cuda_times = []
torch.cuda.synchronize()
for _ in range(num_trials):
    # torch.cuda.synchronize()
    start = time.perf_counter()
    _ = cuda_hadamard(data)
    cuda_times.append(time.perf_counter() - start)

# Test PyTorch Hadamard
pytorch_times = []
torch.cuda.synchronize()
for _ in range(num_trials):
    # torch.cuda.synchronize()
    start = time.perf_counter()
    _ = pytorch_hadamard(data)
    pytorch_times.append(time.perf_counter() - start)

# Test Scipy Hadamard (Matrix Multiplication)
scipy_times = []
torch.cuda.synchronize()
for _ in range(num_trials):
    # torch.cuda.synchronize()
    start = time.perf_counter()
    _ = F.linear(data, had)
    scipy_times.append(time.perf_counter() - start)

# Test Gaussian Matrix Multiplication
torch.cuda.synchronize()
matrix_times = []
for _ in range(num_trials):
    # torch.cuda.synchronize()
    start = time.perf_counter()
    _ = F.linear(data, gaus)
    matrix_times.append(time.perf_counter() - start)

# Calculate statistics (ms)
def calc_stats(times):
    times_ms = np.array(times) * 1000
    return times_ms.mean(), times_ms.std() / np.sqrt(num_trials)

cuda_mean, cuda_std = calc_stats(cuda_times)
pytorch_mean, pytorch_std = calc_stats(pytorch_times)
scipy_mean, scipy_std = calc_stats(scipy_times)
matrix_mean, matrix_std = calc_stats(matrix_times)

print(f"Dao FWHT:")
print(f"Mean time: {cuda_mean:.2e} ms ± {cuda_std:.2e} ms")
print(f"\nPyTorch FWHT:")
print(f"Mean time: {pytorch_mean:.2e} ms ± {pytorch_std:.2e} ms")
print(f"\nWHT matmul:")
print(f"Mean time: {scipy_mean:.2e} ms ± {scipy_std:.2e} ms")
print(f"\nMatmul:")
print(f"Mean time: {matrix_mean:.2e} ms ± {matrix_std:.2e} ms")

# Get transform results once
cuda_out = cuda_hadamard(data)
pytorch_out = pytorch_hadamard(data)
scipy_out = F.linear(data, had)

# Compare all three
all_equal = (
    torch.allclose(cuda_out, pytorch_out, rtol=1e-3, atol=1e-3) and
    torch.allclose(cuda_out, scipy_out, rtol=1e-3, atol=1e-3)
)

print(f"\nAll transforms equal: {all_equal}")
