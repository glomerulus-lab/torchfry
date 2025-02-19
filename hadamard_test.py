import torch
import numpy as np
import time
from fast_hadamard_transform.fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform as cuda_hadamard
# Using your provided function as pytorch_hadamard

def pytorch_hadamard(u, normalize=False):
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

# Set up test data
dim = 16_384
batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create random test data
data = torch.randn(batch_size, dim, device=device)

# Warm up the GPU
for _ in range(10):
    _ = cuda_hadamard(data)
    _ = pytorch_hadamard(data)

# Test CUDA version
cuda_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = cuda_hadamard(data)
    torch.cuda.synchronize()
    cuda_times.append(time.perf_counter() - start)

# Test PyTorch version
pytorch_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = pytorch_hadamard(data)
    torch.cuda.synchronize()
    pytorch_times.append(time.perf_counter() - start)

# Calculate statistics
cuda_mean = np.mean(cuda_times) * 1000  # Convert to ms
cuda_std = np.std(cuda_times) * 1000
pytorch_mean = np.mean(pytorch_times) * 1000
pytorch_std = np.std(pytorch_times) * 1000

print(f"CUDA Implementation:")
print(f"Mean time: {cuda_mean:.2f} ms ± {cuda_std:.2f} ms")
print(f"\nPyTorch Implementation:")
print(f"Mean time: {pytorch_mean:.2f} ms ± {pytorch_std:.2f} ms")
print(f"\nSpeedup: {pytorch_mean/cuda_mean:.2f}x")

# Verify outputs match
cuda_output = cuda_hadamard(data)
pytorch_output = pytorch_hadamard(data)
max_diff = torch.max(torch.abs(cuda_output - pytorch_output))
print(f"\nMaximum difference between outputs: {max_diff:.2e}")