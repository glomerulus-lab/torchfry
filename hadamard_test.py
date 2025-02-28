import torch
import numpy as np
import time
from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform as cuda_hadamard
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
dim = 16_384
batch_size = 8_192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create random test data
data = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
# Hadamard matrix
had = torch.tensor(scipy.linalg.hadamard(dim), device=device, dtype=torch.float32)
# Random Gaussian Matrix
gaus = torch.randn((dim, dim), device=device)


# Warm up the GPU
for _ in range(10):
    _ = cuda_hadamard(data)
    _ = pytorch_hadamard(data)
    F.linear(data, had)
    result = data @ gaus 
    
# Test DAO version
cuda_times = []
for _ in range(40):
    start = time.perf_counter()
    _ = cuda_hadamard(data)
    cuda_times.append(time.perf_counter() - start)

# Test Recursive version
pytorch_times = []
for _ in range(40):
    start = time.perf_counter()
    _ = pytorch_hadamard(data)
    pytorch_times.append(time.perf_counter() - start)

# Scipy version
scipy_times = []
for _ in range(40):
    start = time.perf_counter()
    F.linear(data, had) 
    scipy_times.append(time.perf_counter() - start)

# Scipy version
matrix_times = []
for _ in range(40):
    start = time.perf_counter()
    result = data @ gaus 
    matrix_times.append(time.perf_counter() - start)

# Calculate statistics
cuda_mean = np.mean(cuda_times) * 1000
cuda_std = np.std(cuda_times) * 1000
pytorch_mean = np.mean(pytorch_times) * 1000
pytorch_std = np.std(pytorch_times) * 1000
scipy_mean = np.mean(scipy_times) * 1000
scipy_std = np.std(scipy_times) * 1000
matrix_mean = np.mean(matrix_times) * 1000
matrix_std = np.std(matrix_times) * 1000


print(f"Dao FWHT:")
print(f"Mean time: {cuda_mean:.2f} ms ± {cuda_std:.2f} ms")
print(f"\nPyTorch FWHT:")
print(f"Mean time: {pytorch_mean:.2f} ms ± {pytorch_std:.2f} ms")
print(f"\nWHT matmul:")
print(f"Mean time: {scipy_mean:.2f} ms ± {scipy_std:.2f} ms")
print(f"\nMatmul:")
print(f"Mean time: {matrix_mean:.2f} ms ± {matrix_std:.2f} ms")