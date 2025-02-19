import torch
import math
import numpy as np
import time
from typing import Tuple

class WalshHadamardTransform:
    def __init__(self, size, coeff=None):
        self.size = size
        self.coeff = coeff
        self.matrix = self._generate_walsh_hadamard()
        # Convert to torch tensor once during initialization
        self.matrix_tensor = torch.tensor(self.matrix, dtype=torch.float32)

    def _generate_walsh_hadamard(self):
        """Generate Walsh-Hadamard matrix with frequencies in ascending order"""
        n = int(math.log2(self.size))
        # Initialize with proper scaling
        row = [1 / (math.sqrt(2) ** n)] * self.size
        matrix = [list(row) for _ in range(self.size)]
        
        # Apply Hadamard pattern
        for i in range(n):
            for j in range(self.size):
                for k in range(self.size):
                    if (j // (2 ** i)) % 2 == 1 and (k // (2 ** i)) % 2 == 1:
                        matrix[j][k] = -matrix[j][k]
                        if self.coeff is not None:
                            if abs(matrix[j][k] - self.coeff) < 1e-6:
                                matrix[j][k] = 0

        # Sort rows by frequency (number of sign changes)
        matrix.sort(key=lambda x: sum(1 for a, b in zip(x[1:], x[:-1]) if a * b < 0))
        return matrix

    def transform(self, u, normalize=False):
        """
        Apply Walsh-Hadamard transform using explicit matrix multiplication.
        """
        original_shape = u.shape
        u_reshaped = u.view(-1, self.size)
        result = torch.matmul(u_reshaped, self.matrix_tensor.T)
        return result.view(original_shape)

def fast_hadamard_transform(u, normalize=False):
    """
    Fast Hadamard transform using butterfly operations.
    Parameters:
        u: Tensor of shape (..., n) where n is a power of 2
        normalize: if True, divide by sqrt(n)
    Returns:
        Tensor of shape (..., n)
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], 
                      x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

def benchmark_transforms(size: int = 16_000, batch_size: int = 2, num_runs: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Benchmark both Hadamard transform implementations.
    """
    print(f"\nBenchmarking with size={size}, batch_size={batch_size}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    x = torch.randn(batch_size, size, device=device)
    
    # Initialize matrix-based transform
    wht = WalshHadamardTransform(size)
    wht.matrix_tensor = wht.matrix_tensor.to(device)
    
    # Warmup runs
    print("Performing warmup runs...")
    _ = fast_hadamard_transform(x)
    _ = wht.transform(x)
    
    # Time fast transform
    start = time.perf_counter()
    for _ in range(num_runs):
        result1 = fast_hadamard_transform(x)
    end = time.perf_counter()
    fast_time = (end - start) / num_runs
    
    # Time matrix-based transform
    start = time.perf_counter()
    for _ in range(num_runs):
            # Initialize matrix-based transform
        wht = WalshHadamardTransform(size)
        wht.matrix_tensor = wht.matrix_tensor.to(device)
        result2 = wht.transform(x)
    end = time.perf_counter()
    matrix_time = (end - start) / num_runs
    
    # Print results
    print(f"\nResults over {num_runs} runs:")
    print(f"Fast butterfly implementation: {fast_time*1000:.2f} ms per run")
    print(f"Matrix-based implementation: {matrix_time*1000:.2f} ms per run")
    print(f"Matrix version is {fast_time/matrix_time:.2f}x slower")
    
    # Verify results match
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    print(f"\nMax difference between implementations: {max_diff:.2e}")
    
    # Memory usage
    if device == 'cuda':
        print("\nMemory Usage:")
        print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    return result1, result2

if __name__ == "__main__":
    # Example usage
    size = 4096
    batch_size = 2
    
    # Create sample data
    data = torch.randn(batch_size, size)
    
    # Run both transforms and benchmark
    results = benchmark_transforms(size=size, batch_size=batch_size, num_runs=5)