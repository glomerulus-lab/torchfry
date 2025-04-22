
"""
Benchmarking Approximation Error for Fastfood and Random Kitchen Sink Methods

This script compares the approximation error of different random feature map methods—
namely Random Kitchen Sinks (RKS) and Fastfood transforms—against the exact RBF kernel.
The benchmark evaluates the kernel approximation error of our implemented Fastfood
and Random Kitchen Sink layers against the exact RBF kernel. Error should be 
independent of computer architecture and choice of CPU or GPU.

Functions:
---------
- exact_rbf_kernel(x, exact, output_dims): 
    Computes the approximation error of scikit-learn's RBFSampler (CPU) against the exact RBF kernel.

- RKS_GPU_layer(x, exact, output_dims): 
    Computes the approximation error of this repo's Random Kitchen Sink layer (GPU) using fastfood-torch.

- FF_GPU_layer(x, exact, output_dims): 
    Computes the approximation error of this repo's Fastfood layer (GPU) using fastfood-torch.

Usage:
-----
The script generates two plots:
1. A linear-scale plot of approximation error vs. output dimension.
2. A log-log scale version of the same plot.

Adjust `input_dim`, `output_dims`, `scale`, or `num_data` to explore different benchmark configurations.

Run:
----
$ python tests/benchmarks/approximation_error.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from fastfood_torch.transforms import FastFoodLayer, RKSLayer

def exact_rbf_kernel(x, exact, output_dims):
    #rks error
    error = []
    for dim in output_dims:
        #rks approx
        rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
        rks.fit(x)
        rks_approx = rks.transform(x)

        rks_approx = rks_approx @ rks_approx.T
        
        difference = np.linalg.norm(exact-rks_approx, 'fro')
        error.append(difference/num_data)
    return error

def RKS_GPU_layer(x, exact, output_dims):
    #ff error
    error = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    for dim in output_dims:
        rks_obj = RKSLayer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
        phi = rks_obj.forward(x)

        rks_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact-rks_approx, 'fro')
        error.append(difference/num_data)
    return error


def FF_GPU_layer(x, exact, output_dims):

    #BIG ff error
    error = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    for dim in output_dims:
        fast_food_obj = FastFoodLayer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device, hadamard='Dao')
        phi = fast_food_obj.forward(x)

        ff_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact-ff_approx, 'fro')
        error.append(difference/num_data)
    return error


if __name__ == '__main__':

    error_names = [
        "Exact RBF",
        "RKS Layer",
        "FF Layer",
    ]
    approx_errors = [
        exact_rbf_kernel,
        RKS_GPU_layer,
        FF_GPU_layer,
    ]

    # Dimensioning
    input_dim = 8
    output_dims = [16,32,64,128,256,512,1024,2048,4096,8192]
    num_data = 2000 
    x = np.random.rand(num_data, input_dim)

    # Universal Scale
    scale = 0.5

    # Exact rbf kernel
    exact = rbf_kernel(x, gamma=(1/(2*scale**2)))

    save_errors = []
    for error_method, name in zip(approx_errors, error_names):
        error = error_method(x, exact, output_dims)
        save_errors.append(error)
        plt.plot(output_dims, error, label=name, marker='o')

    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.title('Approximation Error vs. RBF Kernel')
    plt.legend()
    plt.show()

    for error, name in zip(save_errors, error_names):
        plt.loglog(output_dims, error, label=name, marker='o')

    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.title('Approximation Error vs. RBF Kernel')
    plt.legend()
    plt.show()
