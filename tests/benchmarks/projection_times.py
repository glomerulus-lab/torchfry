"""
Benchmarking Projection Times for Fastfood and Random Kitchen Sink Methods

This script compares the speed of our random feature map methods,
such as Random Kitchen Sinks (RKS) and Fastfood transforms, across varying input dimensions.
It includes both CPU and GPU-based implementations using scikit-learn, scikit-learn-extra,
and torchfry. The results are visualized as log-scaled runtime plots.

Functions:
---------
- exact_rbf_sampler(input_dims, num_runs=10): 
    Measures the average runtime of scikit-learn's RBFSampler on CPU.
    
- other_RKS(input_dims, num_runs=10): 
    Benchmarks a custom RKSLayer implementation (CPU).

- sklearn_ff(input_dims, num_runs=10): 
    Tests the scikit-learn-extra Fastfood implementation (CPU), timing internal operations.

- RKS_GPU_layer(input_dims, num_runs=10): 
    Evaluates the GPU performance of this repo's Random Kitchen Sink layer

- FF_Layer(input_dims, num_runs=10): 
    Evaluates the GPU performance of this repo's Fastfood layer

Usage:
-----
Adjust the `proj_names` and `projection_methods` lists to include the projection methods
you want to benchmark. The benchmark runs each method over a list of input feature dimensions 
(from 128 to 8192) and plots average transformation time over multiple runs.

Run:
----
$ python tests/benchmarks/projection_times.py
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from torchfry.transforms import FastfoodLayer, RKSLayer
import time
import torch

def exact_rbf_sampler(input_dims, num_runs=10):
    times = []
    for dim in input_dims:
        x = np.random.rand(4096, dim)
        output_dim = dim * 4
        rks = RBFSampler(gamma=(1/(2*scale**2)), n_components=output_dim)
        rks.fit(x)

        total_time = 0
        for _ in range(num_runs):
            start = time.time()
            rks.transform(x)
            end = time.time()
            total_time += (end - start)

        avg_time = total_time / num_runs
        times.append(avg_time)

    return times

def other_RKS(input_dims, num_runs=10):
    times = []
    for dim in input_dims:
        x = np.random.rand(4096, dim)
        output_dim = dim * 4
        rks = RKSLayer(output_dim, scale)
        rks.fit(x)

        total_time = 0
        for _ in range(num_runs):
            start = time.time()
            rks.transform(x)
            end = time.time()
            total_time += (end - start)

        avg_time = total_time / num_runs
        times.append(avg_time)

    return times

def sklearn_ff(input_dims, num_runs=10):
    times = []
    for dim in input_dims:
        x = np.random.rand(4096, dim)
        output_dim = dim * 4
        trade = 'accuracy'
        ff = Fastfood(sigma=scale, n_components=output_dim, tradeoff_mem_accuracy=trade)
        ff.fit(x)
        ff.transform(x)

        total_time = 0
        for _ in range(num_runs):
            start = time.time()
            X_padded = ff._pad_with_zeros(x)
            HGPHBX = ff._apply_approximate_gaussian_matrix(
                ff._B, ff._G, ff._P, X_padded
            )
            VX = ff._scale_transformed_data(ff._S, HGPHBX)
            VX = torch.tensor(VX)

            if trade == 'mem':
                VX = torch.cos(VX + ff._U)
                output = VX * np.sqrt(2.0 / VX.shape[1])

            if trade == 'accuracy':
                (1 / np.sqrt(VX.shape[1])) * torch.hstack([torch.cos(VX), torch.sin(VX)])

            end = time.time()
            total_time += (end - start)

        avg_time = total_time / num_runs
        times.append(avg_time)

    return times

def RKS_GPU_layer(input_dims, num_runs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    times = []
    for dim in input_dims:
        x = np.random.rand(4096, dim)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        output_dim = dim * 4
        rks_obj = RKSLayer(input_dim=dim, output_dim=output_dim, scale=scale, device=device)

        total_time = 0
        torch.cuda.synchronize()
        for _ in range(num_runs):
            # torch.cuda.synchronize()
            start = time.perf_counter()
            rks_obj.forward(x)
            total_time += (time.perf_counter()-start)

        avg_time = total_time / num_runs
        times.append(avg_time)

    return times

def FF_Layer(input_dims, num_runs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    times = []
    for dim in input_dims:
        x = np.random.rand(4096, dim)
        x = torch.tensor(x, dtype=torch.float32, device=device)

        output_dim = dim * 4
        fast_food_obj = FastfoodLayer(input_dim=dim, output_dim=output_dim, scale=scale, device=device, hadamard='Torch')

        total_time = 0
        torch.cuda.synchronize()
        for _ in range(num_runs):
            # torch.cuda.synchronize()
            start = time.perf_counter()
            fast_food_obj.forward(x)
            total_time += (time.perf_counter()-start)

        avg_time = total_time / num_runs
        times.append(avg_time)

    return times

if __name__ == '__main__':
    proj_names = [
        # "Exact RBF Kernel (CPU)",
        # "Random Kitchen Sink (CPU)",
        # "Sklearn Fastfood (CPU)",
        # "Random Kitchen Sink (GPU)",
        "Fastfood (GPU)"
    ]
    projection_methods = [
        # exact_rbf_sampler,
        # other_RKS,
        # sklearn_ff,
        # RKS_GPU_layer,
        FF_Layer,
    ]

    input_dims = [128, 128, 128, 128, 256, 512, 1024, 2048, 4096, 8192]
    num_data = 4096
    scale = 20

    for name, proj_method in zip(proj_names, projection_methods):
        proj_time = proj_method(input_dims, num_runs=20)
        plt.plot(input_dims[3:], proj_time[3:], label=name, marker='o')

    plt.xlabel('Input Dimension')
    plt.xticks(input_dims[2:], input_dims[2:], rotation=90)
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('Projection Times')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
