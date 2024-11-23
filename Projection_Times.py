import numpy as np
import sklearn as sc
from FastFood_BaseLine.random_projections import projections
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from FastFood_Layer import FastFood_Layer
from RKS_Layer import RKS_Layer
from BIG_FastFood_Layer import BIG_Fastfood_Layer
import time  
import torch
import math



def exact_rbf_sampler(x, output_dims):
    #rks error
    times = []
    for dim in output_dims:
        #rks approx
        rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
        rks.fit(x)

        start = time.time()
        rks.transform(x)
        end = time.time()

        times.append(end-start)
    # warm up
    return times


def other_RKS(x, output_dims):
    times = []
    #rks error
    for dim in output_dims:
        #rks approx
        rks = projections.rks(dim, scale)
        rks.fit(x)

        start = time.time()
        rks.transform(x)
        end = time.time()

        times.append(end-start)
    # warm up
    return times


def sklearn_ff(x, output_dims):
    #ff error
    times = []
    for dim in output_dims:
        trade = 'accuracy'
        #ff approx
        ff = Fastfood(sigma=scale, n_components=dim, tradeoff_mem_accuracy=trade)
        ff.fit(x)
        ff.transform(x)

        start = time.time()
        X_padded = ff._pad_with_zeros(x)
        HGPHBX = ff._apply_approximate_gaussian_matrix(
            ff._B, ff._G, ff._P, X_padded
        )
        VX = ff._scale_transformed_data(ff._S, HGPHBX)
        VX = torch.tensor(VX)

        #memory
        if trade == 'mem':
            VX = torch.cos(VX + ff._U)
            output = VX * np.sqrt(2.0 / VX.shape[1])

        #accuracy: non non linearity
        if trade == 'accuracy':
            (1 / np.sqrt(VX.shape[1])) * torch.hstack([torch.cos(VX), torch.sin(VX)])

        end = time.time()
        times.append(end-start)
    # warm up
    return times


def fastfood_GPU_layer(x, output_dims):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    times=[]
    for n in output_dims:
        fast_food_obj = FastFood_Layer(input_dim=x.shape[1], output_dim=n, scale=scale, device=device)

        start = time.time()
        phi = fast_food_obj.forward(x)
        end = time.time()

        times.append(end-start)
    # warm up
    return times


def RKS_GPU_layer(x, output_dims):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    times=[]
    for n in output_dims:
        rks_obj = RKS_Layer(input_dim=x.shape[1], output_dim=n, scale=scale, device=device)

        start = time.time()
        phi = rks_obj.forward(x)
        end = time.time()

        times.append(end-start)
    # warm up
    return times


def BIG_ff_layer(x, output_dims):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    times=[]
    for n in output_dims:
        fast_food_obj = BIG_Fastfood_Layer(input_dim=x.shape[1], output_dim=n, scale=scale, device=device)

        start = time.time()
        phi = fast_food_obj.forward(x)
        end = time.time()

        times.append(end-start)
    # warm up
    return times


if __name__ == '__main__':
    
    proj_names = np.array([
        "Exact RBF Kernel (CPU)",
        "Random Kitchen Sink (CPU)",
        "Sklearn Fastfood (CPU)",
        "Classic Fastfood (GPU)",
        "Random Kitchen Sink (GPU)",
        "BIG Fastfood (GPU)"
    ])
    projection_methods = np.array([
        exact_rbf_sampler,
        other_RKS,
        sklearn_ff,
        fastfood_GPU_layer,
        RKS_GPU_layer,
        BIG_ff_layer,
    ])

    # Dimensioning
    input_dim = 256
    output_dims = [256,256,512,1024,2048,4096,]#8192,16384]
    num_data = 10_000 

    # Data
    x = np.random.rand(num_data, input_dim)
    
    # Universal Scale
    scale = 20

    for name, proj_method in zip(proj_names, projection_methods):
        proj_time = proj_method(x, output_dims)
        plt.plot(output_dims[2:], proj_time[2:], label=name, marker='o')

    plt.xlabel('Dimension')
    plt.xticks(output_dims[2:], output_dims[2:], rotation=90)
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('Projection Times')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
