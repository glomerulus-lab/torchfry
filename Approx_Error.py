import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from Layers.FastFood_Layer import FastFood_Layer
from Layers.RKS_Layer import RKS_Layer
import torch
from Layers.Name_Pending_Layer import BIG_Fastfood_Layer



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


def fastfood_GPU_layer(x, exact, output_dims):
    #ff error
    error = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    for dim in output_dims:
        fast_food_obj = FastFood_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
        phi = fast_food_obj.forward(x)

        ff_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact-ff_approx, 'fro')
        error.append(difference/num_data)
    return error


def RKS_GPU_layer(x, exact, output_dims):
    #ff error
    error = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    for dim in output_dims:
        rks_obj = RKS_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
        phi = rks_obj.forward(x)

        rks_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact-rks_approx, 'fro')
        error.append(difference/num_data)
    return error


def BIG_ff_layer(x, exact, output_dims):

    #BIG ff error
    error = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    for dim in output_dims:
        fast_food_obj = BIG_Fastfood_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
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
        "BIG FF Layer",
    ]
    approx_errors = [
        exact_rbf_kernel,
        # fastfood_GPU_layer,
        RKS_GPU_layer,
        BIG_ff_layer,
    ]

    # Dimensioning
    input_dim = 10
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
