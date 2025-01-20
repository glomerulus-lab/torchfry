import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# Methods to compare
from RKS_Layer import RKS_Layer
from BIG_FastFood_Layer import BIG_Fastfood_Layer
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.kernel_approximation import RBFSampler


def exact_rbf_sampler(x, exact, output_dims):
    # Compute the exact rbf kernel. This method is our baseline.
    times = []
    error = []
    for dim in output_dims:
        
        rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
        
        # measure time
        start = time.time()
        rks_approx = rks.fit_transform(x)
        end = time.time()
        times.append(end-start)

        # compute error
        rks_approx = rks_approx @ rks_approx.T
        difference = np.linalg.norm(exact - rks_approx, 'fro')
        error.append(difference / x.shape[0])
        
    return times, error

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
    
    return times, error

def RKS_GPU_layer(x, exact, output_dims):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    times = []
    error = []
    for dim in output_dims:
        rks = RKS_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)

        # measure time
        start = time.time()
        phi = rks.forward(x)
        end = time.time()
        times.append(end-start)

        # compute error
        rks_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact - rks_approx, 'fro')
        error.append(difference / x.shape[0])
        
    return times, error

def BIG_ff_layer(x, exact, output_dims):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)

    times = []
    error = []
    for dim in output_dims:
        fast_food_obj = BIG_Fastfood_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
        
        # mesure time
        start = time.time()
        phi = fast_food_obj.forward(x)
        end = time.time()
        times.append(end-start)

        # compute error
        ff_approx = (phi @ phi.T).cpu().detach().numpy()
        difference = np.linalg.norm(exact - ff_approx, 'fro')
        error.append(difference / x.shape[0])

    return times, error