import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from FastFood_Layer import FastFood_Layer
from RKS_Layer import RKS_Layer
import torch
from BIG_FastFood_Layer import BIG_Fastfood_Layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dimension
d = 16
#data points
num_data = 2000 
x = np.random.rand(num_data, d)

#universal scale
scale = 0.5

#exact rbf kernel
exact_rbf = rbf_kernel(x, gamma=(1/(2*scale**2)))
#new n dimension for rks and ff
dimensions = [16,32,64,128,256,512,1024,2048,4096,8192]

#rks error
rks_error = []
for dim in dimensions:
    #rks approx
    rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
    rks.fit(x)
    rks_approx = rks.transform(x)

    rks_approx = rks_approx @ rks_approx.T
    
    difference = np.linalg.norm(exact_rbf-rks_approx, 'fro')
    rks_error.append(difference/num_data)


x = torch.tensor(x, dtype=torch.float32, device=device)

#ff error
ff_error = []
for dim in dimensions:
    fast_food_obj = FastFood_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
    phi = fast_food_obj.forward(x)

    ff_approx = (phi @ phi.T).cpu().detach().numpy()
    difference = np.linalg.norm(exact_rbf-ff_approx, 'fro')
    ff_error.append(difference/num_data)

#ff error
rks_2_error = []
for dim in dimensions:
    rks_obj = RKS_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
    phi = rks_obj.forward(x)

    rks_approx = (phi @ phi.T).cpu().detach().numpy()
    difference = np.linalg.norm(exact_rbf-rks_approx, 'fro')
    rks_2_error.append(difference/num_data)

#ff error
ff_2_error = []
for dim in dimensions:
    fast_food_obj = BIG_Fastfood_Layer(input_dim=x.shape[1], output_dim=dim, scale=scale, device=device)
    phi = fast_food_obj.forward(x)

    ff_approx = (phi @ phi.T).cpu().detach().numpy()
    difference = np.linalg.norm(exact_rbf-ff_approx, 'fro')
    ff_2_error.append(difference/num_data)


plt.plot(dimensions,rks_error, label='RKS_Approx', marker='o')
plt.plot(dimensions,ff_error, label='FF_Approx', marker='o')
plt.plot(dimensions,ff_2_error, label='FF_Big_PP_Approx', marker='o')
plt.plot(dimensions,rks_2_error, label='RKS_Layer_Approx', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Error')
plt.title('Approximation Error vs. RBF Kernel')
plt.legend()
plt.show()

plt.loglog(dimensions,rks_error, label='RKS_Approx', marker='o')
plt.loglog(dimensions,ff_error, label='FF_Approx', marker='o')
plt.plot(dimensions,rks_2_error, label='RKS_Layer_Approx', marker='o')
plt.plot(dimensions, ff_2_error, label='FF_Big_PP_Approx', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Error')
plt.title('Approximation Error vs. RBF Kernel')
plt.legend()
plt.show()
