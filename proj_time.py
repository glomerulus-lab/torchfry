import numpy as np
import sklearn as sc
from FastFood.random_projections import projections
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from OnLearningTheKernel.fast_transformers.feature_maps.fastfood import FastFoodRandomFeatures
import time
import torch

#work around for a -
import sys
import os

# Add the path to sys.path
sys.path.append(os.path.abspath(r"C:\research\projections\structured-nets"))

# Import the module with hyphen
structure = __import__('pytorch.structure', fromlist=['fastfood'])

# Access the fastfood function
fastfood = structure.fastfood



#dimension
d = 256
#data points
num_data = 20000 
x = np.random.rand(num_data, d)

#universal scale
scale = 20

#new n dimension for rks and ff
dimensions = [1024,2048,4096,8192]

#rks error
rks_time = []
for dim in dimensions:
    #rks approx
    rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
    rks.fit(x)
    start = time.time()
    rks.transform(x)

    end = time.time()
    rks_time.append(end-start)


rks_mine_time = []
#rks error
for dim in dimensions:
    #rks approx
    rks = projections.rks(dim, scale)
    rks.fit(x)
    start = time.time()
    rks.transform(x)

    end = time.time()
    rks_mine_time.append(end-start)

#ff error
ff_time = []
for dim in dimensions:
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
    ff_time.append(end-start)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ff_2_time = []
for n in dimensions:
    S = np.random.randn(n)
    G = np.random.randn(n)
    B = np.random.randn(n)
    P = np.random.permutation(n)
    x = np.random.randn(20000,n)

    S = torch.tensor(S, dtype=torch.float, device=device)
    G = torch.tensor(G, dtype=torch.float, device=device)
    B = torch.tensor(B, dtype=torch.float, device=device)
    P = torch.tensor(P, dtype=torch.long, device=device)
    x = torch.tensor(x, dtype=torch.float, device=device)

    start = time.time()
    fastfood.fastfood_multiply(S,G,B,P,x)

    end = time.time()
    ff_2_time.append(end-start)

ff_3_time=[]
x = torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x
for n in dimensions:
    fast_food_obj = FastFoodRandomFeatures(n)
    fast_food_obj.new_feature_map(device, torch.float32)
    start = time.time()
    fast_food_obj.forward(x)
    end = time.time()
    ff_3_time.append(end-start)


plt.plot(dimensions,rks_time, label='RKS_Time', marker='o')
plt.plot(dimensions,rks_mine_time, label='RKS_Personal_Time', marker='o')
plt.plot(dimensions,ff_time, label='FF_built-in_Time', marker='o')
plt.plot(dimensions,ff_2_time, label='FF_structured-nets_Time', marker='o')
plt.plot(dimensions,ff_3_time, label='FF_OnLearning_Time', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Time (s)')
plt.title('Projection Times')
plt.legend()
plt.show()
