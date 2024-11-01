import numpy as np
import sklearn as sc
from FastFood_BaseLine.random_projections import projections
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood
from FastFood_BaseLine.structured_nets.pytorch.structure import fastfood
from FastFood_Layer import Fastfood_Layer
import time
import torch
import math


#dimension
d = 256
#data points
num_data = 10_000 
x = np.random.rand(num_data, d)

#universal scale
scale = 20

#new n dimension for rks and ff
#first two dimensions are for warm-up
dimensions = [256,256,1024,2048,4096,8192,16384]

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
rks_time = rks_time[2:]


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
rks_mine_time = rks_mine_time[2:]

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
ff_time = ff_time[2:]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.tensor(x, dtype=torch.float32, device=device)

ff_2_time = []
for n in dimensions:
    nd_time = 0
    for _ in range(math.ceil(n/x.shape[1])):
        S = np.random.randn(x.shape[1])
        G = np.random.randn(x.shape[1])
        B = np.random.randn(x.shape[1])
        P = np.random.permutation(x.shape[1])    
        S = torch.tensor(S, dtype=torch.float, device=device)
        G = torch.tensor(G, dtype=torch.float, device=device)
        B = torch.tensor(B, dtype=torch.float, device=device)
        P = torch.tensor(P, dtype=torch.long, device=device)
        start = time.time()
        phi = fastfood.fastfood_multiply(S,G,B,P,x)
        end = time.time()
        nd_time += (end-start)
    ff_2_time.append(nd_time)
ff_2_time = ff_2_time[2:]


ff_3_time=[]
for n in dimensions:
    fast_food_obj = Fastfood_Layer(input_dim=x.shape[1], output_dim=n, scale=scale, device=device)

    start = time.time()
    phi = fast_food_obj.forward(x)
    end = time.time()

    ff_3_time.append(end-start)
ff_3_time = ff_3_time[2:]

#set dimensions avoid graphing the warm-up passes
dimensions = dimensions[2:]

plt.plot(dimensions,rks_time, label='RKS_Time', marker='o')
plt.plot(dimensions,rks_mine_time, label='RKS_Personal_Time', marker='o')
plt.plot(dimensions,ff_time, label='FF_built-in_Time', marker='o')
plt.plot(dimensions,ff_2_time, label='FF_structured-nets_Time (GPU)', marker='o')
plt.plot(dimensions,ff_3_time, label='FastFood_Layer_Time (GPU)', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.title('Projection Times')
plt.legend(loc='best')
plt.show()
