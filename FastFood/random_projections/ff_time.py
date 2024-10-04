import numpy as np
import sklearn as sc
from sklearn.preprocessing import StandardScaler
from sklearn_extra.kernel_approximation import Fastfood
import projections
import time
import torch

#dimension
d = 256
#data points
num_data = 20000 
x = np.random.rand(num_data, d)

#universal scale
scale = 20


trade = 'accuracy'
#ff approx
ff = Fastfood(sigma=scale, n_components=8192, tradeoff_mem_accuracy=trade)
ff.fit(x)
ff.transform(x)

s1 = time.time()
X_padded = ff._pad_with_zeros(x)

s2 = time.time()
HGPHBX = ff._apply_approximate_gaussian_matrix(
    ff._B, ff._G, ff._P, X_padded
)

s3 = time.time()
VX = ff._scale_transformed_data(ff._S, HGPHBX)
VX = torch.tensor(VX)

#memory
if trade == 'mem':
    s4 = time.time()
    VX = torch.cos(VX + ff._U)
    output = VX * np.sqrt(2.0 / VX.shape[1])

#accuracy: non non linearity
if trade == 'accuracy':
    s4 = time.time()
    (1 / np.sqrt(VX.shape[1])) * torch.hstack([torch.cos(VX), torch.sin(VX)])

