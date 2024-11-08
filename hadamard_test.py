
from FastFood_BaseLine.structured_nets.pytorch.structure.hadamard import hadamard_transform_cuda
from FastFood_Layer import hadamard_transform
import time
import numpy as np
import torch

#dimension
d = 256
#data points
num_data = 10_000 
x = np.random.rand(num_data, d)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.tensor(x, dtype=torch.float32, device=device)

start = time.time()
hadamard_transform_cuda(x)
end = time.time()
print(end-start)

start = time.time()
hadamard_transform(x)
end = time.time()
print(end-start)