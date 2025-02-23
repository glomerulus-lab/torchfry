import torch
import math
import time
import numpy as np 
from torch.nn import init
import torch.nn as nn
from math import sqrt
from scipy.stats import chi
import importlib
from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform

class FastFood_Layer(nn.Module):
    """
    Random Fastfood features for the RBF kernel according to [1].

    [1]: "Fastfood - Approximating Kernel Expansions in Loglinear Time" 
    by Quoc Le, Tamas Sarlos and Alexander Smola.

    Arguments
    ---------
        input_dim: int 
            The input data feature dimension.
        output_dim: int
            The output dimension to be projected into.
        scale: float
            Scale factor for normalization
        learn_S: boolean    return np.concatenate([x_even + x_odd, x_even - x_odd])

            If S matrix is to be learnable
        learn_G: boolean
            If G matrix is to be learnable
        learn_B: boolean
            If B matrix is to be learnable
        device: string
            Device for operations
        nonlinearity: boolean
            If internal nonlinearity is used, or defered
    """
    def __init__(self, input_dim, output_dim, scale, learn_S=False, learn_G=False, learn_B=False, device=None, nonlinearity=True):
        super(FastFood_Layer, self).__init__()

        # Initialize parameters for Fastfood function
        self.m = math.ceil(output_dim / input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_S = learn_S
        self.learn_G = learn_G
        self.learn_B = learn_B
        self.device = device
        self.scale = scale
        self.nonlinearity = nonlinearity
        self.P = None
        self.B = None 
        self.G = None
        self.S = None

        # Sample required matrices
        self.new_feature_map(torch.float32)
        
    def new_feature_map(self, dtype):
        """Sample new permutation and scaling matrices for the Fastfood feature map.

        This function initializes the permutation matrix P, the binary scaling 
        matrix B, the Gaussian scaling matrix G, and the scaling matrix S based 
        on the learnable parameters.

        Arguments:
        ----------
        dtype (torch.dtype): The data type for the matrices.
        """
        # Device set 
        device = self.device
        
        # Permutation matrix P
        self.P = torch.stack([torch.randperm(self.input_dim, device=device) for _ in range(self.m)])

        # Learnable B Matrix
        if self.learn_B:
            self.B = nn.Parameter(torch.randn(self.m, self.input_dim, device=device))
        # Non Learnable B
        else:
            self.B = nn.Parameter(torch.tensor(np.random.choice([-1, 1], size=(self.m, self.input_dim)), dtype=dtype, device=device), requires_grad=False)

        # Learnable G Matrix
        if self.learn_G:
            self.G = nn.Parameter(torch.randn(self.m, self.input_dim, device=device))
        # Non Learnable G
        else:
            self.G = nn.Parameter(torch.randn(self.m, self.input_dim, dtype=dtype, device=device), requires_grad=False)

        # Learnable S Matrix
        if self.learn_S:
            self.S = nn.Parameter(torch.empty(self.m, self.input_dim, device=device))
            init.normal_(self.S, mean=sqrt(self.input_dim), std=sqrt(self.input_dim))
        # Non Learnable S
        else:
            self.S = nn.Parameter(torch.tensor(chi.rvs(df=self.input_dim, size=(self.m, self.input_dim)), dtype=dtype, device=device), requires_grad=False)
            # Normalize S rows by corresponding G rows
            row_norms = torch.norm(self.G, dim=1, keepdim=True)
            with torch.no_grad():  
                self.S.div_(row_norms)

    def forward(self, x):
        """
        Compute the Fastfood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        
        Returns:
        -------
        Tensor: The transformed tensor after applying the Fastfood feature map.
        """
        x = x.view(-1, 1, self.input_dim)                                # Reshape to [x, 1, input_dim]
        Bx = x * self.B                                                  # Apply binary scaling, broadcast over 2nd dim to [x, m, input_dim]
        HBx = hadamard_transform(Bx)                                     # Hadamard transform over last dim
        index = self.P.unsqueeze(0).expand(HBx.size(0), -1, -1)          # Add additional dim to Permute, and match size to HBx
        PHBx = HBx.gather(-1, index)                                     # Permute HBx using P on final dim of HBx
        PHBx.mul_(self.G)                                                # Apply Gaussian scaling, element wise mult, no broadcast
        HGPHBx = hadamard_transform(PHBx)                                # Hadamard transform over last dim
        HGPHBx.mul_(self.S)                                              # Final scaling, element wise mult, no broadcast
        norm_factor = (1.0 / (self.scale*sqrt(self.input_dim)))          # Norm factor based on input_dim
        Vx = HGPHBx.view(-1, self.m * self.input_dim).mul_(norm_factor)  # Norm factor applied, reshape into [x, m * input_dim]
        result = Vx[..., :self.output_dim]                               # Trim to exact [x, m * input_dim]
        if self.nonlinearity:                                            # If desired internal nonlinearity
            result = self.phi(result)                                    # Nonlinearity
        return result
    

    def phi(self, x):
        """
        Apply nonlinearity to output.

        Arguments:
        ----------
            x (tensor): Input tensor that will be transformed.
        """

        # Create a uniform distribution between 0 and 2 * pi
        U = 2 * torch.pi * torch.rand(self.output_dim, device=self.device)

        # Add the uniform distribution to x
        # Out of place: x = x + u
        x.add_(U)

        # Apply the cosine function to x, adding U for randomness
        torch.cos_(x)

        # Normalization
        # Out of place: x = x * math.sqrt(2.0 / self.output_dim)
        x.mul_(math.sqrt(2.0 / self.output_dim))

        return x