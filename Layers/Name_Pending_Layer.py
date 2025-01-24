import torch
import math
import time
import numpy as np 
from torch.nn import init
import torch.nn as nn
from math import sqrt
from scipy.stats import chi

def hadamard_transform(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)

    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

class BIG_Fastfood_Layer(nn.Module):
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
        learn_S: boolean
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
        super(BIG_Fastfood_Layer, self).__init__()

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
        
        # Learnable Params
        if self.learn_G:
            self.G = nn.Parameter(torch.empty(self.m, self.input_dim, device=device)) 
            init.normal_(self.G, std=sqrt(1/self.input_dim))
        if self.learn_B:
            self.B = nn.Parameter(torch.empty(self.m, self.input_dim, device=device)) 
            init.normal_(self.B, std=sqrt(1/self.input_dim))
        if self.learn_S: 
            self.S = nn.Parameter(torch.empty(self.m, self.input_dim, device=device)) 
            init.normal_(self.S, std=sqrt(1/self.input_dim))

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
        self.P = torch.zeros((self.m, self.input_dim), device=device, requires_grad=False, dtype=torch.int64)
        for i in range(self.m):
            self.P[i, :] = torch.randperm(self.input_dim, device=device)

        if not self.learn_B:
            # Binary scaling matrix B sampled from {-1, 1}
            self.B = torch.tensor(
                np.random.choice([-1.0, 1.0], 
                    size=(self.m, self.input_dim)
                ),
                dtype=dtype, 
                device=device, 
                requires_grad=False
            )
            
        if not self.learn_G:
            # Gaussian scaling matrix G initialized to random values
            self.G = torch.randn(
                (self.m, self.input_dim), 
                dtype=dtype,
                device=device,
                requires_grad=False
            )

        if not self.learn_S: 
            # Scaling matrix S sampled from a chi-squared distribution
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.input_dim, 
                    size=(self.m, self.input_dim)
                ), 
                dtype=dtype,
                device=device,
                requires_grad=False
                )
            
            # Norm each row of S, with norm of corresponding row of G
            row_norms = torch.norm(self.G, dim=1, keepdim=True)
            self.S /= row_norms
            

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
        x_run = x.view(-1, 1, self.input_dim)                            # Reshape to [x, 1, input_dim]
        Bx = x_run * self.B                                              # Apply binary scaling, broadcast over 2nd dim to [x, m, input_dim]
        HBx = hadamard_transform(Bx)                                     # Hadamard transform over last dim
        index = self.P.unsqueeze(0).expand(HBx.size(0), -1, -1)          # Add additional dim to Permute, and match size to HBx
        PHBx = HBx.gather(-1, index)                                     # Permute HBx using P on final dim of HBx
        GPHBx = PHBx * self.G                                            # Apply Gaussian scaling, element wise mult, no broadcast
        HGPHBx = hadamard_transform(GPHBx)                               # Hadamard transform over last dim
        SHGPHBx = HGPHBx * self.S                                        # Final scaling, element wise mult, no broadcast
        norm_factor = (1.0 / (self.scale * sqrt(self.input_dim)))        # Norm factor based on input_dim
        Vx = (norm_factor * SHGPHBx.view(-1, self.m * self.input_dim))   # Norm factor applied, reshape into [x, m * input_dim]
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