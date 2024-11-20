import torch
import math
import numpy as np 
from torch.nn import init
import torch.nn as nn
from math import sqrt
from torch.nn.parameter import Parameter
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
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


def stack_hadamard_transforms(u, input_dim):
    _, output_dim = u.shape
    result = torch.zeros_like(u)
    
    for i in range(math.ceil(output_dim / input_dim)-1):
        start = i * input_dim
        end = (i+1) * input_dim
        subset = u[start : end]
        result[start : end] = hadamard_transform(subset)

    # Special case. Controls when output_dim is not divisible by input_dim
    start = math.ceil(output_dim / input_dim)-1
    end = -1
    subset = u[start : end]
    padding_size = max(0, input_dim - subset.size(-1))
    subset = nn.functional.pad(subset, (0, padding_size))
    result[start : end] = hadamard_transform(subset)[:(len(result)-start)]

    return result
        

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
    """
    def __init__(self, input_dim, output_dim, scale, learn_S=False, learn_G=False, learn_B=False, device=None):
        super(BIG_Fastfood_Layer, self).__init__()

        # Initialize parameters for Fastfood function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_S = learn_S
        self.learn_G = learn_G
        self.learn_B = learn_B
        self.device = device
        self.scale = scale
        self.P = None
        self.B = None 
        self.G = None
        self.S = None
        
        # Learnable Params
        if self.learn_G:
            self.G = Parameter(torch.Tensor(self.output_dim)) 
            init.normal_(self.G, std=sqrt(1./self.output_dim))
        if self.learn_B:
            self.B = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.B, std=sqrt(1./self.input_dim))
        if self.learn_S: 
            self.S = Parameter(torch.Tensor(self.output_dim)) 
            init.normal_(self.S, std=sqrt(1./self.output_dim))

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
        self.P = torch.randperm(
            self.output_dim, 
            device=device,
            requires_grad=False
        )

        if not self.learn_B:
            # Binary scaling matrix B sampled from {-1, 1}
            self.B = torch.tensor(
                np.random.choice([-1.0, 1.0], 
                    size=self.input_dim
                ),
                dtype=dtype, 
                device=device, 
                requires_grad=False
            )
        if not self.learn_G:
            # Gaussian scaling matrix G initialized to random values
            self.G = torch.zeros(
                self.output_dim, 
                dtype=dtype,
                device=device,
                requires_grad=False
            )
            self.G.normal_()

        if not self.learn_S: 
            # Scaling matrix S sampled from a chi-squared distribution
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.output_dim, 
                    size=self.output_dim
                ), 
                dtype=dtype,
                device=device,
                requires_grad=False
                ) / torch.norm(self.G)

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
        
        
        # pad x with 0's until it reaches output_dim length
        # padding_size = max(0, self.output_dim - x.size(-1))
        # x_padded = nn.functional.pad(x, (0, padding_size))

        # Number of times to repeat the Hadamard
        repetition = int(self.output_dim / self.input_dim)
        assert (repetition & (repetition -1 )) == 0, 'r must be a power of 2'

        # Reshape for Fastfood processing
        x_run = x.view(-1, self.input_dim)

        # Fastfood multiplication steps
        Bx = x_run * self.B                       # Apply binary scaling
        HBx = hadamard_transform(Bx)              # Hadamard transform
        HBx = HBx.repeat(1, repetition)           # Apply repetition
        PHBx = HBx[:, self.P]                     # Apply permutation
        GPHBx = PHBx * self.G                     # Apply Gaussian scaling
        HGPHBx = stack_hadamard_transforms(GPHBx, self.input_dim)        # Another Hadamard transform
        SHGPHBx = HGPHBx * self.S                 # Final scaling

        # Normalize and recover original shape
        Vx = ((1.0/(self.scale * sqrt(self.output_dim))) * SHGPHBx).view(-1, self.output_dim)

        return self.phi(Vx)
    

    def phi(self, x):
        """
        Apply nonlinearity to output.

        Arguments:
        ----------
            x (tensor): Input tensor that will be transformed.
        """
        # Create a uniform distribution between 0 and 2 * pi
        U = 2 * torch.pi * torch.rand(self.output_dim, device=self.device)

        # Apply the cosine function to x, adding U for randomness
        x = torch.cos(x + U)

        # Normalization
        return x * math.sqrt(2.0 / self.output_dim)
