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


class Fastfood_Stack_Object(nn.Module):
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
    def __init__(self, input_dim, scale, learn_S=False, learn_G=False, learn_B=False, device=None):
        super(Fastfood_Stack_Object, self).__init__()

        # Initialize parameters for Fastfood function
        self.input_dim = input_dim
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
            self.G = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.G, std=sqrt(1./self.input_dim))
        if self.learn_B:
            self.B = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.B, std=sqrt(1./self.input_dim))
        if self.learn_S: 
            self.S = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.S, std=sqrt(1./self.input_dim))

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
            self.input_dim, 
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
                self.input_dim, 
                dtype=dtype,
                device=device,
                requires_grad=False
            )
            self.G.normal_()

        if not self.learn_S: 
            # Scaling matrix S sampled from a chi-squared distribution
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.input_dim, 
                    size=self.input_dim
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
        # Original shape
        x_shape = x.shape
        
        # Reshape for Fastfood processing
        x_run = x.view(-1, self.input_dim)

        # Fastfood multiplication steps
        Bx = x_run * self.B                       # Apply binary scaling
        HBx = hadamard_transform(Bx)              # Hadamard transform
        PHBx = HBx[:, self.P]                     # Apply permutation
        GPHBx = PHBx * self.G                     # Apply Gaussian scaling
        HGPHBx = hadamard_transform(GPHBx)        # Another Hadamard transform
        SHGPHBx = HGPHBx * self.S                 # Final scaling

        # Normalize and recover original shape
        Vx = ((1.0/(self.scale * sqrt(self.input_dim))) * SHGPHBx).view(x_shape)

        return Vx

class FastFood_Layer(nn.Module):
    """
    Layer that stacks multiple Fastfood transformations to project input 
    features into a higher dimensional space.

    Arguments:
    ----------
        input_dim (int): The input dimension of the features.
        output_dim (int): The desired output dimension of the layer.
        scale (float): A scaling factor for the output.
        learn_S (bool): If True, allows the scaling matrix S to be learnable.
        learn_G (bool): If True, allows the Gaussian scaling matrices to be learnable.
        learn_B (bool): If True, allows the binary scaling matrices to be learnable.
        device (torch.device, optional): The device on which to allocate the parameters.
    """
    def __init__(self, input_dim, output_dim, scale, learn_S=False, learn_G=False, learn_B=False, device=None):
        super(FastFood_Layer, self).__init__()

        # Create a list of Fastfood stack objects to reach the desired output dimension
        self.stack = nn.ModuleList(
            [Fastfood_Stack_Object(input_dim=input_dim, scale=scale, learn_S=learn_S, learn_G=learn_G, learn_B=learn_B, device=device)
             for _ in range(math.ceil(output_dim / input_dim))]
        )
        self.input_dim = input_dim    # Store input dimension
        self.output_dim = output_dim  # Store the desired output dimension
        self.device = device          # Device to store on
            
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

    def forward(self, x):
        """
        Forward pass through the Fastfood layer.

        This method applies all stacked Fastfood transformations to the input tensor 
        and concatenates the outputs.

        Arguments:
        ----------
        x (Tensor): Input tensor of shape (N, L, H, D).

        Returns:
        -------
        Tensor: The concatenated output tensor of shape (N, L, H, output_dim).
        """
        # Call forward for all objects
        stacked_output = [l(x) for l in self.stack]

        # Concatenate results along the last dimension
        stacked_output = torch.cat(stacked_output, dim=-1)

        # Trim the output to the desired output dimension
        stacked_output = stacked_output[..., :self.output_dim]

        return self.phi(stacked_output)
        
