import torch
import torch.nn as nn
from torch.nn import init
import scipy
from scipy.stats import chi
import numpy as np 
import math

def hadamard_transform_pytorch(u, normalize=False):
    """
    Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.

    Parameters
    ----------
    u: Tensor of shape (..., n)
    normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    
    Returns
    -------
    product: Tensor of shape (..., n)

    Notes
    -----
    This Hadamard function is taken from the following:
    `HazyResearch/structured-nets/pytorch/structure/hadamard.py`
    `cs1160701/OnLearningTheKernel/fast_transformers/feature_maps/fastfood.py` 
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)

    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

class hadamard_transform_matmul:
    """
    Hadamard Transform using explicit Hadamard matrix instantiation and matrix multiplication.

    Parameters
    ----------
    input_dim : int  
        The dimension of the Hadamard matrix, matching the last dimension of the input matrix.  
    device : str  
        The device on which computations will be performed.  
    """
    def __init__(self, input_dim, device):
        self.matrix = nn.Parameter(torch.tensor(scipy.linalg.hadamard(input_dim), device=device, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        return x @ self.matrix
    
class FastFoodLayer(nn.Module):
    """
    Implementation of Fastfood transformation layer for efficient random feature mapping.

    This layer approximates a dense random projection using the Fastfood algorithm,
    which utilizes structured matrices (Hadamard, diagonal random, permutation matrices) 
    to reduce time complexity from Random Kitchen Sink's O(nd) to O(n log d) and space 
    complexity from O(n^2) to O(n), where d is the input_dim and n is the output_dim.

    Parameters
    ----------
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
        learn_B: booleanself.hadamard_matrix = torch.tensor(scipy.linalg.hadamard(input_dim), device=device, dtype=torch.int)
            If B matrix is to be learnable
        device: string
            The device on which computations will be performed
        nonlinearity: boolean
            If internal nonlinearity is used, or defered
        hadamard: string
            Type of hadamard function desired, Dao, Recursive FWHT, or matrix mul. ("Dao", "Matmul", "Torch")

    Notes
    -----
    See "Fastfood | Approximating Kernel Expansions in Loglinear Time" by
    Quoc Le, Tamás Sarlós and Alex Smola.
    """

    def __init__(self, input_dim, output_dim, scale=1, learn_S=False, learn_G=False, learn_B=False, device=None, nonlinearity=True, hadamard=None):
        super(FastFoodLayer, self).__init__()
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

        # Dependancy of Dao-AILab fast-hadamard-transform
        if hadamard == "Dao":
            try:
                from fast_hadamard_transform import hadamard_transform
                self._hadamard = hadamard_transform
            except ImportError:
                print("Dao Hadamard not available, falling back to PyTorch hadamard function")
                self._hadamard = hadamard_transform_pytorch

        # Implicit hadamard matrix instantiation
        elif hadamard == "Matmul":
            matmul_hadamard = hadamard_transform_matmul(input_dim=self.input_dim, device=device)
            self._hadamard = matmul_hadamard.forward

        # Fallback, PyTorch Hadamard function    
        else:
            self._hadamard = hadamard_transform_pytorch

        # Sample required matrices
        self.new_feature_map(torch.float32)
        
    def new_feature_map(self, dtype):
        """Sample new permutation and scaling matrices for the Fastfood feature map.

        This function initializes the permutation matrix P, the binary scaling 
        matrix B, the Gaussian scaling matrix G, and the scaling matrix S based 
        on the learnable parameters.

        Parameters
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
            init.normal_(self.S, mean=math.sqrt(self.input_dim), std=math.sqrt(self.input_dim))
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

        Parameters
        ----------
        x : (N, L, H, D)
            The input tensor.
        
        Returns
        -------
        Tensor: The transformed tensor after applying the Fastfood feature map.
        """
        x = x.view(-1, 1, self.input_dim)                                # Reshape to [x, 1, input_dim]
        Bx = x * self.B                                                  # Apply binary scaling, broadcast over 2nd dim to [x, m, input_dim]
        HBx = self._hadamard(Bx)                                         # Hadamard transform over last dim
        index = self.P.unsqueeze(0).expand(HBx.size(0), -1, -1)          # Add additional dim to Permute, and match size to HBx
        PHBx = HBx.gather(-1, index)                                     # Permute HBx using P on final dim of HBx
        PHBx.mul_(self.G)                                                # Apply Gaussian scaling, element wise mult, no broadcast
        HGPHBx = self._hadamard(PHBx)                                    # Hadamard transform over last dim
        SHGPHBx = HGPHBx * self.S                                        # Final scaling, element wise mult, no broadcast
        norm_factor = (1.0 / (self.scale*math.sqrt(self.input_dim)))          # Norm factor based on input_dim
        Vx = SHGPHBx.view(-1, self.m * self.input_dim).mul_(norm_factor) # Norm factor applied, reshape into [x, m * input_dim]
        result = Vx[..., :self.output_dim]                               # Trim to exact [x, m * input_dim]
        if self.nonlinearity:                                            # If desired internal nonlinearity
            result = self.phi(result)                                    # Nonlinearity
        return result
    

    def phi(self, x):
        """
        Apply nonlinearity to output.

        Parameters
        ----------
            x (tensor): Input tensor that will be transformed.
        
        Returns
        -------
        """

        # Create a uniform distribution between 0 and 2 * pi
        U = 2 * torch.pi * torch.rand(self.output_dim, device=self.device)

        # Add the uniform distribution to x
        x.add_(U)

        # Apply the cosine function to x
        torch.cos_(x)

        # Normalization
        x.mul_(math.sqrt(2.0 / self.output_dim))

        return x