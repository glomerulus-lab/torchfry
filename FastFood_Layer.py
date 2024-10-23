import torch
import numpy as np 
from torch.nn import init
from torch.nn import Module
from math import sqrt, log
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


class FastFood_Layer(Module):
    """
    Random Fastfood features for the RBF kernel according to [1].

    [1]: "Fastfood - Approximating Kernel Expansions in Loglinear Time" 
    by Quoc Le, Tamas Sarlos and Alexander Smola.

    Arguments
    ---------
        input_dim: int 
            The input data feature dimension
        output_dim: int
            The ouput dimension to be projected into
    """
    def __init__(self, input_dim, output_dim, learn_S=False, learn_G_B=False, device=None):
        super(FastFood_Layer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_S = learn_S
        self.learn_G_B = learn_G_B
        self.device = device
        self.P = None
        self.B = None 
        self.G = None
        self.S = None
        
        # Learnable Params
        if self.learn_G_B:
            self.B = Parameter(torch.Tensor(self.input_dim)) 
            self.G = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.B, std=sqrt(1./self.input_dim))
            init.normal_(self.G, std=sqrt(1./self.input_dim))
        if self.learn_S: 
            self.S = Parameter(torch.Tensor(self.input_dim)) 
            init.normal_(self.S, std=sqrt(1./self.input_dim))

    def new_feature_map(self, dtype):
        # Device set 
        device = self.device
        # Permutation matrix P 
        self.P = torch.randperm(
            self.input_dim, 
            device=device 
        )

        if not self.learn_G_B:
            # Binary scaling matrix B 
            self.B = torch.tensor(
                np.random.choice([-1.0, 1.0], 
                    size=self.input_dim
                ),
                dtype=dtype, 
                device=device, 
                requires_grad=True
            )

            # Gaussian scaling matrix G 
            self.G = torch.zeros(
                self.input_dim, 
                dtype=dtype,
                device=device
            )
            self.G.normal_()

        if not self.learn_S: 
            # Scaling matrix S
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.input_dim, 
                    size=self.input_dim
                ), 
                dtype=dtype,
                device=device 
            ) / torch.norm(self.G)

    def forward(self, x):
        """
        Compute the FastFood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        """ 
        # Original shape
        x_shape = x.shape

        #number of times to run to build into higher dimension
        number_runs = self.output_dim // self.input_dim

        #store stacks
        stacked_results = []

        for _ in range(number_runs):
            #generate new matrices
            self.new_feature_map(float)
            # Reshape for Fastfood
            x_run = x.view(-1, self.input_dim)

            # Fastfood multiplication
            Bx = x_run * self.B
            HBx = hadamard_transform(Bx)
            PHBx = HBx[:, self.P]
            GPHBx = PHBx * self.G
            HGPHBx = hadamard_transform(GPHBx)
            SHGPHBx = HGPHBx * self.S

            # Normalize and recover original shape
            Vx = (sqrt(1.0/self.input_dim) * SHGPHBx).view(x_shape)

            # Collect the result for stacking
            stacked_results.append(Vx)
        
        # Combine all matrices
        stacked_output = torch.cat(stacked_results, dim=-1)
        
        # Cut off end to equal output dimension
        stacked_output = stacked_output[..., :self.output_dim]

        return stacked_output

