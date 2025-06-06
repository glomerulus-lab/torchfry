import torch
import torch.nn as nn
from torch.nn import init
import scipy
from scipy.stats import chi
import numpy as np 
import math

def hadamard_transform_pytorch(u, normalize=False):
    """
    Multiply :math:`H_n @ u` where :math:`H_n` is the Hadamard matrix of dimension 
    :math:`n \\times n`. :math:`n` must be a power of 2.

    Parameters
    ----------
    u: torch.Tensor
        Has a shape :math:`(..., n)` where :math:`n` is a power of 2.
    normalize: bool
        If True, divide the result by :math:`2^{m/2}` where :math:`m = \\log_2(n)` to
        normalize the transform. If False, no normalization is applied.

    Returns
    -------
    product: torch.Tensor
        Returns the same tensor in memory with the Hadamard transform applied to it. 

    Notes
    -----
    This Hadamard function is adapted from the following file:
    `https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard.py`
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)

    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)

class hadamard_transform_matmul(nn.Module):
    """
    This Hadamard transformation explicitly stores the Hadamard matrix, then performs 
    matrix multiplication to complete the transformation.

    Parameters
    ----------
    input_dim : int  
        The dimension of the Hadamard matrix, matching the last dimension of the input 
        matrix.
    device : str  
        The device on which computations will be performed.  
    """
    def __init__(self, input_dim, device):
        self.matrix = nn.Parameter(torch.tensor(scipy.linalg.hadamard(input_dim), 
                                                device=device, dtype=torch.float), 
                                                requires_grad=False)

    def forward(self, x):
        return x @ self.matrix
    
class FastfoodLayer(nn.Module):
    """
    FastfoodLayer(input_dim, output_dim, scale=1, learn_S=False, learn_G=False, learn_B=False, device=None, nonlinearity=True, hadamard=None)

    Implementation of Fastfood transformation layer for efficient random feature mapping.

    This layer approximates a dense random projection using the Fastfood algorithm,
    which utilizes structured matrices (Hadamard, diagonal random, permutation matrices) 
    to reduce time complexity from Random Kitchen Sink's :math:`O(nd)` to 
    :math:`O(n \log d)` and space complexity from :math:`O(n^2)` to :math:`O(n)`, where 
    :math:`d` is the input_dim and :math:`n` is the output_dim.

    Parameters
    ----------
        input_dim: int 
            The input data feature dimension. (:math:`d`)
        output_dim: int
            The output dimension to be projected into. (:math:`n`)
        scale: float
            Scalar factor for normalization. (:math:`\sigma`)
        learn_S: bool
            If :math:`S` matrix is to be learnable.
        learn_G: bool
            If :math:`G` matrix is to be learnable.
        learn_B: bool
            If :math:`B` matrix is to be learnable.
        device: torch.device
            The device on which computations will be performed.
        nonlinearity: bool
            If True, apply nonlinearity of :math:`cos(Vx + u)`.
        hadamard: str
            Type of hadamard function desired: Dao, matrix multiplication, or Recursive 
            FWHT (``Dao``, ``Matmul``, ``PyTorch``).
    Notes
    -----
    .. math::

        Vx = \\frac{1}{\\sigma \\sqrt{d}} SHG \\Pi HB

    :math:`S`: Diagonal scaling matrix, allows our rows of :math:`V` to be independent of 
    one another. For Fastfood, this helps us match the radial shape from an RBF Kernel.

    :math:`H`: Hadamard function is a square symmetric matrix of :math:`1` and  :math:`-1`
    where each column is orthogonal. ``torchfry`` comes with three options for Hadamard:
    
        ``Matmul`` explicitly builds a Hadamard matrix of size :math:`d \\times d` and 
        stores it to memory, which will be used for matrix multiplication of the input
        against this matrix to achieve the Hadamard transform. This implementation is 
        extremely quick when leveraging strong GPUs, but requires storing a large matrix.

        ``Dao`` is adapted from https://github.com/Dao-AILab/fast-hadamard-transform, 
        which is written in CUDA and provides a PyTorch interface to leverage GPU power 
        even further.

        ``PyTorch`` repeatedly splits the input tensor into pairs, sums and subtracts 
        them, and concatenates the results step-by-step, applying the Hadamard transform 
        efficiently without explicitly creating the matrix.

    :math:`G`:
    Diagonal Gaussian matrix. Data sampled from a normal distribution with variance  
    proportional to the dimension of the input data.

    :math:`\Pi`:
    Applies a permutation to randomize the order of the rows. After the second  
    Hadamard is applied, the rows are independent of one another.

    :math:`B`:
    Diagonal binary matrix, drawn from a :math:`\{-1,+1\}`, helps input data become dense.

    When nonlinearity is used, the layer is computed as: 

    .. math::

        \cos(Vx + u)
            
    References
    ----------
    .. [1] Le, Q., Sarlós, T., & Smola, A. (2018). Fastfood: Approximate Kernel Expansions in Loglinear Time.
        https://arxiv.org/abs/1408.3060
    
    Examples
    --------
    A simple example of the Fastfood layer on a linear regression dataset with noise.

    >>> import torch
    >>> import torch.nn as nn
    >>> from torchfry.transforms import FastfoodLayer
    >>>
    >>> device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    >>>
    >>> # Linear regression with noise
    >>> x = torch.randn(128, 1, device=device)
    >>> y = 2*x + 3 + 0.1*torch.randn(128, 1, device=device)
    >>>
    >>> model = nn.Sequential(
    >>>     FastfoodLayer(1, 512, scale=1, learn_B=True, learn_G=True, learn_S=True, device=device, hadamard="Torch"),
    >>>     FastfoodLayer(512, 512, scale=1, learn_B=True, learn_G=True, learn_S=True, device=device, hadamard="Torch"),
    >>>     nn.Linear(512, 1)).to(device)
    >>>
    >>> criterion = nn.MSELoss()
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    >>>
    >>> # Training loop for 10 epochs
    >>> epochs = 10
    >>> for epoch in range(epochs):
    >>>     # model.train()
    >>>     optimizer.zero_grad()
    >>>     y_pred = model(x)
    >>>     loss = criterion(y_pred, y)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    Epoch [1/10], Loss: 14.2901
    Epoch [2/10], Loss: 14.2365
    Epoch [3/10], Loss: 14.2638
    Epoch [4/10], Loss: 14.2251
    Epoch [5/10], Loss: 14.2316
    Epoch [6/10], Loss: 14.0655
    Epoch [7/10], Loss: 14.0691
    Epoch [8/10], Loss: 14.0698
    Epoch [9/10], Loss: 14.0007
    Epoch [10/10], Loss: 13.9704
    """
    def __init__(self, input_dim, output_dim, scale=1, learn_S=False, learn_G=False, learn_B=False, device=None, nonlinearity=True, hadamard=None):
        super(FastfoodLayer, self).__init__()
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
        """
        Sample new permutation and scaling matrices for the Fastfood feature map.

        This function initializes the permutation matrix :math:`P`, the binary scaling 
        matrix :math:`B`, the Gaussian scaling matrix :math:`G`, and the scaling matrix 
        :math:`S` based on the learnable parameters.

        Parameters
        ----------
        dtype: torch.dtype
            Specifies the precision of the floats. 
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
        Applies the Fastfood transform to the input tensor following the Fastfood
        formula from the cited paper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        x: torch.Tensor
            Transformed tensor of shape (batch_size, output_dim) after projection, 
            optionally passed through a cosine-based nonlinearity if enabled.
        """
        # Reshape to [x, 1, input_dim]
        x = x.view(-1, 1, self.input_dim)
        
        # Apply binary scaling, broadcast over 2nd dim to [x, m, input_dim]
        Bx = x * self.B

        # Hadamard transform over last dim
        HBx = self._hadamard(Bx)

        # Add additional dim to Permute, and match size to HBx
        index = self.P.unsqueeze(0).expand(HBx.size(0), -1, -1)

        # Permute HBx using P on final dim of HBx
        PHBx = HBx.gather(-1, index)

        # Apply Gaussian scaling, element wise mult, no broadcast
        PHBx.mul_(self.G)

        # Hadamard transform over last dim
        HGPHBx = self._hadamard(PHBx)

        # Final scaling, element wise mult, no broadcast
        SHGPHBx = HGPHBx * self.S

        # Norm factor based on input_dim
        norm_factor = (1.0 / (self.scale*math.sqrt(self.input_dim)))

        # Norm factor applied, reshape into [x, m * input_dim]
        Vx = SHGPHBx.view(-1, self.m * self.input_dim).mul_(norm_factor)

        # Trim to exact [x, m * input_dim]
        result = Vx[..., :self.output_dim]

        # If desired, apply internal nonlinearity
        if self.nonlinearity:
            result = self.phi(result)
        return result
    

    def phi(self, x):
        """
        Apply random Fourier feature mapping using cosine transformation:

        .. math::

            \cos(Vx + u)

        This operation adds a random phase shift to the input tensor and applies
        a cosine nonlinearity, effectively projecting the data into a randomized
        feature space for kernel approximation.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor that will be transformed.
        
        Returns
        -------
        x: torch.Tensor
            Output tensor of the same shape after normalization.
        """
        # Create a uniform distribution between 0 and 2 * pi
        U = 2 * torch.pi * torch.rand(self.output_dim, device=self.device)

        # Add the uniform distribution to x
        # Out of place operation: x = x + u
        x.add_(U)

        # Apply the cosine function to x
        torch.cos_(x)

        # Normalization
        # Out of place: x = x * math.sqrt(2.0 / self.output_dim)
        x.mul_(math.sqrt(2.0 / self.output_dim))

        return x
