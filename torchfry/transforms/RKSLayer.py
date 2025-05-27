import torch
import math
import torch.nn as nn

class RKSLayer(nn.Module):
    """
    This layer explicitly builds a dense matrix of random Gaussian noise. 

    Parameters
    ----------
        input_dim: int 
            The input dimension of the features.
        output_dim: int 
            The desired output dimension of the layer.
        scale: float 
            Scalar factor for normalization
        learn_G: bool 
            If True, allows the Random Gaussian Matrix G to be learnable.
        device: torch.device
            The device on which to allocate the parameters.
    
    References
    ----------
    .. [1] Le, Q., Sarlós, T., & Smola, A. (2018). Fastfood: Approximate Kernel Expansions in Loglinear Time.
        https://arxiv.org/pdf/1408.3060
    
    Examples
    --------
    A simple example of the Random Kitchen Sink Layer on a linear regression dataset with noise.

    >>> import torch
    >>> import torch.nn as nn
    >>> from fastfood_torch.transforms import RKSLayer
    >>>
    >>> device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    >>>
    >>> # Linear regression with noise
    >>> x = torch.randn(128, 1, device=device)
    >>> y = 2*x + 3 + 0.1*torch.randn(128, 1, device=device)
    >>>
    >>> model = nn.Sequential(
    >>>     RKSLayer(1, 512, scale=1, learn_G=True, device=device),
    >>>     RKSLayer(512, 512, scale=1, learn_G=True, device=device),
    >>>     nn.Linear(512, 1)
    >>> ).to(device)
    >>>
    >>> criterion = nn.MSELoss()
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    >>>
    >>> # Training loop for 100 epochs
    >>> epochs = 100
    >>> for epoch in range(epochs):
    >>>     # model.train()
    >>>     optimizer.zero_grad()
    >>>     y_pred = model(x)
    >>>     loss = criterion(y_pred, y)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     
    >>>     if (epoch + 1) % 20 == 0:  # Print every 20 epochs
    >>>         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    """
    def __init__(self, input_dim, output_dim, scale, learn_G=False, device=None, nonlinearity=True):
        super(RKSLayer, self).__init__()

        self.input_dim = input_dim       # Data input dimension
        self.scale = scale               # Non linearity requirement
        self.learn_G = learn_G           # Param for learning weight
        self.output_dim = output_dim     # Projection dimension
        self.device = device             # GPU or CPU   
        self.nonlinearity = nonlinearity # Desired internal nonlinearity

        # If Gaussian is learnable
        if self.learn_G:
            # Make G learnable with normal initialization
            self.G = nn.Parameter((1 / self.scale) * torch.randn(input_dim, output_dim, device=self.device), requires_grad=True)
        else:
            # Use a fixed random Gaussian matrix for G
            self.G = nn.Parameter((1 / self.scale) * torch.randn(input_dim, output_dim, device=self.device), requires_grad=False)

    def phi(self, x):
        """
        Apply random Fourier feature mapping using cosine transformation.

        This operation adds a random phase shift to the input tensor and applies
        a cosine nonlinearity, effectively projecting the data into a randomized
        feature space for kernel approximation.

        Parameters
        ----------
        x: torch.tensor
            Input tensor that will be transformed.
        
        Returns
        -------
        x: torch.tensor
            Output tensor of the same shape after normalization.
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

    

    def forward(self, x):
        """
        The forward function for the Random Kitchen Sink function.
        It is a matrix multiplication with the G matrix. Then an optional
        phi nonlinearity is applied. 

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, L, H, D), where N is the batch size 
            and D is the input feature dimension.

        Returns
        -------
        X: torch.Tensor
            Transformed tensor of shape (N, L, H, output_dim) after projection 
            and optional nonlinearity.
        """

        # Project and add offset
        result = x @ self.G

        if self.nonlinearity:            # If desired internal nonlinearity
            result = self.phi(result)    # Nonlinearity
        return result
