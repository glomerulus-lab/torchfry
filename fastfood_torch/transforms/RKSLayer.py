import torch
import math
import torch.nn as nn

class RKSLayer(nn.Module):
    """
    Layer that stacks multiple Fastfood transformations to project input 
    features into a higher dimensional space.

    Arguments:
    ----------
        input_dim (int): The input dimension of the features.
        output_dim (int): The desired output dimension of the layer.
        scale (float): A scaling factor for the output.
        learn_G (bool): If True, allows the Random Gaussian Matrix G to be learnable.
        device (torch.device, optional): The device on which to allocate the parameters.
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

        # Project and add offset
        result = x @ self.G

        if self.nonlinearity:            # If desired internal nonlinearity
            result = self.phi(result)    # Nonlinearity
        return result
