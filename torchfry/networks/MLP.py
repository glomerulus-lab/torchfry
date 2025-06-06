import torch.nn as nn
from torchfry.transforms import FastfoodLayer

class MLP(nn.Module):
    """
    Multi-Layer Perceptron-based model (MLP) that replaces the stacked FC layers with
    random feature layers (e.g., FastfoodLayer, RKSLayer), each followed by batch 
    normalization and ReLU activation, ending with a FC linear layer for classification. 
    
    Network architecture is as follows::

        (Fastfood/RKS -> BN -> ReLU) * n -> FC Linear (Output)

    Where ``n`` is the desired number of stacked projection layers.

    Parameters
    ----------
    features : int
        The dimension of the input features.
    classes : int
        Number of output classes for classification.
    widths : list of int
        List containing the widths (number of neurons) for each hidden layer.
    proj_args : dict
        Additional keyword arguments to pass to the projection layers.
    projection_layer : nn.Module class, (default=FastfoodLayer)
        The type of projection layer to use in hidden layers.
    
    Notes
    -----
    This model is primarily run on the Fashion MNIST dataset, but supports CIFAR-10 as
    well.
    """
    def __init__(self, features, classes, widths, proj_args, layer=FastfoodLayer):
        super().__init__()
        # Sequential Placeholder
        layers = []
        for width in widths:
            layers.append(layer(input_dim=features, output_dim=width, **proj_args))
            layers.append(nn.BatchNorm1d(width, affine=False))
            layers.append(nn.ReLU())
            features = width

        # Final output layer, learnable
        layers.append(nn.Linear(features, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, features).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (batch_size, classes), representing raw 
            classification scores.
        """
        return self.network(x)
