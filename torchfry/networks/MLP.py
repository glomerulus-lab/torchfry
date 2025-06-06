import torch.nn as nn
from torchfry.transforms import FastfoodLayer

class MLP(nn.Module):
    """
    MLP(features, classes, widths, layer=FastfoodLayer, proj_args={})

    Multi-Layer Perceptron-based model (MLP) that replaces the stacked FC layers with
    random feature layers (e.g., FastfoodLayer, RKSLayer), each followed by batch 
    normalization and ReLU activation, ending with a FC linear layer for classification. 
    
    Network architecture is as follows::

        (Fastfood/RKS -> BN -> ReLU) * n -> FC Linear (Output)

    Where ``n`` is the desired number of stacked projection layers.

    Parameters
    ----------
    features : int
        Number of features for the projection layer.
    classes : int
        Number of output classes for classification.
    widths : list of int
        List containing the widths (number of neurons) for each hidden layer.
    projection_layer : nn.Module
        The type of projection layer to use in hidden layers.
    proj_args : dict
        Additional arguments to pass to the projection layer (e.g., scale,
        device, learnable flags, etc.).
    
    Notes
    -----
    This model is primarily run on the Fashion MNIST dataset, but supports CIFAR-10 as
    well.
    """
    def __init__(self, features, classes, widths, layer=FastfoodLayer, proj_args={}):
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
