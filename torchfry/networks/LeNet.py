import torch.nn as nn
import torch.nn.functional as F
from torchfry.transforms import FastfoodLayer

class LeNet(nn.Module):
    """
    LeNet-based model similar to Deep Fried Convnets. This model replaces the traditional 
    FC layer with a random feature layer (e.g., FastfoodLayer, RKSLayer), followed by 
    batch normalization and ReLU activation, ending with a FC linear layer for 
    classification. 
    
    Network architecture is as follows::

        Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten ->
        
        Fastfood/RKS -> BN -> ReLU -> FC Linear (Output)

    Parameters
    ----------
    projection_layer: nn.Module
        The layer type to use (FastfoodLayer or RKSLayer)
    features: int
        Number of features for the projection layer
    proj_args: list
        Additional arguments to pass to the projection layer (e.g., input_dim, scale,
        device, learnable flags, etc.)
    
    References
    ----------
    .. [1] Yang, Z., Moczulski, M., Denil, M., et al. (2014). Deep Fried Convnets.
        https://arxiv.org/abs/1412.7149

    Notes
    -----
    This model is programmed to run on the CIFAR-10 dataset.
    """
    def __init__(self, projection_layer=FastfoodLayer, features=1024, proj_args={}):
        super(LeNet, self).__init__()
        # Convolutional layers matching Caffe definition
        # Caffe: (Convolutional Architecture for Fast Feature Embedding)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        self.flattened_size = 50 * 5 * 5
        
        # Replace FC layer with projection layer
        self.projection = projection_layer(
            input_dim=self.flattened_size, 
            output_dim=features,
            **proj_args
        )
        
        # Add batch normalization layer
        self.bn = nn.BatchNorm1d(features)
        
        # Output layer
        self.output = nn.Linear(features, 10)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, 3, 32, 32).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (batch_size, classes), representing raw 
            classification scores.

        Notes
        -----
        Input tensor corresponds to the CIFAR-10 dataset, which has :math:`3` color 
        channels and :math:`32 \\times 32` size images.
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten features
        x = x.view(-1, self.flattened_size)
        
        # Apply projection layer
        x = self.projection(x)
        
        # Apply batch normalization and non-linearity
        x = F.relu(self.bn(x))
        
        # Classification
        x = self.output(x)
        
        return x