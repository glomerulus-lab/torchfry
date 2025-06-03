import torch
import torch.nn as nn
from torchvision import models
from torchfry.transforms import FastfoodLayer

class VGG(nn.Module):
    """
    VGG-based model that uses a pre-trained VGG-16BN model from the Visual Geometry Group.
    It consists of 13 convolutional layers, each followed by batch normalization and ReLU 
    activation, some followed by a max-pooling layer for a total of 5, then 2 FC linear 
    layers with ReLU and dropout after each, finally ending with a FC linear layer for
    classification. 
    
    Network architecture is as follows::

        (Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool) * 2 ->

        (Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool) * 3 -> 

        Flatten -> (FC Linear -> ReLU -> Dropout) * 2 -> FC Linear (Output)

    In this implementation, the first two FC linear layers are replaced with one of the 
    random feature layers (``Fastfood/RKS``)::

        ... -> (Fastfood/RKS -> ReLU -> Dropout) * 2 -> FC Linear (Output)

    Parameters
    ----------
    projection_layer : nn.Module
        The type of projection layer to use within the classifier layers
    input_shape : tuple of int
        Shape of the input images in (channels, height, width) format.
    features : int
        Number of features for the classifier layers 
    classes : int
        Number of output classes for classification
    proj_args : dict or None
        Additional keyword arguments to pass to the projection layers.

    Notes
    -----
    This model is programmed to run on the CIFAR-10 dataset.
    """
    def __init__(self, projection_layer=FastfoodLayer, input_shape=(3, 32, 32), features=512, classes=10, proj_args=None):
        super(VGG, self).__init__()
        # Load pre-trained VGG16 with batch normalization
        vgg = models.vgg16_bn(weights=None)
        # Extract feature extractor (convolutional blocks)
        self.features = vgg.features
        
        # Replace classifier with your custom classifier
        self.classifier = nn.Sequential(
            projection_layer(input_dim=features, output_dim=features, **proj_args),
            nn.ReLU(True),
            nn.Dropout(),
            projection_layer(input_dim=features, output_dim=features, **proj_args),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=features, out_features=classes)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (batch_size, classes), representing raw 
            classification scores.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x