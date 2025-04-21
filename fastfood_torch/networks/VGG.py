import torch
import torch.nn as nn
from torchvision import models
from fastfood_torch.transforms import FastFoodLayer, RKSLayer

class VGG(nn.Module):
    """
    Customized VGG11 model with a projection layer and batch normalization.
    Args:
        input_shape (tuple): Shape of the input images as (channels, height, width)
        projection_layer (nn.Module): A projection layer, FastFood or RKS
        features (int): Number of hidden units in the FC layers
        classes (int): Number of output classes
        proj_args (dict): Arguments to pass into the projection layer
    """
    
    def __init__(self, input_shape=(3, 32, 32), projection_layer=FastFoodLayer, features=4096, classes=10, proj_args={}):
        super(VGG, self).__init__()

        # Load VGG11 and replace the first layer for custom input channels
        self.vgg = models.vgg11_bn(weights=None)
        self.vgg.features[0] = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)

        # Dynamically compute flattened feature size
        self.features = self.vgg.features
        self.flattened_size = self._get_flattened_size(input_shape)

        # Replace the classifier
        self.classifier = nn.Sequential(
            projection_layer(input_dim=self.flattened_size, output_dim=features, **proj_args),
            nn.ReLU(),
            nn.Dropout(),
            projection_layer(input_dim=features, output_dim=features, **proj_args),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(features, classes)
        )

    def _get_flattened_size(self, input_shape):
        """
        Passes a dummy input through the feature extractor to determine output size of convs
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.features(dummy_input)
            return out.view(1, -1).shape[1]

    def forward(self, x):
        """
        Forward pass
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
