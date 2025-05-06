import torch
import torch.nn as nn
from torchvision import models
from fastfood_torch.transforms import FastFoodLayer, RKSLayer

class VGG(nn.Module):
    def __init__(self, projection_layer=FastFoodLayer, input_shape=(3, 32, 32), features=512, num_classes=10, proj_args=None):
        super(VGG, self).__init__()
        
        # Load pre-trained VGG11 with batch normalization
        vgg = models.vgg16_bn(weights=None)
        # Extract feature extractor (convolutional blocks)
        self.features = vgg.features
        
        # Replace classifier with your custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(features, features),
            # projection_layer(input_dim=flattened_size, output_dim=features, **proj_args),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(features, features),
            # projection_layer(input_dim=features, output_dim=features, **proj_args),
            nn.ReLU(True),
            nn.Linear(in_features=features, out_features=num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x