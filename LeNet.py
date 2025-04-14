import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Layers.FastFood_Layer import FastFood_Layer
from Layers.RKS_Layer import RKS_Layer

class LeNet(nn.Module):
    def __init__(self, projection_layer=FastFood_Layer, features=1024, proj_args={}):
        """
        Initialize a LeNet model similar to DeepFriedLeNet.
        
        Args:
            projection_layer: The layer type to use (FastFood_Layer or RKS_Layer)
            features: Number of features for the projection layer
            proj_args: Additional arguments to pass to the projection layer
        """
        super(LeNet, self).__init__()
        
        # Convolutional layers matching Caffe definition
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        self.flattened_size = 50 * 4 * 4
        
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
        Forward pass through the network
        
        Args:
            x: Input image tensor
        """
        # Convolutional ;ayers
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