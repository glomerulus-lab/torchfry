import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=10):
        super(VGG, self).__init__()

        # Load VGG11 with batch normalization and customize the first convolutional layer
        self.vgg = models.vgg11_bn(weights=None)
        self.vgg.features[0] = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.features = self.vgg.features

        # Determine the flattened size after adaptive pooling
        self.flattened_size = self._get_flattened_size(input_shape)

        # Define the classifier with linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, classes)
        )

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.features(dummy_input)
            out = self.avgpool(out)
            return out.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
