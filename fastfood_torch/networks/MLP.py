import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, classes, widths, layer, proj_args):
        super().__init__()
        
        # Sequential Placeholder
        layers = []
        for width in widths:
            layers.append(layer(input_dim=input_dim, output_dim=width, **proj_args))
            layers.append(nn.BatchNorm1d(width, affine=False))
            layers.append(nn.ReLU())
            input_dim = width

        # Final output layer, learnable
        layers.append(nn.Linear(input_dim, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
