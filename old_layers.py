import math
import torch 
import torch.nn as nn

import numpy as np

class PFLayerNorm(nn.Module):
    """
    Parameter-free layer normalization layer, more suited when dealing with images.
    """
    def __init__(self, dims=(2,3)):
        super().__init__()

        self.dims = dims 
    
    def forward(self, x):
        mean = x.mean(self.dims, keepdim=True)
        std = x.std(self.dims, keepdim=True)

        return (x - mean) / torch.sqrt(std**2 + 1e-10)

class CLayerNorm(nn.Module):
    """
    A layer normalization layer where you only learn weights for each channel independantly, similarly to Batch Normalization.
    """
    def __init__(self, in_channels, dims=(2,3)):
        super().__init__()

        self.dims = dims

        self.layernorm = PFLayerNorm(dims=dims)
        self.gamma = nn.Parameter(torch.ones(1, in_channels))
        self.beta = nn.Parameter(torch.zeros(1, in_channels))

        for d in dims:
            self.gamma.unsqueeze_(d)
            self.beta.unsqueeze_(d)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.gamma * x + self.beta

        return x

class CPowerNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        nn.init.ones_(self.running_mean)

    def forward(self, X):
        self._check_input_dim(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked += 1

            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # Normalization
        x = x / torch.sqrt(self.running_mean + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        # Update running estimates
        if self.training:
            mean = torch.pow(x, 2).mean((0, 2, 3))
            n = x.numel() / x.size(1)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
        else:
            mean = self.running_mean

        return x