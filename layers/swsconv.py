import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np 

class SWSConv2d(nn.Conv2d):
    """
    Implementation of the Scaled Weight Standardization layer, applied on a convolutional layer.
    https://arxiv.org/abs/2101.08692
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        activation="relu", 
        gamma=None, 
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.gamma = gamma

        if self.gamma is None:
            if activation == "relu":
                self.gamma = np.sqrt(2) / np.sqrt(1 - 1/np.pi)

            elif activation == "swish":
                self.gamma = 1. / 0.63

            elif activation == "mish":
                self.gamma = 1. / 0.56

            elif activation == "linear":
                self.gamma = 1.

            else: 
                raise Exception(f"Activation {activation} not implemented.")

    def forward(self, x):
        mu = self.weight.mean((1,2,3), keepdim=True)
        std = self.weight.std((1,2,3), keepdim=True)

        w = self.gamma * (self.weight - mu) / torch.sqrt(std**2 + 1e-10) / np.sqrt(np.prod(self.weight.size()[1:]))

        return F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
