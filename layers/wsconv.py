import torch 
from torch import nn
from torch.nn import functional as F

class WSConv2d(nn.Conv2d):
    """
    Implementation of the Weight Standardization layer, applied on a convolutional layer.
    It is advised to use this layer in coordination with nn.GroupNorm.
    https://arxiv.org/abs/1903.10520
    """

    def forward(self, x):
        mu = self.weight.mean((1,2,3), keepdim=True)
        std = self.weight.std((1,2,3), keepdim=True)

        w = (self.weight - mu) / torch.sqrt(std**2 + 1e-10)

        return F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
