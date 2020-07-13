import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """
    Mish activation function. Note that this function is greedy in computational ressources.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    """
    Swish activation function.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)