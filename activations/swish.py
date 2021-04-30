import torch
import torch.nn as nn
from torch.nn import functional as F

class Swish(nn.Module):
    """
    Swish activation function.
    """

    def __init__(self, alpha=1., learnable=False):
        super().__init__()

        self.learnable = False

        if learnable:
            self.alpha = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.alpha, alpha)

        else:
            self.alpha = alpha 
    
    def forward(self, x):
        return F.silu(self.alpha * x, inplace=True) / (self.alpha + 1e-10)