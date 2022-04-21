from torch import nn 
from typing import Optional 
from .einsum import einsum

class LayerScale(nn.Module):
    in_channels: int
    layerscale_init: Optional[float] = 0.1

    def __init__(self, in_channels: int, layerscale_init: Optional[float]=0.1):
        super().__init__()

        if layerscale_init is not None: 
            self.layerscale = nn.Parameter(layerscale_init * torch.ones(in_channels))
    
    def forward(self, x):
        if hasattr(self, "layerscale"):
            return einsum("c, b c ... -> b c ...", self.layerscale, x)

        return x
