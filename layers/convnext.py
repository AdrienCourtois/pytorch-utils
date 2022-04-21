from torch import nn
from typing import Optional
from .channelprojection import ChannelProjection2d, ChannelProjection1d
from .layernorm import LayerNorm
from .layerscale import LayerScale

class ConvNeXt2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: int=4, layerscale_init: Optional[float]=0.1):
        super().__init__()

        self.proj = ChannelProjection2d(in_channels, out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels),
            LayerNorm(out_channels),
            nn.Conv2d(out_channels, ratio * out_channels, 1),
            nn.GELU(),
            nn.Conv2d(ratio * out_channels, out_channels, 1),
            LayerScale(out_channels, layerscale_init=layerscale_init)
        )
    
    def forward(self, x):
        x = self.proj(x)

        return x + self.block(x)

class ConvNeXt1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: int=4, layerscale_init: Optional[float]=0.1):
        super().__init__()

        self.proj = ChannelProjection1d(in_channels, out_channels)

        self.block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 7, padding=3, groups=out_channels),
            LayerNorm(out_channels),
            nn.Conv1d(out_channels, ratio * out_channels, 1),
            nn.GELU(),
            nn.Conv1d(ratio * out_channels, out_channels, 1),
            LayerScale(out_channels, layerscale_init=layerscale_init)
        )
    
    def forward(self, x):
        x = self.proj(x)

        return x + self.block(x)
