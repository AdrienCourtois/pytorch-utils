from torch import nn 

class ChannelProjection2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        if in_channels != out_channels:
            self.layer = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        if hasattr(self, "layer"):
            x = self.layer(x)
        
        return x

class ChannelProjection1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        if in_channels != out_channels:
            self.layer = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        if hasattr(self, "layer"):
            x = self.layer(x)
        
        return x
