import torch 
from torch import nn 

class CBAM(nn.Module):
    def __init__(self, in_channels, activation=None, r=2):
        """
        The CBAM module as described in https://arxiv.org/pdf/1807.06521.pdf.
        Args:
        - in_channels (int): Number of input channels
        - activation (nn.Module): Activation function (default: nn.ReLU)
        - r (float): Reduction factor
        """

        super().__init__()

        in_channels = in_channels 
        mid_channels = int(in_channels / r)
        activation = nn.ReLU if activation is None else activation

        # Channel attention module
        self.maxpool_channel = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool_channel = nn.AdaptiveAvgPool2d((1,1))

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            activation(),
            nn.Conv2d(mid_channels, in_channels, 1)
        )

        # Spatial attention module
        self.post_filtering = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        # Channel attention module
        A_channel = torch.sigmoid(
            self.shared_mlp(self.avgpool_channel(x)) + \
            self.shared_mlp(self.maxpool_channel(x))
        )

        x = A_channel * x

        # Spatial attention module
        A_spatial = torch.sigmoid(
            self.post_filtering(
                torch.cat([
                    x.mean(1, keepdim=True),
                    x.max(1, keepdim=True).values
                ], 1)
            )
        )

        x = A_spatial * x

        return x