import math 
from torch import nn 

class SEBlock(nn.Module):
    """
    Implementation of a squeeze and excitation layer.
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels, ratio=16, activation=None):
        """
        - in_channels (int): Number of channels of the input.
        - ratio (int): The number of times we increase the number of channels of the intermediate representation of the input. The smaller the better but the higher associated the computation cost. Default being 16 as suggested in the original paper.
        - activation: A provided activation function. Default being nn.Identity.
        """

        super().__init__()

        mid_channels = math.ceil(in_channels / ratio)

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.Identity() if activation is None else activation()
        )
        self.excite = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.excite(self.squeeze(x))

        return x * A