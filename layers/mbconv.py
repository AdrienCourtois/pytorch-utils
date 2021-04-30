import math 
from torch import nn 
from . import ConvBlock, SEBlock

class MBConv(nn.Module):
    """
    Implementation of a MBConv block, as used in EfficientNet.
    https://arxiv.org/abs/1905.11946
    Inspiration took from https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        expend_ratio=2, 
        kernel_size=(3,3), 
        stride=1,
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode="zeros",
        residual=True, 
        se_ratio=16,
        activation=None,
        eps=1e-3,
        momentum=0.01
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. Note that is in_channels != out_channels, no residual connection can be applied.
        - expend_ratio (int): The number of intermediary channels is computed using expend_ratio * in_channels. Default being 2 as in EfficientNet.
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, groups, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - residual (bool):
        - se_ration (int): The ratio of the Squeeze and Excitation layer. See the documentation of layers.SEBlock for more details.
        - activation: A provided activation function. Default being nn.Identity.
        - eps, momentup: see the nn.BatchNorm2d document of PyTorch.
        """
        
        super().__init__()

        mid_channels = math.ceil(in_channels * expend_ratio)

        self.activation = nn.Identity if activation is None else activation

        self.conv_channelwise = ConvBlock(
            in_channels, 
            mid_channels, 
            1, 
            groups=groups, 
            bias=False, 
            activation=self.activation,
            eps=eps, 
            momentum=momentum
        )

        self.conv_spatialwise = ConvBlock(
            mid_channels, 
            mid_channels,
            kernel_size, 
            stride=stride, 
            padding=padding,
            dilation=dilation, 
            groups=mid_channels, 
            bias=False, 
            padding_mode=padding_mode,
            activation=self.activation, 
            eps=eps, 
            momentum=momentum
        )

        self.se = SEBlock(mid_channels, ratio=se_ratio)
        
        self.conv_channelwise2 = ConvBlock(
            mid_channels, 
            out_channels,
            1,
            groups=groups,
            bias=False,
            eps=eps, 
            momentum=momentum
        )

        self.residual = residual and (in_channels == out_channels) and (stride == 1)
    
    def forward(self, x):
        y = self.conv_channelwise(x)
        y = self.conv_spatialwise(y)
        y = self.se(y)
        y = self.conv_channelwise2(y)

        if self.residual:
            y = y + x
        
        return y
