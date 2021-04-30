from torch import nn 

class ConvBlock(nn.Module):
    """
    Implementation of a sequence of:
    - A convolution layer
    - A Batch Normalization layer
    - An activation function (default being nn.Identity)
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros',
        activation=None,
        eps=1e-3, 
        momentum=0.01
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, groups, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - activation: A provided activation function. Default being nn.Identity.
        - eps, momentup: see the nn.BatchNorm2d document of PyTorch.
        """

        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias, 
            padding_mode=padding_mode
        )
        
        self.bn = nn.BatchNorm2d(
            out_channels, 
            eps=eps, 
            momentum=momentum
        )

        self.activation = nn.Identity() if activation is None else activation()
    
    def forward(self, x):
        return self.activation(
            self.bn(
                self.conv(x)
            )
        )