import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair
import math

class LegendreConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        super().__init__()

        # Checking the validity of the parameters
        if self.groups != 1:
            raise Exception("Not implemented for groups > 1.")

        # Parameters instanciation
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Micro optimization
        self.unfolder = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)

    def reset_parameters(self):
        # Initialization of the parameters based on the traditional conv 
        # TODO: might need to be changed based on the output distribution
        # TODO: as the initialization appears to be a real game-changer 
        # TODO: (cf paper with the COS activation function)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        N, C, H, W = x.size()
        H_ = H - 2*int(self.kernel_size[0]//2) + 2*self.padding[0]
        W_ = W - 2*int(self.kernel_size[1]//2) + 2*self.padding[1]

        x = self.unfolder(x)
        x = (self.weight[None,:,:,None] + x[:,None,:,:]).min(2).values
        x = x.view(N, self.out_channels, H_, W_)

        return x