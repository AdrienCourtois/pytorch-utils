import math
import torch 
import torch.nn as nn
from .activations import Swish

import numpy as np

class Flatten(nn.Module):
    """
    Flatten function, analoguous to the Keras homonym.
    For a given tensor of dimension (N, C, H, W), returns a vector of dimension (N, C*H*W).
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    """
    Implementation of a sequence of:
    - A convolution layer
    - A Batch Normalization layer
    - An activation function (default being activations.Swish)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 activation=None,
                 eps=1e-3, momentum=0.01):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, groups, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - activation: A provided activation function. Default being activations.Mish(). If the provided activation is a module, it has to be initialized (i.e activation=Mish() and not activation=Mish).
        - eps, momentup: see the nn.BatchNorm2d document of PyTorch.
        """

        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.activation = Swish() if activation is None else activation
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """
    Implementation of a squeeze and excitation layer.
    """

    def __init__(self, in_channels, ratio=16, activation=None):
        """
        - in_channels (int): Number of channels of the input.
        - ratio (int): The number of times we increase the number of channels of the intermediate representation of the input. The smaller the better but the higher associated the computation cost. Default being 16 as suggested in the original paper.
        - activation: A provided activation function. Default being Swish(). If the provided activation is a module, it has to be initialized (i.e activation=Mish() and not activation=Mish).
        """

        super().__init__()

        mid_channels = math.ceil(in_channels / ratio)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0)
        self.activation = Swish() if activation is None else activation
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 1, stride=1, padding=0)

    def forward(self, x):
        return x * torch.sigmoid(self.conv2(self.activation(self.conv1(self.global_pool(x)))))


class MBConv(nn.Module):
    """
    Implementation of a MBConv block, as used in EfficientNet.
    """

    def __init__(self, in_channels, out_channels, expend_ratio=2, kernel_size=(3,3), 
                 stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
                 residual=True, se_ratio=16,
                 activation=None,
                 eps=1e-3, momentum=0.01):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. Note that is in_channels != out_channels, no residual connection can be applied.
        - expend_ratio (int): The number of intermediary channels is computed using expend_ratio * in_channels. Default being 2 as in EfficientNet.
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, groups, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - residual (bool):
        - se_ration (int): The ratio of the Squeeze and Excitation layer. See the document of layers.SEBlock for more details.
        - activation: A provided activation function. Default being activation.Swish(). If the provided activation is a module, it has to be initialized (i.e activation=Mish() and not activation=Mish).
        - eps, momentup: see the nn.BatchNorm2d document of PyTorch.
        """
        
        super().__init__()

        mid_channels = math.ceil(in_channels * expend_ratio)

        self.activation = Swish() if activation is None else activation

        self.conv_spatialwise = ConvBlock(in_channels, mid_channels, 1, 
                                              stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
                                              activation=self.activation,
                                              eps=eps, momentum=momentum)
        self.conv_channelwise = ConvBlock(mid_channels, mid_channels, 
                                              kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                                              activation=self.activation,
                                              eps=eps, momentum=momentum)
        self.se = SEBlock(mid_channels, ratio=se_ratio)
        self.conv_spatialwise2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        )

        self.residual = residual and (in_channels == out_channels) and (stride == 1)
    
    def forward(self, x):
        y = self.conv_spatialwise(x)
        y = self.conv_channelwise(y)
        y = self.se(y)
        y = self.conv_spatialwise2(y)

        if self.residual:
            y = y + x
        
        return y


class MultiHeadedSpatialAttention(nn.Module):
    """
    Implementation of the multi-headed spatial attention mechanism for computer vision.
    Note that the `groups` parameter plays the role of the number of heads.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', position=True):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - groups (int): Plays the role of the number of heads. Has to be a multiplier of in_channels and out_channels.
        - position: Toggles the usage of relative positional encoding as described in the paper. The positional encoding is computed using a gaussian kernel.
        """
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = int(out_channels / groups)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.position = position

        self.query_op = nn.Conv2d(in_channels, out_channels, 1, stride=stride, groups=groups, bias=False)
        self.key_op = nn.Conv2d(in_channels, out_channels, 1, stride=stride, groups=groups, bias=False)
        self.value_op = nn.Conv2d(in_channels, out_channels, 1, stride=stride, groups=groups, bias=False)

        self.unfolding = nn.Unfold(kernel_size, dilation=dilation, padding=padding, stride=stride)

        if self.position:
            X, Y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing='ij')
            r = np.concatenate((X[:,:,None], Y[:,:,None]), 2) - np.array([int(kernel_size/2),int(kernel_size/2)])
            r = np.exp(-np.linalg.norm(r, axis=2)) 
            self.position_matrix = torch.Tensor(r)
            
        else:
            self.position_matrix = torch.zeros(kernel_size, kernel_size)
        
        self.position_matrix = self.position_matrix.view(1, 1, 1, kernel_size*kernel_size, 1, 1)
        self.position_matrix = nn.Parameter(self.position_matrix).requires_grad_(False)
    
    def forward(self, x):
        query = self.query_op(x)
        key = self.key_op(x)
        value = self.value_op(x)

        key = self.unfolding(key).view(x.size(0), 
                                       self.groups, self.mid_channels, 
                                       self.kernel_size*self.kernel_size, 
                                       x.size(2), x.size(3))
        value = self.unfolding(value).view(x.size(0), 
                                           self.groups, self.mid_channels, 
                                           self.kernel_size*self.kernel_size, 
                                           x.size(2), x.size(3))
        query = query.view(x.size(0),
                           self.groups, self.mid_channels,
                           1, 
                           x.size(2), x.size(3))

        similarity = (query * (key + self.position_matrix)).sum(2, keepdim=True).softmax(2)
        output = (similarity * value).sum(3)

        output = output.view(output.size(0), 
                             output.size(1) * output.size(2),
                             output.size(3), output.size(4))

        return output


class CoordConv2d(nn.Module):
    """
    Simple implementation of the CoordConv layer.
    Note that the `groups` parameter may not work as you expect, as the position channels won't be add to each sub-group.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """
        See parameters of torch.nn.Conv2d
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size, **kwargs)

    def cat(self, x):
        N, C, H, W = x.size()

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_x = 2 * (grid_x / (W-1.)) - 1
        grid_y = 2 * (grid_y / (H-1.)) - 1

        grid = torch.cat((grid_x.unsqueeze(0), grid_y.unsqueeze(0)), 0)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)

        if x.is_cuda:
            grid = grid.cuda()

        return torch.cat((x, grid), 1)

    def forward(self, x):
        return self.conv(self.cat(x))

class WSConv2d(nn.Conv2d):
    """
    Implementation of the Weight Standardization layer, applied on a convolutional layer.
    It is advised to use this layer in coordination with nn.GroupNorm.
    """
    def forward(self, x):
        mu = self.weight.mean((1,2,3), keepdim=True)
        std = self.weight.std((1,2,3), keepdim=True)

        w = (self.weight - mu) / torch.sqrt(std**2 + 1e-10)

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SWSConv2d(nn.Conv2d):
    """
    Implementation of the Scaled Weight Standardization layer, applied on a convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation="relu", gamma=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.gamma = gamma

        if self.gamma is None:
            if activation == "relu":
                self.gamma = np.sqrt(2) / np.sqrt(1 - 1/np.pi)
            elif activation == "swish":
                self.gamma = 1. / 0.63
            elif activation == "mish":
                self.gamma = 1. / 0.56
            elif activation == "linear":
                self.gamma = 1.
            else: 
                raise Exception(f"Activation {activation} not implemented.")

    def forward(self, x):
        mu = self.weight.mean((1,2,3), keepdim=True)
        std = self.weight.std((1,2,3), keepdim=True)

        w = self.gamma * (self.weight - mu) / torch.sqrt(std**2 + 1e-10) / np.sqrt(np.prod(self.weight.size()[1:]))

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PFLayerNorm(nn.Module):
    """
    Parameter-free layer normalization layer, more suited when dealing with images.
    """
    def __init__(self, dims=(2,3)):
        super().__init__()

        self.dims = dims 
    
    def forward(self, x):
        mean = x.mean(self.dims, keepdim=True)
        std = x.std(self.dims, keepdim=True)

        return (x - mean) / torch.sqrt(std**2 + 1e-10)

class CLayerNorm(nn.Module):
    """
    A layer normalization layer where you only learn weights for each channel independantly, similarly to Batch Normalization.
    """
    def __init__(self, in_channels, dims=(2,3)):
        super().__init__()

        self.dims = dims

        self.layernorm = PFLayerNorm(dims=dims)
        self.gamma = nn.Parameter(torch.ones(1, in_channels))
        self.beta = nn.Parameter(torch.zeros(1, in_channels))

        for d in dims:
            self.gamma.unsqueeze_(d)
            self.beta.unsqueeze_(d)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.gamma * x + self.beta

        return x

class CPowerNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        nn.init.ones_(self.running_mean)

    def forward(self, X):
        self._check_input_dim(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked += 1

            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # Normalization
        x = x / torch.sqrt(self.running_mean + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        # Update running estimates
        if self.training:
            mean = torch.pow(x, 2).mean((0, 2, 3))
            n = x.numel() / x.size(1)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
        else:
            mean = self.running_mean

        return x