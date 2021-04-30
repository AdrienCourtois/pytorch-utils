import torch 
from torch import nn 
import numpy as np 

class MultiHeadedSpatialAttention(nn.Module):
    """
    Implementation of the multi-headed spatial attention mechanism for computer vision.
    Note that the `groups` parameter plays the role of the number of heads.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding=0,
        stride=1,
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros', 
        position=True
    ):
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

        self.to_Q = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            groups=groups, 
            bias=bias
        )
        self.to_K = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            groups=groups, 
            bias=bias
        )
        self.to_V = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            groups=groups, 
            bias=bias
        )

        self.unfolding = nn.Unfold(
            kernel_size, 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        )

        # TODO: Recode
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
        B, C, H, W = x.size() 

        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)

        K = self.unfolding(K).view(
            B, 
            self.groups, 
            self.mid_channels, 
            self.kernel_size*self.kernel_size, 
            H, W
        )
        V = self.unfolding(V).view(
            B, 
            self.groups, 
            self.mid_channels, 
            self.kernel_size*self.kernel_size, 
            H, W
        )
        Q = Q.view(
            B,
            self.groups, 
            self.mid_channels,
            H, W
        )

        A = torch.einsum(
            "b g c h w, b g c p h w -> b g p h w", 
            Q, 
            K + self.position_matrix
        )
        A = A.softmax(2)
        
        y = torch.einsum("b g p h w, b g c p h w -> b g c h w", A, V)
        y = y.view(B, C, H, W)

        return y