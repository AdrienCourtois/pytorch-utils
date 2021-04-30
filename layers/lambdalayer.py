import torch 
from torch import nn 

class LambdaLayer(nn.Module):
    """
    LambdaLayer from Lambda Network, without the positional encoding. 
    TODO: heads
    """

    def __init__(self, in_channels, mid_channels, dim_query, dim_value, bias=True, bn=True):
        # The output is a torch.Tensor of dimension [?, dim_value, H, W].
        
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.dim_query = dim_query
        self.dim_value = dim_value
        self.bn = bn

        self.to_Q = nn.Conv2d(in_channels, dim_query, 1, bias=bias)
        self.to_K = nn.Conv2d(in_channels, mid_channels*dim_query, 1, bias=bias)
        self.to_V = nn.Conv2d(in_channels, mid_channels*dim_value, 1, bias=bias)

        if self.bn:
            self.norm_Q = nn.BatchNorm2d(dim_query)
            self.norm_V = nn.BatchNorm2d(mid_channels*dim_value)
    
    def forward(self, x):
        B, _, H, W = x.size()

        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)

        if self.bn:
            Q = self.norm_Q(Q)
            V = self.norm_V(V)

        Q = Q.view(B, self.dim_query, H*W)
        K = K.view(B, self.mid_channels, self.dim_query, H*W)
        V = V.view(B, self.mid_channels, self.dim_value, H*W)

        K = K.softmax(3)

        context = torch.einsum('b u k m, b u v m -> b k v', K, V)
        y = torch.einsum('b k n, b k v -> b v n', Q, context)
        y = y.view(B, self.dim_value, H, W)

        return y