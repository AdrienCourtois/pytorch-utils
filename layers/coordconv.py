import torch 
from torch import nn 

class CoordConv2d(nn.Module):
    """
    Simple implementation of the CoordConv layer.
    https://arxiv.org/abs/1807.03247
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """
        See parameters of torch.nn.Conv2d
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels+2, 
            out_channels, 
            kernel_size, 
            **kwargs
        )

        self.grid = nn.Parameter(torch.Tensor(1, 1, 1, 1), requires_grad=False)

    def generate_grid(self, H, W):
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_x = 2 * (grid_x / (W-1.)) - 1
        grid_y = 2 * (grid_y / (H-1.)) - 1

        grid = torch.cat((grid_x.unsqueeze(0), grid_y.unsqueeze(0)), 0)
        grid = grid.unsqueeze(0)
        
        self.grid.data = grid.to(self.grid.device)

    def forward(self, x):
        if x.size(2) != self.grid.size(2) or x.size(3) != self.grid.size(3):
            self.generate_grid(x.size(2), x.size(3))
        
        x = torch.cat([x, self.grid.repeat(x.size(0), 1, 1, 1)], 1)

        return self.conv(x)