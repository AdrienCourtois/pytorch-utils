import torch 
from torch import nn 

class GlobalContextLayer(nn.Module):
    def __init__(self, in_features, r=2):
        super().__init__()

        mid_features = int(in_features / r)
        
        self.W = nn.Conv2d(in_features, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1),
            nn.LayerNorm([mid_features, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_features, in_features, 1)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        context = self.W(x).view(B, -1)
        context = torch.softmax(context, 1).view(B, H, W)
        context = torch.einsum("b c h w, b h w -> b c", x, context).view(B, C, 1, 1)

        t = self.transform(context)

        return x + t
