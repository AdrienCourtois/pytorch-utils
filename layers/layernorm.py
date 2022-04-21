from torch import nn
from .einsum import einsum

def pad_as(x, ref):
    _, _, *dims = ref.size()

    for _ in range(len(dims)):
        x = x.unsqueeze(dim=-1)
    
    return x

class LayerNorm(nn.Module):
    """
    LayerNorm in the "channel first" mode.
    """

    def __init__(self, in_channels: int, eps: float=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        return einsum("c, b c ... -> b c ...", self.weight, x) + pad_as(self.bias, x)
