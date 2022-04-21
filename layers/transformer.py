import torch
from torch import nn
from typing import Optional
from .einsum import einsum
from .layernorm import LayerNorm
from .layerscale import LayerScale
from .invertedbottleneck import InvertedBottleneck1d

class Transformer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int=8, layerscale_init: Optional[float]=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.heads = heads

        self.prenorm = LayerNorm(in_channels)
        self.to_QKV = nn.Conv1d(in_channels, 3 * out_channels, 1)
        self.layerscale = LayerScale(out_channels, layerscale_init=layerscale_init)

        self.postprocess = InvertedBottleneck(out_channels, out_channels, layerscale_init=layerscale_init)

        self.pe = nn.Parameter(torch.ones(1, 1, 1), requires_grad=False)
    
    def generate_posenc(self, H, W):
        channels = int(np.ceil(self.in_channels / 4) * 2)
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2, dtype=torch.float) / channels))

        pos_x = torch.arange(H, dtype=torch.float)
        pos_y = torch.arange(W, dtype=torch.float)
        sin_inp_x = einsum("i, j -> i j", inv_freq, pos_x)
        sin_inp_y = einsum("i, j -> i j", inv_freq, pos_y)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)

        emb = torch.zeros(self.in_channels, H, W, dtype=torch.float)
        emb[0::4] = sin_inp_x.cos()[:, :, None]
        emb[1::4] = sin_inp_x.sin()[:, :, None]
        emb[2::4] = sin_inp_y.cos()[:, None]
        emb[3::4] = sin_inp_y.sin()[:, None]

        return emb
    
    def forward(self, x):
        B, C, H, W = x.size()

        if (self.pe.size(1), self.pe.size(2)) != (H, W):
            self.pe.data = self.generate_posenc(H, W).to(self.pe.device)
        
        x = x.view(B, C, H * W)

        y = self.prenorm(x)

        QKV = self.to_QKV(y)
        Q, K, V = torch.chunk(QKV, 3, dim=1)
        Q, K, V = map(lambda x: x.view(B, self.heads, -1, H * W), (Q, K, V))

        pe = self.pe.view(self.heads, -1, H * W)

        A = (einsum("b h c n, b h c m -> b h n m", Q, K) + einsum("h c n, h c m -> h n m", pe, pe))/ np.sqrt(C)
        A = A.softmax(-1)
        
        y = einsum("b h n m, b h c m -> b h c n", A, V)
        y = y.reshape(B, -1, H * W)

        x = x + self.layerscale(y)
        x = x + self.postprocess(x)

        return x.view(B, C, H, W)
