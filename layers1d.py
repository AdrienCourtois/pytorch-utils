import torch
from torch import nn 
import numpy as np 
import math 

class ReverseLayerNorm(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_features) # to fit the official transformer

    def forward(self, x):
        return self.layer_norm(x.permute(0,2,1)).permute(0,2,1).contiguous()


class Transformer(nn.Module):
    def __init__(self, in_features, heads=8, K=100):
        super().__init__()

        self.in_features = in_features
        self.mid_features = in_features // heads
        self.heads = heads
        self.K = K

        self.to_Q = nn.Conv1d(in_features, in_features, 1, groups=heads)
        self.to_K = nn.Conv1d(in_features, in_features, 1, groups=heads)
        self.to_V = nn.Conv1d(in_features, in_features, 1, groups=heads)
        self.ln1 = ReverseLayerNorm(in_features)

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_features, in_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_features, in_features, 1)
        )
        self.ln2 = ReverseLayerNorm(in_features)

        self.initialize()

        self.pos_encoding = nn.Parameter(self.generate_positional_encoding(), requires_grad=True)
    
    def generate_positional_encoding(self):
        """
        Cosine positional encoding
        Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Returns:
            pe: torch.Tensor of dimension [1, C, K] containing the positional encoding.
        """
        pe = torch.zeros(self.K, self.in_features)
        position = torch.arange(0, self.K, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.in_features, 2).float() * (-math.log(10000.0) / self.in_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.transpose(0, 1).unsqueeze(0)
    
    def initialize(self):
        nn.init.normal_(self.to_Q.weight, std=2.*self.in_features/5.)
        nn.init.normal_(self.to_K.weight, std=2.*self.in_features/5.)
        nn.init.normal_(self.to_V.weight, std=2.*self.in_features/5.)
        nn.init.normal_(self.feed_forward[0].weight, std=2.*self.in_features/5.)
        nn.init.normal_(self.feed_forward[2].weight, std=2.*self.in_features/5.)
    
    def forward(self, x):
        B, C, K = x.size()

        y = self.ln1(x) # Note: pre-norm

        # Positional encoding
        y = y + self.pos_encoding # Note: *sqrt(C) is more unstable

        # Self-Attention
        q = self.to_Q(y).view(B, self.heads, self.mid_features, K)
        k = self.to_K(y).view(B, self.heads, self.mid_features, K)
        v = self.to_V(y).view(B, self.heads, self.mid_features, K)

        A = torch.einsum("b h c k, b h c l -> b h k l", q, k) / np.sqrt(self.mid_features)
        A = A.softmax(3)
        y = torch.einsum("b h k l, b h c l -> b h c k", A, v).view(B, self.in_features, K)

        x = x + y

        # Feed Forward
        y = self.ln2(x) # Note: pre-norm
        y = self.feed_forward(y)

        x = x + y

        return x