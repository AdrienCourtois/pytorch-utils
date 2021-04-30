from torch import nn 
import warnings

class Flatten(nn.Module):
    """
    Flatten function, analoguous to the Keras homonym.
    For a given tensor of dimension (N, C, H, W), returns a vector of dimension (N, C*H*W).
    [Deprecated] Please use nn.Flatten instead.
    """

    def __init__(self):
        super().__init__()

        warnings.warn(
            "Flatten is deprecated, please use nn.Flatten instead.",
            DeprecationWarning
        )
    
    def forward(self, x):
        return x.view(x.size(0), -1)