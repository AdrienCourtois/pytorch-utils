# Utilitaries layers for Deep Learning applied to Computer Vision - PyTorch
This GitHub aims at providing implementations of deep learning layers presented recently and that could be used for Computer Vision. The goal of this repository is to provide layers that have not been yet added to the main version of PyTorch.

## Features
### Layers
- Squeeze and Excitation layer [1] <br>
*A normalization layer that can be used with or in replacement of Batch Normalization* <br>
**Usage:** 
```python
from pytorch-utils.layers import SEBlock
from pytorch-utils.activations import Mish

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  Mish(),
  SEBlock()
)
```
- MBConv <br>
*As used in EfficientNet [2], decorrelates the channel and the spatial processing in convolutions* <br>
**Usage:**
```python
from pytorch-utils.layers import MBConv

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  nn.ReLU(),
  MBConv(64, 64, activation=nn.ReLU())
)
```
- Multi-Headed Self-Attention <br>
*As presented in [3], this layer enables to leverage the attention mechanism for computer vision-related tasks* <br>
**Usage:**
```python
from pytorch-utils.layers import MultiHeadedSpatialAttention
from pytorch-utils.activations import Mish

model = nn.Sequential(
  nn.Conv2d(3, 128, 1),
  Mish(),
  MultiHeadedSpatialAttention(128, 128, 7, padding=3, groups=4),
  Mish(),
  MultiHeadedSpatialAttention(128, 128, 7, padding=3, groups=4),
  Mish()
)
```
- Flatten <br>
*As in Keras but lacking in PyTorch, flattens an input image to put it in the formalism of a vector*<br>
**Usage:**
```python
from pytorch-utils.layers import Flatten

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  nn.ReLU(),
  Flatten(),
  nn.Linear(128, 1)
)
```
### Activation functions
- Mish [4] <br>
*Implementation of Mish*<br>
**Usage:**
```python
from pytorch-utils.layers import Flatten
from pytorch-utils.activations import Mish

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  Mish(),
  Flatten(),
  nn.Linear(128, 1)
)
```
- Swish [5] <br>
*Implementation of Swish*<br>
**Usage:**
```python
from pytorch-utils.layers import Flatten
from pytorch-utils.activations import Swish

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  Swish(),
  Flatten(),
  nn.Linear(128, 1)
)
```

## References
- [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, *Squeeze-and-Excitation Networks*, https://arxiv.org/abs/1709.01507
- [2] Mingxing Tan, Quoc V. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, https://arxiv.org/abs/1905.11946
- [3] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jonathon Shlens, *Stand-Alone Self-Attention in Vision Models*, https://arxiv.org/abs/1906.05909
- [4] Diganta Misra, *Mish: A Self Regularized Non-Monotonic Neural Activation Function*, https://arxiv.org/abs/1908.08681
- [5] Prajit Ramachandran, Barret Zoph, Quoc V. Le, *Searching for Activation Functions*, https://arxiv.org/abs/1710.05941
