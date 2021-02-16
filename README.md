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

- Weight Standardization [8] <br>
*Possible improvement of batch normalization for small and big batches when used in coordination with Group Normalization [7], as suggested in [9].*<br>
**Usage:**
```python
from pytorch-utils.layers import WSConv2d

model = nn.Sequential(
  WSConv2d(1, 64, 3, padding=1),
  nn.GroupNorm(64),
  nn.ReLU(),
  Flatten(),
  nn.Linear(128, 1)
)
```

- Scaled Weight Standardization [10] <br>
*Used to make sure the output of a convolutional layer follows a centered reduced gaussian distribution when the input does too. It is the first step to a network Batch-Normalization-free.*<br>
**Usage:**
```python
from pytorch-utils.layers import SWSConv2d

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(),
  SWSConv2d(1, 64, 3, padding=1, activation="relu"), # you have to specify the preceding activation
  nn.ReLU(),
  Flatten(),
  nn.Linear(128, 1)
)
```

- Parameter-Free Layer Normalization <br>
*In the case of images, one cannot usually afford to learn HW weights for each normalization layer. We propose a parameter-free layer normalization layer easier to instantiate than PyTorch's original one.*<br>
**Usage:**
```python
from pytorch-utils.layers import PFLayerNorm

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  PFLayerNorm(),
  nn.ReLU()
)
```

- Layer Normalization parametrized along channels <br>
*We propose this mix of Parameter-Free Layer Normalization with the parametrization of Batch Normalization, where the weights are learned along the channel axis.*<br>
**Usage:**
```python
from pytorch-utils.layers import CLayerNorm

model = nn.Sequential(
  nn.Conv2d(1, 64, 3, padding=1),
  CLayerNorm(64),
  nn.ReLU()
)
```

- CoordConv solution <br>
*As presented in [6], this layer proposes to concatenate channels containing the position of each pixel to a given feature map, and to perform a normal convolution on top of that. This allows for tasks where the position is important to be performed with a CNN.*<br>
**Usage:**
```python
from pytorch-utils.layers import CoordConv2d

model = nn.Sequential(
  CoordConv(1, 64, 3, padding=1),
  nn.ReLU(),
  Flatten(),
  nn.Linear(128, 1)
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
- [6] Rosanne Liu, Joel Lehman, Piero Molino, Felipe Petroski Such, Eric Frank, Alex Sergeev, Jason Yosinski, *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, https://arxiv.org/abs/1807.03247
- [7] Yuxin Wu, Kaiming He, *Group normalization*, Proceedings of the European conference on computer vision (ECCV), pages 3-19, 2018
- [8] Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, Alan Yuille, *Micro-Batch Training with Batch-Channel Normalization and Weight Standardization, https://arxiv.org/abs/1903.10520
- [9] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby, *Big Transfer (BiT): General Visual Representation Learning*, https://arxiv.org/abs/1912.11370
- [10] Andrew Brock, Soham De, Samuel L. Smith, *Characterizing signal propagation to close the performance gap in unnormalized ResNets*, https://arxiv.org/abs/2101.08692