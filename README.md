Spectral Normalization for Keras
================================
The **simple** Keras implementation of ICLR 2018 paper, [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/forum?id=B1QRgziT-)

![CIFAR10 epoch 118](./generated_img_CIFAR10/icon48.png)
How to use?
----
1. Move SpectralNormalizationKeras.py in your dir
2. Import these layer class
``` python
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
```
3. Use these layers in your discriminator as usual



Issue
-----
1. Add ResNet architecture 
2. Compare with SELU and WGAN-GP
3. Projection Discriminator