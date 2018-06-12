Spectral Normalization for Keras
================================
The **simple** Keras implementation of ICLR 2018 paper, [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/forum?id=B1QRgziT-)

**CIFAR10 epoch 245 (ResNet architecture)**
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/generated_img_CIFAR10_ResNet/SN_epoch_245.png)

**CIFAR10 epoch 245 (DCGAN architecture)**
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/generated_img_CIFAR10_DCGAN/SN_epoch_245.png)

How to use?
----
1. Move SpectralNormalizationKeras.py in your dir
2. Import these layer class
``` python
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
```
3. Use these layers in your discriminator as usual

Example notebook
------
[CIFAR10 with DCGAN architecture](http://nbviewer.jupyter.org/github/ishengfang/SpectralNormalizationKeras/blob/master/CIFAR10%28DCGAN%20Structure%29.ipynb)
[CIFAR10 with ResNet architecture](http://nbviewer.jupyter.org/github/ishengfang/SpectralNormalizationKeras/blob/master/CIFAR10%28ResNet%29.ipynb)

Model Detail
-------------------------

### Architecture
### DCGAN 
#### Generator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/DCGAN_Generator.png)
#### Discriminator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/DCGAN_Generator.png)
### ResNet GAN
#### Generator 
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/ResNet_Generator.png)
##### Generator UpSampling ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Generator_resblock_1.png)
#### Dicriminator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/ResNet_Dicriminator.png)
##### Discriminator DownSampling ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Discriminator_resblock_Down_1.png)
##### Discriminator ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Discriminator_resblock_1.png)

Issue
-----
1. Compare with SELU and WGAN-GP
2. Projection Discriminator
