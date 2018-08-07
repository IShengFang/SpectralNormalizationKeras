Spectral Normalization for Keras
================================
The **simple** Keras implementation of ICLR 2018 paper, [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/forum?id=B1QRgziT-)

Result
-----------------------------
### CIFAR10
#### DCGAN architecture

##### with Spectral Normalization

**100 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_with_SN/epoch_099.png)

**200 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_with_SN/epoch_199.png)

**300 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_with_SN/epoch_299.png)

##### with Gradeint Penalty

**100 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_without_SN/epoch_099.png)

**200 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_without_SN/epoch_199.png)

**300 epoch**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_without_SN/epoch_299.png)


#### ResNet architecture

##### with Spectral Normalization

**epoch 100**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_with_SN/epoch_099.png)

**epoch 200**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_with_SN/epoch_199.png)


**epoch 300**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_with_SN/epoch_299.png)

##### with Gradeint Penalty

**epoch 100**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_without_SN/epoch_099.png)

**epoch 200**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_without_SN/epoch_199.png)

**epoch 300**

![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_without_SN/epoch_299.png)



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
[CIFAR10 with DCGAN architecture](http://nbviewer.jupyter.org/github/ishengfang/SpectralNormalizationKeras/blob/master/CIFAR10%28DCGAN%29.ipynb)

[CIFAR10 with ResNet architecture](http://nbviewer.jupyter.org/github/ishengfang/SpectralNormalizationKeras/blob/master/CIFAR10%28ResNet%29.ipynb)

Model Detail
-------------------------

### Architecture
### DCGAN 
#### Generator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/DCGAN_Generator.png)
#### Discriminator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/DCGAN_Discriminator.png)
### ResNet GAN
#### Generator 
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/ResNet_Generator.png)
##### Generator UpSampling ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Generator_resblock_1.png)
#### Dicriminator
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/ResNet_Discriminator.png)
##### Discriminator DownSampling ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Discriminator_resblock_Down_1.png)
##### Discriminator ResBlock
![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/model/Discriminator_resblock_1.png)

Issue
-----
- [x] Compare with WGAN-GP
- [ ] Projection Discriminator

Acknowledgment
-----
- Thank @anshkapil pointed out and @IFeelBloated corrected this implementation.