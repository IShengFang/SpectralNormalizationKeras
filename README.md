Spectral Normalization for Keras
================================
The **simple** Keras implementation of ICLR 2018 paper, Spectral Normalization for Generative Adversarial Networks.
[[openreview]](https://openreview.net/forum?id=B1QRgziT-)[[arixiv]](https://arxiv.org/abs/1802.05957)[[original code(chainer)]](https://github.com/pfnet-research/sngan_projection)

[[Hackmd]](https://hackmd.io/s/BkW34Lje7#)[[github]](https://github.com/IShengFang/SpectralNormalizationKeras)

Result
-----------------------------
### CIFAR10
#### DCGAN architecture

| 10epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_009.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_009.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_009.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_009.png)|

| 100epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_099.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_099.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_099.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_099.png)|

| 200epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_199.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_199.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_199.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_199.png)|

| 300epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_299.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_299.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_299.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_299.png)|

| 400epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_399.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_399.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_399.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_399.png)|

| 500epoch | with SN |without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/epoch_499.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/epoch_499.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/epoch_499.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/epoch_499.png)|

| Loss | with SN |without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_GP/loss.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_GP/loss.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_SN_noGP/loss.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_dcgan_noSN_noGP/loss.png)|

#### ResNet architecture

| 10epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_009.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_009.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_009.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_009.png)|

| 100epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_099.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_099.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_099.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_099.png)|

| 200epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_199.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_199.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_199.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_199.png)|

| 300epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_299.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_299.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_299.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_299.png)|

| 400epoch | With SN |Without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_399.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_399.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_399.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_399.png)|

| 500epoch | with SN |without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/epoch_499.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/epoch_499.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/epoch_499.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/epoch_499.png)|

| Loss | with SN |without SN |
|:-------:|:-------:|:---------:|
|**With GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_GP/loss.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_GP/loss.png)|
|**Without GP**|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_SN_noGP/loss.png)|![](https://raw.githubusercontent.com/IShengFang/SpectralNormalizationKeras/master/img/generated_img_CIFAR10_resnet_noSN_noGP/loss.png)|

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
