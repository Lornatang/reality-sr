# reality-sr
Real world super resolution library

## Train

### Introduction

We follow the standard GAN training process, which is a two-stage training. The first stage trains the generative model based on PSNR loss, and the second stage trains the generative model and adversarial model based on PSNR loss, perceptual loss and GAN loss.

### Data Preparation

Usually, we recommend using the df2k dataset, which is a combined dataset consisting of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)+[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).

Since we use the degradation method, only the GT(Ground-Truth) image is needed.

#### 1- Download df2k dataset

```bash
mkdir -p ./datasets/df2k
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -O ./datasets/DIV2K_train_HR.zip
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar -O ./datasets/Flickr2K.tar
unzip ./datasets/DIV2K_train_HR.zip -d ./datasets
tar -xf ./datasets/Flickr2K.tar -C ./datasets
```

#### 2- Multi-scale image generation

Multi-scale image training helps the model learn features at different scales, thereby showing better generalization in real-world scenarios.

```bash
python3 ./tools/misc/generate_multi_scale_dataset.py -i ./datasets/DIV2K_train_HR -o ./datasets/df2k
python3 ./tools/misc/generate_multi_scale_dataset.py -i ./datasets/Flickr2K -o ./datasets/df2k
```

#### 3- Combine datasets

Fusion of multi-scale datasets and original datasets to further improve model generalization.

```bash
mv ./datasets/DIV2K_train_HR/* ./datasets/df2k
mv ./datasets/Flickr2K/* ./datasets/df2k
```