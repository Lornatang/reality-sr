# Training

## Train DF2K

### Dataset Preparation

DF2K: Combine the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset with the [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

**Step 1: Download the DF2K dataset and extract it to `./data`.**
    
```shell
cd scripts
bash get_df2k_datasets.sh
```

**Step 2: Download the test dataset and extract it to `./data`.**

```shell
cd scripts
# will download Set5
bash get_test_datasets.sh
```

**Step 3: Crop to sub images.**

The main purpose is to reduce the IO consumption of hard drives and speed up training.

```shell
python tools/misc/slice_sub_image.py -i data/DF2K/DF2K -o data/DF2K/DF2K_sub_image_384x384 --crop-size 384 --step 192 --thresh-size 384
```

If the machine has enough performance, you can increase the number of threads to speed up processing.

```shell
python tools/misc/slice_sub_image.py -i data/DF2K/DF2K -o data/DF2K/DF2K_sub_image_384x384 --crop-size 384 --step 192 --thresh-size 384 --num-workers 32
```

**Step 4: Download the pretrained model weights to `./results/pretrained_models`.**

```shell
wget https://github.com/Lornatang/Real_ESRGAN-PyTorch/releases/download/0.1.0/realesrnet_x4-df2k_degradation.pkl -O results/pretrained_models/realesrnet_x4-df2k_degradation.pkl
wget https://github.com/Lornatang/Real_ESRGAN-PyTorch/releases/download/0.1.0/realesrgan_x4-df2k_degradation.pkl -O results/pretrained_models/realesrgan_x4-df2k_degradation.pkl
```

### Train

#### Finetune (Recommended)

**Step 1: Finetune the realesrnet_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrnet_x4_degradation-finetune.yaml
# Results will be saved in `./results/train/realesrnet_x4_degradation-finetune`
```

**Step 2: Finetune the realesrgan_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrgan_x4_degradation-finetune.yaml
# Results will be saved in `./results/train/realesrgan_x4_degradation-finetune`
```

#### Train from scratch

**Step 1: train the realesrnet_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrnet_x4_degradation.yaml
# Results will be saved in `./results/train/realesrnet_x4_degradation`
```

**Step 2: train the realesrgan_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrgan_x4_degradation.yaml
# Results will be saved in `./results/train/realesrgan_x4_degradation`
```

## Train custom dataset

### Dataset Preparation

**Step 1: Put the training data in the `./data` directory.**

The data structure is as follows:

```shell
|---data
    |---custom
        |---gt
            |---0001.png
            |---0002.png
            |---...
        |---gt_sub_image_384x384
            |---0001-0001_0001.png
            |---0001-021_0301.png
            |---...
```

**Step 2: Crop to sub images.**

The main purpose is to reduce the IO consumption of hard drives and speed up training.

```shell
python tools/misc/slice_sub_image.py -i data/custom/gt -o data/custom/gt_sub_image_384x384 --crop-size 384 --step 192 --thresh-size 384
```

If the machine has enough performance, you can increase the number of threads to speed up processing.

```shell
python tools/misc/slice_sub_image.py -i data/custom/gt -o data/custom/gt_sub_image_384x384 --crop-size 384 --step 192 --thresh-size 384 --num-workers 32
```

**Step 3: Download the test dataset and extract it to `./data`.**

```shell
cd scripts
# will download Set5
bash get_test_datasets.sh
```

### Train

#### Finetune (Recommended)

**Step 1: Change realesrnet_x4 yaml.**

Find the `realesrnet_x4_degradation-finetune.yaml` file in the `configs/Real_ESRGAN/realesrnet_x4_degradation-finetune.yaml` directory and follower modify.

```shell
# line 48
TRAIN_GT_IMAGES_DIR: "data/custom/gt_sub_image_384x384"  # 178574 images
```

**Step 2: Finetune the realesrnet_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrnet_x4_degradation-finetune.yaml
# Results will be saved in `./results/train/realesrnet_x4_degradation-finetune`
```

**Step 3: Change realesrgan_x4 yaml.**

Find the `realesrgan_x4_degradation-finetune.yaml` file in the `configs/Real_ESRGAN/realesrgan_x4_degradation-finetune.yaml` directory and follower modify.

```shell
# line 48
TRAIN_GT_IMAGES_DIR: "data/custom/gt_sub_image_384x384"  # 178574 images
```

**Step 4: Finetune the realesrgan_x4 model.**

```shell
python tools/train.py configs/Real_ESRGAN/realesrgan_x4_degradation-finetune.yaml
# Results will be saved in `./results/train/realesrgan_x4_degradation-finetune`
```