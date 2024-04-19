# Mammogram segmentation 
## Description 

## Installation
Use the environment configuration file to create a conda environment:

```shell
conda env create -f environment.yml
```
Activate the environment:
```
conda activate mammo-segmentation
```
Install the following packages: 
```shell
# torch 1.12.1 for cuda 11.3, adapt to your system accordingly
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchmetrics==1.3.2

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install openmim
mim install mmsegmentation
```
## Dataset structure

```shell
CsawS
├── anonymized_dataset
│   ├── 000 -> each folder contains original image and different masks for components
│   ├── 001
│   ├── ...
│   ├── 279
│   └── 280
├── original_images -> image used for training
└── test_data
    ├── annotator_1 -> annotations from expert 1
    ├── annotator_2 -> annotations from expert 2
    ├── annotator_3 -> annotations from expert 3
    ├── anonymized_dataset -> don't need to care
    └── original_images -> image used for testing

```
## Preprocesing data
First, the input dataset is preprocessesed by performing the following steps:
1. **Binarization**: Converts grayscale images to binary based on a threshold.
2. **Morphological Operations**: Cleans up images using specified morphological operations and dilation.
3. **Blob Selection**: Retains only the largest contiguous regions.
4. **CLAHE**: Enhances contrast using Contrast Limited Adaptive Histogram Equalization.
   





