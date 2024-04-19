# Mammogram segmentation 
## Description 

## Installation
Use the environment configuration file to create a conda environment:

```shell
conda env create -f environment.yml
```
Install the following packages:
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchmetrics==1.3.2
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install openmim
mim install mmsegmentation
```

Activate the environment:

