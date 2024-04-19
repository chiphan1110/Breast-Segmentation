![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Mammogram segmentation 
## Description 
Segmentation model to segment nipple and pectoral muscle in mammograms. 
Model are trained on [CSAW-S Dataset](https://github.com/ChrisMats/CSAW-S) using [SegFormer](https://github.com/NVlabs/SegFormer). 
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
├── original_images 
└── test_data
    ├── annotator_1 -> annotations from expert 1
    ├── annotator_2 -> annotations from expert 2
    ├── annotator_3 -> annotations from expert 3
    ├── anonymized_dataset 
    └── original_images 

```
## Data Preprocessing
First, adjust the dataset directory by changing the `ROOT` path at `config.py`. Adjust preprocessing params if needed. Then run:
```shell
python preprocess_data.py # default is preprocessing train data
python preprocessing_data.py --data_mode "test"
```
The above command with conduct preprocessing for train set and test set by performing the following steps:
1. **Binarization**: Converts grayscale images to binary based on a threshold.
2. **Morphological Operations**: Cleans up images using specified morphological operations and dilation.
3. **Blob Selection**: Retains only the largest contiguous regions.
4. **CLAHE**: Enhances contrast using Contrast Limited Adaptive Histogram Equalization.

After preprocessing, the data directory will have structure like this:
```shell
CsawS
├── anonymized_dataset
│   ├── 000 
│   ├── 001
│   ├── ...
│   ├── 279
│   └── 280
├── original_images 
├── test_data
│   ├── annotator_1 -> I used this annotation as grouthtruth mask for test set
│   ├── annotator_2 
│   ├── annotator_3 
│   ├── anonymized_dataset 
│   ├── original_images 
│   └── test_images -> image used for testing
└── train_images -> image used for testing
```
## Training
### Loading pretrained model:
- For basic reproducing, just download the pretrained model weights from this link and adjust the variable `PRETRAINED_MODEL_PATH` at `config.py` file.
- For exploring more pretrained models, follow these steps:
  - Download weights pretrained on ImageNet-1K used for Training given by [SegFormer repo](https://github.com/open-mmlab/mmsegmentation.git), or by accessing this link directly [[onedrive]](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ). 
  - Install MMSegmentation toolbox:
    ```shell
    git clone -b main https://github.com/open-mmlab/mmsegmentation.git
    ```
  - Convert SegFormer pretrained model to MMSegmentation style:
    ```shell
    python mmsegmentation/tools/model_converters/mit2mmseg.py ${SegFormer_PRETRAINED_PATH} ${MMSeg_CONVERTED_PATH}
    ```
  - Note:
    - MMSeg_CONVERTED_PATH will later be used as pretrained model path for training. Please adjust the adjust the variable `PRETRAINED_MODEL_PATH` at `config.py` file by this value. 
    - Further information for SegFormer configuration can be found at this [link](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer).
### Training model:
Run train code:
```shell
python train.py
```
After training, the training logs and model weights are saved to the `/output` folder in the data directory like this:
```shell
CsawS
├── anonymized_dataset
│   ├── 000 
│   ├── 001
│   ├── ...
│   ├── 279
│   └── 280
├── original_images 
├── output
│   ├── logs -> train and validation log
│   ├── models -> model 
├── test_data
│   ├── annotator_1 
│   ├── annotator_2 
│   ├── annotator_3 
│   ├── anonymized_dataset 
│   ├── original_images 
│   └── test_images 
└── train_images 
```

### Testing
Adjust `BEST_MODEL_DIR` in `config.py` file according to path to model that is wanted to test on and run:
```shell
python test.py
```
The test log and prediction mask are saved at folder `output\predictions\{BEST_MODEL_NAME}\`

### Inference on new images
Adjust `RAW_INFER_IMAGE` in `config.py` file according to the folder that contain images that model will predict on. 
Follow the code inside `visualization_test_infer`. 
The predicted masked will be saved inside the data folder. 





   





