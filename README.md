![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Mammogram segmentation 
## Description 
This repository contains a model for segmenting the nipple and pectoral muscle in mammograms. The model is trained on the [CSAW-S Dataset](https://github.com/ChrisMats/CSAW-S) using the [SegFormer architecture](https://github.com/NVlabs/SegFormer).

## Installation
1. **Create and activate a Conda environment**:
    ```shell
    conda env create -f environment.yml
    conda activate mammo-segmentation
    ```

2. **Install necessary packages**:
    ```shell
    # Install PyTorch for CUDA 11.3 (adjust according to your CUDA version)
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    
    # Install additional required packages
    pip install torchmetrics==1.3.2
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
    pip install openmim
    mim install mmsegmentation
    ```
## Dataset structure
Ensure your dataset directory structure resembles the following layout:
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
Adjust the dataset directory (ROOT path) in config.py. Modify preprocessing parameters as needed. Run the following commands to preprocess the data:
```shell
python preprocess_data.py  # Default: preprocessing train data
python preprocess_data.py --data_mode "test"  # For test data
```
Processing involves:
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
- For basic reproducing, just download the pretrained model weights from this link [Google Drive](https://drive.google.com/drive/folders/10Vd66VJpKvRhSyc1H1U5AZnYGztCXeMM?usp=sharing) and adjust the variable `PRETRAINED_MODEL_PATH` at `config.py` file.
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
Run the training script:
```shell
python train.py
```
Training outputs (logs and model weights) are saved in the `/output` directory inside CsawS folder.

### Testing
Adjust BEST_MODEL_DIR in config.py to point to your best model for testing and run:
```shell
python test.py
```
Test results are stored in the `output/predictions/{BEST_MODEL_NAME}/` directory.

### Inference on new images
Update `RAW_INFER_IMAGE` in `config.py` to point to the directory containing new images to predict on. Follow the code inside `visualization_test_infer.ipynb`. 
The predicted masked will be saved inside the data folder. 

### Model
The current model has been trained for 30 epochs and achieves an Intersection over Union (IOU) score of 0.75 and a Dice Score of 0.78. Model weights, mask predictions, and logs for both training and testing are available for access on [Google Drive](https://drive.google.com/drive/folders/10Vd66VJpKvRhSyc1H1U5AZnYGztCXeMM?usp=sharing).






   





