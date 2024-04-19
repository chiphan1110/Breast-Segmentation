import os
import albumentations as A
from albumentations.pytorch import ToTensorV2 

# Hardware Configuration 
CUDA_VISIBLE_DEVICES = "0"

# Path
ROOT = "/home/phanthc/Chi/Data/CsawS/"

RAW_TRAIN_IMAGE= os.path.join(ROOT, "original_images/")
TRAIN_IMAGE = os.path.join(ROOT, "train_images/")
TRAIN_MASK = os.path.join(ROOT, "anonymized_dataset/")

TEST_ROOT = os.path.join(ROOT, "test_data/")
RAW_TEST_IMAGE= os.path.join(TEST_ROOT, "original_images/")
TEST_IMAGE= os.path.join(TEST_ROOT, "test_images/")
TEST_MASK = os.path.join(TEST_ROOT, "annotator_1/")

OUTPUT_DIR = os.path.join(ROOT, "output/")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models/")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs/")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions/")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model.pth")
BEST_MODEL_NAME = BEST_MODEL_DIR.split('/')[-1].split('.')[0]

RAW_INFER_IMAGE= os.path.join(TEST_ROOT, "inference/raw/")
INFER_IMAGE= os.path.join(TEST_ROOT, "inference/preprocessed/")
INFER_MASK_PRED = os.path.join(TEST_ROOT, "inference/", BEST_MODEL_NAME + "/")

PRETRAINED_MODEL_PATH = "/home/phanthc/Chi/Code/pretrained/mit_b4_mmseg.pth"

# Data Preprocessing 
THRESHOLD = 0.1
MAXVAL = 1.0
KERNEL_SIZE = 23
OPERATION = "open"
TOP_X = 1
REVERSE = True
CLIP = 2.0
TILE = 8


# Data Augmentation
INPUT_SIZE = 512

TRAIN_TRANSFORM = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean = (0.485), std = (0.229)),
    A.ToRGB(),
    ToTensorV2()
])
VAL_TRANSFORM = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean = (0.485), std = (0.229)),
    A.ToRGB(),
    ToTensorV2()
])
TEST_TRANSFORM = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean = (0.485), std = (0.229)),
    A.ToRGB(),
    ToTensorV2()
])

# Dataset Params
VAL_FRACTION = 0.2
SEED = 42

# Training 
N_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
STEP_SIZE = 20
GAMMA = 0.5
EARLY_STOPPING = 10

# Visualization
CUSTOM_COLORMAP = [
    (0, 0, 0),  # Background
    (255, 0, 0),  # Nipple
    (0, 0, 255)  # Pectoral muscle
]


