import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from config import *

class CSAWS(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __combined_mask__(self, img_name):
        img_name = img_name.split('.')[0]
        mask_path = os.path.join(self.mask_dir, img_name.split('_')[0])
        mask_nipple_name = os.path.join(mask_path, img_name + "_nipple.png")
        mask_pectoral_muscle_name = os.path.join(mask_path, img_name + "_pectoral_muscle.png")

        mask_nipple = Image.open(mask_nipple_name).convert('L')
        mask_pectoral_muscle = Image.open(mask_pectoral_muscle_name).convert('L')

        mask_nipple = np.array(mask_nipple) > 0
        mask_pectoral_muscle = np.array(mask_pectoral_muscle) > 0

        
        h, w = mask_nipple.shape
        combined_mask = np.zeros((h, w, 3), dtype=np.uint8)  # 3 classes

        combined_mask[:, :, 0] = ~(mask_nipple | mask_pectoral_muscle)
        combined_mask[:, :, 1] = mask_nipple
        combined_mask[:, :, 2] = mask_pectoral_muscle
        combined_mask = combined_mask.astype(np.float32)

        return combined_mask   
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = self.__combined_mask__(img_name)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return img, mask.argmax(dim=2).squeeze()


class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_name).convert("L")  
        original_size = img.size
        if self.transform:
            image = self.transform(img)
        return image, original_size


if __name__ == "__main__":
    # Example usage
    dataset = CSAWS(TRAIN_IMAGE, TRAIN_MASK, transform=TRAIN_TRANSFORM)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for data in data_loader:
        images, masks = data[0], data[1]
        print(images.shape)   
        print(masks.shape) 
        break   
