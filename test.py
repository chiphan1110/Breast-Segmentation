import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from load_data import CSAWS
from utils import *
from config import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Testing a segmentation model')
    parser.add_argument('--init_model_file', default=BEST_MODEL_DIR, help='Path to the trained model file', dest='init_model_file')
    parser.add_argument('--test_image_dir', default=TEST_IMAGE, help='Path to the test data file', dest='test_image_dir')
    parser.add_argument('--test_mask_dir', default=TEST_MASK, help='Path to the test mask file', dest='test_mask_dir')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size for testing')
    parser.add_argument('--transform', type=A.Compose, default=TEST_TRANSFORM, help='Data augmentation')
    parser.add_argument('--pred_log_dir', type=str, default=PRED_DIR, help='File to save test predictions score', dest='pred_log_dir')
    args = parser.parse_args()

    return args

def initialize_test_env(args):
    create_dir(args.pred_log_dir)
    model_name = args.init_model_file.split('/')[-1].split('.')[0]
    pred_mask_dir = os.path.join(args.pred_log_dir, model_name)
    create_dir(pred_mask_dir)
    log_file = os.path.join(pred_mask_dir, f"log_{model_name}.txt")
    return log_file, pred_mask_dir

def initialize_test_log_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("test_loss\ttest_IoU\ttest_dice\n")

def load_test_dataset(args):
    test_dataset = CSAWS(args.test_image_dir, args.test_mask_dir, args.transform)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return data_loader

def load_model(model_file):
    model = torch.load(model_file)
    return model

def test_model(model, test_loader, test_loss_meter, test_intersection_meter, test_union_meter, test_target_meter, device, criterion, pred_mask_dir):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device).float()
            masks = masks.to(device).long()
            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear').squeeze(1)
            loss = criterion(outputs, masks)

            test_loss_meter.update(loss.item())
            output_mask = outputs.argmax(1).squeeze(1)
            intersection, union, target = intersectionAndUnionGPU(output_mask.float(), masks.float(), 3)
            test_intersection_meter.update(intersection)
            test_union_meter.update(union)
            test_target_meter.update(target)

            output_mask_np = output_mask.cpu().numpy().squeeze()
            color_mask = np.zeros((*output_mask_np.shape, 3), dtype=np.uint8)
            for idx, color in enumerate(CUSTOM_COLORMAP):
                color_mask[output_mask_np == idx] = color

            mask_image = Image.fromarray(color_mask)
            mask_image.save(os.path.join(pred_mask_dir, f'{i}.png'))

    test_loss_avg = test_loss_meter.avg    
    test_iou = test_intersection_meter.sum / (test_union_meter.sum + 1e-10)
    test_dice = 2 * test_intersection_meter.sum / (test_target_meter.sum + test_union_meter.sum + 1e-10)
    test_mIoU = torch.mean(test_iou)
    test_mDice = torch.mean(test_dice)

    return test_loss_avg, test_mIoU, test_mDice 

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_log, pred_mask_dir = initialize_test_env(args)
    initialize_test_log_file(test_log)
    test_loader = load_test_dataset(args)
    model = load_model(args.init_model_file)
    criterion = nn.CrossEntropyLoss()

    test_loss_meter = AverageMeter()
    test_intersection_meter = AverageMeter()
    test_union_meter = AverageMeter()
    test_target_meter = AverageMeter()

    print('Predicting on test data...')
    test_loss_avg, test_mIoU, test_mDice  = test_model(model, test_loader, test_loss_meter, test_intersection_meter, test_union_meter, test_target_meter, device, criterion, pred_mask_dir)
    with open(test_log, 'a') as f:
        f.write(f"{test_loss_avg}\t{test_mIoU}\t{test_mDice}n")
    print(f"Test Loss: {test_loss_avg}, Test mIoU: {test_mIoU}, Test mDice: {test_mDice}")
    print("Testing complete!")

if __name__ == "__main__":
    main()