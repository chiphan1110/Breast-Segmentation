import os
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

def argument_parser():
    parser = argparse.ArgumentParser(description='Testing a segmentation model')
    parser.add_argument('--init_model_file', default=BEST_MODEL_DIR, help='Path to the trained model file', dest='init_model_file')
    parser.add_argument('--test_image_dir', default=TEST_IMAGE, help='Path to the test data file', dest='test_data_dir')
    parser.add_argument('--test_mask_dir', default=TEST_MASK, help='Path to the test mask file', dest='mask_dir')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size for testing')
    parser.add_argument('--transform', type=A.Compose, default=TEST_TRANSFORM, help='Data augmentation')
    parser.add_argument('--pred_log_dir', type=str, default=PRED_DIR, help='File to save test predictions score', dest='pred_log_dir')
    
    return parser.parse_args()

def initialize_test_env(args):
    create_dir(args.pred_log_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = os.path.join(args.pred_log_dir, f"log_{current_time}.txt")
    return log_file

def initialize_test_log_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("test_loss\ttest_IoU\ttest_dice\n")

def load_test_dataset(args):
    test_dataset = CSAWS(args.test_image_dir, args.test_mask_dir, args.transform)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return data_loader

def load_model(model_file, device):
    model = torch.load(model_file)
    return model

def test_model(model, test_loader, test_loss_meter, test_intersection_meter, test_union_meter, test_target_meter, device, criterion):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device).float()
            masks = masks.to(device).long()
            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear').squeeze(1)
            loss = criterion(outputs, masks)

            predictions.append(outputs)
            test_loss_meter.update(loss.item())
            output_mask = outputs.argmax(1).squeeze(1)
            intersection, union, target = intersectionAndUnionGPU(output_mask.float(), masks.float(), 3)
            test_intersection_meter.update(intersection)
            test_union_meter.update(union)
            test_target_meter.update(target)

    test_loss_avg = test_loss_meter.avg    
    test_iou = test_intersection_meter.sum / (test_union_meter.sum + 1e-10)
    test_dice = 2 * test_intersection_meter.sum / (test_target_meter.sum + test_union_meter.sum + 1e-10)
    test_mIoU = torch.mean(test_iou)
    test_mDice = torch.mean(test_dice)

    return predictions, test_loss_avg, test_mIoU, test_mDice 


def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_log = initialize_test_env(args)
    initialize_test_log_file(test_log)
    test_loader = load_test_dataset(args)
    model = load_model(args.init_model_file, device)
    criterion = nn.CrossEntropyLoss()

    test_loss_meter = AverageMeter()
    test_intersection_meter = AverageMeter()
    test_union_meter = AverageMeter()
    test_target_meter = AverageMeter()

    print('Predicting on test data...')
    predictions, test_loss_avg, test_mIoU, test_mDice  = test_model(model, test_loader, test_loss_meter, test_intersection_meter, test_union_meter, test_target_meter, device, criterion, test_log)
    with open(test_log, 'a') as f:
        f.write(f"{test_loss_avg}\t{test_mIoU}\t{test_mDice}n")


if __name__ == "__main__":
    main()