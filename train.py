import os
from datetime import datetime
import albumentations as A
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from tqdm import tqdm
from load_data import CSAWS
from config import *
from utils import *
from model import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train model for image segmentation")
    parser.add_argument("--image_dir", type=str, default=TRAIN_IMAGE, help="Path to the image directory")
    parser.add_argument("--mask_dir", type=str, default=TRAIN_MASK, help="Path to the mask directory")
    parser.add_argument("--transform", type=A.Compose, default=TRAIN_TRANSFORM, help="Data augmentation")
    parser.add_argument("--val_fraction", type=float, default=VAL_FRACTION, help="Fraction of the dataset to use for validation")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=STEP_SIZE, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Gamma for learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=EARLY_STOPPING, help="Number of epochs to wait before early stopping")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to the model directory")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Path to the log directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    return args

def initialize_training_env(args):
    create_dir(args.model_dir)
    create_dir(args.log_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = os.path.join(args.log_dir, f"log_{current_time}.txt")
    return log_file

def initialize_train_log_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("epoch\ttrain_loss\ttrain_IoU\ttrain_Dice\tval_loss\tval_IoU\tval_Dice\n")

def load_dataset(args):
    print("Loading dataset...")

    full_dataset = CSAWS(args.image_dir, args.mask_dir, args.transform)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=args.val_fraction, random_state=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

def train_one_epoch(model, device, trainloader, optimizer, criterion, meters, scheduler):
    model.train()
    train_loss_meter, intersection_meter, union_meter, target_meter = meters
    train_loss_meter.reset()
    intersection_meter.reset()
    union_meter.reset()
    target_meter.reset()
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc="Training")

    for batch_id, (x, y) in progress_bar:
        x = x.to(device).float()
        y = y.to(device).long()
        preds = model(x)
        preds = F.interpolate(preds, size=y.shape[1:], mode='bilinear')
        loss = criterion(preds, y)
        loss.backward()
        optimizer.zero_grad()
        with torch.no_grad():
            train_loss_meter.update(loss.item())
            preds_mask = preds.argmax(1).squeeze(1)
            intersection, union, target = intersectionAndUnionGPU(preds_mask.float(), y.float(), 3)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
    
    train_iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    train_dice = 2 * intersection_meter.sum / (target_meter.sum + union_meter.sum + 1e-10)
    train_mIoU = torch.mean(train_iou)
    train_mDice = torch.mean(train_dice)
    
    scheduler.step()
    return train_loss_meter.avg, train_mIoU, train_mDice

def validate(model, device, valloader, criterion, meters):
    model.eval()
    val_loss_meter, intersection_meter, union_meter, target_meter = meters
    val_loss_meter.reset()
    intersection_meter.reset()
    union_meter.reset()
    target_meter.reset()
    progress_bar = tqdm(enumerate(valloader), total=len(valloader), desc="Validation")

    with torch.no_grad():
        for batch_id, (x, y) in progress_bar:
            x = x.to(device).float()
            y = y.to(device).long()
            preds = model(x)
            preds = F.interpolate(preds, size=y.shape[1:], mode='bilinear')
            loss = criterion(preds, y)

            val_loss_meter.update(loss.item())
            preds_mask = preds.argmax(1).squeeze(1)
            intersection, union, target = intersectionAndUnionGPU(preds_mask.float(), y.float(), 3)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
    
    val_iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    val_dice = 2 * intersection_meter.sum / (target_meter.sum + union_meter.sum + 1e-10)
    val_mIoU = torch.mean(val_iou)
    val_mDice = torch.mean(val_dice)

    return val_loss_meter.avg, val_mIoU, val_mDice

def train_model(args, model, device, trainloader, valloader, optimizer, criterion, scheduler):
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M%_S")
    log_file = initialize_training_env(args)
    initialize_train_log_file(log_file)

    train_loss_meter = AverageMeter()
    train_intersection_meter = AverageMeter()
    train_union_meter = AverageMeter()
    train_target_meter = AverageMeter()

    val_loss_meter = AverageMeter()
    val_intersection_meter = AverageMeter()
    val_union_meter = AverageMeter()
    val_target_meter = AverageMeter()

    train_meters = (train_loss_meter, train_intersection_meter, train_union_meter, train_target_meter)
    val_meters = (val_loss_meter, val_intersection_meter, val_union_meter, val_target_meter)

    early_stopping = args.early_stopping
    best_val_loss = float('inf')
    no_improvement_ep = 0

    for epoch in range(1, args.epochs + 1):
        print(f"### Epoch {epoch} ###")
        train_loss, train_mIoU, train_mDice = train_one_epoch(model, device, trainloader, optimizer, criterion, train_meters, scheduler)
        val_loss, val_mIoU, val_mDice = validate(model, device, valloader, criterion, val_meters)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_ep = 0
        else:
            no_improvement_ep += 1
        
        if no_improvement_ep > early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Train Loss: {train_loss:.6f}, Train IoU: {train_mIoU:.6f}, Train Dice: {train_mDice:.6f}")
        print(f"Val Loss: {val_loss:.6f}, Val IoU: {val_mIoU:.6f}, Val Dice: {val_mDice:.6f}")

        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{train_mIoU:.6f}\t{train_mDice:.6f}\t{val_loss:.6f}\t{val_mIoU:.6f}\t{val_mDice:.6f}\n")
    
    if epoch % 10 == 0 or epoch == args.epochs:
        final_model_file = os.path.join(args.model_dir, f"model_{epoch}_{current_time}.pth")
        save_model(model, final_model_file)
        print("Training complete!")

def main():
    args = parse_args()
    train_loader, val_loader = load_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device) 
    model.init_weights()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_model(args, model, device, train_loader, val_loader, optimizer, criterion, scheduler)

if __name__ == "__main__":
    main()