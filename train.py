import os
from datetime import datetime
import albumentations as A
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader
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


        
def load_dataset(args):
    print("Loading dataset...")

    full_dataset = CSAWS(args.image_dir, args.mask_dir, args.transform)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=args.val_fraction, random_state=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, device, trainloader, valloader, optimizer, criterion, scheduler, n_epochs, early_stopping):
    train_loss_meter = AverageMeter()
    train_intersection_meter = AverageMeter()
    train_union_meter = AverageMeter()
    train_target_meter = AverageMeter()

    val_loss_meter = AverageMeter()
    val_intersection_meter = AverageMeter()
    val_union_meter = AverageMeter()
    val_target_meter = AverageMeter()

    early_stopping = args.early_stopping
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M%_S")
    log_file = initialize_training_env(args)
    initialize_train_log_file(log_file)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement > early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1,}\t{train_logs['dice_loss']:.6f}\t{train_logs['iou_score']:.6f}\t{valid_logs['dice_loss']:.6f}\t{valid_logs['iou_score']:.6f}\n")
    
    final_model_file = os.path.join(args.model_dir, f"model_{args.single_label}_{current_time}.pth")
    save_model(model, final_model_file)
    print("Training complete!")

def main():
    args = parse_args()
    train_loader, val_loader = load_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device) 
    model.init_weights()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.paramters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_model(model, device, train_loader, val_loader, optimizer, criterion, scheduler, args.epochs, args.early_stopping)

if __name__ == "__main__":
    main()