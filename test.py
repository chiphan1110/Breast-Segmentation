import os
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from datetime import datetime
import numpy as np
from load_data import CSAWS
from utils import *
from config import *
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Test a UNET model')
    parser.add_argument('--init_model_file', default=BEST_MODEL_DIR, help='Path to the trained model file', dest='init_model_file')
    parser.add_argument('--test_data_dir', default=SAVE_TEST_PATH, help='Path to the test data file', dest='test_data_dir')
    parser.add_argument('--label_list', type=list, default=CLASSES, help='List of labels', dest='label_list')
    parser.add_argument('--single_label', type=str, default=SINGLE_LABEL, help="Single label")

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size for testing')
    # parser.add_argument('--transform', type=transforms.Compose, default=TRANSFORM, help='Data augmentation')
    parser.add_argument('--pred_log_dir', type=str, default=PRED_DIR, help='File to save test predictions score', dest='pred_log_dir')
    
    return parser.parse_args()

def initialize_test_env(args):
    create_dir(args.pred_log_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = os.path.join(args.pred_log_dir, f"log_{current_time}.txt")
    return log_file

def load_test_dataset(args):
    test_dataset = CSAWS(args.test_data_dir, args.label_list, args.single_label)
    return test_dataset

def load_model(model_file, device):
    model = torch.load(model_file)
    return model

def test_model(best_model, test_loader, loss, metrics, device, test_log):
    test_epoch = smp.utils.train.ValidEpoch(
        best_model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
    )

    test_logs = test_epoch.run(test_loader)
    with open(test_log, 'a') as f:
            f.write(f"{test_logs['dice_loss']:.4f}\t{test_logs['iou_score']:.4f}\n")
    

def visualize_pred(dataset, model, device, num_samples=5):
    model.eval()
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, gt_mask = dataset[idx]

        if torch.is_tensor(image):
            image = to_pil_image(image)
        if torch.is_tensor(gt_mask):
            gt_mask = to_pil_image(gt_mask)

        if dataset.transform is not None:
            x_tensor = dataset.transform(image).unsqueeze(0).to(device)
        else:
            x_tensor = to_pil_image.to_tensor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pr_mask = model(x_tensor)
            pr_mask = pr_mask.squeeze().cpu()
        
        if torch.is_tensor(pr_mask):
            pr_mask = to_pil_image(pr_mask)
        
        visualize(image, gt_mask, pr_mask)

def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_log = initialize_test_env(args)
    initialize_test_log_file(test_log)
    test_dataset = load_test_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = utils.losses.DiceLoss()
    metrics = [utils.metrics.IoU(threshold=0.5),]
    model = torch.load(args.init_model_file)

    print('Predicting on test data...')
    test_model(model, test_loader, criterion, metrics, device, test_log)

    visualize_pred(test_dataset, model, device)


if __name__ == "__main__":
    main()