# Image: Remove artefacts -> clahe enhancement 
import argparse
import numpy as np
import cv2
import os
from PIL import Image
import skimage
from config import *
from utils import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process image")
    parser.add_argument("--raw_data_dir", type=str, default=RAW_TRAIN_IMAGE, help="Path to the image")
    parser.add_argument("--processed_data_dir", type=str, default=TRAIN_IMAGE, help="Path to save the processed image")
    parser.add_argument("--threshold", type=int, default=THRESHOLD, help="Threshold value for binarisation")
    parser.add_argument("--maxval", type=int, default=MAXVAL, help="Maximum value for binarisation")
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE, help="Size of the kernel")
    parser.add_argument("--operation", type=str, default=OPERATION, help="Morphological operation")
    parser.add_argument("--top_x", type=int, default=TOP_X, help="Number of largest blobs to keep")
    parser.add_argument("--reverse", type=bool, default=REVERSE, help="Reverse the operation")
    parser.add_argument("--clip", type=float, default=CLIP, help="Clip limit for CLAHE")
    parser.add_argument("--tile", type=int, default=TILE, help="Tile size for CLAHE")
    
    return parser.parse_args()

def globalBinarise(img, thresh, maxval):
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    
    return binarised_img

def editMask(mask, ksize, operation):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    
    return edited_mask


def SortContoursByArea(contours, reverse=True):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours] 
    return sorted_contours, bounding_boxes


def XLargestBlobs(mask, top_x=None, reverse=True):
    contours, _ = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    
    n_contours = len(contours)
    
    if n_contours > 0:
        if n_contours < top_x or top_x == None:
            top_x = n_contours
        sorted_contours, bounding_boxes = SortContoursByArea(contours=contours,
                                                             reverse=reverse)
        X_largest_contours = sorted_contours[0:top_x]

        to_draw_on = np.zeros(mask.shape, np.uint8)

        X_largest_blobs = cv2.drawContours(image=to_draw_on, 
                                           contours=X_largest_contours, 
                                           contourIdx=-1, 
                                           color=1, 
                                           thickness=-1) 
        
    return n_contours, X_largest_blobs

def applyMask(img, mask):
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    
    return masked_img

def clahe(img, clip=2.0, tile=(8, 8)):
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def preprocess_image(args, img):   
    binarised_img = globalBinarise(img, args.threshold, args.maxval)
    edited_mask = editMask(mask=binarised_img, ksize=(args.kernel_size, args.kernel_size), operation=args.operation)
    _, xlargest_mask = XLargestBlobs(mask=edited_mask, top_x=args.top_x, reverse=args.reverse)
    masked_img = applyMask(img, mask=xlargest_mask)
    clahe_img = clahe(masked_img, clip=args.clip, tile=(args.tile, args.tile))
    processed_img = clahe_img
    return processed_img

def process_dataset(args, data_mode):

    if data_mode == "train":
        args.raw_data_dir = RAW_TRAIN_IMAGE
        args.processed_data_dir = TRAIN_IMAGE
    elif data_mode == "test":
        args.raw_data_dir = RAW_TEST_IMAGE
        args.processed_data_dir = TEST_IMAGE

    create_dir(args.processed_data_dir)
    images = [img for img in os.listdir(args.raw_data_dir) if img.endswith('.png')]
    for img_name in tqdm(images, desc="Processing Images"):
        img_path = os.path.join(args.raw_data_dir, img_name)
        img_name = img_name.split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        processed_img = preprocess_image(args, img)
        save_path = os.path.join(args.processed_data_dir, f"{img_name}.png")
        cv2.imwrite(save_path, processed_img)
    
    print(f"Preprocessing {data_mode} set done!")

def main():
    args = parse_args()
    # process_dataset(args, "train")
    process_dataset(args, "test")

if __name__ == "__main__":
    main()
    


