"""
Script to prepare and organize the training data.
"""

import os
import shutil
from pathlib import Path
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

def setup_directory_structure():
    """Create the necessary directory structure."""
    base_dir = Path('data/foggy_cityscapes')
    dirs = [
        base_dir / 'train' / 'images',
        base_dir / 'train' / 'annotations',
        base_dir / 'val' / 'images',
        base_dir / 'val' / 'annotations'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def organize_dataset(source_dir: str, train_split: float = 0.8):
    """
    Organize the dataset by splitting into train and validation sets.
    
    Args:
        source_dir: Path to the source directory containing foggy images
        train_split: Fraction of images to use for training
    """
    source_path = Path(source_dir)
    target_path = Path('data/foggy_cityscapes')
    
    # Get all image files
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {source_dir}")
        return
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process training images
    print("\nProcessing training images...")
    process_image_set(train_files, target_path / 'train', 'train')
    
    # Process validation images
    print("\nProcessing validation images...")
    process_image_set(val_files, target_path / 'val', 'val')

def process_image_set(image_files, target_dir, split):
    """Process a set of images and create annotations."""
    for img_file in tqdm(image_files, desc=f"Processing {split} set"):
        # Read and process image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to read image: {img_file}")
            continue
            
        # Save image
        target_img_path = target_dir / 'images' / img_file.name
        cv2.imwrite(str(target_img_path), img)
        
        # Create simple annotation (placeholder - you should replace with actual annotations)
        annotation = create_placeholder_annotation(img)
        
        # Save annotation
        annotation_path = target_dir / 'annotations' / f"{img_file.stem}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)

def create_placeholder_annotation(img):
    """
    Create a placeholder annotation for an image.
    In a real scenario, this should be replaced with actual object detection annotations.
    """
    height, width = img.shape[:2]
    
    # Example annotation format (similar to COCO)
    annotation = {
        'image': {
            'height': height,
            'width': width,
        },
        'annotations': []
    }
    
    return annotation

def main():
    """Main function to prepare the dataset."""
    print("Setting up directory structure...")
    setup_directory_structure()
    
    source_dir = r"C:\Users\ARROL\Desktop\foggy_img"
    print(f"\nOrganizing dataset from: {source_dir}")
    organize_dataset(source_dir)
    
    print("\nDataset preparation complete!")
    print("Please check the data/foggy_cityscapes directory for the organized dataset.")

if __name__ == '__main__':
    main() 