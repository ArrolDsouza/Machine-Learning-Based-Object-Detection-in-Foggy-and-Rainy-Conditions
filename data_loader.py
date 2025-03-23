"""
Data loader module for handling multiple weather condition datasets.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.preprocessing_config import DATASET_PATHS, BATCH_SIZE
from src.preprocessing.image_processor import WeatherImageProcessor

class WeatherDataset(Dataset):
    """Custom dataset for weather-affected images."""
    
    def __init__(self, dataset_name: str, split: str = 'train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataset_name: Name of the dataset ('foggy_cityscapes', 'reside', or 'rainy_coco')
            split: Data split ('train' or 'val')
            transform: Optional transform to be applied on images
        """
        self.dataset_path = Path(DATASET_PATHS[dataset_name])
        self.split = split
        self.transform = transform
        self.processor = WeatherImageProcessor()
        
        # Load image paths and annotations
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset paths and annotations."""
        samples = []
        
        # Dataset-specific loading logic
        if self.dataset_path.name == 'foggy_cityscapes':
            img_dir = self.dataset_path / self.split / 'images'
            ann_dir = self.dataset_path / self.split / 'annotations'
            
            for img_path in img_dir.glob('*.png'):
                ann_path = ann_dir / f"{img_path.stem}.json"
                if ann_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'annotation_path': str(ann_path)
                    })
                    
        elif self.dataset_path.name == 'reside':
            img_dir = self.dataset_path / self.split
            for img_path in img_dir.glob('*.jpg'):
                # RESIDE has paired hazy and clear images
                clear_path = img_dir / 'clear' / f"{img_path.stem}_clear.jpg"
                if clear_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'clear_image_path': str(clear_path)
                    })
                    
        elif self.dataset_path.name == 'rainy_coco':
            with open(self.dataset_path / f"{self.split}.json", 'r') as f:
                annotations = json.load(f)
            
            for item in annotations['images']:
                img_path = self.dataset_path / 'images' / item['file_name']
                if img_path.exists():
                    ann_list = [ann for ann in annotations['annotations'] 
                              if ann['image_id'] == item['id']]
                    samples.append({
                        'image_path': str(img_path),
                        'annotations': ann_list
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        image = self.processor.preprocess_image(image)
        
        # Apply additional transforms if specified
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Load and process annotations based on dataset type
        if 'annotation_path' in sample:
            with open(sample['annotation_path'], 'r') as f:
                annotations = json.load(f)
            target = self._process_annotations(annotations)
        elif 'annotations' in sample:
            target = self._process_annotations(sample['annotations'])
        else:
            target = {}
            
        return image, target
    
    def _process_annotations(self, annotations):
        """Process annotations into the required format."""
        # Convert annotations to YOLO format
        # [class_id, x_center, y_center, width, height]
        processed = []
        
        if isinstance(annotations, list):
            # COCO format
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                
                # Convert to relative coordinates
                x_center = (bbox[0] + bbox[2]/2) / self.processor.image_size[0]
                y_center = (bbox[1] + bbox[3]/2) / self.processor.image_size[1]
                width = bbox[2] / self.processor.image_size[0]
                height = bbox[3] / self.processor.image_size[1]
                
                processed.append([category_id, x_center, y_center, width, height])
        else:
            # Custom format - adjust based on your annotation format
            pass
            
        return torch.tensor(processed, dtype=torch.float32)

def create_dataloaders(dataset_names, batch_size=BATCH_SIZE):
    """Create DataLoader instances for training and validation."""
    dataloaders = {}
    
    for dataset_name in dataset_names:
        for split in ['train', 'val']:
            dataset = WeatherDataset(
                dataset_name=dataset_name,
                split=split,
                transform=WeatherImageProcessor().get_augmentation_pipeline()
            )
            
            dataloaders[f"{dataset_name}_{split}"] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=4,
                pin_memory=True
            )
    
    return dataloaders 