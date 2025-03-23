"""
Image preprocessing module for handling foggy and rainy conditions.
Implements various enhancement and augmentation techniques.
"""

import cv2
import numpy as np
import torch
import albumentations as A
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.preprocessing_config import (
    IMAGE_SIZE,
    AUGMENTATION_PARAMS,
    PREPROCESSING_STEPS
)

class WeatherImageProcessor:
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.aug_params = AUGMENTATION_PARAMS
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Processed image
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        for step in PREPROCESSING_STEPS:
            if hasattr(self, step):
                image = getattr(self, step)(image)
        return image
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.image_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        return image
    
    def dehaze(self, image: np.ndarray) -> np.ndarray:
        """
        Remove haze using Dark Channel Prior algorithm.
        Based on He et al.'s paper "Single Image Haze Removal Using Dark Channel Prior"
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if len(image.shape) != 3:
            return image
            
        # Parameters
        w = 0.95  # dehazing strength
        t0 = 0.1  # minimum transmission
        window_size = 15
        
        # Get dark channel
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        # Estimate atmospheric light
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, 3)
        top_pixels = np.argpartition(flat_dark, -int(flat_dark.size * 0.001))[-int(flat_dark.size * 0.001):]
        atmospheric = np.max(flat_image[top_pixels], axis=0)
        
        # Estimate transmission
        transmission = 1 - w * dark_channel / np.max(atmospheric)
        transmission = np.maximum(transmission, t0)
        
        # Recover the scene radiance
        result = np.empty_like(image, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (image[:, :, i].astype(np.float32) - atmospheric[i]) / transmission + atmospheric[i]
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_synthetic_fog(self, image: np.ndarray) -> np.ndarray:
        """Add synthetic fog effect to image."""
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        intensity = np.random.uniform(
            self.aug_params['fog']['min_intensity'],
            self.aug_params['fog']['max_intensity']
        )
        
        fog = np.ones_like(image) * 255
        return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    
    def add_synthetic_rain(self, image: np.ndarray) -> np.ndarray:
        """Add synthetic rain effect to image."""
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        params = self.aug_params['rain']
        h, w = image.shape[:2]
        
        # Create rain layer
        rain_layer = np.zeros_like(image)
        
        # Generate random rain drops
        n_drops = int(w * h * params['drop_density'])
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Draw rain drop line
            x2 = x + np.random.randint(-params['drop_length'], params['drop_length'])
            y2 = y + params['drop_length']
            
            cv2.line(
                rain_layer,
                (x, y),
                (x2, y2),
                params['drop_color'],
                params['drop_width']
            )
        
        # Blend rain layer with original image
        return cv2.addWeighted(image, 0.8, rain_layer, 0.2, 0)
    
    def get_augmentation_pipeline(self) -> A.Compose:
        """Create Albumentations pipeline for training augmentations."""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_params['brightness_contrast']['brightness_limit'],
                contrast_limit=self.aug_params['brightness_contrast']['contrast_limit'],
                p=self.aug_params['brightness_contrast']['p']
            ),
            A.OneOf([
                A.Lambda(image=self.add_synthetic_fog),
                A.Lambda(image=self.add_synthetic_rain)
            ], p=0.5)
        ]) 