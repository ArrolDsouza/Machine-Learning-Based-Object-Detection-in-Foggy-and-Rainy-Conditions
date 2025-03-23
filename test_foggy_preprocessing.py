"""
Test script for foggy image preprocessing pipeline.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.image_processor import WeatherImageProcessor

def test_foggy_preprocessing():
    """Test the preprocessing pipeline on a foggy image."""
    try:
        # Get a random image from the training set
        train_dir = Path('data/foggy_cityscapes/train/images')
        logging.info(f"Looking for images in: {train_dir.absolute()}")
        
        if not train_dir.exists():
            logging.error(f"Training directory not found: {train_dir}")
            return
            
        image_files = list(train_dir.glob('*.jpg'))
        logging.info(f"Found {len(image_files)} images")
        
        if not image_files:
            logging.error("No images found in the training directory")
            return
            
        test_image = random.choice(image_files)
        logging.info(f"Selected test image: {test_image}")
        
        # Initialize processor
        processor = WeatherImageProcessor()
        
        # Load image
        image = cv2.imread(str(test_image))
        if image is None:
            logging.error(f"Failed to load image: {test_image}")
            return
            
        logging.info(f"Image shape: {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure for visualization
        plt.style.use('dark_background')  # Better visualization for foggy images
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Foggy Image Preprocessing Test', size=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Foggy Image', size=12)
        
        # Enhanced contrast
        logging.info("Applying contrast enhancement...")
        enhanced = processor.enhance_contrast(image.copy())
        axes[0, 1].imshow(enhanced)
        axes[0, 1].set_title('Enhanced Contrast', size=12)
        
        # Defogged image
        logging.info("Applying defogging...")
        defogged = processor.dehaze(image.copy())
        axes[1, 0].imshow(defogged)
        axes[1, 0].set_title('Defogged Image', size=12)
        
        # Full preprocessing pipeline
        logging.info("Applying full preprocessing pipeline...")
        preprocessed = processor.preprocess_image(image.copy())
        axes[1, 1].imshow(preprocessed)
        axes[1, 1].set_title('Full Preprocessing Pipeline', size=12)
        
        # Adjust display
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        output_dir = Path('results/foggy_preprocessing_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'foggy_preprocessing_test.png'
        
        logging.info(f"Saving results to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info("Results saved successfully")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error during preprocessing test: {str(e)}", exc_info=True)

if __name__ == '__main__':
    test_foggy_preprocessing()