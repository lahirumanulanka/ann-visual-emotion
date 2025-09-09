"""
Enhanced Dataset Module for 224x224 Enhanced Emotion Recognition Images
Extends the base emotion dataset to work with AI-enhanced images.
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class EnhancedEmotionDataset(Dataset):
    """
    Dataset class for enhanced emotion recognition images (224x224).
    
    This dataset works with images that have been enhanced from 48x48 to 224x224
    using the image enhancement pipeline.
    """
    
    def __init__(self, dataframe, enhanced_root_dir, original_root_dir=None, 
                 transform=None, label_map=None, fallback_to_original=True):
        """
        Initialize the enhanced emotion dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels
            enhanced_root_dir (str or Path): Root directory containing enhanced 224x224 images
            original_root_dir (str or Path, optional): Fallback to original images if enhanced not found
            transform (callable, optional): Optional transform to be applied on images
            label_map (dict, optional): Dictionary mapping emotion names to indices
            fallback_to_original (bool): Whether to fall back to original images if enhanced not found
        """
        self.df = dataframe.reset_index(drop=True)
        self.enhanced_root_dir = Path(enhanced_root_dir)
        self.original_root_dir = Path(original_root_dir) if original_root_dir else None
        self.transform = transform
        self.label_map = label_map
        self.fallback_to_original = fallback_to_original
        
        # Auto-detect column names for flexibility
        self._detect_columns()
        
        # Verify enhanced dataset exists
        self._verify_enhanced_dataset()
        
        logger.info(f"Enhanced dataset initialized:")
        logger.info(f"- Samples: {len(self.df)}")
        logger.info(f"- Enhanced root: {self.enhanced_root_dir}")
        logger.info(f"- Original root: {self.original_root_dir}")
        logger.info(f"- Fallback enabled: {self.fallback_to_original}")
        if label_map:
            logger.info(f"- Classes: {list(label_map.keys())}")
    
    def _detect_columns(self):
        """Auto-detect image path and label columns."""
        # Find path/image column
        path_candidates = [c for c in self.df.columns if any(
            keyword in c.lower() for keyword in ['path', 'file', 'image', 'img', 'filepath']
        )]
        self.path_col = path_candidates[0] if path_candidates else self.df.columns[0]
        
        # Find label/emotion column
        label_candidates = [c for c in self.df.columns if any(
            keyword in c.lower() for keyword in ['label', 'class', 'emotion', 'target']
        )]
        self.label_col = label_candidates[0] if label_candidates else self.df.columns[1]
        
        if len(self.df.columns) < 2:
            raise ValueError("DataFrame must have at least 2 columns (image path and label)")
    
    def _verify_enhanced_dataset(self):
        """Verify that enhanced images exist."""
        if not self.enhanced_root_dir.exists():
            if self.fallback_to_original and self.original_root_dir:
                logger.warning(f"Enhanced dataset not found at {self.enhanced_root_dir}")
                logger.warning("Will fall back to original images")
            else:
                raise FileNotFoundError(f"Enhanced dataset not found at {self.enhanced_root_dir}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is the enhanced 224x224 RGB image tensor
                   and label is the corresponding emotion class index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label from dataframe
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        label = row[self.label_col]
        
        # Convert label to index if using string labels
        if self.label_map and isinstance(label, str):
            if label in self.label_map:
                label_idx = self.label_map[label]
            else:
                raise ValueError(f"Unknown emotion label: {label}")
        else:
            label_idx = int(label)
        
        # Try to load enhanced image first
        enhanced_img_path = self.enhanced_root_dir / rel_path
        image = None
        
        try:
            if enhanced_img_path.exists():
                image = Image.open(enhanced_img_path).convert('RGB')
            else:
                raise FileNotFoundError(f"Enhanced image not found: {enhanced_img_path}")
                
        except Exception as e:
            # Fall back to original image if available
            if self.fallback_to_original and self.original_root_dir:
                original_img_path = self.original_root_dir / rel_path
                try:
                    original_image = Image.open(original_img_path).convert('L')
                    # Resize to 224x224 and convert to RGB
                    image = original_image.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
                    logger.debug(f"Used original image (resized): {original_img_path}")
                except Exception as e2:
                    logger.warning(f"Failed to load both enhanced and original images: {e}, {e2}")
                    # Create dummy image as last resort
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                logger.warning(f"Failed to load enhanced image: {e}")
                # Create dummy image
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        if self.label_map:
            # Use string labels
            class_counts = self.df[self.label_col].value_counts().to_dict()
        else:
            # Use numeric labels
            class_counts = self.df[self.label_col].value_counts().to_dict()
        
        return class_counts
    
    def get_sample_weights(self):
        """Calculate sample weights for balanced training."""
        class_counts = self.get_class_distribution()
        total_samples = len(self.df)
        
        # Calculate weight for each class (inverse frequency)
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        # Assign weight to each sample
        sample_weights = []
        for idx in range(len(self.df)):
            label = self.df.iloc[idx][self.label_col]
            sample_weights.append(class_weights[label])
        
        return torch.FloatTensor(sample_weights)
    
    def get_enhanced_availability(self):
        """Check how many samples have enhanced images available."""
        available_count = 0
        missing_count = 0
        
        for idx in range(len(self.df)):
            rel_path = self.df.iloc[idx][self.path_col]
            enhanced_img_path = self.enhanced_root_dir / rel_path
            
            if enhanced_img_path.exists():
                available_count += 1
            else:
                missing_count += 1
        
        return {
            'available': available_count,
            'missing': missing_count,
            'total': len(self.df),
            'coverage': available_count / len(self.df) * 100
        }


def create_enhanced_dataloader(csv_path, enhanced_data_dir, original_data_dir=None,
                              batch_size=32, num_workers=4, shuffle=True, 
                              transform=None, label_map=None):
    """
    Create a DataLoader for enhanced emotion recognition dataset.
    
    Args:
        csv_path (str): Path to CSV file containing image paths and labels
        enhanced_data_dir (str): Directory containing enhanced 224x224 images
        original_data_dir (str, optional): Directory containing original images for fallback
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        transform (callable, optional): Transforms to apply to images
        label_map (dict, optional): Mapping from emotion names to indices
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the enhanced dataset
    """
    from torch.utils.data import DataLoader
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Create dataset
    dataset = EnhancedEmotionDataset(
        dataframe=df,
        enhanced_root_dir=enhanced_data_dir,
        original_root_dir=original_data_dir,
        transform=transform,
        label_map=label_map
    )
    
    # Check enhanced dataset coverage
    availability = dataset.get_enhanced_availability()
    logger.info(f"Enhanced dataset coverage: {availability['coverage']:.1f}% "
               f"({availability['available']}/{availability['total']})")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader