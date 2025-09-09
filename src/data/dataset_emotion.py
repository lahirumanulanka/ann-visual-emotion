# PyTorch Dataset for Emotion Recognition
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Custom dataset class for emotion recognition that supports both grayscale and RGB images.
    
    This dataset is designed to work with both traditional CNN models (grayscale) and 
    transfer learning models (RGB) for visual emotion recognition.
    """
    
    def __init__(self, dataframe, root_dir, transform=None, label_map=None, rgb=False):
        """
        Initialize the emotion dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels
            root_dir (str or Path): Root directory containing images
            transform (callable, optional): Optional transform to be applied on images
            label_map (dict, optional): Dictionary mapping emotion names to indices
            rgb (bool): If True, convert images to RGB; if False, use grayscale
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = label_map
        self.rgb = rgb
        
        # Auto-detect column names for flexibility
        self._detect_columns()
        
        print(f"Dataset initialized:")
        print(f"- Samples: {len(self.df)}")
        print(f"- Image mode: {'RGB' if rgb else 'Grayscale'}")
        print(f"- Path column: '{self.path_col}'")
        print(f"- Label column: '{self.label_col}'")
        if label_map:
            print(f"- Classes: {list(label_map.keys())}")
    
    def _detect_columns(self):
        """Auto-detect image path and label columns."""
        # Find path/image column
        path_candidates = [c for c in self.df.columns if any(
            keyword in c.lower() for keyword in ['path', 'file', 'image', 'img']
        )]
        self.path_col = path_candidates[0] if path_candidates else self.df.columns[0]
        
        # Find label/emotion column
        label_candidates = [c for c in self.df.columns if any(
            keyword in c.lower() for keyword in ['label', 'class', 'emotion', 'target']
        )]
        self.label_col = label_candidates[0] if label_candidates else self.df.columns[1]
        
        if len(self.df.columns) < 2:
            raise ValueError("DataFrame must have at least 2 columns (image path and label)")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is the transformed image tensor
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
        
        # Load image
        img_path = self.root_dir / rel_path
        
        try:
            # Load image in appropriate mode
            if self.rgb:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('L')
        except Exception as e:
            # Create a dummy image in case of loading error
            print(f"Warning: Error loading image {img_path}: {e}")
            if self.rgb:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                image = Image.new('L', (48, 48), color=128)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Class distribution with emotion names as keys and counts as values
        """
        if self.label_map:
            # Convert numeric labels back to emotion names
            label_counts = self.df[self.label_col].value_counts()
            if isinstance(label_counts.index[0], str):
                # Labels are already strings
                return label_counts.to_dict()
            else:
                # Labels are numeric, convert to emotion names
                reverse_map = {v: k for k, v in self.label_map.items()}
                return {reverse_map.get(k, f"class_{k}"): v 
                       for k, v in label_counts.to_dict().items()}
        else:
            return self.df[self.label_col].value_counts().to_dict()
    
    def get_sample_info(self, idx):
        """
        Get detailed information about a specific sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample information including path, label, etc.
        """
        row = self.df.iloc[idx]
        info = {
            'index': idx,
            'image_path': self.root_dir / row[self.path_col],
            'relative_path': row[self.path_col],
            'label': row[self.label_col],
            'rgb_mode': self.rgb
        }
        
        if self.label_map and isinstance(row[self.label_col], str):
            info['label_idx'] = self.label_map[row[self.label_col]]
        else:
            info['label_idx'] = int(row[self.label_col])
        
        return info


class EmotionDatasetWithAugmentation(EmotionDataset):
    """
    Enhanced emotion dataset with built-in data augmentation options.
    """
    
    def __init__(self, dataframe, root_dir, transform=None, label_map=None, 
                 rgb=False, augment_factor=1):
        """
        Initialize dataset with augmentation.
        
        Args:
            augment_factor (int): Factor by which to augment the dataset size
        """
        super().__init__(dataframe, root_dir, transform, label_map, rgb)
        self.augment_factor = augment_factor
        
        if augment_factor > 1:
            print(f"Dataset augmented by factor {augment_factor} -> {len(self)} samples")
    
    def __len__(self):
        """Return augmented dataset size."""
        return len(self.df) * self.augment_factor
    
    def __getitem__(self, idx):
        """Get item with augmentation support."""
        # Map augmented index to original index
        original_idx = idx % len(self.df)
        return super().__getitem__(original_idx)


def create_emotion_datasets(train_csv, val_csv, test_csv, data_dir, label_map_path=None,
                           train_transform=None, val_transform=None, rgb=False):
    """
    Factory function to create train, validation, and test datasets.
    
    Args:
        train_csv (str): Path to training CSV file
        val_csv (str): Path to validation CSV file  
        test_csv (str): Path to test CSV file
        data_dir (str): Root directory containing images
        label_map_path (str, optional): Path to label mapping JSON file
        train_transform (callable, optional): Transform for training data
        val_transform (callable, optional): Transform for validation/test data
        rgb (bool): Whether to use RGB images
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, label_map)
    """
    import json
    
    # Load label map if provided
    label_map = None
    if label_map_path and Path(label_map_path).exists():
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
    
    # Load dataframes
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df, data_dir, transform=train_transform, 
        label_map=label_map, rgb=rgb
    )
    
    val_dataset = EmotionDataset(
        val_df, data_dir, transform=val_transform, 
        label_map=label_map, rgb=rgb
    )
    
    test_dataset = EmotionDataset(
        test_df, data_dir, transform=val_transform, 
        label_map=label_map, rgb=rgb
    )
    
    print(f"\nDatasets created successfully:")
    print(f"- Training: {len(train_dataset)} samples")
    print(f"- Validation: {len(val_dataset)} samples")
    print(f"- Test: {len(test_dataset)} samples")
    
    if label_map:
        print(f"- Classes: {len(label_map)} -> {list(label_map.keys())}")
    
    return train_dataset, val_dataset, test_dataset, label_map


# Example usage and testing
if __name__ == "__main__":
    print("Testing EmotionDataset...")
    
    # Create a dummy dataset for testing
    dummy_data = pd.DataFrame({
        'image_path': [f'image_{i}.jpg' for i in range(10)],
        'emotion': ['happiness', 'sadness', 'anger', 'fear', 'surprise'] * 2
    })
    
    dummy_label_map = {
        'happiness': 0, 'sadness': 1, 'anger': 2, 
        'fear': 3, 'surprise': 4
    }
    
    # Test grayscale dataset
    print("\n1. Testing grayscale dataset:")
    gray_dataset = EmotionDataset(
        dummy_data, 
        root_dir='/tmp', 
        label_map=dummy_label_map,
        rgb=False
    )
    
    print(f"Dataset length: {len(gray_dataset)}")
    print(f"Class distribution: {gray_dataset.get_class_distribution()}")
    print(f"Sample info: {gray_dataset.get_sample_info(0)}")
    
    # Test RGB dataset
    print("\n2. Testing RGB dataset:")
    rgb_dataset = EmotionDataset(
        dummy_data, 
        root_dir='/tmp', 
        label_map=dummy_label_map,
        rgb=True
    )
    
    print(f"Dataset length: {len(rgb_dataset)}")
    print(f"RGB mode: {rgb_dataset.rgb}")
    
    print("\n✓ EmotionDataset tests completed!")
