#!/usr/bin/env python3
"""
Test the enhanced dataset functionality with actual data
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.enhanced_dataset import EnhancedEmotionDataset, create_enhanced_dataloader
from torchvision import transforms

def create_test_csv():
    """Create a test CSV from the enhanced test data."""
    
    # Find all enhanced images
    enhanced_dir = Path('/tmp/test_data_enhanced')
    image_files = list(enhanced_dir.rglob('*.jpg'))
    
    # Create DataFrame
    data = []
    for img_path in image_files:
        # Extract emotion from path
        rel_path = img_path.relative_to(enhanced_dir)
        emotion = rel_path.parts[1]  # train/emotion/file.jpg
        
        data.append({
            'filepath': str(rel_path),
            'label': emotion,
            'split': 'train'
        })
    
    df = pd.DataFrame(data)
    
    # Create label mapping
    label_map = {emotion: idx for idx, emotion in enumerate(df['label'].unique())}
    
    # Add label_id
    df['label_id'] = df['label'].map(label_map)
    
    # Save files
    df.to_csv('/tmp/test_enhanced.csv', index=False)
    
    with open('/tmp/test_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Created test CSV with {len(df)} samples")
    print(f"Emotions: {list(label_map.keys())}")
    print(f"Label mapping: {label_map}")
    
    return '/tmp/test_enhanced.csv', '/tmp/test_label_map.json', label_map

def test_enhanced_dataset():
    """Test the enhanced dataset functionality."""
    
    # Create test data
    csv_path, label_map_path, label_map = create_test_csv()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nDataset info:")
    print(df.head())
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create enhanced dataset
    dataset = EnhancedEmotionDataset(
        dataframe=df,
        enhanced_root_dir='/tmp/test_data_enhanced',
        original_root_dir='/tmp/test_data_orig',
        transform=transform,
        label_map=label_map
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Check enhanced availability
    availability = dataset.get_enhanced_availability()
    print(f"Enhanced coverage: {availability['coverage']:.1f}%")
    print(f"Available: {availability['available']}, Missing: {availability['missing']}")
    
    # Test loading a few samples
    print("\nTesting sample loading:")
    for i in range(min(3, len(dataset))):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")
    
    # Test DataLoader
    print("\nTesting DataLoader:")
    dataloader = create_enhanced_dataloader(
        csv_path=csv_path,
        enhanced_data_dir='/tmp/test_data_enhanced',
        original_data_dir='/tmp/test_data_orig',
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        shuffle=True,
        transform=transform,
        label_map=label_map
    )
    
    # Get first batch
    for batch_images, batch_labels in dataloader:
        print(f"Batch shape: {batch_images.shape}")
        print(f"Batch labels: {batch_labels}")
        break
    
    print("\n✅ Enhanced dataset test completed successfully!")
    return True

if __name__ == "__main__":
    test_enhanced_dataset()