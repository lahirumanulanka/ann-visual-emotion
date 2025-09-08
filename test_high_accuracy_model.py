#!/usr/bin/env python3
"""
Test script for High-Accuracy Emotion Recognition Model
Quick validation that everything works correctly
"""

import os
import torch
import pandas as pd
from pathlib import Path
from high_accuracy_emotion_model import (
    Config, create_label_mapping, HighAccuracyEmotionDataset, 
    HighAccuracyEmotionModel, set_seed
)

def test_model():
    """Test the high-accuracy model setup"""
    set_seed(42)
    cfg = Config()
    
    print("🧪 Testing High-Accuracy Emotion Recognition Model")
    print("=" * 50)
    
    # Check if data files exist
    for path in [cfg.TRAIN_CSV, cfg.VAL_CSV, cfg.TEST_CSV]:
        if not Path(path).exists():
            print(f"❌ Data file not found: {path}")
            return False
        print(f"✅ Found: {path}")
    
    # Test label mapping
    try:
        label_to_idx, idx_to_label = create_label_mapping(cfg.TRAIN_CSV)
        cfg.NUM_CLASSES = len(idx_to_label)
        print(f"✅ Label mapping created: {cfg.NUM_CLASSES} classes")
    except Exception as e:
        print(f"❌ Error creating label mapping: {e}")
        return False
    
    # Test dataset creation
    try:
        # Load a small sample for testing
        df = pd.read_csv(cfg.TRAIN_CSV)
        sample_df = df.head(100)  # Use only first 100 samples for test
        
        # Save sample to temporary CSV
        temp_csv = Path(cfg.OUT_DIR) / "temp_train_sample.csv"
        Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(temp_csv, index=False)
        
        train_dataset = HighAccuracyEmotionDataset(
            csv_path=str(temp_csv),
            images_root=cfg.IMAGES_ROOT,
            label_map=label_to_idx,
            img_size=cfg.IMG_SIZE,
            is_training=True
        )
        print(f"✅ Dataset created: {len(train_dataset)} samples")
        
        # Test dataset access
        image, label = train_dataset[0]
        print(f"✅ Dataset access test: image shape {image.shape}, label {label}")
        
        # Clean up temp file
        temp_csv.unlink()
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False
    
    # Test model creation
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HighAccuracyEmotionModel(
            model_name=cfg.MODEL_NAME,
            num_classes=cfg.NUM_CLASSES,
            pretrained=cfg.PRETRAINED,
            dropout_rate=cfg.DROPOUT_RATE,
            use_attention=cfg.USE_ATTENTION
        ).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model created and tested")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Device: {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    print("\n🎉 All tests passed! Model is ready for training.")
    print(f"   Target accuracy: 80%+")
    print(f"   Model: {cfg.MODEL_NAME}")
    print(f"   Classes: {cfg.NUM_CLASSES}")
    print(f"   Device: {device}")
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model validation successful!")
        print("Ready to train the high-accuracy emotion recognition model.")
    else:
        print("\n❌ Model validation failed!")
        exit(1)