#!/usr/bin/env python3
"""
Test script to validate the enhanced notebook can run without errors.
This tests the basic imports and configuration setup.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path('/home/runner/work/ann-visual-emotion/ann-visual-emotion')
sys.path.append(str(project_root))

def test_basic_imports():
    """Test that all required libraries can be imported"""
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix
        from tqdm import tqdm
        from PIL import Image
        import json
        print("✅ All basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_availability():
    """Test that the data files are accessible"""
    data_root = project_root / "data/processed/EmoSet_splits"
    required_files = [
        "train.csv",
        "val.csv", 
        "test.csv",
        "label_map.json",
        "stats.json"
    ]
    
    all_found = True
    for file_name in required_files:
        file_path = data_root / file_name
        if file_path.exists():
            print(f"✅ Found {file_name}")
        else:
            print(f"❌ Missing {file_name}")
            all_found = False
    
    return all_found

def test_data_loading():
    """Test basic data loading functionality"""
    try:
        data_root = project_root / "data/processed/EmoSet_splits"
        
        # Load CSV files
        train_df = pd.read_csv(data_root / "train.csv")
        val_df = pd.read_csv(data_root / "val.csv") 
        test_df = pd.read_csv(data_root / "test.csv")
        
        # Load label mapping
        with open(data_root / "label_map.json", 'r') as f:
            label_to_idx = json.load(f)
        
        print(f"✅ Data loading successful:")
        print(f"   • Training samples: {len(train_df):,}")
        print(f"   • Validation samples: {len(val_df):,}")
        print(f"   • Test samples: {len(test_df):,}")
        print(f"   • Classes: {len(label_to_idx)}")
        print(f"   • Class names: {list(label_to_idx.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_pytorch_setup():
    """Test PyTorch setup and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch setup:")
        print(f"   • PyTorch version: {torch.__version__}")
        print(f"   • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • Number of CPU cores: {torch.get_num_threads()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch setup error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced CNN Transfer Learning Notebook Setup")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Availability", test_data_availability),
        ("Data Loading", test_data_loading),
        ("PyTorch Setup", test_pytorch_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if not result:
            all_passed = False
    
    print("\n🎯 Overall Result:")
    if all_passed:
        print("✅ All tests passed! The notebook is ready to run.")
        print("🚀 You can now execute the Enhanced CNN Transfer Learning notebook.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("🔧 Fix the issues before running the notebook.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)