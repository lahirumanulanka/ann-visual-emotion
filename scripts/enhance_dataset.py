#!/usr/bin/env python3
"""
Dataset Enhancement Script using Generative AI

This script processes the emotion recognition dataset and enhances the 48x48 images
to 224x224 using generative AI techniques (SRCNN/Enhanced SRCNN) for better
transfer learning performance.

Usage:
    python scripts/enhance_dataset.py --input_dir data/raw/EmoSet --output_dir data/enhanced/EmoSet
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from genai.synth_data import (
    EmotionImageEnhancer, 
    enhance_emotion_dataset,
    compare_enhancement_methods
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhance emotion dataset using generative AI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, 
                       default='data/raw/EmoSet',
                       help='Input directory containing original images')
    
    parser.add_argument('--output_dir', type=str, 
                       default='data/enhanced/EmoSet',
                       help='Output directory for enhanced images')
    
    parser.add_argument('--splits_dir', type=str,
                       default='data/processed/EmoSet_splits',
                       help='Directory containing train/val/test split CSV files')
    
    parser.add_argument('--model_type', type=str, 
                       choices=['srcnn', 'enhanced_srcnn'],
                       default='enhanced_srcnn',
                       help='Type of enhancement model to use')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    
    parser.add_argument('--splits', nargs='+', 
                       default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create comparison images showing different enhancement methods')
    
    parser.add_argument('--sample_images', type=int, default=5,
                       help='Number of sample images for comparison (if --create_comparison is used)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for processing')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    return device


def load_dataset_splits(splits_dir):
    """Load dataset split information."""
    splits_dir = Path(splits_dir)
    splits_data = {}
    
    for split_name in ['train', 'val', 'test']:
        csv_path = splits_dir / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            splits_data[split_name] = df
            print(f"✓ Loaded {split_name} split: {len(df)} images")
        else:
            print(f"⚠ Missing {split_name} split file: {csv_path}")
    
    # Load label map if available
    label_map_path = splits_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        print(f"✓ Loaded label map: {list(label_map.keys())}")
    else:
        label_map = None
        print("⚠ No label map found")
    
    return splits_data, label_map


def create_enhanced_csv_files(splits_data, enhanced_output_dir, original_splits_dir):
    """Create new CSV files pointing to enhanced images."""
    enhanced_splits_dir = Path(enhanced_output_dir).parent / "enhanced_splits"
    enhanced_splits_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating enhanced split files in: {enhanced_splits_dir}")
    
    for split_name, df in splits_data.items():
        # Create copy of dataframe
        enhanced_df = df.copy()
        
        # Update paths to point to enhanced images (if path column exists)
        path_columns = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower() or 'image' in col.lower()]
        
        if path_columns:
            path_col = path_columns[0]
            print(f"Updating {path_col} column for {split_name} split")
            
            # Note: The paths remain the same since we're creating enhanced images in the same structure
            # but in a different root directory. The actual path change happens during data loading.
        
        # Save enhanced CSV
        enhanced_csv_path = enhanced_splits_dir / f"{split_name}.csv"
        enhanced_df.to_csv(enhanced_csv_path, index=False)
        print(f"✓ Saved enhanced {split_name} CSV: {enhanced_csv_path}")
    
    # Copy additional files (label_map, stats, etc.)
    original_splits_path = Path(original_splits_dir)
    for additional_file in ['label_map.json', 'stats.json', 'stats_balanced.json']:
        source_file = original_splits_path / additional_file
        target_file = enhanced_splits_dir / additional_file
        
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, target_file)
            print(f"✓ Copied {additional_file}")
    
    return enhanced_splits_dir


def enhance_dataset_splits(splits_data, input_dir, output_dir, model_type, device):
    """Enhance all dataset splits using the specified model."""
    total_stats = {'total': 0, 'success': 0, 'failed': 0}
    split_stats = {}
    
    # Initialize enhancer
    enhancer = EmotionImageEnhancer(model_type=model_type, device=device)
    
    print(f"\n{'='*60}")
    print(f"ENHANCING EMOTION DATASET")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    
    # Process each split
    for split_name, df in splits_data.items():
        print(f"\n{'-'*40}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'-'*40}")
        
        start_time = time.time()
        
        # Enhance the split
        stats = enhancer.enhance_dataset(
            dataset_df=df,
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=16  # Process in batches for efficiency
        )
        
        # Update total stats
        total_stats['total'] += stats['total']
        total_stats['success'] += stats['success']
        total_stats['failed'] += stats['failed']
        split_stats[split_name] = stats
        
        elapsed_time = time.time() - start_time
        print(f"✓ {split_name} completed in {elapsed_time:.1f}s")
        print(f"✓ Success rate: {stats['success']/stats['total']*100:.1f}%")
    
    return total_stats, split_stats


def create_comparison_samples(splits_data, input_dir, output_dir, num_samples=5):
    """Create comparison images showing different enhancement methods."""
    comparison_dir = Path(output_dir).parent / "enhancement_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'-'*40}")
    print(f"Creating enhancement comparisons")
    print(f"{'-'*40}")
    
    # Get sample images from train split
    if 'train' in splits_data:
        train_df = splits_data['train']
        
        # Sample random images
        sample_indices = train_df.sample(min(num_samples, len(train_df))).index
        
        for i, idx in enumerate(sample_indices):
            row = train_df.loc[idx]
            
            # Get image path (assume first column or detect path column)
            path_columns = [col for col in train_df.columns if 'path' in col.lower()]
            if path_columns:
                img_path = Path(input_dir) / row[path_columns[0]]
            else:
                img_path = Path(input_dir) / row.iloc[0]  # First column
            
            if img_path.exists():
                sample_dir = comparison_dir / f"sample_{i+1}"
                sample_dir.mkdir(exist_ok=True)
                
                print(f"Creating comparison for: {img_path.name}")
                
                try:
                    compare_enhancement_methods(
                        image_path=str(img_path),
                        output_dir=str(sample_dir)
                    )
                except Exception as e:
                    print(f"Error creating comparison for {img_path}: {e}")
    
    print(f"✓ Comparison images saved to: {comparison_dir}")


def print_final_summary(total_stats, split_stats, start_time):
    """Print final summary of the enhancement process."""
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"ENHANCEMENT SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total images processed: {total_stats['total']}")
    print(f"Successfully enhanced: {total_stats['success']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"Success rate: {total_stats['success']/total_stats['total']*100:.1f}%")
    
    if total_stats['success'] > 0:
        avg_time_per_image = total_time / total_stats['success']
        print(f"Average time per image: {avg_time_per_image:.2f}s")
    
    print(f"\nSplit-wise statistics:")
    for split_name, stats in split_stats.items():
        success_rate = stats['success'] / stats['total'] * 100
        print(f"- {split_name}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    print(f"\n✓ Dataset enhancement completed successfully!")


def main():
    """Main function to enhance emotion dataset."""
    args = parse_arguments()
    
    print("Emotion Dataset Enhancement using Generative AI")
    print("=" * 60)
    
    # Setup
    start_time = time.time()
    device = setup_device(args.device)
    
    # Load dataset splits
    print(f"\nLoading dataset splits from: {args.splits_dir}")
    splits_data, label_map = load_dataset_splits(args.splits_dir)
    
    if not splits_data:
        print("❌ No dataset splits found. Exiting.")
        return
    
    # Filter splits to process
    splits_to_process = {k: v for k, v in splits_data.items() if k in args.splits}
    
    if not splits_to_process:
        print(f"❌ None of the requested splits {args.splits} were found. Available: {list(splits_data.keys())}")
        return
    
    print(f"Will process splits: {list(splits_to_process.keys())}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enhance dataset
    total_stats, split_stats = enhance_dataset_splits(
        splits_data=splits_to_process,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        device=device
    )
    
    # Create enhanced CSV files
    enhanced_splits_dir = create_enhanced_csv_files(
        splits_data=splits_to_process,
        enhanced_output_dir=args.output_dir,
        original_splits_dir=args.splits_dir
    )
    
    # Create comparison samples if requested
    if args.create_comparison:
        create_comparison_samples(
            splits_data=splits_to_process,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_samples=args.sample_images
        )
    
    # Print final summary
    print_final_summary(total_stats, split_stats, start_time)
    
    # Save enhancement report
    report_path = Path(args.output_dir).parent / "enhancement_report.json"
    report_data = {
        'model_type': args.model_type,
        'device': device,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'total_stats': total_stats,
        'split_stats': split_stats,
        'processing_time_seconds': time.time() - start_time,
        'enhanced_splits_dir': str(enhanced_splits_dir)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✓ Enhancement report saved to: {report_path}")
    print(f"✓ Enhanced dataset available at: {args.output_dir}")
    print(f"✓ Enhanced split files available at: {enhanced_splits_dir}")


if __name__ == "__main__":
    main()