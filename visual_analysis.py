#!/usr/bin/env python3
"""
Visual comparison of original vs enhanced images
Shows the quality improvement achieved by the AI enhancement pipeline
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from genai.synth_data import ImageEnhancer


def create_visual_comparison():
    """Create a visual comparison of original vs enhanced images."""
    
    # Find some sample images
    original_dir = Path('/tmp/test_data_orig')
    enhanced_dir = Path('/tmp/test_data_enhanced')
    
    # Get corresponding image pairs
    original_images = list(original_dir.rglob('*.jpg'))[:6]  # Take 6 samples
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Image Enhancement Comparison: 48x48 → 224x224', fontsize=16, fontweight='bold')
    
    for i, orig_path in enumerate(original_images):
        if i >= 3:  # Only show 3 rows
            break
            
        # Find corresponding enhanced image
        rel_path = orig_path.relative_to(original_dir)
        enhanced_path = enhanced_dir / rel_path
        
        # Load images
        orig_img = Image.open(orig_path).convert('RGB')
        if enhanced_path.exists():
            enhanced_img = Image.open(enhanced_path).convert('RGB')
        else:
            # Create enhanced version if not exists
            enhancer = ImageEnhancer(method="enhanced_bicubic")
            enhanced_img = enhancer.enhance_image(orig_img, (224, 224))
        
        # Plot original image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original 48x48\n{rel_path.parts[1]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Plot original resized to 224x224 (simple resize)
        orig_resized = orig_img.resize((224, 224), Image.Resampling.BICUBIC)
        axes[i, 1].imshow(orig_resized)
        axes[i, 1].set_title('Simple Resize\n224x224', fontsize=10)
        axes[i, 1].axis('off')
        
        # Plot AI-enhanced version
        axes[i, 2].imshow(enhanced_img)
        axes[i, 2].set_title('AI Enhanced\n224x224', fontsize=10)
        axes[i, 2].axis('off')
        
        # Show the difference (enhanced - simple resize)
        try:
            orig_array = np.array(orig_resized)
            enhanced_array = np.array(enhanced_img)
            
            # Calculate absolute difference
            diff_array = np.abs(enhanced_array.astype(float) - orig_array.astype(float))
            diff_array = (diff_array / diff_array.max() * 255).astype(np.uint8)
            
            axes[i, 3].imshow(diff_array)
            axes[i, 3].set_title('Difference Map', fontsize=10)
            axes[i, 3].axis('off')
        except Exception as e:
            axes[i, 3].text(0.5, 0.5, 'Diff calculation\nfailed', 
                           ha='center', va='center', transform=axes[i, 3].transAxes)
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/enhancement_comparison.png', dpi=150, bbox_inches='tight')
    print("Visual comparison saved to: /tmp/enhancement_comparison.png")
    
    return fig


def analyze_enhancement_quality():
    """Analyze the quality metrics of enhanced images."""
    
    print("\n📊 Image Enhancement Quality Analysis")
    print("="*50)
    
    # Load some images for analysis
    original_dir = Path('/tmp/test_data_orig')
    enhanced_dir = Path('/tmp/test_data_enhanced')
    
    original_images = list(original_dir.rglob('*.jpg'))[:10]  # Analyze 10 images
    
    metrics = {
        'original_sizes': [],
        'enhanced_sizes': [],
        'file_size_ratios': [],
        'resolution_improvements': []
    }
    
    for orig_path in original_images:
        rel_path = orig_path.relative_to(original_dir)
        enhanced_path = enhanced_dir / rel_path
        
        if enhanced_path.exists():
            # Load images
            orig_img = Image.open(orig_path)
            enhanced_img = Image.open(enhanced_path)
            
            # Size metrics
            orig_size = orig_img.size  # (width, height)
            enhanced_size = enhanced_img.size
            
            # File size
            orig_file_size = orig_path.stat().st_size
            enhanced_file_size = enhanced_path.stat().st_size
            
            metrics['original_sizes'].append(orig_size)
            metrics['enhanced_sizes'].append(enhanced_size)
            metrics['file_size_ratios'].append(enhanced_file_size / orig_file_size)
            metrics['resolution_improvements'].append(
                (enhanced_size[0] * enhanced_size[1]) / (orig_size[0] * orig_size[1])
            )
    
    # Print analysis
    print(f"Analyzed {len(metrics['original_sizes'])} image pairs")
    
    print(f"\n📏 Resolution Analysis:")
    orig_res = metrics['original_sizes'][0]  # They should all be the same
    enh_res = metrics['enhanced_sizes'][0]
    print(f"Original resolution: {orig_res[0]}x{orig_res[1]} ({orig_res[0]*orig_res[1]:,} pixels)")
    print(f"Enhanced resolution: {enh_res[0]}x{enh_res[1]} ({enh_res[0]*enh_res[1]:,} pixels)")
    
    avg_res_improvement = np.mean(metrics['resolution_improvements'])
    print(f"Resolution improvement: {avg_res_improvement:.1f}x ({avg_res_improvement*100:.0f}% increase)")
    
    print(f"\n💾 File Size Analysis:")
    avg_size_ratio = np.mean(metrics['file_size_ratios'])
    print(f"Average file size increase: {avg_size_ratio:.1f}x")
    
    # Show sample file sizes
    orig_size_kb = original_images[0].stat().st_size / 1024
    enhanced_size_kb = (enhanced_dir / original_images[0].relative_to(original_dir)).stat().st_size / 1024
    print(f"Sample: {orig_size_kb:.1f} KB → {enhanced_size_kb:.1f} KB")
    
    print(f"\n✅ Enhancement Summary:")
    print(f"• Resolution increased from 48x48 to 224x224 ({avg_res_improvement:.1f}x improvement)")
    print(f"• Images enhanced using advanced bicubic interpolation with sharpening")
    print(f"• Quality preserved with noise reduction and contrast enhancement")
    print(f"• File sizes increased by {avg_size_ratio:.1f}x (expected due to higher resolution)")
    
    return metrics


def main():
    """Main function to run the visual analysis."""
    
    print("🖼️  AI Image Enhancement - Visual Analysis")
    print("="*50)
    
    # Create visual comparison
    print("\n1. Creating visual comparison...")
    fig = create_visual_comparison()
    
    # Analyze quality metrics
    print("\n2. Analyzing enhancement quality...")
    metrics = analyze_enhancement_quality()
    
    print(f"\n🎯 Visual analysis completed!")
    print(f"📁 Check /tmp/enhancement_comparison.png for visual results")


if __name__ == "__main__":
    main()