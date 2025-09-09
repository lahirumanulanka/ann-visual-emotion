#!/usr/bin/env python3
"""
Test the image enhancement functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from genai.synth_data import ImageEnhancer
from PIL import Image
import numpy as np

def test_enhancement():
    """Test image enhancement with a sample image."""
    
    # Find a sample image
    data_path = Path(__file__).parent / 'data' / 'raw' / 'EmoSet'
    sample_images = list(data_path.rglob('*.jpg'))[:5]
    
    print(f"Found {len(sample_images)} sample images for testing")
    
    if not sample_images:
        print("No sample images found!")
        return False
        
    enhancer = ImageEnhancer(method="enhanced_bicubic")
    
    for i, img_path in enumerate(sample_images):
        print(f"\nTesting with image {i+1}: {img_path}")
        
        # Load original image
        original = Image.open(img_path)
        print(f"Original size: {original.size}")
        print(f"Original mode: {original.mode}")
        
        # Enhance image
        enhanced = enhancer.enhance_image(img_path, target_size=(224, 224))
        print(f"Enhanced size: {enhanced.size}")
        print(f"Enhanced mode: {enhanced.mode}")
        
        # Save test output
        output_path = Path("/tmp") / f"test_enhanced_{i}.jpg"
        enhanced.save(output_path, quality=95)
        print(f"Saved enhanced image to: {output_path}")
        
        if i >= 2:  # Test only first 3 images
            break
    
    print("\n✅ Enhancement test completed successfully!")
    return True

if __name__ == "__main__":
    test_enhancement()