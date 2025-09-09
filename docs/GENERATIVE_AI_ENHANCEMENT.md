# Generative AI Enhanced Visual Emotion Recognition

This document provides a comprehensive guide for using generative AI to enhance the visual emotion recognition dataset and improve model performance.

## Overview

The enhanced system addresses the problem of training CNN transfer learning models on small (48x48) grayscale facial emotion images by using generative AI techniques to create high-quality 224x224 pixel images suitable for pre-trained models.

### Key Improvements

- **Generative AI Enhancement**: Uses Enhanced SRCNN (Super-Resolution CNN) to upscale 48x48 images to 224x224
- **Quality Preservation**: Maintains facial emotion features during upscaling
- **Transfer Learning Ready**: Creates images compatible with ImageNet pre-trained models
- **Performance Comparison**: Tools to compare original vs enhanced dataset performance

## Architecture

### Enhancement Pipeline

1. **Input**: 48x48 grayscale emotion images
2. **Enhancement Model**: Enhanced SRCNN with attention mechanism
3. **Post-processing**: Sharpening and contrast enhancement
4. **Output**: 224x224 high-quality grayscale images (converted to RGB for training)

### Models Available

- **SRCNN**: Basic Super-Resolution CNN
- **Enhanced SRCNN**: Advanced version with:
  - Residual connections
  - Attention mechanism for facial features
  - Batch normalization
  - Enhanced post-processing

## Quick Start

### 1. Enhance Your Dataset

```bash
# Enhance the entire emotion dataset
python scripts/enhance_dataset.py \
    --input_dir data/raw/EmoSet \
    --output_dir data/enhanced/EmoSet \
    --model_type enhanced_srcnn \
    --create_comparison

# This will:
# - Process all train/val/test splits
# - Create enhanced 224x224 images
# - Generate comparison samples
# - Create new CSV files pointing to enhanced images
```

### 2. Train with Enhanced Data

```bash
# Train using enhanced dataset
python scripts/train_enhanced_cnn.py \
    --use_enhanced \
    --backbone vgg16 \
    --epochs 30 \
    --output_dir results/enhanced

# Compare original vs enhanced performance
python scripts/train_enhanced_cnn.py \
    --compare_both \
    --backbone vgg16 \
    --epochs 20 \
    --output_dir results/comparison
```

## Detailed Usage

### Dataset Enhancement

#### Basic Enhancement
```bash
python scripts/enhance_dataset.py \
    --input_dir data/raw/EmoSet \
    --output_dir data/enhanced/EmoSet
```

#### Advanced Options
```bash
python scripts/enhance_dataset.py \
    --input_dir data/raw/EmoSet \
    --output_dir data/enhanced/EmoSet \
    --model_type enhanced_srcnn \
    --device cuda \
    --create_comparison \
    --sample_images 10
```

#### Parameters
- `--model_type`: Choose between 'srcnn' and 'enhanced_srcnn'
- `--device`: 'auto', 'cuda', or 'cpu'
- `--create_comparison`: Generate before/after comparison images
- `--sample_images`: Number of comparison samples to create

### Training Models

#### Train on Enhanced Data Only
```bash
python scripts/train_enhanced_cnn.py \
    --use_enhanced \
    --enhanced_data data/enhanced/EmoSet \
    --enhanced_splits data/enhanced/enhanced_splits \
    --backbone vgg16 \
    --epochs 30 \
    --batch_size 32
```

#### Compare Original vs Enhanced
```bash
python scripts/train_enhanced_cnn.py \
    --compare_both \
    --original_data data/raw/EmoSet \
    --enhanced_data data/enhanced/EmoSet \
    --backbone vgg16 \
    --epochs 20
```

#### Training Parameters
- `--backbone`: 'vgg16', 'vgg19', or 'alexnet'
- `--freeze_backbone`: Freeze pre-trained weights
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--backbone_lr`: Learning rate for backbone (default: 1e-5)
- `--classifier_lr`: Learning rate for classifier (default: 1e-3)

## Expected Performance Improvements

Based on the enhanced approach, you can expect:

### Accuracy Improvements
- **Baseline CNN (48x48)**: ~66% accuracy (current)
- **Transfer Learning + Simple Upsampling**: ~66-70% accuracy
- **Transfer Learning + Generative AI Enhancement**: ~70-75% accuracy (target)

### Quality Benefits
- Better preservation of facial features during upscaling
- Reduced artifacts compared to simple interpolation
- Enhanced edge definition for emotion detection
- Optimized for CNN feature extraction

## File Structure After Enhancement

```
data/
├── raw/
│   └── EmoSet/                 # Original 48x48 images
├── processed/
│   └── EmoSet_splits/          # Original CSV files
├── enhanced/
│   ├── EmoSet/                 # Enhanced 224x224 images
│   ├── enhanced_splits/        # Enhanced CSV files
│   └── enhancement_comparison/ # Before/after samples
└── results/
    ├── enhanced/               # Enhanced model results
    ├── original/               # Original model results
    └── comparison/             # Comparison results
```

## Programmatic Usage

### Using the Enhancement API

```python
from src.genai.synth_data import EmotionImageEnhancer

# Initialize enhancer
enhancer = EmotionImageEnhancer('enhanced_srcnn', device='cuda')

# Enhance single image
enhancer.enhance_image_file('input.jpg', 'enhanced.jpg', target_size=(224, 224))

# Enhance dataset
import pandas as pd
df = pd.read_csv('dataset.csv')
stats = enhancer.enhance_dataset(df, 'input_dir', 'output_dir')
```

### Custom Training Loop

```python
from scripts.train_enhanced_cnn import train_model

config = {
    'name': 'my_enhanced_model',
    'data_type': 'enhanced',
    'data_config': {
        'data_dir': 'data/enhanced/EmoSet',
        'splits_dir': 'data/enhanced/enhanced_splits'
    },
    'model': {'backbone': 'vgg16', 'freeze_backbone': False},
    # ... other config options
}

results = train_model(config, device='cuda')
```

## Performance Monitoring

The system provides comprehensive monitoring:

### Training Metrics
- Training/validation loss and accuracy curves
- Learning rate schedules
- Early stopping based on validation performance

### Comparison Reports
- Side-by-side accuracy comparison
- Training time analysis  
- Per-class performance breakdown
- Confusion matrices

### Visualization
- Enhancement quality comparison samples
- Training progress plots
- Performance comparison charts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 16
   
   # Use CPU if necessary
   --device cpu
   ```

2. **Missing Data Files**
   - Ensure your dataset follows the expected structure
   - Check CSV files have correct column names
   - Verify image paths are relative to data directory

3. **Poor Enhancement Quality**
   - Try different enhancement models (srcnn vs enhanced_srcnn)
   - Adjust post-processing parameters
   - Ensure input images are properly formatted

### Performance Tips

1. **GPU Usage**: Use CUDA if available for faster enhancement
2. **Batch Processing**: Adjust batch size based on available memory  
3. **Parallel Processing**: Use multiple workers for data loading
4. **Model Selection**: Try different backbones (VGG16 vs VGG19 vs AlexNet)

## Contributing

To extend the enhancement system:

1. **Add New Models**: Implement in `src/genai/synth_data.py`
2. **Improve Post-processing**: Modify `_apply_postprocessing` method
3. **Add Metrics**: Extend evaluation in training scripts
4. **Custom Datasets**: Adapt `EmotionDataset` class

## References

- Super-Resolution CNN (SRCNN): [Dong et al., 2014]
- Transfer Learning for Computer Vision: [Razavian et al., 2014]
- Facial Emotion Recognition: [Goodfellow et al., 2013]

## License

This enhancement system is part of the ann-visual-emotion project and follows the same license terms.