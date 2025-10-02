# Comprehensive Exploratory Data Analysis (03_exploratory_data_analysis.ipynb)

This document describes the comprehensive EDA workflow implemented in `notebooks/03_exploratory_data_analysis.ipynb`.

---

## Overview

The comprehensive EDA notebook extends the basic dataset inspection performed in `01_eda.ipynb` by adding:
1. Advanced feature extraction from images
2. Statistical outlier detection and treatment
3. SMOTE-based class balancing
4. Feature engineering with domain-specific features
5. Complete justification of preprocessing choices

---

## Workflow Steps

### 1. Data Loading
- Loads train, validation, and test splits from `data/processed/EmoSet_splits/`
- Combines all splits for comprehensive analysis
- Total dataset: ~80,000 images across 6 emotion classes

### 2. Missing Value Analysis
- Checks for missing values in all columns
- Generates heatmap visualization
- **Finding**: No missing values detected in the dataset

### 3. Class Distribution Analysis
- Computes class counts and percentages
- Calculates imbalance ratio (max/min class size)
- Visualizes distribution with bar plots and pie charts
- **Finding**: Imbalance ratio of ~1.11x (relatively balanced after preprocessing)

### 4. Image Feature Extraction

Extracts statistical features from images:

**Basic Statistics:**
- Mean intensity (overall and per RGB channel)
- Standard deviation (overall and per RGB channel)
- Min/Max pixel values
- Image dimensions (width, height)

**Implementation Notes:**
- Processes images in batches for efficiency
- Handles different image formats (RGB, grayscale)
- Converts container paths to local filesystem paths

### 5. Outlier Detection

Uses two complementary methods:

**IQR (Interquartile Range) Method:**
- Detects values beyond Q1 - 1.5×IQR and Q3 + 1.5×IQR
- Good for symmetric distributions

**Z-Score Method:**
- Identifies values beyond 3 standard deviations
- Effective for Gaussian-like distributions

**Key Features Analyzed:**
- `mean_intensity`: Overall brightness
- `std_intensity`: Overall contrast

### 6. Outlier Treatment

**Chosen Strategy: Capping (not removal)**

**Justification:**
- Extreme brightness/contrast may represent valid emotional expressions
- Removing outliers would reduce dataset size
- Capping at 1st and 99th percentiles preserves data while limiting extreme values
- Maintains natural variation in emotional expressions

**Process:**
```python
lower_cap = df[column].quantile(0.01)
upper_cap = df[column].quantile(0.99)
df[column] = df[column].clip(lower=lower_cap, upper=upper_cap)
```

### 7. Feature Engineering

Creates 6 new domain-specific features:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| **brightness** | Mean intensity | Emotional correlation with lighting (happy=bright, sad=dark) |
| **contrast** | Std intensity | Strong emotions may have higher contrast |
| **color_balance_rb** | mean_r / (mean_b + ε) | Warm (red) vs cool (blue) color psychology |
| **edge_density** | std / (mean + ε) | More edges = more expressive faces |
| **rgb_variance** | Var(R, G, B) | Color diversity indicates environmental context |
| **color_saturation** | max(R,G,B) - min(R,G,B) | Vivid vs muted emotional perception |

**Why These Features?**
- **Brightness**: Research shows darker images correlate with negative emotions
- **Contrast**: High contrast indicates more dynamic facial expressions
- **Color Psychology**: Warm/cool tones affect emotional perception
- **Edge Content**: More facial details suggest stronger expressions
- **Color Diversity**: Environmental context affects emotion interpretation
- **Saturation**: Color intensity influences emotional impact

### 8. SMOTE (Synthetic Minority Over-sampling Technique)

**Purpose:** Balance class distribution without simple duplication

**How it Works:**
1. Identifies minority classes
2. For each minority sample, finds k nearest neighbors
3. Creates synthetic samples by interpolating between neighbors
4. Generates enough samples to balance all classes

**Parameters Used:**
- `random_state=42`: Reproducibility
- `k_neighbors=5`: Standard choice for interpolation

**Results:**
- Before: Imbalance ratio ~1.11x
- After: Perfect balance (1.00x)
- All classes: 875 samples each
- Synthetic samples generated: ~250 (5% of dataset)

**Why SMOTE over alternatives?**
- **vs Random Oversampling**: Creates diverse samples, reduces overfitting
- **vs Class Weights**: Actual balanced data better for some algorithms
- **vs Undersampling**: Preserves all original data, no information loss
- **vs ADASYN**: Simpler, more stable for this use case

### 9. Visualization

Generates multiple visualizations:

1. **Class Distribution** (before/after SMOTE)
   - Bar plots showing count changes
   - Saved as PNG for documentation

2. **Feature Distributions**
   - Histograms for all features
   - Identifies skewness and outliers

3. **Box Plots by Class**
   - Shows feature ranges per emotion
   - Reveals class-specific patterns

4. **Correlation Matrix**
   - Heatmap of feature correlations
   - Identifies redundant features

### 10. Output Artifacts

All saved to `data/processed/EmoSet_splits_eda/`:

| File | Description |
|------|-------------|
| `balanced_dataset_with_features.csv` | Complete dataset (5,250 rows × 17 columns) |
| `feature_info.json` | Feature names, types, and label encoding |
| `eda_summary.json` | Statistics: sizes, ratios, distributions |
| `EDA_REPORT.md` | Human-readable summary report |
| `class_distribution_before_balancing.png` | Initial class distribution |
| `class_distribution_comparison.png` | Before/after SMOTE comparison |

---

## Usage

### Running the Notebook

```bash
cd notebooks
jupyter notebook 03_exploratory_data_analysis.ipynb
```

### Configuration Options

Key variables to adjust:

```python
# Sample size for feature extraction (trade speed vs completeness)
USE_SAMPLE = True
SAMPLE_SIZE = 5000  # Set higher or False for full dataset

# Outlier capping percentiles
lower_percentile = 1   # Lower bound
upper_percentile = 99  # Upper bound

# SMOTE parameters
k_neighbors = 5  # Number of neighbors for interpolation
```

### Expected Runtime

- **Sample mode (5,000 images)**: ~5-10 minutes
- **Full dataset (80,000 images)**: ~60-90 minutes

Runtime depends on:
- CPU speed
- Image I/O performance
- Memory availability

---

## Integration with Model Training

The balanced dataset can be used in training notebooks:

```python
import pandas as pd

# Load balanced dataset with features
df = pd.read_csv('../data/processed/EmoSet_splits_eda/balanced_dataset_with_features.csv')

# Load feature metadata
import json
with open('../data/processed/EmoSet_splits_eda/feature_info.json') as f:
    feature_info = json.load(f)

# Use engineered features
engineered_features = feature_info['engineered_features']
X = df[engineered_features].values
y = df['label'].values
```

---

## Preprocessing Justification Summary

### 1. No Missing Value Imputation Needed
- Dataset is clean with no missing values
- All images successfully loaded

### 2. Outlier Capping (Not Removal)
- Preserves dataset size
- Maintains natural variation
- Prevents extreme values from dominating

### 3. SMOTE for Balancing
- Creates synthetic samples through interpolation
- Better than duplication (reduces overfitting)
- Better than undersampling (no data loss)
- Achieves perfect class balance

### 4. Feature Engineering
- 6 domain-specific features
- Based on emotion recognition literature
- Captures illumination, color, and texture properties
- Validated through correlation analysis

### 5. Sample-Based Processing
- Allows faster iteration during development
- Can scale to full dataset when needed
- Maintains statistical representativeness

---

## Next Steps

After running this EDA:

1. **Model Training**: Use `balanced_dataset_with_features.csv` for training
2. **Feature Selection**: Analyze feature importance from trained models
3. **Hyperparameter Tuning**: Optimize with balanced classes
4. **Evaluation**: Compare performance with/without engineered features

---

## References

- **SMOTE**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
- **Color Psychology**: Elliot & Maier (2014) - "Color psychology: effects of perceiving color on psychological functioning"
- **Emotion Recognition**: Li & Deng (2020) - "Deep Facial Expression Recognition: A Survey"

---

*Last updated: 2025-10-02*
*Notebook version: 1.0*
