# Quick Start Guide: Comprehensive EDA Notebook

This guide helps you run the new `03_exploratory_data_analysis.ipynb` notebook and use its outputs for model training.

---

## 🚀 Quick Start

### 1. Run the EDA Notebook

```bash
cd notebooks
jupyter notebook 03_exploratory_data_analysis.ipynb
```

**Or run all cells programmatically:**

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 03_exploratory_data_analysis.ipynb
```

### 2. Check the Outputs

After execution, check the output directory:

```bash
ls -lh ../data/processed/EmoSet_splits_eda/
```

You should see:
- ✓ `balanced_dataset_with_features.csv` (balanced dataset)
- ✓ `feature_info.json` (feature metadata)
- ✓ `eda_summary.json` (statistics)
- ✓ `EDA_REPORT.md` (summary report)
- ✓ PNG visualization files

---

## 📊 What the Notebook Does

### Step-by-Step Process

1. **Loads Data** - Reads train/val/test splits from `data/processed/EmoSet_splits/`
2. **Missing Values** - Checks for and reports any missing data
3. **Class Distribution** - Analyzes imbalance across 6 emotion classes
4. **Feature Extraction** - Extracts 10 statistical features from images
5. **Outlier Detection** - Uses IQR and Z-score methods
6. **Outlier Treatment** - Caps outliers at 1st/99th percentiles
7. **Feature Engineering** - Creates 6 new features (brightness, contrast, etc.)
8. **SMOTE Balancing** - Balances classes using synthetic oversampling
9. **Visualization** - Generates comparison plots
10. **Saves Results** - Exports balanced dataset and metadata

### Key Outputs

**Balanced Dataset**
- 5,250 samples (from 5,000 original)
- 875 samples per class (perfect balance)
- 17 features total (10 extracted + 6 engineered + 1 label)

**New Engineered Features**
1. `brightness` - Mean pixel intensity
2. `contrast` - Standard deviation of intensity
3. `color_balance_rb` - Red/Blue ratio
4. `edge_density` - Texture complexity
5. `rgb_variance` - Color diversity
6. `color_saturation` - Color intensity

---

## 🔗 Integration with Model Training

### Option 1: Use the Balanced Dataset Directly

If you want to train on the balanced dataset with engineered features:

```python
import pandas as pd
import json

# Load balanced dataset
df = pd.read_csv('../data/processed/EmoSet_splits_eda/balanced_dataset_with_features.csv')

# Load feature info
with open('../data/processed/EmoSet_splits_eda/feature_info.json') as f:
    feature_info = json.load(f)

# Use engineered features
X = df[feature_info['engineered_features']].values
y = df['label'].values

# Now split and train your model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Option 2: Apply the Same Preprocessing to New Data

If you want to apply the same feature engineering to the full dataset:

```python
def engineer_features(df):
    """Apply the same feature engineering as in the EDA notebook"""
    df['brightness'] = df['mean_intensity']
    df['contrast'] = df['std_intensity']
    df['color_balance_rb'] = df['mean_r'] / (df['mean_b'] + 1e-5)
    df['edge_density'] = df['std_intensity'] / (df['mean_intensity'] + 1e-5)
    df['rgb_variance'] = df[['mean_r', 'mean_g', 'mean_b']].var(axis=1)
    df['color_saturation'] = (df[['mean_r', 'mean_g', 'mean_b']].max(axis=1) - 
                              df[['mean_r', 'mean_g', 'mean_b']].min(axis=1))
    return df

# Apply to your dataset
df_processed = engineer_features(df_original)
```

### Option 3: Use in CNN Training Notebook

Update `CNN_with_Transfer_Learning.ipynb` to use the balanced dataset:

```python
# Instead of loading from original splits
# train_df = pd.read_csv('../data/processed/EmoSet_splits/train.csv')

# Load the balanced dataset
balanced_df = pd.read_csv('../data/processed/EmoSet_splits_eda/balanced_dataset_with_features.csv')

# Split back into train/val/test
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(balanced_df, test_size=0.3, stratify=balanced_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
```

---

## ⚙️ Configuration Options

### Processing Sample Size

By default, the notebook processes 5,000 images for speed. To process all images:

```python
# In the notebook, change:
USE_SAMPLE = False  # Process all ~80,000 images
# or
SAMPLE_SIZE = 10000  # Process 10,000 images
```

### SMOTE Parameters

Adjust class balancing behavior:

```python
# In the notebook, modify:
smote = SMOTE(
    random_state=42,      # For reproducibility
    k_neighbors=5,        # Number of neighbors (default: 5)
    sampling_strategy='auto'  # Balance all classes equally
)
```

### Outlier Treatment

Change the capping percentiles:

```python
# In the notebook, modify:
def cap_outliers(df, column, lower_percentile=1, upper_percentile=99):
    # Change to lower_percentile=5, upper_percentile=95 for less aggressive capping
```

---

## 📈 Interpreting the Results

### Check the EDA Report

```bash
cat ../data/processed/EmoSet_splits_eda/EDA_REPORT.md
```

Key metrics to look for:
- **Imbalance Ratio**: Should be close to 1.0 after SMOTE
- **Synthetic Samples**: Number of new samples generated
- **Feature Count**: Should show 16 features (10 extracted + 6 engineered)

### Visualizations

View the generated plots:
- `class_distribution_before_balancing.png` - Original distribution
- `class_distribution_comparison.png` - Before/after SMOTE

---

## 🔍 Troubleshooting

### Issue: "Module not found: tqdm"

```bash
pip install tqdm
```

### Issue: "Module not found: imbalanced-learn"

```bash
pip install imbalanced-learn
```

### Issue: "File not found" errors

Make sure you're running from the `notebooks/` directory:

```bash
cd /path/to/ann-visual-emotion/notebooks
jupyter notebook 03_exploratory_data_analysis.ipynb
```

### Issue: Notebook takes too long

Reduce the sample size:

```python
USE_SAMPLE = True
SAMPLE_SIZE = 1000  # Faster for testing
```

### Issue: Out of memory

Process in smaller batches or reduce sample size:

```python
SAMPLE_SIZE = 2000  # Reduce memory usage
```

---

## 📚 Next Steps

After running the EDA:

1. **Review the Report** - Check `EDA_REPORT.md` for insights
2. **Validate Balance** - Ensure classes are properly balanced
3. **Model Training** - Use the balanced dataset in `CNN_with_Transfer_Learning.ipynb`
4. **Feature Selection** - Analyze which engineered features improve performance
5. **Iterate** - Adjust parameters based on model performance

---

## 📖 Documentation

For detailed information:
- **Workflow Details**: See `docs/comprehensive_eda.md`
- **Feature Engineering Rationale**: See `docs/comprehensive_eda.md` Section 7
- **SMOTE Explanation**: See `docs/comprehensive_eda.md` Section 8

---

## 💡 Tips

1. **Start Small**: Use sample mode (5,000 images) for initial runs
2. **Check Outputs**: Always verify the generated files before training
3. **Save Results**: Keep the JSON files for reproducibility
4. **Document Changes**: If you modify parameters, note them in your experiments
5. **Version Control**: Commit the outputs for tracking

---

## 🤝 Questions?

If you encounter issues or need clarification:
1. Check the detailed documentation in `docs/comprehensive_eda.md`
2. Review the inline comments in the notebook
3. Examine the output files in `data/processed/EmoSet_splits_eda/`

---

*Last updated: 2025-10-02*
