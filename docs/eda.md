<!--
  Auto-generated / updated from `notebooks/01_eda.ipynb`.
  If you re-run the notebook, re-run the data indexing cells and then refresh this file.
-->

# 📊 Exploratory Data Analysis (EDA) — Emotion Recognition Dataset

This report consolidates statistics from the raw emotion image dataset and (legacy) curated split metadata.

## 1. Raw Dataset Overview

The raw dataset (scanned directly from `data/raw/FullDataEmoSet/`) contains:

- **Total images (readable):** 43,756
- **Detected classes:** 6 (from folder names)

| Class | Count | Percent |
|-------|-------|---------|
| angry | 5,089 | 11.63% |
| fearful | 4,589 | 10.49% |
| happy | 13,370 | 30.56% |
| neutral | 8,268 | 18.90% |
| sad | 7,504 | 17.15% |
| surprised | 4,936 | 11.28% |

**Descriptive statistics (counts per class):**

- Min: 4,589  
- Max: 13,370  
- Mean: 7,292.67  
- Std Dev: 3,336.16  
- Quartiles (25% / 50% / 75%): 4,974.25 / 6,296.50 / 8,077.00

### Notes on Class Schema
The earlier curated split metadata referenced 8 labels (`amusement`, `anger`, `awe`, `contentment`, `disgust`, `excitement`, `fear`, `sadness`). The raw folder scan currently surfaces **6 emotion categories** (likely a consolidated / alternative mapping: e.g., `happy` vs amusement/contentment, `neutral`, `surprised` vs excitement/awe, etc.). A reconciliation step is recommended (see Recommendations section).

## 2. Image Format & Mode

| Property | Observation |
|----------|-------------|
| Modes | All images detected as grayscale (`L`) — 43,756 |
| Formats | PNG: 28,273 (64.6%), JPEG: 15,483 (35.4%) |
| Corrupt / unreadable | 0 |
| Small images (<32×32) | 0 |
| Extreme aspect ratio (<0.5 or >2.0) | 0 |

All images can be safely loaded and resized. Uniform grayscale mode suggests either original grayscale sources or prior preprocessing. If models expect 3-channel input, a channel repeat / conversion to RGB will be needed.

## 3. Pixel & Dimension Characteristics

The notebook collects width, height, area, and aspect ratio per image (distribution visualizations exist in the notebook). Key takeaways:

- No pathological tiny images.
- Aspect ratios fall within a constrained band (no extreme distortions), simplifying augmentation policy design.
- Area distribution likely skewed (typical for heterogeneous web-sourced datasets); heavy-tail images can be uniformly resized to 224×224 without severe artifacts.

## 4. Legacy Split Metadata (Historical Reference)

Previous curated splits (not regenerated in this run) reported 8-class distributions across train/val/test (example schema below). These are retained for backward compatibility and documentation but do **not** match the 6-folder raw taxonomy discovered now.

| (Historical) Emotion | Train | Val | Test | Total |
|----------------------|-------|-----|------|-------|
| Amusement | 3,046 | 381 | 381 | 3,808 |
| Anger | 1,119 | 140 | 140 | 1,399 |
| Awe | 491 | 62 | 61 | 614 |
| Contentment | 2,512 | 314 | 314 | 3,140 |
| Disgust | 21 | 3 | 3 | 27 |
| Excitement | 4,910 | 614 | 614 | 6,138 |
| Fear | 362 | 45 | 45 | 452 |
| Sadness | 1,403 | 175 | 176 | 1,754 |

Severe sparsity of the Disgust class (n=27) made robust modeling across all 8 categories challenging; consolidation may have been intentional.

## 5. Data Quality Summary

- ✅ 100% of indexed files load without error.  
- ✅ Consistent single-channel mode (handled via expansion to 3-channels at training time if using pretrained CNNs).  
- ✅ No extreme resolutions or aspect ratios requiring filtering.  
- ⚠ Class imbalance persists (largest class `happy` ~2.9× the smallest `fearful`).  
- ⚠ Schema divergence (6 raw vs 8 historical) requires explicit label mapping documentation.

## 6. Recommended Next Steps

1. Label Mapping Audit:
	- Create a JSON mapping (e.g. `label_mapping.json`) clarifying how legacy 8-class labels collapse into 6 (if applicable).
2. Stratified Split Regeneration:
	- Recompute train/val/test using the 6-class taxonomy to avoid leakage and mismatch.
3. Class Imbalance Mitigation:
	- Techniques: weighted loss (CrossEntropy with class weights), mixup, focal loss, or minority oversampling.
4. Channel Handling:
	- Implement deterministic conversion `img = img.convert('RGB')` during dataset load.
5. Provenance Tracking:
	- Store a manifest (CSV/Parquet) with hash + original path for reproducibility.
6. Augmentation Policy:
	- Light geometric + photometric (avoid over-distorting expressions). Consider histogram equalization or CLAHE if contrast varies.

## 7. Planned / Current Preprocessing Pipeline

| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Load & verify | Fail fast on corrupt assets |
| 2 | Convert to RGB | Match pretrained backbone expectations |
| 3 | Resize to 224×224 | Standard input size for CNN/ViT small variants |
| 4 | (Optional) Center crop / pad | Normalize framing |
| 5 | Augment (train only) | Improve generalization |
| 6 | Normalize (ImageNet mean/std) | Align with pretrained weights |

Normalization stats (ImageNet):
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

## 8. Reproducibility Metadata

- Notebook: `notebooks/01_eda.ipynb`
- Execution date: 2025-09-26
- Environment: Python (see project `pyproject.toml` / `requirements.txt`), executed via repo virtual environment.
- Parameters: `MAX_IMAGES_PER_CLASS = None` (full scan)

## 9. Changelog

| Date | Change |
|------|--------|
| 2025-09-26 | Rebuilt EDA from raw folders; added 6-class statistics and legacy schema reconciliation. |

---

For deeper visual summaries (histograms, sample grids), refer to the executed notebook outputs.
