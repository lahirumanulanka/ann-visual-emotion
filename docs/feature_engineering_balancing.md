<!--
	Summary generated from `notebooks/02_feature_engineering_balancing.ipynb` (static code & intent review; notebook not executed here).
	Re-run the notebook to obtain live numbers, plots, and CSV artifacts, then refresh this document if needed.
-->

# 🛠️ Feature Engineering & Balancing Pipeline

This document distills the design and intended outputs of the feature engineering + dataset balancing workflow captured in the notebook `02_feature_engineering_balancing.ipynb`.

## 1. Objectives
1. Scan raw dataset (`data/raw/FullDataEmoSet`) and quantify per-class distribution.
2. Define a balancing target (max / median / custom per-class dict).
3. Oversample minority classes via on-the-fly augmentation (configurable cap per original image).
4. (Optionally) undersample majority classes (disabled by default).
5. Resize all images to a uniform spatial resolution (default 224×224) into a working output tree.
6. Compute RGB per-channel mean/std BEFORE and AFTER transformations (for normalization).
7. Provide sample visual sanity checks of augmented images.
8. Produce stratified train / val / test splits with reproducibility (seeded) and export CSV manifests.
9. Generate balancing diagnostics (max deviation %, tolerance compliance) and per-split distributions.

## 2. Key Configuration Parameters

| Name | Purpose | Default / Example |
|------|---------|-------------------|
| `RAW_ROOT` | Source dataset root | `../data/raw/FullDataEmoSet` |
| `WORK_ROOT` | Balanced output root | `../data/processed/FullDataEmoSet_balanced` |
| `IN_PLACE` | Toggle writing into raw tree | `False` |
| `RUN_WRITE` | Master switch for any disk writes | `True` (set to `False` for dry-run) |
| `BALANCE_STRATEGY` | Target sizing per class (`'max'`, `'median'`, or dict) | `'max'` |
| `UNDERSAMPLE` | Reduce majority classes to target | `False` |
| `AUG_PER_IMAGE_LIMIT` | Max synthetic variants per original | `4` |
| `TARGET_SIZE` | Standardized output image size | `(224, 224)` |
| `INTERPOLATION` | Resize interpolation mode | `'bilinear'` |
| `TRAIN_PCT, VAL_PCT, TEST_PCT` | Split fractions (must sum to 1) | `0.7, 0.15, 0.15` |
| `SEED` | Random seed for reproducibility | `42` |
| `SAMPLE_PER_CLASS` | Sample size for mean/std estimation (pre & post) | `200` |

## 3. Workflow Overview

1. **Scan & Index**: Recursively enumerate image files per class (extensions: jpg/jpeg/png/bmp/gif). Build `df(path,label)`.
2. **Baseline Distribution**: Compute counts, percentages, descriptive stats.
3. **Pre-Transform Stats**: Random sample (per class) → RGB conversion → channel mean & std accumulation.
4. **Target Derivation**: Build `target_per_class` mapping:
	 - `max`: all classes boosted to size of current max class.
	 - `median`: boost to median class count.
	 - `dict`: explicit numeric targets.
5. **Augmentation Loop**:
	 - Always copy & resize originals first.
	 - If deficit vs. target remains: cycle originals, generate augmented variants (bounded by `AUG_PER_IMAGE_LIMIT` per source).
	 - PIL-based transforms (probabilistic): horizontal flip, vertical flip, rotation (±20°), brightness, contrast, sharpness adjustments.
6. **(Optional) Undersampling**: If enabled & class exceeds target, simply skip writing excess images (no deletion when out-of-place).
7. **Post-Transform Stats**: Re-sample from output tree for updated per-channel mean/std + visual panel of up to 8 samples.
8. **Split Generation**: Stratified train/val/test using scikit-learn `train_test_split` (two-stage with preserved proportions).
9. **Manifest Export** (if `RUN_WRITE` and balanced output exists):
	 - CSVs: `train.csv`, `val.csv`, `test.csv` → columns: `path,label` (paths remapped to a container-friendly root).
	 - `status.csv`: Each class → target vs. actual realized count.
	 - `status.json`: Structured summary (see schema below).
10. **Diagnostics**: Balance check function reports min/max/mean/std + maximum % deviation and tolerance compliance (±2%).

## 4. Augmentation Strategy Details

| Transform | Parameterization | Notes |
|-----------|------------------|-------|
| Horizontal flip | 20% prob | Mirrors expression (generally safe) |
| Vertical flip | 10% prob | Use cautiously; faces inverted may be unrealistic |
| Rotation | Uniform integer in [-20°, +20°] | Mild geometric variance |
| Brightness | Factor ∈ [0.7, 1.3] | Illumination robustness |
| Contrast | Factor ∈ [0.7, 1.3] | Dynamic range variation |
| Sharpness | Factor ∈ [0.7, 1.5] | Texture emphasis/de-emphasis |

Safeguards:
- Limit per-source augmentations via `AUG_PER_IMAGE_LIMIT`.
- Cycle iterator ensures broad coverage of different seed images.

## 5. Output Directory Structure

Assuming out-of-place processing (`IN_PLACE=False`):

```
data/
	raw/FullDataEmoSet/<class>/<orig>.png
	processed/FullDataEmoSet_balanced/
		<class>/  # originals resized + augmented files
	processed/EmoSet_splits/
		raw_balanced/<class>/*.png  # normalized copy for manifest stability
		train.csv
		val.csv
		test.csv
		status.csv
		status.json
```

## 6. `status.json` Schema

```json
{
	"total_images": <int>,
	"class_distribution": {"train": {"class": count, ...}, "val": {...}, "test": {...}},
	"overall_class_distribution": {"class": total_count, ...},
	"splits_fraction": {"train": 0.7, "val": 0.15, "test": 0.15},
	"image_size": {"width": 224, "height": 224},
	"created_timestamp": "<ISO8601 UTC>",
	"seed": 42,
	"target_count_per_class": 12345 | {"class": target_count, ...}
}
```

## 7. Reproducibility Considerations
- Deterministic splits via consistent `SEED`.
- Augmentation randomness still seeded but may differ if code or library versions change.
- Container path remapping ensures training pipelines can rely on stable `/data/...` paths.
- `status.csv` allows quick verification of whether balancing achieved intent.

## 8. Potential Improvements
| Area | Suggestion | Benefit |
|------|------------|---------|
| Aug diversity | Add color jitter hue/sat (bounded) | Broader invariance |
| QA | Hash original + augmented files | Integrity & dedupe detection |
| Logging | Emit per-class augmentation yield stats | Debug minority saturation |
| Performance | Parallelize augmentation with multiprocessing | Faster large-class expansion |
| Normalization | Persist computed mean/std to JSON for training reuse | Eliminates recomputation |
| Skew handling | Option for focal loss weight JSON export | Smooth integration downstream |

## 9. Usage Modes

| Mode | Settings | Outcome |
|------|----------|---------|
| Dry run | `RUN_WRITE=False` | Plan review without filesystem changes |
| Balance to max | `BALANCE_STRATEGY='max'` | Uniform dataset at majority size |
| Balance to median | `BALANCE_STRATEGY='median'` | Less aggressive oversampling |
| Custom targets | `BALANCE_STRATEGY={'happy':8000,'sad':8000,...}` | Fine-grained control |
| In-place resize only | `IN_PLACE=True`, disable augmentation | Standardize resolution w/o new files |

## 10. Failure / Edge Cases Handled
- Corrupt images: silently skipped (wrapped in try/except with continue).
- Missing raw root: assertion halts early with clear message.
- Non-summing split fractions: assertion guard.
- Empty output on first run with `RUN_WRITE=False`: Split logic transparently falls back to raw paths.
- Duplicate augmentations: capped by per-source limit + cycle break when quota reached.

## 11. How to Refresh This Report
1. Open and execute the notebook sequentially (set `RUN_WRITE=True` when ready to materialize changes).
2. Inspect generated `status.json` and CSVs.
3. Optionally capture actual numeric class distributions and paste them into a new section below.

## 12. Placeholder for Actual Run Metrics
Add real metrics below after first full execution:

```
Run date: <YYYY-MM-DD>
Total balanced images: <int>
Target per class: <int or dict>
Final split sizes: train=<n>, val=<n>, test=<n>
Pre mean/std: [R,G,B]=[...] / [...]
Post mean/std: [R,G,B]=[...] / [...]
Max deviation after balancing: <pct>
Balanced within tolerance (±2%): <True/False>
```

---

For raw dataset exploratory statistics see `docs/eda.md`.
