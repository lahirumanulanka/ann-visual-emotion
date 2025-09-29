## Emotion Dataset – Exploratory Data Analysis (EDA)

This document summarizes the exploratory data analysis performed in the notebook `notebooks/01_eda.ipynb` on the raw emotion image dataset (`data/raw/FullDataEmoSet`). It is meant to be a concise, human‐readable reference for methodology, reproducibility, key diagnostics, and recommended follow‑ups — without needing to open the notebook.

---

### Objectives

1. Inspect raw dataset structure (class folders) and count images per class.
2. Validate image readability / integrity.
3. Quantify class imbalance (counts & percentages + basic descriptive stats).
4. Capture pixel geometry statistics (width, height, area, aspect ratio).
5. Visual sanity check via sampled image grid.
6. Flag potential quality issues (very small images, extreme aspect ratios, uncommon modes or formats, unreadable files).

---

### Data Source & Layout

Root folder scanned:

```
data/raw/FullDataEmoSet/
└── <class_name>/  (each immediate subdirectory = emotion class)
```

All immediate subfolders are treated as labels. Images are collected recursively under each class directory. Supported extensions in the current notebook logic: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif` (case‑insensitive). Additional formats (e.g., WebP, TIFF) are ignored by design for simplicity.

---

### Core Variables & Parameters

| Variable | Purpose |
|----------|---------|
| `RAW_ROOT` | Resolved path to the raw dataset root (`../data/raw/FullDataEmoSet`). |
| `labels` | Sorted list of class names (folder names). |
| `IMG_EXTS` | Whitelist of accepted file extensions. |
| `MAX_IMAGES_PER_CLASS` | Optional cap (set to `None` in current run → uses all images). |
| `raw_df` | Main DataFrame of successfully read images and metadata. |
| `bad_images` | List of unreadable/corrupted or otherwise failing images (path + error). |
| `MIN_W`, `MIN_H` | Minimum width & height thresholds to flag small images (32×32). |
| `ASPECT_MIN`, `ASPECT_MAX` | Aspect ratio (w/h) thresholds for extremes (0.5 – 2.0). |

Each row in `raw_df` includes:

```
path, label, width, height, area, aspect, format, mode
```

Where:
* `area = width * height`
* `aspect = width / height`
* `format` & `mode` come from PIL (two opens: one with `verify()` for integrity, second to extract size).

---

### Processing Workflow (Notebook Steps)

1. **Scan Class Folders** – Enumerate subdirectories under `RAW_ROOT` and treat them as labels.
2. **Image Index & Integrity Check**  
   * Recursively gather files with allowed extensions.  
   * (Optional) truncate per class if `MAX_IMAGES_PER_CLASS` is set.  
   * For each file: 
	 * Open with `PIL.Image.open(...); im.verify()` to catch structural corruption. 
	 * Re-open to extract `(width, height)`, plus format/mode.  
   * Collect failures (path + exception) in `bad_images`.
3. **Aggregate Metadata** – Build `raw_df` DataFrame of readable images.  
4. **Class Distribution Analysis** – Compute counts, percentages, and descriptive statistics (mean, std, min, quartiles, max).  
5. **Visualization** – Bar charts for counts & percentages; histograms for width, height, clipped area (99th percentile), and aspect ratio (also clipped at 99th percentile).  
6. **Sample Image Grid** – For up to 6 classes (configurable), randomly sample (deterministic seed) a small number of images (default: 4) to visually confirm labeling consistency and content quality.  
7. **Format / Mode & Quality Flags** – Tabulate `mode` and `format`; flag small images and those with extreme aspect ratios for potential pre‑processing decisions (e.g., cropping, padding, exclusion).  

---

### Key Diagnostics (What to Look For)

| Diagnostic | Insight / Interpretation |
|------------|--------------------------|
| Class Counts vs. Percentages | Identify imbalance; large skew may require re‑sampling, class weights, or augmentation. |
| Descriptive Stats of Counts | High std / large min→max gap confirms imbalance severity. |
| Pixel Geometry Histograms | Reveal outliers (e.g. huge images dominating area distribution). |
| Aspect Ratio Distribution | Non‑portrait/landscape extremes → may distort when resized; consider cropping or padding. |
| Mode / Format Diversity | Mixed color spaces (e.g., `RGBA`, `L`) may need conversion to uniform `RGB`. |
| Small Images | Might contribute noise; decide on upscaling vs. exclusion. |
| Corrupted / Unreadable Files | Should be removed or quarantined before training pipelines. |

---

### Example Pseudocode Snippets

Below are condensed patterns (not copy‑paste identical to the notebook, simplified for clarity):

```python
from pathlib import Path
from PIL import Image
import pandas as pd

RAW_ROOT = Path('data/raw/FullDataEmoSet').resolve()
labels = sorted([p.name for p in RAW_ROOT.iterdir() if p.is_dir()])

records, bad = [], []
for lab in labels:
	for img_path in (RAW_ROOT / lab).rglob('*'):
		if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
			continue
		try:
			with Image.open(img_path) as im:  # integrity check
				im.verify()
			with Image.open(img_path) as im2:
				w, h = im2.size
				records.append({
					'path': str(img_path), 'label': lab,
					'width': w, 'height': h,
					'area': w * h,
					'aspect': w / h if h else float('nan'),
					'format': im2.format, 'mode': im2.mode
				})
		except Exception as e:
			bad.append({'path': str(img_path), 'label': lab, 'error': str(e)})

raw_df = pd.DataFrame(records)
```

---

### Interpreting Pixel Statistics

| Metric | Why It Matters |
|--------|----------------|
| Width / Height | Guides resizing strategy; extremely large images could be downscaled early to save compute. |
| Area | Proxy for potential memory footprint; helps detect outlier megapixel images. |
| Aspect (w/h) | Extreme ratios may stretch after naive resizing; consider letterboxing or center‑cropping. |

The notebook clips area and aspect ratio histograms at the 99th percentile to reduce the visual dominance of long tails.

---

### Quality Flags & Threshold Rationale

| Flag | Threshold | Rationale |
|------|-----------|-----------|
| Small Images | width < 32 or height < 32 | Very low information content; may degrade model training or require special augmentation. |
| Extreme Aspect | aspect < 0.5 or > 2.0 | Could distort face/emotion features after uniform scaling; review individually. |

Thresholds are deliberately conservative and should be refined once downstream preprocessing (e.g., face cropping, alignment) is finalized.

---

### Reproducibility Notes

Run the notebook end‑to‑end to regenerate all numbers & plots:

1. Ensure the raw dataset exists at `data/raw/FullDataEmoSet`.
2. Open `notebooks/01_eda.ipynb`.
3. (Optional) Set `MAX_IMAGES_PER_CLASS` before executing the indexing cell if you want a faster exploratory pass.
4. Execute cells sequentially.

If you need a non‑interactive script version, you could extract the scanning + summary logic into a future module (e.g., `src/data/scan_raw.py`).

---

### Potential Enhancements (Next Steps)

1. Duplicate / near‑duplicate detection (e.g., perceptual hashing) to reduce redundancy.
2. Face detection coverage analysis (how many images contain valid faces?).
3. Color statistics (mean / std per channel) for normalization baselines.
4. Augmentation impact preview (simulate flips, crops, brightness jitter) on class balance.
5. Stratified train/val/test leakage checks (ensure no near‑duplicates across splits).
6. Automatic exclusion list generation for corrupted / extreme outlier images.
7. Reporting script that exports a markdown or HTML snapshot (including plots) for CI artefacts.

---

### Quick Summary

The EDA pipeline reliably: (a) enumerates classes, (b) filters & validates images, (c) quantifies imbalance, and (d) flags geometry and format anomalies. These diagnostics should be reviewed before finalizing preprocessing (resize strategy, augmentation, class balancing). Implementing the proposed enhancements will further harden dataset quality and improve downstream model robustness.

---

*Last updated: automated documentation sync from `01_eda.ipynb`.*

---

### Current Dataset Statistics Snapshot

This snapshot was generated from the raw folder at the time of documentation update.

Timestamp (UTC): 2025-09-29T08:42:47Z

**Global Overview**

- Total images: 43,756
- Number of classes: 6
- Classes: angry, fearful, happy, neutral, sad, surprised
- Corrupted / unreadable images detected: 0

**Class Distribution**

| Class | Count | Percentage |
|-------|-------|------------|
| angry | 5,089 | 11.63% |
| fearful | 4,589 | 10.49% |
| happy | 13,370 | 30.56% |
| neutral | 8,268 | 18.90% |
| sad | 7,504 | 17.15% |
| surprised | 4,936 | 11.28% |
| **Total** | **43,756** | **100%** |

**Class Count Descriptive Stats**

| Metric | Value |
|--------|-------|
| count | 6 |
| mean | 7,292.67 |
| std | 3,336.16 |
| min | 4,589 |
| 25% | 4,974.25 |
| 50% (median) | 6,296.50 |
| 75% | 8,077.00 |
| max | 13,370 |

Interpretation: Strong imbalance — the largest class (happy) is ~2.9× the smallest (fearful). Consider class weighting, augmentation, or sampling adjustments.

**Pixel Geometry (All Images)**

| Dimension | count | mean | std | min | 25% | 50% | 75% | max |
|-----------|-------|------|-----|-----|-----|-----|-----|-----|
| width | 43,756 | 74.52 | 70.49 | 48 | 48 | 48 | 100 | 640 |
| height | 43,756 | 74.52 | 70.46 | 48 | 48 | 48 | 100 | 640 |
| area | 43,756 | 10,520.47 | 44,964.60 | 2,304 | 2,304 | 2,304 | 10,000 | 409,600 |
| aspect | 43,756 | 1.0000 | 0.00121 | 0.9223 | 1.0 | 1.0 | 1.0 | 1.0802 |

Notes:
1. A very large proportion of images are square 48×48 (classic FER sizing) — reflected by identical quartiles for width/height/area.
2. Presence of larger images (up to 640×640) introduces scale variance; downstream pipeline should standardize resolution early.
3. Aspect ratios are extremely tight around 1.0 (std ≈ 0.0012), simplifying augmentation (no letterboxing needed for most samples).

**Image Formats & Modes**

| Format | Count | Share |
|--------|-------|-------|
| PNG | 28,273 | 64.58% |
| JPEG | 15,483 | 35.42% |

| Mode | Count | Share |
|------|-------|-------|
| L (grayscale) | 43,756 | 100.0% |

Implication: All images are grayscale. If models expect 3‑channel input, conversion will simply replicate channels or map to 1→3 tensor. Color augmentations are not applicable; focus on geometric / intensity transforms.

**Imbalance Considerations**

- Majority class (happy) accounts for 30.56% of data; minority classes hover near 10–12%.
- Potential strategies: class-weighted loss, focal loss, oversampling of minority classes, or targeted augmentation (e.g., mixup, random erasing) biased toward underrepresented labels.

**Recommended Preprocessing Based on Stats**

1. Standardize all images to a consistent resolution (e.g., 48×48 or upscale to 96×96 with interpolation + light sharpening if needed).
2. Normalize pixel intensities (mean/std computed from training subset only; global grayscale simplifies channel stats).
3. Evaluate benefit of histogram equalization / CLAHE for contrast normalization, especially for subtle emotional cues.
4. Introduce augmentation tuned for facial expression robustness: slight rotations (±10°), horizontal flips (if semantics unchanged), mild Gaussian noise, brightness/contrast jitter.
5. Monitor per-class effective sample count post-augmentation to avoid overfitting to synthetic variants of minority classes.

**Future Enhancements to This Section**

- Persist snapshot JSON alongside markdown for change tracking across dataset revisions.
- Add plots (distribution charts) exported from notebook under `docs/figures/eda/` and embed.
- Track temporal drift if dataset is periodically updated.

---


