# 🧪 GenAI Synthetic Data Generation Findings

<!--
	Derived from `notebooks/03_synthetic_gen_ai_generation.ipynb` (static inspection only; notebook not re-executed here).
	Update this file after a fresh run to replace placeholder metrics with actual outputs.
-->

## 1. Objective
Augment the existing emotion recognition dataset with high-quality, labeled synthetic facial expression images to reach a larger, more balanced corpus while maintaining quality, diversity, and ethical safeguards.

## 2. Target & Scope
| Aspect | Value / Strategy |
|--------|------------------|
| Target total images | 83,000 (`TARGET_TOTAL`) *(Notebook text header mentions 100k; config currently set to 83k → confirm alignment)* |
| Base image size | 224×224 pixels |
| Color mode | Grayscale (`L`) enforced post-generation |
| Generation model | `runwayml/stable-diffusion-v1-5` (Diffusers) |
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Hardware | CUDA if available; falls back to CPU |
| Seed | 20250924 (deterministic reproducibility) |
| Synthetic fraction cap (train) | 0.60 per class |
| Blur variance threshold | ≥ 60.0 (Laplacian var) |
| Near-duplicate rejection | pHash Hamming distance ≤ 4 rejected |
| Face detection policy | Accept exactly 1 face when `FACE_DETECT_SINGLE_ONLY=True` |

## 3. Pipeline Stages
1. Load original balanced dataset (`RAW_ROOT`) and compute per-class counts.
2. Plan synthetic needs per class to reach uniform per-class target (ceil of target/num_classes).
3. Build emotion-specific prompts + stylistic diversity suffixes.
4. Generate images sequentially (batch size 1 for memory stability).
5. Apply filtering: single-face detection → blur threshold → perceptual hash dedup.
6. Convert to grayscale + resize (enforced again in a safety pass).
7. (Optional) Face cropping variant produced into a separate `*_cropped` directory.
8. Aggregate metadata (`synthetic_meta.csv` / `.jsonl`).
9. Merge original + synthetic; compute per-class synthetic fractions.
10. Apply stratified splitting with synthetic fraction enforcement for training split.
11. Export curated image copies (or hardlinks) + CSV manifests + extended `status.json`.
12. Mirror curated set back into original splits directory (label map & stats refresh).
13. Diagnostics: balance deviation, synthetic ratio plots, heatmaps, per-class stacks.
14. Integrity checks (mode, size, grayscale consistency, sample stats).
15. Environment & reproducibility capture (`env_report.json`).
16. Disk usage summary & final counts.

## 4. Prompt Engineering
Prompts are emotion-specific natural language portrait descriptors with additive stylistic suffixes to encourage diversity (lighting, lens quality, detail). Suffixes cycle deterministically by index. A SHA-256 hash of the base prompt dict (`prompt_dict_hash`) is stored to track prompt versioning for model provenance.

Example structure:
```
"portrait photo of a person smiling, expressive happy face, natural skin texture, neutral background, well lit, ultra detailed, photorealistic"
```

## 5. Quality & Diversity Controls
| Control | Rationale |
|---------|-----------|
| Single-face enforcement | Avoid multi-face ambiguity for label association |
| Laplacian blur variance | Remove overly smooth / defocused generations |
| Perceptual hash dedup | Reduce redundant near-identical samples |
| Grayscale normalization | Align with downstream model expectations & reduce color confounders |
| Synthetic fraction cap | Prevent synthetic over-dominance in training signal |
| Deterministic seeding | Reproducibility / auditability |

## 6. Metadata Outputs
| File | Description |
|------|-------------|
| `synthetic_meta.csv` | Tabular synthetic sample metadata |
| `synthetic_meta.jsonl` | Line-delimited JSON for each accepted synthetic image |
| `synthetic_meta_cropped.csv` | Cropped face variant metadata (if cropping step executed) |
| `status.json` | Extended dataset composition metrics (original vs synthetic) |
| `env_report.json` | Environment + library + parameter provenance |
| `label_map.json` | Label → index mapping (mirrored splits) |
| `train/val/test.csv` | Final merged curated splits (container path normalized) |

### `status.json` (Extended) Structure
```
{
	total_images,
	total_original,
	total_synthetic,
	synthetic_fraction,
	per_class: { label: { original, synthetic, total, synthetic_fraction } },
	splits_fraction: { train, val, test },
	image_size: { width, height, mode },
	seed,
	generation_model_id,
	prompts_version_hash,
	quality_thresholds: { blur_var_min, hash_dist_max, phash_size },
	created_timestamp
}
```

## 7. Balance & Diagnostics
Diagnostics compute deviation from the mean class count and mark whether within a ±2% tolerance. Additional visual diagnostics described in the notebook (not reproduced here) include:
* Stacked bar: original vs synthetic counts per class.
* Synthetic fraction bar chart.
* Heatmap of synthetic fraction per class per split.
* Grid: original vs synthetic sample thumbnails (qualitative drift inspection).

## 8. Integrity & Statistical Checks
Steps ensure:
* All grayscale images sized exactly 224×224 after enforcement pass.
* Random subsample mean/std for original vs synthetic grayscale distributions (used for monitoring potential distributional divergence).
* Environment capture secures Python / Torch / Diffusers versions for reproducibility.

## 9. Risk & Ethical Considerations
| Area | Consideration | Mitigation |
|------|---------------|------------|
| Demographic bias | Prompts may implicitly bias phenotypes | Expand prompt pool with explicit diversity descriptors in future runs |
| Overfitting to synthetic artifacts | Models may learn generation quirks | Cap synthetic proportion; mix real-first augmentation |
| Semantic drift | Emotion semantics may blur (e.g., fear vs surprise) | Add classifier-in-the-loop validation or CLIP-based semantic scoring |
| Privacy | Synthetic only; no real user facial data added | Maintain clear documentation and hashes for audit |
| Duplicate leakage | Near-duplicate acceptance reduces variance | pHash + threshold, optionally add perceptual embedding distance |

## 10. Observed / Potential Metrics (Placeholders)
Populate after execution:
```
Original count: <int>
Synthetic generated: <int>
Combined total: <int>
Train/Val/Test sizes: <t>/<v>/<te>
Overall synthetic fraction: <float>
Per-class synthetic fractions: {label: fraction, ...}
Max class deviation (%): <float>
Within ±2% tolerance: <bool>
Mean/std grayscale (orig): mean=<m1>, std=<s1>
Mean/std grayscale (synthetic): mean=<m2>, std=<s2>
Disk usage (orig / synthetic / curated): <sizes>
Rejected (multi-face / blur / dup): <counts if logged>
```

## 11. Recommended Enhancements
| Category | Improvement | Benefit |
|----------|------------|---------|
| Prompt diversity | Add demographic + age + lighting variants cycles | Reduce bias & improve generalization |
| Validation | Integrate automated emotion classifier scoring filter | Filter mismatched generations |
| Deduplication | Embed-based (e.g., CLIP / FaceNet) vector similarity threshold | Stronger semantic duplicate filtering |
| Parallelism | Use multiprocessing or async generation queue | Throughput scalability |
| Drift tracking | Track FID/KID between original & synthetic subsets | Quantify realism & shift |
| Provenance | Log per-image generation config snapshot | Forensically trace anomalies |

## 12. Reproducibility Artifacts
| Artifact | Purpose |
|----------|---------|
| Seed constants (`BASE_SEED`) | Deterministic generation ordering |
| `prompt_dict_hash` | Versioning of emotional prompt schema |
| Environment JSON | Cross-run comparability |
| Hardlink copy strategy | Space efficiency & integrity (inode-level) |

## 13. Open Questions
1. Should target be restored to 100k (header) vs 83k (config)?
2. Is grayscale optimal vs retaining RGB for subtle cues?
3. Should vertical flips be disallowed (facial orientation realism)?
4. Do we enforce expression classifier agreement before acceptance?

## 14. Execution Checklist (For Next Run)
1. Confirm `TARGET_TOTAL` matches strategic dataset size goal.
2. Expand prompt coverage (demographics + lighting + accessories neutrality).
3. Run generation loop; capture real metrics and insert into Section 10.
4. Validate per-class synthetic fraction cap not exceeded post-split.
5. Export updated findings + link FID/KID evaluation (if added).

---

For upstream raw dataset analysis see `docs/eda.md`; for balancing logic see `docs/feature_engineering_balancing.md`.
