<!--
	Consolidated project README assembled from existing documentation files:
	- docs/eda.md
	- docs/feature_engineering_balancing.md
	- docs/genai_findings.md
	- docs/model_design.md
	- docs/xai_comparison.md
	This is a high-level entry point. For deep dives, follow the linked docs.
-->

# 😃 ANN Visual Emotion Recognition Platform

End‑to‑end system for building, extending, explaining, and deploying a facial emotion recognition model. The repository integrates dataset exploration, balancing & augmentation, synthetic data generation with GenAI, advanced transfer learning, explainability (XAI), and multi‑target deployment (API, real‑time app, mobile via Flutter + ONNX).

---
## 1. High‑Level Architecture
```
					Raw Dataset (6 classes)            Legacy 8-class (historical)
										 |                                 |
								 EDA Scan ------------------------------+
										 |
						Balancing & Augmentation (02_*)
										 |
				+------------+-------------+
				|                          |
 Synthetic Gen (Stable Diffusion)  |
				| (quality filters, capping)|
				+------------+--------------+
										 |
						Merged Processed Splits (train/val/test CSVs, label_map)
										 |
					 Transfer Learning Training (CNN/Vit, staged FT, EMA)
										 |
					Explainability (Grad-CAM, SHAP, LIME) & Metrics
										 |
				Export & Deployment (PyTorch .pth, ONNX, HF Hub, Real-time App)
										 |
				 Mobile / API / Web Inference (Flutter app, realtime_app FastAPI)
```

---
## 2. Repository Structure (Key Directories)
| Path | Purpose |
|------|---------|
| `data/raw/FullDataEmoSet` | Source emotion images (6 current classes) |
| `data/processed/EmoSet_splits_gen` | Generated merged splits (after balancing + synthetic) |
| `notebooks/` | EDA, balancing, synthetic generation, transfer learning experiments |
| `docs/` | Generated documentation (EDA, model design, balancing, XAI, GenAI) |
| `scripts/` | Data prep utilities (splits, grayscale, ONNX export, etc.) |
| `models/` | Saved artifacts (`best_model.pth`, `model.onnx`, `label_map.json`) |
| `realtime_app/` | Python real‑time server (FastAPI + inference) |
| `app/emotion_detector/` | Flutter mobile application consuming ONNX model |
| `docker/` | Container definitions for training / API serving |
| `deliverables/` | Presentation, packaging, demo collateral |

---
## 3. Data Lifecycle
| Stage | Description | Source Notebook / Script | Output Artifacts |
|-------|-------------|--------------------------|------------------|
| EDA | Scan counts, format, quality metrics | `01_eda.ipynb` | `docs/eda.md` |
| Balancing & Aug | Oversample minorities with augmentations, split | `02_feature_engineering_balancing.ipynb` | Balanced tree + `train/val/test.csv` |
| Synthetic GenAI | Diffusion-based class expansion (filtered) | `03_synthetic_gen_ai_generation.ipynb` | Synthetic images + `status.json` |
| Merge & Splits | Combine original + synthetic within cap | GenAI notebook + scripts | Updated splits & label map |
| Training | Staged fine‑tuning with explainability hooks | `CNN_with_Transfer_Learning.ipynb` | `best_model.pth`, `model_fixed.onnx` |
| Deployment | Export, HF Hub, mobile integration | Notebook + `scripts/export_onnx.py` | HF model repo / ONNX asset |

---
## 4. Current Dataset Snapshot (from EDA)
| Class | Count | Percent |
|-------|-------|---------|
| angry | 5,089 | 11.63% |
| fearful | 4,589 | 10.49% |
| happy | 13,370 | 30.56% |
| neutral | 8,268 | 18.90% |
| sad | 7,504 | 17.15% |
| surprised | 4,936 | 11.28% |

Total images: 43,756 (all grayscale). Legacy 8‑class schema retained for historical references (see `docs/eda.md`).

---
## 5. Model Overview (Transfer Learning)
Key techniques:
* Two‑stage fine‑tuning (head warmup → full unfreeze with discriminative LRs)
* Mixed precision + gradient clipping + EMA
* Label smoothing + class weighting + MixUp (default) / optional CutMix
* Rich monitoring: loss, accuracy, macro F1, LR schedules, per‑epoch weight/grad stats
* ONNX export with operator fallback logic
* Hugging Face publishing pipeline automation

Details: see `docs/model_design.md`.

---
## 6. Explainability (XAI) Toolkit
| Method | Use Case |
|--------|----------|
| Grad-CAM | Fast saliency sanity checks |
| Grad-CAM++ | Finer localization when multiple micro‑regions matter |
| SHAP | Global + local additive attribution, bias audits |
| LIME | Local explanation for misclassifications |

Extended comparison + evaluation guidance: `docs/xai_comparison.md`.

---
## 7. Synthetic Data Generation (GenAI)
Implemented with Stable Diffusion (`runwayml/stable-diffusion-v1-5`), prompt templating, and multi‑stage quality filters:
| Filter | Purpose |
|--------|---------|
| Single face detection | Avoid ambiguous labels |
| Blur variance threshold | Remove low-detail outputs |
| Perceptual hash deduplication | Eliminate near duplicates |
| Synthetic fraction cap | Control distributional drift |

Status & reproducibility metadata: `docs/genai_findings.md`.

---
## 8. Real‑Time & Mobile Deployment
| Component | Path | Description |
|-----------|------|-------------|
| FastAPI Inference | `realtime_app/` | Python service loading PyTorch or ONNX model |
| Flutter App | `app/emotion_detector/` | Device camera → ONNX inference; environment variable for API fallback |
| ONNX Assets | `models/model_fixed.onnx` (or `model.onnx`) | Shared inference format |

Dockerfiles at `docker/` support containerized training & serving.

---
## 9. Getting Started
### 9.1 Environment Setup (Python)
Use the provided `pyproject.toml` / `requirements.txt` (one may supersede the other; prefer the former if Poetry or PEP 621 tooling is used).

### 9.2 Data Preparation
```
python scripts/prepare_data.py        # if required to stage processed splits
python scripts/split_dataset.py       # alternative splitting utility
```

### 9.3 Training (Notebook)
Open `notebooks/CNN_with_Transfer_Learning.ipynb`, adjust paths in `CFG` if not on Colab, run sequentially. After run, update metrics placeholders in `docs/model_design.md` & this README (Section 11).

### 9.4 ONNX Export (Script Shortcut)
```
python scripts/export_onnx.py --checkpoint models/best_model.pth --out models/model.onnx
```

### 9.5 Real-Time API
```
uvicorn realtime_app.main:app --reload
```

### 9.6 Flutter App (iOS/Android)
Requires Flutter SDK & platform toolchains:
```
cd app/emotion_detector
flutter pub get
flutter run -d <device_id> --dart-define=HF_API_URL=<inference_endpoint>
```

---
## 10. Configuration Highlights
| Domain | Key Knobs |
|--------|-----------|
| Balancing | `BALANCE_STRATEGY`, `AUG_PER_IMAGE_LIMIT`, `SEED` |
| Generation | `TARGET_TOTAL`, blur var threshold, pHash distance |
| Training | `freeze_backbone_epochs`, `lr_backbone`, `lr_head`, `use_mixup`, `use_ema` |
| Explainability | `run_gradcam`, `run_shap`, `run_lime` |
| Export | ONNX opset fallback list, `dynamo=True` toggle |

---
## 11. Metrics (Placeholders – Fill After Training Run)
```
Best validation macro F1: <float>
Best validation accuracy: <float>
Epoch of best model: <int>
Final test accuracy: <float>
Final test macro F1: <float>
Synthetic fraction overall: <float>
Train/Val/Test sizes: <t>/<v>/<te>
Average epoch time: <sec>
Total training wall time: <min>
```

---
## 12. Quality & Risk Controls
| Aspect | Control |
|--------|---------|
| Data integrity | Hashing (planned), format/mode validation |
| Class imbalance | Weighted loss, augmentation, synthetic capping |
| Overfitting | Early stopping (macro F1), EMA, MixUp |
| Drift / bias | XAI audits (SHAP distributions), synthetic prompt diversity plan |
| Reproducibility | Seeded config snapshot in checkpoint, prompt hash, environment capture |

---
## 13. Roadmap
| Area | Next Step | Impact |
|------|-----------|--------|
| Data | Finalize 6↔8 class mapping artifact | Historical continuity |
| Synthetic | Add embedding (CLIP/FaceNet) dedup layer | Stronger uniqueness |
| Training | Add focal / LDAM experiment script | Minority performance |
| XAI | Implement deletion/insertion quantitative harness | Attribution rigor |
| Deployment | Provide quantized ONNX (INT8) build path | Mobile latency |
| Automation | Convert notebook to `scripts/train.py` + Hydra configs | CI reproducibility |

---
## 14. Contributing
1. Fork & create a feature branch.
2. Run lint & tests:
```
ruff check .
pytest -q
```
3. Update relevant docs sections (EDA/model/XAI) if behavior changes.
4. Open PR referencing any updated metrics or artifacts.

---
## 15. Related Documentation Index
| File | Focus |
|------|-------|
| `docs/eda.md` | Raw dataset statistics & schema reconciliation |
| `docs/feature_engineering_balancing.md` | Balancing & augmentation strategy |
| `docs/genai_findings.md` | Synthetic generation pipeline & quality gates |
| `docs/model_design.md` | Transfer learning architecture & training strategy |
| `docs/xai_comparison.md` | Explainability method comparison & usage |

---
## 16. License
Project license: MIT (see `LICENSE`). Models or datasets may have additional upstream licenses—verify before redistribution.

---
## 17. Acknowledgements
* Pretrained models via `torchvision` & `timm`.
* Diffusion pipeline (`diffusers`) for synthetic generation.
* XAI libraries: `pytorch-grad-cam`, `shap`, `lime`.
* Community open-source tooling that enabled rapid iteration.

---
_For detailed experimental reasoning and future design considerations, explore the documents in `docs/` and notebooks in `notebooks/`._

