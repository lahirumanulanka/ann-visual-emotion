<!--
	Generated from static inspection of `notebooks/CNN_with_Transfer_Learning.ipynb`.
	Notebook NOT executed in this pass; metrics below that depend on runtime are placeholders.
	After running training/evaluation, replace placeholders with real numbers.
-->

# 🧠 Emotion Recognition Model Design (Transfer Learning Pipeline)

This document summarizes the advanced transfer learning training notebook for facial emotion classification. The design emphasizes progressive fine‑tuning, rich monitoring, and explainability tooling to produce a production‑ready model (PyTorch → ONNX → mobile / API deployment).

## 1. Objectives
- Build a high‑performing multi-class facial emotion classifier using modern pretrained backbones.
- Mitigate catastrophic forgetting via staged unfreezing + discriminative learning rates.
- Provide transparent training diagnostics (loss, accuracy, macro F1, LR curves, timing).
- Offer model interpretability (Grad-CAM / Grad-CAM++, SHAP, LIME) for qualitative validation.
- Export portable artifacts (PyTorch checkpoint, ONNX) and publish to Hugging Face Hub.

## 2. Supported Backbones
| Type | Identifier (`cfg.model_type`) | Notes |
|------|-------------------------------|-------|
| ResNet | `resnet50` | Default; custom deeper head (Linear→ReLU→LayerNorm→Dropout→Linear) |
| ConvNeXt | `convnext_base` | Via `timm` (if installed) |
| EfficientNet | `tf_efficientnet_b3_ns` | Noisy student pretrained weights |
| Vision Transformer | `vit_base_patch16_224` | Self-attention interpretability (optional) |

## 3. Progressive Training Strategy
| Stage | Epoch Span | Action | Purpose |
|-------|------------|--------|---------|
| 1 (Warmup) | `freeze_backbone_epochs` (default 3) | Freeze all backbone layers; train classification head | Stabilize new head; prevent abrupt weight drift |
| 2 (Fine‑tune) | Remainder | Unfreeze all layers; apply discriminative LR (backbone < head) | Adapt high-level features while preserving general representations |

Learning rate schedule: Linear warmup (Stage 1) → Cosine Annealing decay (Stage 2). Two optimizer parameter groups (backbone vs head) maintained with separate base LRs (`lr_backbone`, `lr_head`).

## 4. Key Hyperparameters (from `CFG` dataclass)
| Parameter | Value (Default) | Rationale |
|-----------|-----------------|-----------|
| `img_size` | 224 | Standard for many ImageNet pretrained models |
| `batch_size` | 32 | Trade-off between stability and VRAM footprint |
| `epochs` | 80 | Provides room for cosine schedule convergence |
| `freeze_backbone_epochs` | 3 | Short head warmup without overfitting head only |
| `lr_backbone` | 1e-4 | Conservative adaptation |
| `lr_head` | 1e-3 | Faster convergence of randomly initialized head |
| `weight_decay` | 1e-4 | Regularization for overfitting control |
| `label_smoothing` | 0.05 | Softens targets; improves calibration |
| `use_mixup` | True (α=0.4) | Regularize decision boundary; combat label noise |
| `use_cutmix` | False | Disabled to limit semantic mixing complexity |
| `grad_clip_norm` | 1.0 | Stabilize occasional large updates |
| `use_ema` | True (decay 0.999) | Smoother final weights; better generalization |
| `patience` | 5 | Early stopping on `macro_f1` |
| `use_class_weights` | True | Mitigate residual class imbalance |

## 5. Data Pipeline
| Phase | Transform / Logic | Notes |
|-------|-------------------|-------|
| Load | Resolve path; open with PIL; convert to RGB | Original raw source may be grayscale |
| Train Aug | RandomResizedCrop(0.75–1.0), HFlip, Rotation(±15°), ColorJitter, Normalize, RandomErasing | Balanced robustness vs expression fidelity |
| Val/Test | Resize (1.14×) → CenterCrop 224 → Normalize | Deterministic evaluation |
| Class Weights | Inverse(freq^α) (α=0.6) normalized | Fed to CrossEntropyLoss |

## 6. Regularization & Optimization Features
- MixUp / CutMix conditional application (mutually exclusive sampling)
- Label smoothing integrated into loss.
- Gradient clipping and mixed precision (`torch.amp.autocast`) for speed + stability.
- EMA shadow weights updated per step; applied before validation / final evaluation.

## 7. Monitoring & Logging
Per epoch printed columns: `Epoch | Phase | Train Loss/Acc | Val Loss/Acc/F1 | LR(backbone/head) | Time | Status`.

Tracked history arrays: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, `lr_backbone`, `lr_head`, `epoch_times` → visualized as three plots (Loss, Accuracy & Macro F1, LR schedule). Weight and gradient statistics (mean/std/min/max, L2) optionally printed each epoch (configurable filters: head/all).

## 8. Early Stopping & Checkpointing
- Metric monitored: `macro_f1` (configurable to accuracy).
- Patience counter halts training if no improvement > `min_delta` (default 0) after `patience` epochs.
- Best checkpoint saved as `models/best_model.pth` with: weights, epoch, validation metric, config snapshot, training history.

## 9. Explainability Toolkit
| Method | Implementation | Purpose |
|--------|---------------|---------|
| Grad-CAM | `pytorch-grad-cam` | Spatial saliency of final conv block |
| Grad-CAM++ | Same library | Refined multi-object importance weighting |
| SHAP (DeepExplainer) | Direct PyTorch | Global & local feature attribution (channel/region) |
| LIME | `lime_image` | Local surrogate for a few test samples |

Target layers for CAM auto-selected (e.g., `layer4[-1]` for ResNet50; last block for ViT). Visualization captions show predicted vs true label.

## 10. ONNX Export Flow
Modern exporter invocation (`torch.onnx.export(..., dynamo=True)`) with:
- In-place op disabling + hook clearing pre-pass.
- Progressive fallback across opset versions (17→11) & constant folding toggles.
- Optional output clone wrapper to avoid view/inplace aliasing issues.

Produces `models/model_fixed.onnx` (or configured name). Intended for mobile / edge inference (e.g., bridging to Flutter via ONNX runtime or conversion pipelines).

## 11. Hugging Face Publishing
Automated cells:
1. Install `huggingface_hub` & `python-dotenv` if absent.
2. Acquire token from Colab Secrets (`HF_TOKEN`).
3. Create / update repo `hirumunasinghe/emotion_face_detector`.
4. Upload: `best_model.pth`, `model.onnx`, `label_map.json`, and a generated `README.md` with model card YAML (tags, license, labels list if available).

## 12. Safety / Reproducibility Measures
| Aspect | Mechanism |
|--------|-----------|
| Seeding | Unified `set_seed(seed)` for Python, NumPy, Torch (CUDNN benchmark True for speed) |
| Config Snapshot | `asdict(cfg)` embedded into checkpoint |
| Weight/Grad Stats | Optional dumps (`final_weights.txt`) for forensic inspection |
| EMA | Deterministic update rule (decay constant) |
| Label Map | Loaded from JSON or inferred from training CSV for stable class index order |

## 13. Potential Failure Points & Mitigations
| Risk | Mitigation |
|------|-----------|
| Path resolution mismatch (absolute vs relative) | `resolve_path` attempts multiple resolutions before raising |
| Class imbalance residual | Weighted loss (and option for sampler) |
| Overfitting head in warmup | Short freeze window + discriminative LRs post-unfreeze |
| Mixed precision numerical instability | GradScaler + gradient clipping |
| ONNX export operator incompatibility | Iterative opset + constant folding fallback loop |
| SHAP memory/time blow-up | Small background subset only |

## 14. Placeholder Performance Metrics (To Fill After Execution)
```
Best validation macro F1: <float>
Best validation accuracy: <float>
Epoch of best model: <int>
Final test accuracy: <float>
Final test macro F1: <float>
Train/Val/Test sizes: <t>/<v>/<te>
Average epoch time: <sec>
Total training wall time: <min>
```

## 15. Interpretability Observations (Template)
Populate after running visualization cells:
```
Grad-CAM: Regions emphasized (e.g., mouth corners for happy, brow furrow for angry).
SHAP: Consistent positive contribution of periocular shadows for fearful vs neutral.
LIME: Local masks align with core expression muscle groups.
Misclass patterns: <brief notes>.
```

## 16. Deployment Considerations
- ONNX model amenable to quantization (dynamic or static) for mobile.
- If latency critical, consider pruning or switching backbone to EfficientNet-B0 / MobileViT.
- Export script already detaches custom hooks and in-place ops → safer downstream conversions (CoreML / TFLite via intermediate tooling).

## 17. Next Improvement Opportunities
| Area | Enhancement | Expected Benefit |
|------|-------------|------------------|
| Data | Integrate synthetic GenAI set (capped fraction) | Broader expression variability |
| Loss | Add Focal or LDAM loss experiment | Better handling of minority confusion |
| Scheduler | One-cycle LR or cosine restarts | Potential faster convergence |
| Calibration | Temperature scaling post-training | Improved probabilistic outputs |
| Monitoring | WandB / MLflow integration | Persistent experiment tracking |
| Distillation | Teacher-student with larger ViT | Smaller deployable model accuracy retention |

## 18. Open Questions
1. Should we keep grayscale conversion or leverage color cues (if available) for subtle expressions?
2. Is MixUp still beneficial with facial expression semantics (risk of blended ambiguous emotions)?
3. Do we need a fairness audit across demographic axes once synthetic diversity expands?

## 19. Changelog
| Date | Change |
|------|--------|
| 2025-09-26 | Initial model design summary from notebook static inspection. |

## 20. Quick Start (Reproduction After Cloning)
1. Ensure processed splits CSVs exist (see `data/processed/EmoSet_splits_gen/{train,val,test}.csv`).
2. Open `notebooks/CNN_with_Transfer_Learning.ipynb`.
3. (Optional) Adjust `CFG` paths if running outside Colab (replace `/content/ann-visual-emotion` with repo root).
4. Run cells in order:
	- Dependency install
	- Imports & configuration
	- Path checks / dataset load
	- Training loop (wait for completion or early stop)
	- Evaluation (validation + test)
	- Explainability (Grad-CAM / SHAP / LIME)
	- ONNX export
	- (Optional) Hugging Face publish
5. Copy real metrics into Section 14 & interpretability notes into Section 15.

## 21. Artifact Mapping & Downstream Usage
| Artifact | Produced By | Consumed In | Purpose |
|----------|-------------|-------------|---------|
| `models/best_model.pth` | Training loop checkpoint save | Further fine-tuning / conversion scripts | Full state (weights + config + history) |
| `models/model_fixed.onnx` | ONNX export cell | Mobile / API runtime (`onnxruntime`) | Framework‑agnostic inference |
| `models/label_map.json` | Label map load/save logic | Real-time app (`realtime_app/`, Flutter assets) | Consistent class index mapping |
| `final_weights.txt` (optional) | Weight dump step | Auditing / diffing future runs | Reproducibility for research |
| Training history (in checkpoint) | Serialization of `training_history` | Dashboard / plotting utilities | Post-hoc analysis |

## 22. Evaluation / Metrics Insertion Guide
After a full run, gather values from the final console logs:
| Console Label | Section 14 Field |
|---------------|------------------|
| `Best validation macro F1` | Best validation macro F1 |
| `Best Validation macro_f1` (checkpoint print) | Cross-check for consistency |
| `Validation (Best) Loss/Acc/MacroF1` | (Optional detailed appendix) |
| `Test Loss ... Acc ... MacroF1 ...` | Final test accuracy / macro F1 |
| `Total Training Time` | Total training wall time |
| `Average Time per Epoch` | Average epoch time |
| `Epoch` stored in checkpoint | Epoch of best model |

Paste values into the placeholder block and remove the angle brackets.

## 23. Suggested Lightweight Validation Before Explainability
| Step | Command / Action | Goal |
|------|------------------|------|
| Sanity batch | Forward pass on one mini-batch | Confirm shapes / class count |
| Overfit tiny subset | Train on 32 samples for 5 epochs | Ensure model can overfit (debugging) |
| Class distribution check | Inspect `train_df[label]` counts | Verify weighting logic |
| Gradient norm check | Monitor clip events (optional print) | Detect instability early |

## 24. Integration with Synthetic & Balancing Pipelines
- If synthetic generation (see `genai_findings.md`) increases class counts, re-run balancing to regenerate `train/val/test.csv` before training.
- Recompute class weights automatically (already handled) — no manual config change required unless class taxonomy changes.
- Document new synthetic fraction in Section 14 when reporting final metrics for transparency.

## 25. Troubleshooting Quick Reference
| Symptom | Likely Cause | Mitigation |
|---------|--------------|-----------|
| Rapid overfitting (val F1 drops) | Insufficient augmentation / early freeze too long | Increase MixUp α, reduce freeze epochs to 1–2 |
| No improvement after warmup | Head LR too low / backbone still frozen | Verify unfreezing at correct epoch; raise `lr_head` 2× |
| Export failure (ONNX) | Unsupported op or in-place op leftover | Lower opset, ensure inplace ops disabled, retry with fewer features |
| NaNs in loss | Mixed precision instability | Temporarily disable AMP or reduce LR |
| CAM layer not found | New backbone naming | Manually specify last conv/attention block in `select_target_layers_for_cam` |

## 26. Future Automation Hooks
Potential standalone scripts (not yet implemented):
| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Headless training using same config object |
| `scripts/export_onnx.py` (exists) | Could be extended to load latest checkpoint automatically |
| `scripts/evaluate.py` | Batch inference + metrics on external holdout set |
| `scripts/explain.py` | Generate and archive CAM / SHAP panels for a fixed probe set |

---

---

Related docs: `docs/eda.md`, `docs/feature_engineering_balancing.md`, `docs/genai_findings.md`

