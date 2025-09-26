<!--
	Generated via static inspection of `notebooks/CNN_with_Transfer_Learning.ipynb` (no execution).
	Populate metric placeholders after running notebook visualization cells.
-->

# 🧪 Explainable AI (XAI) Method Comparison for Emotion Classification

This document contrasts the interpretability techniques integrated into the transfer learning pipeline and provides guidance for their practical use in auditing emotion recognition behavior.

## 1. Methods Covered
| Method | Library / Implementation | Input Requirement | Output Form | Granularity |
|--------|--------------------------|-------------------|-------------|-------------|
| Grad-CAM | `pytorch-grad-cam` | Model + target layer + input tensor | Heatmap overlay | Coarse spatial |
| Grad-CAM++ | `pytorch-grad-cam` | Same as Grad-CAM | Enhanced heatmap | Finer spatial (multi-instance) |
| SHAP (DeepExplainer) | `shap` | Background samples + model forward | Per-pixel/channel contribution values | Pixel / region |
| LIME (Image) | `lime_image` | Callable `predict_fn` returning class probs | Superpixel importance mask | Superpixel (local) |

## 2. Conceptual Foundations
| Method | Core Idea | Strength | Limitation |
|--------|----------|----------|------------|
| Grad-CAM | Class-specific gradient weights pooled over spatial feature maps | Fast, intuitive heatmaps | Resolution limited to last conv layer |
| Grad-CAM++ | Weights higher-order gradient terms for better separation | Improves small salient region focus | Slightly slower; still layer-resolution bound |
| SHAP | Shapley value approximation of feature contribution to prediction | Theoretically grounded attribution consistency | Expensive; sensitive to background choice |
| LIME | Local linear surrogate on perturbed superpixels | Model-agnostic; simple visual masks | Instability; depends on segmentation & sampling seed |

## 3. When to Use What
| Goal | Recommended Method(s) | Rationale |
|------|-----------------------|-----------|
| Quick sanity check of focus regions | Grad-CAM | Rapid, low overhead |
| Distinguish subtle facial micro-regions | Grad-CAM++ | Finer saliency where multiple small features matter |
| Auditing attribution consistency across samples | SHAP | Additive decomposition & comparability |
| Explaining a single surprising misclassification | LIME + Grad-CAM | Local surrogate plus spatial heatmap context |
| Bias / spurious feature probe | SHAP | Detect persistent over-attribution to background zones |

## 4. Practical Integration Notes
- Target Layer Selection (Grad-CAM/++): For `resnet50`, final block `layer4[-1]` selected; adjust for different backbones (ConvNeXt: last stage, ViT: last block norm/attention module).
- SHAP Background: Use a small, diverse set (e.g., 16–32 neutral expressions) to reduce noise while capturing base distribution; too small → high variance, too large → slow.
- LIME Predict Function: Ensure preprocessing (normalize with ImageNet mean/std) replicates training path; mismatch yields misleading explanations.
- EMA Weights: Apply before explanation for stability; gradients reflect smoothed parameters.

## 5. Quality Assessment Checklist
| Check | Action | Pass Criteria |
|-------|--------|---------------|
| Saliency localization | Compare heatmaps across classes | Emotion-relevant facial regions dominate (eyes, mouth, brows) |
| Attribution sparsity (SHAP) | Plot distribution of absolute SHAP values | Heavy tail with clear top regions; minimal background dominance |
| LIME stability | Run LIME twice with same seed | Near-identical superpixel importance ordering |
| Class contrast | Inspect Grad-CAM between two different emotions | Distinct patterns (e.g., happy → mouth, angry → brow) |
| Spurious artifact detection | Look for border / background highlights | Minimal non-face emphasis |

## 6. Potential Failure Modes & Mitigations
| Issue | Symptom | Mitigation |
|-------|---------|------------|
| Layer mismatch | Blank / uniform heatmap | Select earlier conv layer or appropriate ViT block |
| Over-smoothing (Grad-CAM) | Broad diffuse regions | Try Grad-CAM++ or higher-res intermediate layer |
| SHAP noise | Highly unstable per-pixel signs | Increase background size moderately or apply image blurring pre-aggregation |
| LIME artifact focus | Highlights cropped borders | Adjust segmentation parameters; reduce `num_samples` noise |
| Attribution drift post fine-tune | Heatmaps shift unpredictably across epochs | Fix random seeds & log intermediate checkpoints |

## 7. Metrics & (Placeholder) Quantitative Proxies
Quantitative evaluation of XAI is non-trivial; placeholders below can be filled if you add instrumentation.
```
Deletion AUC (Grad-CAM): <float>
Insertion AUC (Grad-CAM++): <float>
Top-k (k=10%) pixel retention accuracy drop: <float>
SHAP attribution concentration (Gini): <float>
LIME stability score (IoU between runs): <float>
```
Suggested implementation: implement deletion/insertion curves by progressively masking most salient regions and measuring probability decline.

## 8. Workflow to Capture & Archive Explanations
| Step | Tool | Output |
|------|------|--------|
| 1 | Select probe set (balanced across classes) | List of image paths |
| 2 | Run Grad-CAM/++ on probe batch | Heatmap PNG grid per method |
| 3 | Run SHAP (background + probe subset) | Attribution arrays saved (.npy) + summary plot |
| 4 | Run LIME for misclassified samples | Superpixel overlays PNG |
| 5 | Store JSON manifest | Method → file paths, model hash, timestamp |
| 6 | (Optional) Compute deletion/insertion curves | CSV metrics per sample |

## 9. Governance / Bias Audit Hooks
| Dimension | Probe Strategy | Signal |
|-----------|---------------|--------|
| Demographic variation | Group images by synthetic prompt attributes | Check saliency distribution shift |
| Lighting conditions | Augment probe with varied brightness | Attribution resilience |
| Occlusions (glasses, mask) | Mask overlay test | Robustness vs. partial occlusion |
| Expression intensity | Bucket subtle vs. extreme | Saliency tightness vs. diffusion |

## 10. Recommended Enhancements
| Enhancement | Benefit |
|-------------|---------|
| Add Integrated Gradients | Path-based attribution for baseline comparison |
| Add Captum library wrapper | Unified interface for new methods |
| Persist saliency stats (mean focus radius) | Track model drift over retrains |
| Batch explanation caching | Reduce recomputation cost |
| UI dashboard (e.g., Streamlit) | Interactive auditing & filtering |

## 11. Open Questions
1. Should explanations be part of an automated acceptance gate (e.g., saliency coverage threshold)?
2. Do we need per-class attribution diversity metrics to avoid feature collapse?
3. How to best quantify fairness in spatial attention for synthetic vs. real faces?

## 12. Changelog
| Date | Change |
|------|--------|
| 2025-09-26 | Initial comparison authored from static notebook capabilities. |

---

See also: `docs/model_design.md` (Sections 9 & 15) for integration context.
