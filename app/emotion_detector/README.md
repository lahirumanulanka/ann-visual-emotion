## Emotion Detector (Flutter Mobile Client)

Multi-mode Flutter application for emotion recognition, face-aware realtime analysis, heuristic / simulated inference, and emotion-based image transformation/generation—designed to integrate with (or eventually replace heuristics with) the PyTorch → ONNX model exported in the root project.

---
### 🎯 Goals
1. Provide on-device user experience for emotion detection from: 
   * Static images (gallery selection)
   * Realtime camera stream (continuous detection & smoothing)
2. Support multiple inference strategies while the full ONNX runtime integration is finalized:
   * Heuristic / simulated classifiers (brightness / contrast / edge stats)
   * Face landmark–aware heuristics (using ML Kit: smile & eye openness)
   * Realtime streaming mode with exponential moving average smoothing
   * Procedural emotion image transformation / generation (placeholder for future generative models)
3. Abstract model I/O to allow seamless swap-in of a true ONNX runtime backend once platform plugin is wired.

---
### 🗂 Structure

```
app/emotion_detector/
├── assets/
│   ├── model.onnx           # Exported classifier (ResNet50 head) – not yet consumed directly
│   └── label_map.json       # Emotion label → index mapping (6-class)
├── lib/
│   ├── main.dart            # (Selects / demonstrates available screens or modes)
│   ├── main_simple.dart     # Simple, minimal mock emotion detection demo
│   ├── main_simple_test.dart# Alternate quick test harness
│   ├── main_emotion.dart    # Image-based detection variant
│   ├── main_enhanced.dart   # Uses EnhancedEmotionDetectionService
│   ├── main_real_onnx.dart  # Placeholder for future actual ONNX inference wiring
│   ├── main_backup.dart     # Legacy / backup entry
│   ├── models/
│   │   └── emotion_result.dart
│   ├── services/
│   │   ├── emotion_detection_service.dart          # Basic heuristic classifier (offline)
│   │   ├── emotion_detection_service_mock.dart     # Duplicate mock (could be consolidated)
│   │   ├── enhanced_emotion_service.dart           # Adds richer image feature heuristics
│   │   ├── face_detection_emotion_service.dart     # Uses ML Kit face landmarks & probabilities
│   │   ├── realtime_emotion_service.dart           # Streaming camera pipeline
│   │   └── real_emotion_detection_service.dart     # Stub for ONNX runtime implementation
│   └── screens/
│       ├── realtime_camera_screen.dart             # Live camera overlay & detection
│       └── emotion_image_generator_screen.dart     # Emotion transform / procedural gen
├── pubspec.yaml
└── analysis_options.yaml
```

---
### 🧠 Emotion Classes
Loaded from `assets/label_map.json`:

```
angry, fearful, happy, neutral, sad, surprised
```

These align with the distilled set used by the backend PyTorch model (see project root README for mapping provenance).

---
### ⚙️ Services Overview

| Service | Purpose | Core Signals / Features | Notes |
|---------|---------|-------------------------|-------|
| `emotion_detection_service.dart` | Simple image-level heuristic baseline | Brightness & contrast → probability shaping | Fast offline demo |
| `emotion_detection_service_mock.dart` | Same as above; mock alias | Duplicate of baseline | Can be removed / merged |
| `enhanced_emotion_service.dart` | Adds center brightness, edge density, local feature heuristics | Pseudo-probabilities, more nuanced distribution | No ML Kit dependency |
| `face_detection_emotion_service.dart` | Face cropping + ML Kit landmarks (smile, eyes) | Weighted rules for emotion scoring | Returns face meta (count, detection state) |
| `realtime_emotion_service.dart` | Continuous camera frames + EMA smoothing | Landmark-driven features + orientation handling | Supports front/back camera & bounding box overlay |
| `real_emotion_detection_service.dart` | Placeholder ONNX runtime adapter | (Stub result) | Swap with true inference layer later |
| `EmotionImageGeneratorService` (in generator screen) | Procedural emotion-themed image synthesis | Palette selection + shapes + deterministic noise | Placeholder for future generative model |

---
### 📱 Screens

| Screen | File | Description |
|--------|------|-------------|
| Realtime Camera | `realtime_camera_screen.dart` | Live detection, bounding box, probabilities, cooldown gating to reduce spam |
| Emotion Image Transformer | `emotion_image_generator_screen.dart` | Pick image → detect base emotion → apply visual transform to simulate target emotion |

Entry `main_*.dart` variants allow experimenting with specific service permutations during development.

---
### 🔁 Realtime Pipeline (High-Level)
1. Initialize camera (YUV420 stream) and ML Kit face detector.
2. For each frame (throttled implicitly by analysis flag):
   * Convert planes → `InputImage` with correct rotation & metadata.
   * Detect faces; choose largest.
   * Derive pseudo emotion scores (smile probability, eye openness, head pose heuristics).
   * Apply Exponential Moving Average smoothing to stabilize UI.
3. Emit `RealtimeEmotionResult` → UI overlay draws bounding box + bars.

Cooldown logic prevents repeating identical audible / UI announcements rapidly.

---
### 🧪 Heuristic Scoring Features
Different services compose subsets of:
* Global brightness / contrast (proxy for mood / intensity)
* Center brightness (facial region saliency)
* Edge density (feature richness, potential expression intensity)
* Smile probability (ML Kit)
* Eye openness (surprise / fear cues)
* Head yaw / roll (pose – may correlate with neutral / disengaged states)
* Temporal smoothing (EMA) for realtime jitter reduction

These act as explainable stand-ins until the true model is invoked on-device.

---
### 🧩 Planned ONNX Runtime Integration
Target approach:
1. Add `onnxruntime` (via a Flutter plugin or FFI) dependency.
2. Implement tensor pre-processing:
   * Resize to 224×224
   * Convert RGB → float32
   * Normalize ImageNet mean/std
   * Shape: `[1, 3, 224, 224]`
3. Run inference → logits → softmax.
4. Replace heuristic scores with model outputs while retaining fallback path (feature flag or dev mode switch).

Potential plugins / strategies (evaluate):
* Platform channels wrapping native Android/iOS onnxruntime libraries
* Use `tflite_flutter` only if model converted to TFLite (secondary option)

---
### 🚀 Getting Started

#### Prerequisites
* Flutter SDK (channel stable; Dart >= 3.8.x per `pubspec.yaml`)
* Device or emulator with camera permissions (for realtime mode)

#### Install Dependencies
```bash
flutter pub get
```

#### Run (choose a main variant if desired)
```bash
flutter run -t lib/main.dart
```
Other examples:
```bash
flutter run -t lib/main_simple.dart
flutter run -t lib/main_enhanced.dart
flutter run -t lib/main_real_onnx.dart   # (Currently stub)
```

#### Hot Reload / Debug
Use VS Code / Android Studio or CLI `r` in terminal.

---
### 🔐 Permissions
Add / confirm in Android & iOS:
* Camera
* (Optional) Storage / Photos for gallery picker

Android example (already generated in base project): `android/app/src/main/AndroidManifest.xml` should include:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

---
### 🧪 Testing Strategy (Proposed)
| Area | Test Idea |
|------|-----------|
| Heuristic services | Unit test probability normalization & monotonic influence of smile probability |
| Realtime service | Mock face detector → verify smoothing & bounding box mapping |
| Image generator | Deterministic output given fixed label & seed entropy |
| ONNX adapter (future) | Golden test: known input tensor → expected class ordering |

---
### 🔄 Migration Path to Real Model
| Step | Action |
|------|--------|
| 1 | Implement ONNX runtime wrapper (Dart FFI or plugin) |
| 2 | Add `RealEmotionDetectionService` with preprocessing & prediction |
| 3 | Inject via simple service locator / factory (env flag) |
| 4 | A/B compare heuristic vs real outputs (telemetry) |
| 5 | Remove duplicate mock services once stable |

---
### 🧹 Cleanup Opportunities
* Consolidate `emotion_detection_service.dart` and `emotion_detection_service_mock.dart`.
* Extract shared brightness/contrast utilities into a helper.
* Introduce interface (abstract class) `EmotionBackend` to formalize `initialize()`, `detectEmotion()` signatures.
* Add logging abstraction (toggle verbose diagnostics in debug mode only).

---
### 🐞 Known Limitations
* All current emotion scores are synthetic; they do not reflect learned CNN outputs yet.
* Multi-face handling picks only largest face; no multi-person overlay.
* Lighting and orientation heuristics can misclassify edge cases (e.g. high contrast neutral face → angry/surprised bias).
* Image transformation screen does not use actual generative modeling—purely procedural.

---
### 📈 Roadmap (Mobile-Specific)
| Priority | Feature |
|----------|---------|
| High | ONNX runtime integration & GPU acceleration |
| High | Real model inference parity test vs backend FastAPI |
| Medium | Multi-face simultaneous tracking & per-face emotion chips |
| Medium | Offline batching for captured photos |
| Medium | Local caching of last N results (session analytics) |
| Low | Haptic feedback tied to confidence thresholds |
| Low | In-app tutorial overlay explaining confidence bars |

---
### 🤝 Integration With Backend
Short-term (fallback): Keep heuristic local while backend API (FastAPI) can be queried for ground truth comparison (add future `RemoteEmotionService`).

Long-term: All primary inference local (privacy + latency) with optional remote re-labeling for continuous improvement.

---
### 📄 Licensing / Attribution
Uses:
* `google_mlkit_face_detection` for face & landmark probabilities
* `camera` plugin for realtime frame streaming
* `image` for pixel-level processing & procedural generation

Refer to root project LICENSE for umbrella terms.

---
### 🙋 Support / Questions
Open an issue in the main repository with the `[flutter]` prefix describing:
* Device model & OS
* Steps to reproduce
* Logs (if crash)

---

_Generated mobile README (2025-09-24)._
