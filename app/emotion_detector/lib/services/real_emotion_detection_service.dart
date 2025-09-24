// Placeholder for the ONNX-runtime based service implementation.
// The real implementation depends on the onnxruntime package,
// which isn't included right now. This keeps the build green.
import 'dart:io';

class EmotionDetectionService {
  bool _initialized = false;

  Future<void> initialize() async {
    _initialized = true;
  }

  Future<_StubEmotionResult> detectEmotion(File imageFile) async {
    if (!_initialized) {
      throw Exception('Service not initialized');
    }
    // Return a neutral stub result to satisfy analyzer and optional debug runs.
    return _StubEmotionResult(
      emotion: 'neutral',
      confidence: 0.0,
      allPredictions: const {'neutral': 1.0},
    );
  }

  void dispose() {}
}

class _StubEmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;

  const _StubEmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
  });
}

// Export-like alias so imports that expect EmotionResult still work in example screens
typedef EmotionResult = _StubEmotionResult;
