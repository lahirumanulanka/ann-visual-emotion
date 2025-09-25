import 'dart:io';
import 'package:cross_file/cross_file.dart';
import 'package:image_picker/image_picker.dart';
import '../services/hf_emotion_api_service.dart';
import '../config/api_config.dart';

// API-only replacement: Detect emotions by sending the whole image to the HF endpoint.
class FaceDetectionEmotionService {
  late final HFEmotionApiService _api;
  bool _isInitialized = false;

  Future<void> initialize() async {
    _api = HFEmotionApiService(
        apiUrl: ApiConfig.apiUrl, apiToken: ApiConfig.apiToken);
    _isInitialized = true;
  }

  Future<EmotionResult> detectEmotion(File imageFile) async {
    if (!_isInitialized) {
      throw Exception('Service not initialized. Call initialize() first.');
    }
    final xfile = XFile(imageFile.path);
    final result = await _api.detectEmotion(xfile);
    return EmotionResult(
      emotion: result.emotion,
      confidence: result.confidence,
      allPredictions: result.allPredictions,
      faceDetected: true,
      faceCount: 1,
    );
  }

  void dispose() {}
}

class EmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;
  final bool faceDetected;
  final int faceCount;

  EmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
    this.faceDetected = false,
    this.faceCount = 0,
  });

  @override
  String toString() {
    return 'EmotionResult(emotion: $emotion, confidence: ${(confidence * 100).toStringAsFixed(1)}%, '
        'faceDetected: $faceDetected, faceCount: $faceCount)';
  }
}
