import 'dart:async';
import 'package:camera/camera.dart';
import '../services/hf_emotion_api_service.dart';
import '../config/api_config.dart';

// API-only real-time emotion service: periodically captures photos from the camera
// and sends them to the Hugging Face API. No on-device ML or face detection.
class RealtimeEmotionService {
  final HFEmotionApiService _api = HFEmotionApiService(
      apiUrl: ApiConfig.apiUrl, apiToken: ApiConfig.apiToken);

  bool _isInitialized = false;
  bool _isAnalyzing = false;
  Timer? _timer;
  StreamController<RealtimeEmotionResult>? _emotionStreamController;

  // Camera parameters (only kept for UI orientation/mirroring metadata)
  int _rotationDegrees = 0; // 0, 90, 180, 270
  bool _isFrontCamera = false;

  // Configure capture interval to avoid overwhelming the API
  Duration captureInterval = const Duration(milliseconds: 1200);

  Future<void> initialize() async {
    _isInitialized = true; // nothing to init for API client
  }

  // Update camera parameters when camera is (re)initialized or switched
  void updateCameraParams(
      {required int rotationDegrees, required bool isFrontCamera}) {
    if (![0, 90, 180, 270].contains(rotationDegrees)) {
      rotationDegrees = 0;
    }
    _rotationDegrees = rotationDegrees;
    _isFrontCamera = isFrontCamera;
  }

  // Starts periodic capture using the provided camera controller
  Stream<RealtimeEmotionResult> startRealtimeDetection(
      CameraController controller) {
    if (!_isInitialized) {
      throw Exception('Service not initialized. Call initialize() first.');
    }
    _emotionStreamController =
        StreamController<RealtimeEmotionResult>.broadcast();

    _timer?.cancel();
    _timer = Timer.periodic(captureInterval, (_) async {
      if (_isAnalyzing) return;
      if (!controller.value.isInitialized || controller.value.isTakingPicture)
        return;
      _isAnalyzing = true;
      try {
        final XFile file = await controller.takePicture();
        final result = await _api.detectEmotion(file);
        _emotionStreamController?.add(
          RealtimeEmotionResult(
            emotion: result.emotion,
            confidence: result.confidence,
            allPredictions: result.allPredictions,
            faceDetected:
                true, // We assume a face is present if prediction returned
            faceCount: 1,
            imageRotationDegrees: _rotationDegrees,
            isFrontCamera: _isFrontCamera,
          ),
        );
      } catch (_) {
        // Ignore individual capture/send errors to keep stream alive
      } finally {
        _isAnalyzing = false;
      }
    });

    return _emotionStreamController!.stream;
  }

  void stopRealtimeDetection() {
    _timer?.cancel();
    _timer = null;
    _emotionStreamController?.close();
    _emotionStreamController = null;
  }

  void dispose() {
    stopRealtimeDetection();
  }
}

class RealtimeEmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;
  final bool faceDetected;
  final int faceCount;
  final int? imageWidth;
  final int? imageHeight;
  final int? imageRotationDegrees; // 0,90,180,270
  final bool? isFrontCamera; // true if front/selfie camera

  const RealtimeEmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
    this.faceDetected = false,
    this.faceCount = 0,
    this.imageWidth,
    this.imageHeight,
    this.imageRotationDegrees,
    this.isFrontCamera,
  });

  @override
  String toString() {
    return 'RealtimeEmotionResult(emotion: $emotion, confidence: '
        '${(confidence * 100).toStringAsFixed(1)}%, faceDetected: $faceDetected, '
        'faceCount: $faceCount)';
  }
}
