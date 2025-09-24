import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart'; // for rootBundle, WriteBuffer
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class RealtimeEmotionService {
  late Map<String, int> _labelMap;
  late List<String> _emotions;
  late FaceDetector _faceDetector;
  bool _isInitialized = false;

  StreamController<RealtimeEmotionResult>? _emotionStreamController;
  Timer? _analysisTimer;
  bool _isAnalyzing = false;

  // Camera parameters for correct ML Kit orientation and overlay mapping
  int _rotationDegrees = 0; // 0, 90, 180, 270
  bool _isFrontCamera = false;

  // Exponential moving average for smoothing predictions
  final double _emaAlpha = 0.6; // higher = more responsive, lower = smoother
  Map<String, double>? _emaScores;

  Future<void> initialize() async {
    try {
      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableLandmarks: true,
          enableClassification: true,
          enableTracking: true,
          minFaceSize: 0.15,
          performanceMode: FaceDetectorMode.fast,
        ),
      );

      final labelMapString =
          await rootBundle.loadString('assets/label_map.json');
      _labelMap = Map<String, int>.from(json.decode(labelMapString));

      _emotions = List.filled(_labelMap.length, '');
      _labelMap.forEach((emotion, index) {
        _emotions[index] = emotion;
      });

      _isInitialized = true;
    } catch (e) {
      throw Exception('Failed to initialize real-time emotion service: $e');
    }
  }

  Stream<RealtimeEmotionResult> startRealtimeDetection() {
    if (!_isInitialized) {
      throw Exception('Service not initialized. Call initialize() first.');
    }
    _emotionStreamController =
        StreamController<RealtimeEmotionResult>.broadcast();
    return _emotionStreamController!.stream;
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

  Future<void> processFrame(CameraImage cameraImage) async {
    if (!_isInitialized || _isAnalyzing || _emotionStreamController == null)
      return;
    _isAnalyzing = true;
    try {
      final inputImage = _convertCameraImage(cameraImage);
      if (inputImage == null) return;
      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isNotEmpty) {
        final bestFace = _selectBestFace(faces);
        final emotionResult = await _analyzeRealtimeFace(bestFace);

        // Apply EMA smoothing
        final smoothed = <String, double>{};
        if (_emaScores == null) {
          _emaScores = Map<String, double>.from(emotionResult.allPredictions);
        } else {
          for (final e in emotionResult.allPredictions.entries) {
            final prev = _emaScores![e.key] ?? e.value;
            smoothed[e.key] = _emaAlpha * e.value + (1 - _emaAlpha) * prev;
          }
          _emaScores = smoothed;
        }

        // Recompute top emotion from smoothed scores
        final predictions = _emaScores ?? emotionResult.allPredictions;
        String topEmotion = predictions.entries.first.key;
        double topScore = predictions.entries.first.value;
        for (final kv in predictions.entries) {
          if (kv.value > topScore) {
            topEmotion = kv.key;
            topScore = kv.value;
          }
        }
        _emotionStreamController?.add(RealtimeEmotionResult(
          emotion: topEmotion,
          confidence: topScore,
          allPredictions: predictions,
          faceDetected: true,
          faceCount: faces.length,
          trackingId: bestFace.trackingId,
          boundingBox: bestFace.boundingBox,
          landmarks: bestFace.landmarks,
          imageWidth: cameraImage.width,
          imageHeight: cameraImage.height,
          imageRotationDegrees: _rotationDegrees,
          isFrontCamera: _isFrontCamera,
        ));
      } else {
        _emotionStreamController?.add(RealtimeEmotionResult(
          emotion: 'neutral',
          confidence: 0.3,
          allPredictions: const {'neutral': 0.3},
          faceDetected: false,
          faceCount: 0,
          imageWidth: cameraImage.width,
          imageHeight: cameraImage.height,
          imageRotationDegrees: _rotationDegrees,
          isFrontCamera: _isFrontCamera,
        ));
      }
    } catch (_) {
      // swallow errors per frame
    } finally {
      _isAnalyzing = false;
    }
  }

  InputImage? _convertCameraImage(CameraImage cameraImage) {
    try {
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in cameraImage.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();
      final imageSize =
          Size(cameraImage.width.toDouble(), cameraImage.height.toDouble());
      final rotation = _toImageRotation(_rotationDegrees);
      const format = InputImageFormat.yuv420;
      final metadata = InputImageMetadata(
        size: imageSize,
        rotation: rotation,
        format: format,
        bytesPerRow: cameraImage.planes[0].bytesPerRow,
      );
      return InputImage.fromBytes(bytes: bytes, metadata: metadata);
    } catch (_) {
      return null;
    }
  }

  InputImageRotation _toImageRotation(int degrees) {
    switch (degrees) {
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      case 0:
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  Face _selectBestFace(List<Face> faces) {
    Face bestFace = faces.first;
    double maxArea = _calculateFaceArea(bestFace);
    for (final face in faces) {
      final area = _calculateFaceArea(face);
      if (area > maxArea) {
        maxArea = area;
        bestFace = face;
      }
    }
    return bestFace;
  }

  double _calculateFaceArea(Face face) =>
      face.boundingBox.width * face.boundingBox.height;

  Future<EmotionAnalysisResult> _analyzeRealtimeFace(Face face) async {
    final smileProbability = face.smilingProbability ?? 0.5;
    final leftEyeOpenProbability = face.leftEyeOpenProbability ?? 0.5;
    final rightEyeOpenProbability = face.rightEyeOpenProbability ?? 0.5;

    final scores = _calculateRealtimeEmotionScores(
      smileProbability,
      leftEyeOpenProbability,
      rightEyeOpenProbability,
      face.headEulerAngleY ?? 0.0,
      face.headEulerAngleZ ?? 0.0,
    );

    int maxIndex = 0;
    double maxScore = scores[0];
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIndex = i;
      }
    }

    final detectedEmotion = _emotions[maxIndex];
    final allPredictions = <String, double>{};
    for (int i = 0; i < _emotions.length; i++) {
      allPredictions[_emotions[i]] = scores[i];
    }
    return EmotionAnalysisResult(
      emotion: detectedEmotion,
      confidence: maxScore,
      allPredictions: allPredictions,
    );
  }

  List<double> _calculateRealtimeEmotionScores(
    double smileProbability,
    double leftEyeOpenProbability,
    double rightEyeOpenProbability,
    double headYaw,
    double headRoll,
  ) {
    final scores = <double>[];
    final random = Random();
    final avgEyeOpen = (leftEyeOpenProbability + rightEyeOpenProbability) / 2;
    const baseRandomness = 0.02;
    for (final emotion in _emotions) {
      double score = baseRandomness + (random.nextDouble() * 0.03);
      switch (emotion.toLowerCase()) {
        case 'happy':
          score += smileProbability * 0.6;
          if (avgEyeOpen > 0.7) score += 0.15;
          break;
        case 'sad':
          score += (1.0 - smileProbability) * 0.4;
          if (avgEyeOpen < 0.3) score += 0.2;
          if (headRoll.abs() > 5) score += 0.1;
          break;
        case 'angry':
          score += (1.0 - smileProbability) * 0.3;
          if (avgEyeOpen > 0.8) score += 0.2;
          if (headYaw.abs() > 10) score += 0.1;
          break;
        case 'surprised':
          if (avgEyeOpen > 0.9) score += 0.4;
          if (smileProbability > 0.3 && smileProbability < 0.7) score += 0.2;
          break;
        case 'fearful':
          if (avgEyeOpen > 0.8) score += 0.3;
          score += (1.0 - smileProbability) * 0.2;
          if (headYaw.abs() > 15) score += 0.15;
          break;
        case 'neutral':
          final balanceScore = 1.0 -
              ((smileProbability - 0.5).abs() +
                  (avgEyeOpen - 0.7).abs() +
                  (headYaw.abs() / 30.0) +
                  (headRoll.abs() / 30.0));
          score += balanceScore * 0.4;
          break;
      }
      scores.add(score.clamp(0.0, 1.0));
    }
    final sum = scores.reduce((a, b) => a + b);
    if (sum > 0) {
      for (int i = 0; i < scores.length; i++) {
        scores[i] = scores[i] / sum;
      }
    }
    return scores;
  }

  void stopRealtimeDetection() {
    _analysisTimer?.cancel();
    _emotionStreamController?.close();
    _emotionStreamController = null;
    _emaScores = null;
  }

  void dispose() {
    stopRealtimeDetection();
    if (_isInitialized) {
      _faceDetector.close();
    }
  }
}

class RealtimeEmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;
  final bool faceDetected;
  final int faceCount;
  final int? trackingId;
  final Rect? boundingBox;
  final Map<FaceLandmarkType, FaceLandmark?>? landmarks;
  final int? imageWidth;
  final int? imageHeight;
  final int? imageRotationDegrees; // 0,90,180,270
  final bool? isFrontCamera; // true if front/selfie camera

  RealtimeEmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
    this.faceDetected = false,
    this.faceCount = 0,
    this.trackingId,
    this.boundingBox,
    this.landmarks,
    this.imageWidth,
    this.imageHeight,
    this.imageRotationDegrees,
    this.isFrontCamera,
  });

  @override
  String toString() {
    return 'RealtimeEmotionResult(emotion: $emotion, confidence: '
        '${(confidence * 100).toStringAsFixed(1)}%, faceDetected: $faceDetected, '
        'faceCount: $faceCount, trackingId: $trackingId)';
  }
}

class EmotionAnalysisResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;

  EmotionAnalysisResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
  });
}
