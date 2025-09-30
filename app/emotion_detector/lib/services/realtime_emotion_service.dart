import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart'; // for rootBundle, WriteBuffer
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class RealtimeEmotionService {
  late Map<String, int> _labelMap;
  late List<String> _emotions;
  late FaceDetector _faceDetector;
  bool _isInitialized = false;

  // iOS-only face crop saving configuration
  bool _saveCropsToGallery = false; // disabled by default
  Duration _cropSaveInterval = const Duration(seconds: 3);
  int _lastCropSaveMs = 0;
  String _cropFilePrefix = 'rt_emotion_face';

  void enableRealtimeGallerySaving({
    bool enabled = true,
    Duration? minInterval,
    String? prefix,
  }) {
    _saveCropsToGallery = enabled;
    if (minInterval != null) _cropSaveInterval = minInterval;
    if (prefix != null) _cropFilePrefix = prefix;
  }

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
        final result = RealtimeEmotionResult(
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
        );
        _emotionStreamController?.add(result);

        // Attempt to save crop (iOS only, throttled)
        if (_saveCropsToGallery) {
          _maybeSaveFaceCrop(cameraImage, bestFace, result.emotion);
        }
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

  // --- iOS ONLY: Save cropped face to Photos for validation ---
  Future<void> _maybeSaveFaceCrop(
      CameraImage image, Face face, String emotion) async {
    try {
      if (!Platform.isIOS) return; // Only do this on iOS per requirement
      final now = DateTime.now().millisecondsSinceEpoch;
      if (now - _lastCropSaveMs < _cropSaveInterval.inMilliseconds) return;

      // Permissions (try add-only then broader Photos)
      var status = await Permission.photosAddOnly.request();
      if (!status.isGranted) {
        status = await Permission.photos.request();
        if (!status.isGranted) return;
      }

      // Convert YUV420 -> RGB
      final rgb = _yuv420ToImage(image);
      if (rgb == null) return;

      // Rotate to upright orientation
      img.Image oriented = rgb;
      switch (_rotationDegrees) {
        case 90:
          oriented = img.copyRotate(rgb, angle: 90);
          break;
        case 180:
          oriented = img.copyRotate(rgb, angle: 180);
          break;
        case 270:
          oriented = img.copyRotate(rgb, angle: 270);
          break;
      }
      // Mirror if front camera (after rotation)
      if (_isFrontCamera) {
        oriented = img.flipHorizontal(oriented);
      }

      // Bounding box should correspond to upright orientation (because we passed rotation to ML Kit)
      final bbox = face.boundingBox;
      int x = bbox.left.floor();
      int y = bbox.top.floor();
      int w = bbox.width.floor();
      int h = bbox.height.floor();
      // Clamp
      if (x < 0) x = 0;
      if (y < 0) y = 0;
      if (x + w > oriented.width) w = oriented.width - x;
      if (y + h > oriented.height) h = oriented.height - y;
      if (w <= 0 || h <= 0) return;

      final crop = img.copyCrop(oriented, x: x, y: y, width: w, height: h);
      final ts = now;
      final fname = '${_cropFilePrefix}_${emotion.toLowerCase()}_$ts.jpg';
      final jpg = img.encodeJpg(crop, quality: 90);
      final saveRes = await ImageGallerySaver.saveImage(
        Uint8List.fromList(jpg),
        quality: 90,
        name: fname,
      );
      if (saveRes is Map &&
          (saveRes['isSuccess'] == true || saveRes['success'] == true)) {
        _lastCropSaveMs = now;
      }
    } catch (_) {
      // Swallow saving errors to avoid disrupting real-time pipeline
    }
  }

  // Basic YUV420 (planar) to RGB conversion (performance trade-off: called only when throttled)
  img.Image? _yuv420ToImage(CameraImage image) {
    try {
      final width = image.width;
      final height = image.height;
      final yPlane = image.planes[0];
      final uPlane = image.planes.length > 1 ? image.planes[1] : null;
      final vPlane = image.planes.length > 2 ? image.planes[2] : null;
      // If planes missing, abort
      if (uPlane == null || vPlane == null) return null;

      final img.Image rgbImage = img.Image(width: width, height: height);
      final int uvRowStride = uPlane.bytesPerRow;
      final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

      for (int y = 0; y < height; y++) {
        final yRow = y * yPlane.bytesPerRow;
        final uvRow = (y >> 1) * uvRowStride;
        for (int x = 0; x < width; x++) {
          final int yIndex = yRow + x;
          final int uvIndex = uvRow + (x >> 1) * uvPixelStride;
          final yp = yPlane.bytes[yIndex];
          final up = uPlane.bytes[uvIndex];
          final vp = vPlane.bytes[uvIndex];

          final double Y = yp.toDouble();
          final double U = up.toDouble() - 128.0;
          final double V = vp.toDouble() - 128.0;

          // BT.601 conversion
          double r = Y + 1.402 * V;
          double g = Y - 0.344136 * U - 0.714136 * V;
          double b = Y + 1.772 * U;
          if (r < 0)
            r = 0;
          else if (r > 255) r = 255;
          if (g < 0)
            g = 0;
          else if (g > 255) g = 255;
          if (b < 0)
            b = 0;
          else if (b > 255) b = 255;
          rgbImage.setPixelRgba(x, y, r.toInt(), g.toInt(), b.toInt(), 255);
        }
      }
      return rgbImage;
    } catch (_) {
      return null;
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
