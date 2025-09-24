import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/realtime_emotion_service.dart';

class RealtimeCameraScreen extends StatefulWidget {
  const RealtimeCameraScreen({super.key});

  @override
  State<RealtimeCameraScreen> createState() => _RealtimeCameraScreenState();
}

class _RealtimeCameraScreenState extends State<RealtimeCameraScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraInitialized = false;
  bool _isProcessingStarted = false;

  final RealtimeEmotionService _emotionService = RealtimeEmotionService();
  StreamSubscription<RealtimeEmotionResult>? _emotionSubscription;

  RealtimeEmotionResult? _currentResult;
  String _statusMessage = 'Initializing camera...';
  bool _isServiceInitialized = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeServices();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopCamera();
    _emotionService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeServices() async {
    try {
      setState(() {
        _statusMessage = 'Initializing emotion detection...';
      });

      await _emotionService.initialize();

      setState(() {
        _isServiceInitialized = true;
        _statusMessage = 'Initializing camera...';
      });

      await _initializeCamera();
    } catch (e) {
      setState(() {
        _statusMessage = 'Error initializing: $e';
      });
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _statusMessage = 'No cameras available');
        return;
      }
      final camera = _cameras.length > 1 ? _cameras[1] : _cameras[0];
      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _cameraController!.initialize();

      // Inform service about camera params for orientation/mirroring
      final rotationDegrees = _cameraController!.description.sensorOrientation;
      final isFront = _cameraController!.description.lensDirection ==
          CameraLensDirection.front;
      _emotionService.updateCameraParams(
          rotationDegrees: rotationDegrees, isFrontCamera: isFront);

      setState(() {
        _isCameraInitialized = true;
        _statusMessage = 'Ready! Tap "Start Detection" to begin';
      });
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  void _startRealtimeDetection() {
    if (!_isServiceInitialized || !_isCameraInitialized) return;
    setState(() {
      _isProcessingStarted = true;
      _statusMessage = 'Detecting emotions in real-time...';
    });
    _emotionSubscription = _emotionService.startRealtimeDetection().listen(
          (result) => mounted ? setState(() => _currentResult = result) : null,
          onError: (error) =>
              setState(() => _statusMessage = 'Detection error: $error'),
        );
    _cameraController!.startImageStream((CameraImage image) {
      _emotionService.processFrame(image);
    });
  }

  void _stopRealtimeDetection() {
    setState(() {
      _isProcessingStarted = false;
      _statusMessage = 'Detection stopped. Tap "Start Detection" to resume.';
      _currentResult = null;
    });
    _cameraController?.stopImageStream();
    _emotionSubscription?.cancel();
    _emotionService.stopRealtimeDetection();
  }

  void _stopCamera() {
    _stopRealtimeDetection();
    _cameraController?.dispose();
    _cameraController = null;
    _isCameraInitialized = false;
  }

  Widget _buildEmotionIcon(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
        return const Icon(Icons.sentiment_very_satisfied,
            color: Colors.green, size: 48);
      case 'sad':
        return const Icon(Icons.sentiment_very_dissatisfied,
            color: Colors.blue, size: 48);
      case 'angry':
        return const Icon(Icons.sentiment_dissatisfied,
            color: Colors.red, size: 48);
      case 'surprised':
        return const Icon(Icons.sentiment_neutral,
            color: Colors.orange, size: 48);
      case 'fearful':
        return const Icon(Icons.sentiment_dissatisfied,
            color: Colors.purple, size: 48);
      case 'neutral':
        return const Icon(Icons.sentiment_neutral,
            color: Colors.grey, size: 48);
      default:
        return const Icon(Icons.help_outline, color: Colors.grey, size: 48);
    }
  }

  Widget _buildFaceOverlay() {
    if (_currentResult?.boundingBox == null || !_isCameraInitialized) {
      return const SizedBox.shrink();
    }
    final boundingBox = _currentResult!.boundingBox!;
    final controller = _cameraController!;
    final previewSize = controller.value.previewSize;
    if (previewSize == null) return const SizedBox.shrink();

    final screenWidth = MediaQuery.of(context).size.width;
    final previewAspect =
        controller.value.aspectRatio; // width/height from camera
    final previewHeight = screenWidth / previewAspect;

    // Source image dimensions & rotation from the service result
    final imageW =
        (_currentResult!.imageWidth ?? previewSize.width.toInt()).toDouble();
    final imageH =
        (_currentResult!.imageHeight ?? previewSize.height.toInt()).toDouble();
    final rot = _currentResult!.imageRotationDegrees ??
        controller.description.sensorOrientation;
    final isFront = _currentResult!.isFrontCamera ??
        (controller.description.lensDirection == CameraLensDirection.front);

    // ML Kit coordinates are in the input image's orientation. If rotated 90/270, swap w/h.
    final bool swapWH = rot == 90 || rot == 270;
    final srcW = swapWH ? imageH : imageW;
    final srcH = swapWH ? imageW : imageH;

    final scaleX = screenWidth / srcW;
    final scaleY = previewHeight / srcH;

    double left = boundingBox.left * scaleX;
    final top = boundingBox.top * scaleY;
    final width = boundingBox.width * scaleX;
    final height = boundingBox.height * scaleY;

    // Mirror for front camera
    if (isFront) {
      left = screenWidth - (left + width);
    }

    return Positioned(
      left: left,
      top: top,
      width: width,
      height: height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(
            color: _currentResult!.faceDetected ? Colors.green : Colors.red,
            width: 3,
          ),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Container(
          padding: const EdgeInsets.all(4),
          child: Text(
            _currentResult!.emotion.toUpperCase(),
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 12,
              backgroundColor: Colors.black54,
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Real-time Emotion Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_isCameraInitialized)
            IconButton(
              icon: Icon(_isProcessingStarted ? Icons.stop : Icons.play_arrow),
              onPressed: _isProcessingStarted
                  ? _stopRealtimeDetection
                  : _startRealtimeDetection,
            ),
        ],
      ),
      body: Column(
        children: [
          // Status message
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12.0),
            decoration: BoxDecoration(
              color: _currentResult?.faceDetected == true
                  ? Colors.green.shade50
                  : Colors.blue.shade50,
              border: Border(
                bottom: BorderSide(
                  color: _currentResult?.faceDetected == true
                      ? Colors.green
                      : Colors.blue,
                  width: 1.0,
                ),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  _currentResult?.faceDetected == true
                      ? Icons.face
                      : _isProcessingStarted
                          ? Icons.search
                          : Icons.info,
                  color: _currentResult?.faceDetected == true
                      ? Colors.green
                      : Colors.blue,
                ),
                const SizedBox(width: 8.0),
                Expanded(
                  child: Text(
                    _currentResult != null
                        ? (_currentResult!.faceDetected
                            ? 'Face detected! Emotion: ${_currentResult!.emotion} (${(_currentResult!.confidence * 100).toStringAsFixed(1)}%)'
                            : 'Looking for faces...')
                        : _statusMessage,
                    style: TextStyle(
                      color: _currentResult?.faceDetected == true
                          ? Colors.green.shade700
                          : Colors.blue.shade700,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                if (_currentResult?.faceDetected == true &&
                    (_currentResult!.faceCount > 0)) ...[
                  const SizedBox(width: 8.0),
                  Chip(
                    label: Text(
                        '${_currentResult!.faceCount} face${_currentResult!.faceCount > 1 ? 's' : ''}'),
                    backgroundColor: Colors.green.shade100,
                  ),
                ],
              ],
            ),
          ),

          // Camera preview (fixed aspect, no stretch)
          Expanded(
            flex: 3,
            child: _isCameraInitialized && _cameraController != null
                ? LayoutBuilder(
                    builder: (context, constraints) {
                      final w = constraints.maxWidth;
                      final aspect = _cameraController!.value.aspectRatio;
                      final h = w / aspect;
                      return Stack(
                        children: [
                          // Use FittedBox to ensure the preview maintains aspect ratio without stretch
                          SizedBox(
                            width: w,
                            height: h,
                            child: FittedBox(
                              fit: BoxFit.cover,
                              clipBehavior: Clip.hardEdge,
                              child: SizedBox(
                                width:
                                    _cameraController!.value.previewSize!.width,
                                height: _cameraController!
                                    .value.previewSize!.height,
                                child: CameraPreview(_cameraController!),
                              ),
                            ),
                          ),
                          if (_isProcessingStarted) _buildFaceOverlay(),
                        ],
                      );
                    },
                  )
                : Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 16),
                        Text(_statusMessage),
                      ],
                    ),
                  ),
          ),

          // Control buttons
          Container(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isCameraInitialized && !_isProcessingStarted
                      ? _startRealtimeDetection
                      : null,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Start Detection'),
                  style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      foregroundColor: Colors.white),
                ),
                ElevatedButton.icon(
                  onPressed:
                      _isProcessingStarted ? _stopRealtimeDetection : null,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop Detection'),
                  style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white),
                ),
              ],
            ),
          ),

          // Current emotion result
          if (_currentResult != null)
            Expanded(
              flex: 2,
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: Colors.grey.shade50,
                  border: Border(top: BorderSide(color: Colors.grey.shade300)),
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _buildEmotionIcon(_currentResult!.emotion),
                        const SizedBox(width: 16.0),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Current Emotion:',
                                style: Theme.of(context).textTheme.titleMedium),
                            Text(
                              _currentResult!.emotion.toUpperCase(),
                              style: Theme.of(context)
                                  .textTheme
                                  .headlineSmall
                                  ?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context).primaryColor,
                                  ),
                            ),
                            Text(
                                'Confidence: ${(_currentResult!.confidence * 100).toStringAsFixed(1)}%',
                                style: Theme.of(context).textTheme.bodyLarge),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 16.0),
                    Expanded(
                      child: SingleChildScrollView(
                        child: Column(
                          children: _currentResult!.allPredictions.entries
                              .map((entry) {
                            final percentage = entry.value * 100;
                            return Padding(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 2.0),
                              child: Row(
                                children: [
                                  SizedBox(
                                    width: 80,
                                    child: Text(entry.key,
                                        style: const TextStyle(
                                            fontWeight: FontWeight.w500)),
                                  ),
                                  Expanded(
                                    child: LinearProgressIndicator(
                                      value: entry.value,
                                      backgroundColor: Colors.grey.shade300,
                                      valueColor: AlwaysStoppedAnimation<Color>(
                                        entry.key == _currentResult!.emotion
                                            ? Theme.of(context).primaryColor
                                            : Colors.grey,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 8.0),
                                  SizedBox(
                                    width: 45,
                                    child: Text(
                                        '${percentage.toStringAsFixed(1)}%',
                                        textAlign: TextAlign.right,
                                        style: const TextStyle(
                                            fontWeight: FontWeight.w500)),
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
