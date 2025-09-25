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
  // Disposal guard to avoid setState after dispose
  bool _isDisposing = false;
  // Cooldown handling: pause frame processing for a period after an emotion is announced
  static const Duration _cooldownDuration = Duration(seconds: 10);
  bool _cooldownActive = false;
  DateTime? _cooldownEndsAt;
  Timer? _cooldownTimer;
  int _cooldownRemaining = 0; // seconds

  final RealtimeEmotionService _emotionService = RealtimeEmotionService();
  StreamSubscription<RealtimeEmotionResult>? _emotionSubscription;

  RealtimeEmotionResult? _currentResult;
  String _statusMessage = 'Initializing camera...';
  bool _isServiceInitialized = false;

  void _safeSetState(VoidCallback fn) {
    if (!mounted || _isDisposing) return;
    setState(fn);
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeServices();
  }

  @override
  void dispose() {
    _isDisposing = true;
    WidgetsBinding.instance.removeObserver(this);
    // Stop streams/timers safely without triggering UI updates
    try {
      _cooldownTimer?.cancel();
      _cooldownTimer = null;
      _emotionSubscription?.cancel();
      _emotionSubscription = null;
      _emotionService.stopRealtimeDetection();
      _emotionService.dispose();
      _cameraController?.dispose();
      _cameraController = null;
    } catch (_) {}
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
      _safeSetState(() {
        _statusMessage = 'Initializing emotion detection...';
      });

      await _emotionService.initialize();

      _safeSetState(() {
        _isServiceInitialized = true;
        _statusMessage = 'Initializing camera...';
      });

      await _initializeCamera();
    } catch (e) {
      _safeSetState(() {
        _statusMessage = 'Error initializing: $e';
      });
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        _safeSetState(() => _statusMessage = 'No cameras available');
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

      _safeSetState(() {
        _isCameraInitialized = true;
        _statusMessage = 'Ready! Tap "Start Detection" to begin';
      });
    } catch (e) {
      _safeSetState(() => _statusMessage = 'Camera error: $e');
    }
  }

  void _startRealtimeDetection() {
    if (!_isServiceInitialized || !_isCameraInitialized) return;
    _safeSetState(() {
      _isProcessingStarted = true;
      _statusMessage = 'Detecting emotions in real-time...';
    });
    _emotionSubscription =
        _emotionService.startRealtimeDetection(_cameraController!).listen(
      (result) {
        if (!mounted || _isDisposing) return;
        // Update current result always so bounding box & overlay stay fresh
        _safeSetState(() => _currentResult = result);
        // If we're already in cooldown, don't trigger a new announcement/cooldown
        if (_cooldownActive) return;
        // Start cooldown now that we've "announced" this emotion
        _startCooldown();
      },
      onError: (error) =>
          _safeSetState(() => _statusMessage = 'Detection error: $error'),
    );
  }

  void _stopRealtimeDetection() {
    _safeSetState(() {
      _isProcessingStarted = false;
      _statusMessage = 'Detection stopped. Tap "Start Detection" to resume.';
      _currentResult = null;
    });
    _emotionSubscription?.cancel();
    _emotionService.stopRealtimeDetection();
    _cancelCooldown();
  }

  void _stopCamera() {
    _stopRealtimeDetection();
    _cameraController?.dispose();
    _cameraController = null;
    _isCameraInitialized = false;
  }

  void _startCooldown() {
    _cooldownActive = true;
    _cooldownEndsAt = DateTime.now().add(_cooldownDuration);
    _cooldownRemaining = _cooldownDuration.inSeconds;
    _cooldownTimer?.cancel();
    _cooldownTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted || _isDisposing) {
        t.cancel();
        return;
      }
      final now = DateTime.now();
      final remaining = _cooldownEndsAt!.difference(now).inSeconds;
      if (remaining <= 0) {
        _safeSetState(() {
          _cooldownActive = false;
          _cooldownRemaining = 0;
        });
        t.cancel();
      } else {
        _safeSetState(() => _cooldownRemaining = remaining);
      }
    });
  }

  void _cancelCooldown() {
    _cooldownTimer?.cancel();
    _cooldownActive = false;
    _cooldownRemaining = 0;
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

  // No face overlay in API-only mode
  Widget _buildFaceOverlay() => const SizedBox.shrink();

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
                            ? _cooldownActive
                                ? 'Emotion: ${_currentResult!.emotion} (${(_currentResult!.confidence * 100).toStringAsFixed(1)}%)  (cooldown ${_cooldownRemaining}s)'
                                : 'Emotion: ${_currentResult!.emotion} (${(_currentResult!.confidence * 100).toStringAsFixed(1)}%)'
                            : 'Capturing...')
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
