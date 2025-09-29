import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/realtime_emotion_service.dart';
import '../services/explanation_service.dart';
import 'dart:typed_data';

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
  bool _showHeatmap = true;
  bool _gradCamPlusPlus = false;
  Uint8List? _currentHeatmap;
  bool _isGeneratingHeatmap = false;
  bool _showExplanationPanel = true;
  bool _loadingAttributions = false;
  List<RegionAttribution>? _attributions;
  bool _showLimeMask = false; // toggles between SHAP list and LIME mask
  Uint8List? _limeMask;
  final ExplanationService _explanationService = ExplanationService();

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
      await _explanationService.initialize();

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
      (result) {
        if (!mounted) return;
        // Update current result always so bounding box & overlay stay fresh
        setState(() => _currentResult = result);
        if (_showHeatmap && !_isGeneratingHeatmap) {
          _generateHeatmap();
        }
        // If we're already in cooldown, don't trigger a new announcement/cooldown
        if (_cooldownActive) return;
        // Start cooldown now that we've "announced" this emotion
        _startCooldown();
      },
      onError: (error) =>
          setState(() => _statusMessage = 'Detection error: $error'),
    );
    _cameraController!.startImageStream((CameraImage image) {
      // Skip frame processing while on cooldown to avoid re-announcing same emotion
      if (_cooldownActive) return;
      _emotionService.processFrame(image);
    });
  }

  Future<void> _generateHeatmap() async {
    if (!_isCameraInitialized) return;
    setState(() => _isGeneratingHeatmap = true);
    try {
      // Use preview size as target for simplicity
      final w = _cameraController!.value.previewSize!.width.toInt();
      final h = _cameraController!.value.previewSize!.height.toInt();
      final hm = await _explanationService.generateHeatmap(
          width: w, height: h, gradCamPlusPlus: _gradCamPlusPlus);
      if (!mounted) return;
      setState(() => _currentHeatmap = hm);
    } finally {
      if (mounted) setState(() => _isGeneratingHeatmap = false);
    }
  }

  Future<void> _fetchShap() async {
    setState(() {
      _loadingAttributions = true;
      _showLimeMask = false;
    });
    try {
      final att = await _explanationService.shapAttributions();
      if (!mounted) return;
      setState(() => _attributions = att);
    } finally {
      if (mounted) setState(() => _loadingAttributions = false);
    }
  }

  Future<void> _fetchLime() async {
    setState(() {
      _loadingAttributions = true;
      _showLimeMask = true;
    });
    try {
      final w = 112;
      final h = 112; // downsized mask
      final mask = await _explanationService.limeMask(w, h);
      if (!mounted) return;
      setState(() => _limeMask = mask);
    } finally {
      if (mounted) setState(() => _loadingAttributions = false);
    }
  }

  Widget _buildHeatmapOverlay() {
    if (!_showHeatmap || _currentHeatmap == null || !_isProcessingStarted)
      return const SizedBox.shrink();
    final controller = _cameraController!;
    final previewSize = controller.value.previewSize!;
    return Positioned.fill(
      child: Opacity(
        opacity: 0.55,
        child: Image.memory(
          _currentHeatmap!,
          width: previewSize.width,
          height: previewSize.height,
          fit: BoxFit.cover,
        ),
      ),
    );
  }

  Widget _buildExplanationPanel() {
    if (!_showExplanationPanel || _currentResult == null)
      return const SizedBox.shrink();
    final predictions = _currentResult!.allPredictions.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(16), topRight: Radius.circular(16)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.psychology, color: Colors.white70),
              const SizedBox(width: 8),
              Expanded(
                  child: Text('Explanation',
                      style: const TextStyle(
                          color: Colors.white, fontWeight: FontWeight.bold))),
              IconButton(
                onPressed: () => setState(() => _showExplanationPanel = false),
                icon: const Icon(Icons.close, color: Colors.white70),
              )
            ],
          ),
          SizedBox(
            height: 90,
            child: ListView.builder(
              itemCount: predictions.length,
              itemBuilder: (c, i) {
                final p = predictions[i];
                return Row(
                  children: [
                    SizedBox(
                        width: 70,
                        child: Text(p.key,
                            style: const TextStyle(color: Colors.white70))),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: p.value,
                        backgroundColor: Colors.white24,
                        valueColor: AlwaysStoppedAnimation(
                            p.key == _currentResult!.emotion
                                ? Colors.orangeAccent
                                : Colors.blueGrey),
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text('${(p.value * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 10)),
                  ],
                );
              },
            ),
          ),
          const SizedBox(height: 4),
          Text(_buildSummarySentence(),
              style: const TextStyle(color: Colors.white, fontSize: 12)),
          const SizedBox(height: 8),
          Wrap(spacing: 8, children: [
            ElevatedButton.icon(
              onPressed: _loadingAttributions ? null : _fetchShap,
              icon: const Icon(Icons.bar_chart),
              label: const Text('Explain more (SHAP)'),
              style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  foregroundColor: Colors.white,
                  textStyle: const TextStyle(fontSize: 12)),
            ),
            ElevatedButton.icon(
              onPressed: _loadingAttributions ? null : _fetchLime,
              icon: const Icon(Icons.grid_on),
              label: const Text('LIME Mask'),
              style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal,
                  foregroundColor: Colors.white,
                  textStyle: const TextStyle(fontSize: 12)),
            ),
          ]),
          const SizedBox(height: 6),
          if (_loadingAttributions)
            const Center(child: CircularProgressIndicator(color: Colors.white))
          else
            _buildAttributionContent(),
          const SizedBox(height: 4),
          Text(
              'Disclaimer: Model outputs are probabilistic; explanations approximate internal reasoning.',
              style: const TextStyle(color: Colors.white60, fontSize: 10)),
        ],
      ),
    );
  }

  String _buildSummarySentence() {
    final e = _currentResult?.emotion ?? 'emotion';
    return 'Key facial regions influenced the $e prediction.'; // Placeholder
  }

  Widget _buildAttributionContent() {
    if (_showLimeMask) {
      if (_limeMask == null) return const SizedBox.shrink();
      // render as small grid image using black/white squares
      final w = 112;
      final h = 112;
      return SizedBox(
        height: 100,
        child: GridView.builder(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 16, mainAxisSpacing: 1, crossAxisSpacing: 1),
          itemCount: w * h,
          physics: const NeverScrollableScrollPhysics(),
          itemBuilder: (c, i) {
            final val = _limeMask![i];
            return Container(
                color: val > 0 ? Colors.orangeAccent : Colors.black12);
          },
        ),
      );
    }
    if (_attributions == null) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: _attributions!.take(5).map((a) {
        final positive = a.contribution >= 0;
        return Row(children: [
          Icon(positive ? Icons.add_circle : Icons.remove_circle,
              size: 14,
              color: positive ? Colors.greenAccent : Colors.redAccent),
          const SizedBox(width: 4),
          Expanded(
              child: Text(a.region,
                  style: const TextStyle(color: Colors.white70, fontSize: 12))),
          Text(a.contribution.toStringAsFixed(2),
              style: TextStyle(
                  color: positive ? Colors.greenAccent : Colors.redAccent,
                  fontSize: 11)),
        ]);
      }).toList(),
    );
  }

  void _showHelpDialog() {
    showDialog(
        context: context,
        builder: (_) => AlertDialog(
              title: const Text('Interpreting Explanations'),
              content: const SizedBox(
                width: 400,
                child: SingleChildScrollView(
                    child: Text(
                        'Heatmap colors: red/yellow highlight regions most influencing the predicted emotion; blue areas contribute less. SHAP list shows facial regions with positive (green) or negative (red) impact. LIME mask highlights super-pixels important for the prediction. These are model approximations, not absolute truths.')),
              ),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Close'))
              ],
            ));
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
      if (!mounted) {
        t.cancel();
        return;
      }
      final now = DateTime.now();
      final remaining = _cooldownEndsAt!.difference(now).inSeconds;
      if (remaining <= 0) {
        if (mounted) {
          setState(() {
            _cooldownActive = false;
            _cooldownRemaining = 0;
          });
        }
        t.cancel();
      } else {
        if (mounted) {
          setState(() => _cooldownRemaining = remaining);
        }
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
          IconButton(
              onPressed: _showHelpDialog, icon: const Icon(Icons.help_outline)),
          if (_isCameraInitialized)
            IconButton(
              icon: Icon(_isProcessingStarted ? Icons.stop : Icons.play_arrow),
              onPressed: _isProcessingStarted
                  ? _stopRealtimeDetection
                  : _startRealtimeDetection,
            ),
          if (_isProcessingStarted)
            IconButton(
              tooltip: 'Toggle heatmap overlay',
              icon:
                  Icon(_showHeatmap ? Icons.visibility : Icons.visibility_off),
              onPressed: () => setState(() => _showHeatmap = !_showHeatmap),
            ),
          if (_isProcessingStarted)
            IconButton(
              tooltip: 'Switch Grad-CAM / Grad-CAM++',
              icon: Icon(_gradCamPlusPlus
                  ? Icons.auto_awesome
                  : Icons.auto_awesome_outlined),
              onPressed: () {
                setState(() => _gradCamPlusPlus = !_gradCamPlusPlus);
                _generateHeatmap();
              },
            ),
          if (_isProcessingStarted)
            IconButton(
              tooltip: _showExplanationPanel
                  ? 'Hide explanation'
                  : 'Show explanation',
              icon: Icon(_showExplanationPanel
                  ? Icons.keyboard_arrow_down
                  : Icons.keyboard_arrow_up),
              onPressed: () => setState(
                  () => _showExplanationPanel = !_showExplanationPanel),
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
                                : 'Face detected! Emotion: ${_currentResult!.emotion} (${(_currentResult!.confidence * 100).toStringAsFixed(1)}%)'
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
                          if (_isProcessingStarted) _buildHeatmapOverlay(),
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
              child: Stack(
                children: [
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16.0),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade50,
                      border:
                          Border(top: BorderSide(color: Colors.grey.shade300)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildEmotionIcon(_currentResult!.emotion),
                        const SizedBox(width: 12),
                        Expanded(
                            child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Current Emotion:',
                                style: Theme.of(context).textTheme.titleMedium),
                            Text(_currentResult!.emotion.toUpperCase(),
                                style: Theme.of(context)
                                    .textTheme
                                    .headlineSmall
                                    ?.copyWith(
                                        fontWeight: FontWeight.bold,
                                        color: Theme.of(context).primaryColor)),
                            Text(
                                'Confidence: ${(_currentResult!.confidence * 100).toStringAsFixed(1)}%'),
                            const SizedBox(height: 8),
                            if (!_showExplanationPanel)
                              Text(
                                  'Tap the arrow icon in the app bar to open detailed explanation panel.',
                                  style: TextStyle(
                                      color: Colors.grey.shade600,
                                      fontSize: 12)),
                          ],
                        )),
                      ],
                    ),
                  ),
                  Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: _buildExplanationPanel(),
                  )
                ],
              ),
            ),
        ],
      ),
    );
  }
}
