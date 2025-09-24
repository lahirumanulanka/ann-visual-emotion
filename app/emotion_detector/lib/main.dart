import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'services/face_detection_emotion_service.dart';
import 'screens/realtime_camera_screen.dart';

void main() {
  runApp(const EmotionDetectorApp());
}

class EmotionDetectorApp extends StatelessWidget {
  const EmotionDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Detector Application',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const MainTabScreen(),
    );
  }
}

class MainTabScreen extends StatefulWidget {
  const MainTabScreen({super.key});

  @override
  State<MainTabScreen> createState() => _MainTabScreenState();
}

class _MainTabScreenState extends State<MainTabScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Emotion Detector'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(
              icon: Icon(Icons.photo_camera),
              text: 'Photo Analysis',
            ),
            Tab(
              icon: Icon(Icons.video_camera_front),
              text: 'Real-time Detection',
            ),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: const [
          EmotionDetectorHome(),
          RealtimeCameraScreen(),
        ],
      ),
    );
  }
}

class EmotionDetectorHome extends StatefulWidget {
  const EmotionDetectorHome({super.key});

  @override
  State<EmotionDetectorHome> createState() => _EmotionDetectorHomeState();
}

class _EmotionDetectorHomeState extends State<EmotionDetectorHome> {
  File? _selectedImage;
  String? _detectedEmotion;
  double? _confidence;
  Map<String, double>? _allPredictions;
  bool _isLoading = false;
  String _statusMessage = '';
  bool _faceDetected = false;
  int _faceCount = 0;

  final ImagePicker _picker = ImagePicker();
  final FaceDetectionEmotionService _emotionService =
      FaceDetectionEmotionService();

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  @override
  void dispose() {
    _emotionService.dispose();
    super.dispose();
  }

  Future<void> _initializeService() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Initializing emotion detection...';
    });

    try {
      await _emotionService.initialize();
      setState(() {
        _isLoading = false;
        _statusMessage = 'Ready to detect emotions with face detection!';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Error initializing: $e';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(source: source);
      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
          _detectedEmotion = null;
          _confidence = null;
          _allPredictions = null;
          _faceDetected = false;
          _faceCount = 0;
          _statusMessage =
              'Image selected. Tap "Analyze Emotion" to start detection.';
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error picking image: $e';
      });
    }
  }

  Future<void> _analyzeEmotion() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
      _statusMessage = 'Detecting faces and analyzing emotion...';
    });

    try {
      final result = await _emotionService.detectEmotion(_selectedImage!);

      setState(() {
        _detectedEmotion = result.emotion;
        _confidence = result.confidence;
        _allPredictions = result.allPredictions;
        _faceDetected = result.faceDetected;
        _faceCount = result.faceCount;
        _isLoading = false;

        if (result.faceDetected) {
          _statusMessage =
              'Face detected! Emotion: ${result.emotion} (${(result.confidence * 100).toStringAsFixed(1)}%)';
        } else {
          _statusMessage =
              'No faces detected. Analyzed full image: ${result.emotion} (${(result.confidence * 100).toStringAsFixed(1)}%)';
        }
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Error analyzing emotion: $e';
      });
    }
  }

  Widget _buildEmotionIcon(String emotion) {
    IconData iconData;
    Color iconColor;

    switch (emotion.toLowerCase()) {
      case 'happy':
        iconData = Icons.sentiment_very_satisfied;
        iconColor = Colors.green;
        break;
      case 'sad':
        iconData = Icons.sentiment_very_dissatisfied;
        iconColor = Colors.blue;
        break;
      case 'angry':
        iconData = Icons.sentiment_dissatisfied;
        iconColor = Colors.red;
        break;
      case 'surprised':
        iconData = Icons.sentiment_neutral;
        iconColor = Colors.orange;
        break;
      case 'fearful':
        iconData = Icons.sentiment_dissatisfied;
        iconColor = Colors.purple;
        break;
      case 'neutral':
        iconData = Icons.sentiment_neutral;
        iconColor = Colors.grey;
        break;
      default:
        iconData = Icons.help_outline;
        iconColor = Colors.grey;
    }

    return Icon(iconData, color: iconColor, size: 48);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Status message
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color:
                    _faceDetected ? Colors.green.shade50 : Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8.0),
                border: Border.all(
                  color: _faceDetected ? Colors.green : Colors.blue,
                  width: 1.0,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    _faceDetected ? Icons.face : Icons.info,
                    color: _faceDetected ? Colors.green : Colors.blue,
                  ),
                  const SizedBox(width: 8.0),
                  Expanded(
                    child: Text(
                      _statusMessage,
                      style: TextStyle(
                        color: _faceDetected
                            ? Colors.green.shade700
                            : Colors.blue.shade700,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  if (_faceDetected && _faceCount > 0) ...[
                    const SizedBox(width: 8.0),
                    Chip(
                      label: Text(
                          '${_faceCount} face${_faceCount > 1 ? 's' : ''}'),
                      backgroundColor: Colors.green.shade100,
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20.0),

            // Image selection buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed:
                      _isLoading ? null : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ],
            ),
            const SizedBox(height: 20.0),

            // Selected image
            if (_selectedImage != null) ...[
              Container(
                width: double.infinity,
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8.0),
                  child: Image.file(
                    _selectedImage!,
                    fit: BoxFit.contain,
                  ),
                ),
              ),
              const SizedBox(height: 20.0),

              // Analyze button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _isLoading ? null : _analyzeEmotion,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.psychology),
                  label: Text(_isLoading ? 'Analyzing...' : 'Analyze Emotion'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16.0),
                  ),
                ),
              ),
            ],

            // Results
            if (_detectedEmotion != null && _confidence != null) ...[
              const SizedBox(height: 30.0),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildEmotionIcon(_detectedEmotion!),
                          const SizedBox(width: 16.0),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Detected Emotion:',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                              Text(
                                _detectedEmotion!.toUpperCase(),
                                style: Theme.of(context)
                                    .textTheme
                                    .headlineSmall
                                    ?.copyWith(
                                      fontWeight: FontWeight.bold,
                                      color: Theme.of(context).primaryColor,
                                    ),
                              ),
                              Text(
                                'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                                style: Theme.of(context).textTheme.bodyLarge,
                              ),
                            ],
                          ),
                        ],
                      ),

                      // All predictions
                      if (_allPredictions != null) ...[
                        const SizedBox(height: 20.0),
                        const Divider(),
                        const SizedBox(height: 10.0),
                        Text(
                          'All Predictions:',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 10.0),
                        ..._allPredictions!.entries.map((entry) {
                          final percentage = entry.value * 100;
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4.0),
                            child: Row(
                              children: [
                                SizedBox(
                                  width: 80,
                                  child: Text(
                                    entry.key,
                                    style: const TextStyle(
                                        fontWeight: FontWeight.w500),
                                  ),
                                ),
                                Expanded(
                                  child: LinearProgressIndicator(
                                    value: entry.value,
                                    backgroundColor: Colors.grey.shade300,
                                    valueColor: AlwaysStoppedAnimation<Color>(
                                      entry.key == _detectedEmotion
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
                                        fontWeight: FontWeight.w500),
                                  ),
                                ),
                              ],
                            ),
                          );
                        }).toList(),
                      ],
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 20.0),
          ],
        ),
      ),
    );
  }
}
