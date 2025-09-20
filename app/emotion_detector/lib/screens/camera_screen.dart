import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import '../services/emotion_detection_service.dart';

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;
  final EmotionDetectionService emotionService;

  const CameraScreen({
    Key? key,
    required this.camera,
    required this.emotionService,
  }) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  StreamSubscription? _imageStreamSubscription;
  
  EmotionResult? _lastEmotionResult;
  bool _isDetecting = false;
  bool _isStreamingStarted = false;
  
  // Performance tracking
  DateTime? _lastDetectionTime;
  int _detectionCount = 0;
  double _averageFps = 0.0;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _imageStreamSubscription?.cancel();
    _controller.dispose();
    super.dispose();
  }

  void _startImageStream() {
    if (_isStreamingStarted) return;
    
    _controller.startImageStream((CameraImage image) {
      if (!_isDetecting) {
        _detectEmotion(image);
      }
    });
    
    setState(() {
      _isStreamingStarted = true;
    });
  }

  void _stopImageStream() {
    if (!_isStreamingStarted) return;
    
    _controller.stopImageStream();
    setState(() {
      _isStreamingStarted = false;
    });
  }

  Future<void> _detectEmotion(CameraImage image) async {
    _isDetecting = true;
    
    try {
      final result = await widget.emotionService.detectEmotion(image);
      
      // Update FPS calculation
      final now = DateTime.now();
      if (_lastDetectionTime != null) {
        _detectionCount++;
        final timeDiff = now.difference(_lastDetectionTime!).inMilliseconds;
        if (timeDiff > 0) {
          final fps = 1000.0 / timeDiff;
          _averageFps = (_averageFps * (_detectionCount - 1) + fps) / _detectionCount;
        }
      }
      _lastDetectionTime = now;
      
      setState(() {
        _lastEmotionResult = result;
      });
    } catch (e) {
      print('Error detecting emotion: $e');
    } finally {
      _isDetecting = false;
    }
  }

  Widget _buildEmotionDisplay() {
    if (_lastEmotionResult == null) {
      return Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 8),
            Text(
              'Detecting emotion...',
              style: TextStyle(color: Colors.white),
            ),
          ],
        ),
      );
    }

    final result = _lastEmotionResult!;
    final primaryEmotion = result.emotion;
    final confidence = result.confidence;

    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Primary emotion display
          Row(
            children: [
              _getEmotionIcon(primaryEmotion),
              SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    primaryEmotion.toUpperCase(),
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    '${(confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            ],
          ),
          
          SizedBox(height: 16),
          
          // All emotions with confidence scores
          Text(
            'All Emotions:',
            style: TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
          ),
          SizedBox(height: 8),
          
          ...result.allScores.entries.map((entry) {
            final emotion = entry.key;
            final score = entry.value;
            final isMain = emotion == primaryEmotion;
            
            return Padding(
              padding: EdgeInsets.symmetric(vertical: 2),
              child: Row(
                children: [
                  Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: isMain ? Colors.blue : Colors.grey,
                      shape: BoxShape.circle,
                    ),
                  ),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      emotion,
                      style: TextStyle(
                        color: isMain ? Colors.white : Colors.white70,
                        fontSize: 12,
                        fontWeight: isMain ? FontWeight.w500 : FontWeight.normal,
                      ),
                    ),
                  ),
                  Text(
                    '${(score * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: isMain ? Colors.white : Colors.white70,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          
          if (_averageFps > 0) ...[
            SizedBox(height: 12),
            Text(
              'FPS: ${_averageFps.toStringAsFixed(1)}',
              style: TextStyle(
                color: Colors.white54,
                fontSize: 10,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _getEmotionIcon(String emotion) {
    IconData iconData;
    Color color;
    
    switch (emotion.toLowerCase()) {
      case 'happy':
        iconData = Icons.sentiment_very_satisfied;
        color = Colors.green;
        break;
      case 'sad':
        iconData = Icons.sentiment_very_dissatisfied;
        color = Colors.blue;
        break;
      case 'angry':
        iconData = Icons.sentiment_dissatisfied;
        color = Colors.red;
        break;
      case 'surprised':
        iconData = Icons.sentiment_neutral;
        color = Colors.orange;
        break;
      case 'fearful':
        iconData = Icons.sentiment_dissatisfied;
        color = Colors.purple;
        break;
      case 'neutral':
      default:
        iconData = Icons.sentiment_neutral;
        color = Colors.grey;
        break;
    }
    
    return Icon(
      iconData,
      color: color,
      size: 32,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // Start image stream after camera is initialized
            if (!_isStreamingStarted) {
              WidgetsBinding.instance.addPostFrameCallback((_) {
                _startImageStream();
              });
            }
            
            return Stack(
              children: [
                // Camera preview
                Positioned.fill(
                  child: CameraPreview(_controller),
                ),
                
                // Top controls
                Positioned(
                  top: MediaQuery.of(context).padding.top + 16,
                  left: 16,
                  right: 16,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                        onPressed: () => Navigator.of(context).pop(),
                        icon: Icon(Icons.arrow_back, color: Colors.white),
                        style: IconButton.styleFrom(
                          backgroundColor: Colors.black54,
                        ),
                      ),
                      Text(
                        'Emotion Detector',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      IconButton(
                        onPressed: () {
                          if (_isStreamingStarted) {
                            _stopImageStream();
                          } else {
                            _startImageStream();
                          }
                        },
                        icon: Icon(
                          _isStreamingStarted ? Icons.pause : Icons.play_arrow,
                          color: Colors.white,
                        ),
                        style: IconButton.styleFrom(
                          backgroundColor: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
                
                // Emotion display
                Positioned(
                  bottom: MediaQuery.of(context).padding.bottom + 32,
                  left: 16,
                  right: 16,
                  child: _buildEmotionDisplay(),
                ),
              ],
            );
          } else {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text(
                    'Initializing camera...',
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
            );
          }
        },
      ),
    );
  }
}