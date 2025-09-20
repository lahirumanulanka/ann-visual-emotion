import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/emotion_detection_service.dart';
import 'screens/camera_screen.dart';

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Request camera permissions
  await Permission.camera.request();
  
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
  }
  
  runApp(EmotionDetectorApp());
}

class EmotionDetectorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Real-time Emotion Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: EmotionDetectorHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class EmotionDetectorHome extends StatefulWidget {
  @override
  _EmotionDetectorHomeState createState() => _EmotionDetectorHomeState();
}

class _EmotionDetectorHomeState extends State<EmotionDetectorHome> {
  late EmotionDetectionService _emotionService;
  bool _isLoading = true;
  String _error = '';

  @override
  void initState() {
    super.initState();
    _initializeEmotionDetection();
  }

  Future<void> _initializeEmotionDetection() async {
    try {
      _emotionService = EmotionDetectionService();
      await _emotionService.initialize();
      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to initialize emotion detection: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Initializing emotion detection...'),
            ],
          ),
        ),
      );
    }

    if (_error.isNotEmpty) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error, size: 64, color: Colors.red),
              SizedBox(height: 16),
              Text(_error, textAlign: TextAlign.center),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _error = '';
                    _isLoading = true;
                  });
                  _initializeEmotionDetection();
                },
                child: Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (cameras.isEmpty) {
      return Scaffold(
        body: Center(
          child: Text('No cameras available'),
        ),
      );
    }

    return CameraScreen(
      camera: cameras.first,
      emotionService: _emotionService,
    );
  }

  @override
  void dispose() {
    _emotionService.dispose();
    super.dispose();
  }
}