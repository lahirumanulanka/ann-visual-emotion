// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:emotion_detector/main.dart';

void main() {
  testWidgets('Emotion Detector app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(EmotionDetectorApp());

    // Verify that the app starts and shows the loading or initialization state
    expect(find.text('Initializing emotion detection...'), findsOneWidget);
    
    // Verify app title
    expect(find.text('Real-time Emotion Detector'), findsNothing); // Should not be visible in loading state
  });
}
