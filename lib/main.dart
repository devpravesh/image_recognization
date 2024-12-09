import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_recognization/homescreen.dart';
// import 'package:firebase_core/firebase_core.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // await Firebase.initializeApp();
  // TFLiteModel model = TFLiteModel();

  // // Load the model
  // await model.loadModel();

  // // Load and preprocess an image
  // File imageFile = File('assets/apple.jpg');
  // Uint8List inputImage = preprocessImage(imageFile);

  // // Run inference
  // await model.runInference(inputImage);
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(home: HomeScreen());
  }
}
