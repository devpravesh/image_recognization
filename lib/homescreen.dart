import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  XFile? _image;

  Future<void> pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    setState(() {
      _image = image;
    });
  }

  List<String> predictions = [];

  void getTopPredictions(List<double> probabilities, Map<int, String> labels) {
    final indexedProbs = probabilities.asMap();
    final topPredictions = indexedProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value))
      ..sublist(0, 5); // Select top 5 predictions

    setState(() {
      predictions = topPredictions.map((entry) {
        int index = entry.key;
        double confidence = entry.value; // Convert to percentage
        String label = labels[index] ?? "Unknown";

        return "Class: $label, Confidence: ${confidence.toStringAsFixed(2)}";
      }).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Image Recognition App")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image != null
                ? Image.file(File(_image!.path))
                : const Text("No image selected"),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: pickImage,
              child: const Text("Pick Image"),
            ),
            ElevatedButton(
              onPressed: () async {
                try {
                  TFLiteModel model = TFLiteModel();
                  await model.loadModel();

                  File imageFile = File(_image!.path);
                  Float32List inputImage = preprocessImage(imageFile);

                  List<double> probabilities =
                      await model.runInference(inputImage);

                  Map<int, String> labels = await model
                      .loadLabels('assets/imagenet_class_index.json');

                  getTopPredictions(probabilities, labels);
                } catch (e) {
                  print("Error: $e");
                }
              },
              child: const Text("Submit"),
            ),
            const SizedBox(height: 20),
            predictions.isNotEmpty
                ? Expanded(
                    child: ListView.builder(
                      itemCount: predictions.length,
                      itemBuilder: (context, index) {
                        return ListTile(
                          title: Text(predictions[index]),
                        );
                      },
                    ),
                  )
                : const Text("No predictions yet"),
          ],
        ),
      ),
    );
  }
}

class TFLiteModel {
  late Interpreter _interpreter;

  // Load the TFLite model
  Future<void> loadModel() async {
    try {
      // Load the model
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      print('Model loaded successfully!');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  // Run inference
  Future<List<double>> runInference(Float32List inputImage) async {
    try {
      // Prepare input and output tensors
      var input = inputImage.reshape([1, 224, 224, 3]);
      var output = List.generate(1,
          (_) => List<double>.filled(1001, 0.0)); // Match model's output size

      // Run inference
      _interpreter.run(input, output);

      // Return the first row of predictions as a List<double>
      return output[0];
    } catch (e) {
      print("Inference Error: $e");
      rethrow;
    }
  }

  // Load labels from JSON file
  Future<Map<int, String>> loadLabels(String filePath) async {
    try {
      final jsonString = await rootBundle.loadString(filePath);
      final jsonData = json.decode(jsonString) as Map<String, dynamic>;

      return jsonData.map<int, String>((key, value) {
        // Extract the second element (class_name) from the array
        final List<dynamic> labelData = value as List<dynamic>;
        return MapEntry(int.parse(key), labelData[1] as String);
      });
    } catch (e) {
      print("Error loading labels: $e");
      rethrow;
    }
  }
}

// Preprocess the image
Float32List preprocessImage(File imageFile) {
  // Decode the image
  final originalImage = img.decodeImage(imageFile.readAsBytesSync());
  if (originalImage == null) {
    throw Exception("Failed to decode image.");
  }

  // Resize the image to 224x224 (model's expected size)
  final resizedImage = img.copyResize(originalImage, width: 224, height: 224);

  // Allocate buffer for input tensor [1, 224, 224, 3]
  final Float32List buffer = Float32List(224 * 224 * 3);

  // Extract RGB values and normalize them to [0, 1]
  int bufferIndex = 0;
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final pixel = resizedImage.getPixel(x, y);

      // Extract R, G, B channels and normalize
      buffer[bufferIndex++] = img.getRed(pixel) / 255.0; // R channel
      buffer[bufferIndex++] = img.getGreen(pixel) / 255.0; // G channel
      buffer[bufferIndex++] = img.getBlue(pixel) / 255.0; // B channel
    }
  }

  return buffer;
}

// Debugging function for input tensor
void debugInput(Float32List inputImage) {
  print("Input Tensor Length: ${inputImage.length}");
  print("Expected Tensor Length: ${1 * 224 * 224 * 3}");
}
