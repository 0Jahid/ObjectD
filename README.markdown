```markdown
# ObjectD - Real-Time Object Detection Android App

This sample implements an Activity that performs real-time object detection on live camera frames. It performs the following operations:
- Initializes camera preview and image analysis frame streams using CameraX.
- Loads a quantized MobileNet model using TensorFlow Lite.
- Converts each incoming frame to the RGB colorspace and resizes it to 224x224 pixels.
- Performs inference on the transformed frames and reports the object predicted on the screen.


## Features
- **Real-Time Object Detection**: Detects objects in live camera frames with bounding boxes and confidence scores.
- **CameraX Integration**: Uses CameraX for robust camera preview and image analysis streams.
- **TensorFlow Lite**: Employs a quantized MobileNet model for lightweight inference.
- **Frame Processing**: Converts YUV frames to RGB and resizes to 224x224 pixels for model input.
- **High Performance**: Maintains 30 FPS on a Pixel 3 XL.
- **User Interface**: Displays predictions (e.g., "BED", "CELL PHONE", "BANANA", "LAPTOP") with confidence scores.

## Screenshots
Below are screenshots showcasing the app's object detection capabilities:

1. **Bed Detection**  
   ![Bed Detection](ss/bed.jpg)  
   Detects a bed with a confidence score of 0.75.

2. **Cell Phone Detection**  
   ![Cell Phone Detection](ss/cellphone.jpg)  
   Identifies a cell phone with a confidence score of 0.69.

3. **Banana Detection**  
   ![Banana Detection](ss/banana.jpg)  
   Recognizes bananas with a confidence score of 0.63.

4. **Laptop Detection**  
   ![Laptop Detection](ss/laptop.jpg)  
   Detects a laptop with a confidence score of 0.74.

*Note*: Ensure the screenshot files (`bed_detection.jpg`, `cell_phone_detection.jpg`, `banana_detection.jpg`, `laptop_detection.jpg`) are placed in the `ss` directory of the repository.

## Installation

### Prerequisites
- **Android Device/Emulator**: API level 26  or higher.
- **Git**: To clone the repository.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/0Jahid/ObjectD.git
   cd ObjectD
   ```

2. **Open in Android Studio**:
   - Launch Android Studio and select **Open an existing project**.
   - Navigate to the `ObjectD` directory and click **OK**.

3. **Add Model and Label Files**:
   - Place the quantized MobileNet model (`mobilenet_v1_1.0_quant.tflite`) and label file (`labels.txt`) in `app/src/main/assets`.
   - Example structure:
     ```
     app/src/main/assets/
     ├── mobilenet_v1_1.0_quant.tflite
     ├── labels.txt
     ```

4. **Sync Project**:
   - Click **Sync Project with Gradle Files** in Android Studio to resolve dependencies.

5. **Build and Run**:
   - Connect an Android device or start an emulator.
   - Click **Run > Run 'app'** to build and install the app.

### Dependencies
The project uses the following key dependencies (defined in `app/build.gradle`):
```gradle
dependencies {
    implementation ("androidx.camera:camera-core:1.3.0")
    implementation ("androidx.camera:camera-camera2:1.3.0")
    implementation ("androidx.camera:camera-lifecycle:1.3.0")
    implementation ("androidx.camera:camera-view:1.3.0")
    implementation ("org.tensorflow:tensorflow-lite:2.10.0")
    implementation ("org.tensorflow:tensorflow-lite-support:0.4.2")
}
```

## Usage
1. **Launch the App**:
   - Open the app on your Android device.
   - Grant camera permissions when prompted.

2. **Real-Time Detection**:
   - Point the camera at objects (e.g., bed, cell phone, banana, laptop).
   - The app displays bounding boxes with labels and confidence scores (e.g., "0.75 BED").



## Project Structure
- **app/src/main/java/com/jahid/objectd/**:
  - `CameraActivity.kt`: Main activity for camera preview and object detection.
  - `YuvToRgbConverter.kt`: Converts YUV camera frames to RGB bitmaps.
  - `ObjectDetectionHelper.kt`: Processes TensorFlow Lite model outputs for predictions.
- **app/src/main/res/layout/**:
  - `activity_camera.xml`: UI layout for the camera preview and detection results.
- **app/src/main/assets/**:
  - TensorFlow Lite model and label files.

