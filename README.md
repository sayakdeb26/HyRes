# HybridTestBed: Real-Time Hand Gesture & ROS 2 Pipeline

This repository contains a complete end-to-end framework for real-time hand gesture recognition and its integration into a ROS 2 autonomous robotics pipeline.

## 📂 Repository Structure

### 1. [hand_gesture_lab/](./hand_gesture_lab/)
The research and training environment for the gesture classification model.
- **Goal**: Trains a high-accuracy 5-class gesture classifier.
- **Model**: Bidirectional LSTM with temporal attention and logit smoothing.
- **Features**: 191D vectors (Position, Velocity, Acceleration, Arm Extension).
- **Key Scripts**:
  - `train.py`: Main training orchestration with advanced feature engineering.
  - `fix_dataset.py`: Temporal carry-forward imputation for missing keypoints.
  - `ab_test.py`: Fast diagnostic training and feature comparison.

### 2. [gesture_ws/](./gesture_ws/)
The ROS 2 Humble workspace for live robot control and inference.
- **Packages**:
  - `keypoint_extractor_pkg`: Uses MediaPipe to extract coordinates from `/image`.
  - `sequence_buffer_pkg`: Maintains a rolling 30-frame window of features.
  - `lstm_inference_pkg`: Performs real-time classification and publishes `/prediction`.
  - `data_source_pkg`: Handles real-time image acquisition.

## 🚀 Key Technologies
- **MediaPipe**: Hybrid Pose + Hands extraction for high-ROI stability.
- **PyTorch**: Deep learning backend for Bi-LSTM sequence classification.
- **ROS 2 Humble**: Message-passing and node orchestration.
- **NVIDIA CUDA**: Accelerated inference and training on RTX 5070.

## 🛠️ Getting Started

### Training the Model
1. Navigate to `hand_gesture_lab/`.
2. Ensure data is prepared in `data/processed/`.
3. Run `python3 fix_dataset.py` to repair any corrupted/dropped frames.
4. Run `python3 train.py` to train the 5-class motion model.

### Launching the ROS 2 Pipeline
1. Source your ROS 2 environment.
2. Build the workspace: `colcon build`.
3. Source the overlay: `source install/setup.bash`.
4. Launch the pipeline (refer to package-specific launch files).

---
**Maintained by Sayak Deb**  
*Project Focused on: Gesture-Based Robot Control and Temporal Motion Analysis.*