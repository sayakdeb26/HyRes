# Real-Time Hand Gesture Recognition

A modular, robust prototype for real-time hand gesture recognition using MediaPipe and PyTorch LSTMs. Designed to be easily portable into ROS2 pipelines.

## Features
- **Two-Stage Detection**: Uses MediaPipe Pose to locate shoulders and wrists, creating a dynamic Region of Interest (ROI) for MediaPipe Hands. This significantly improves accuracy and reduces false positives (like face misclassification).
- **Temporal Smoothing**: Uses Exponential Moving Average (EMA) and jump thresholding to stabilize hand keypoints.
- **Normalized Features**: Keypoints are translated to wrist-origin and scaled based on hand size, making gestures invariant to distance from the camera.
- **LSTM Sequence Classifier**: Uses a lightweight 2-layer LSTM for classifying 30-frame keypoint sequences.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Live Demo

To run the live webcam inference (uses random weights if no trained model is provided):

```bash
python src/main.py --camera 0
```
*Press `q` to quit the live feed.*

If you have trained weights:
```bash
python src/main.py --camera 0 --weights models/gesture_lstm.pth
```

## Training on the Jester Dataset

### 1. Data Preprocessing
Ensure you have the Jester dataset extracted (so each video is a folder of `.jpg` frames) and the CSV labels (`Train.csv`, `Validation.csv`).

Run the preprocessor to extract 30-frame `.npy` sequences. You can use the `--debug` flag to save visual representations of the pipeline.
```bash
python dataset/preprocess_jester.py \
    --input_dir /path/to/archive/Train \
    --csv_path /path/to/archive/Train.csv \
    --output_dir data/processed \
    --seq_len 30 \
    --stride 5 \
    --workers 4
```

### 2. Training
Run the training script on the processed `.npy` files:
```bash
python training/train.py --data_dir data/processed --epochs 50 --batch_size 32
```
This will:
- Train the LSTM model.
- Save the best weights to `models/gesture_lstm.pth`.
- Export an ONNX version to `models/gesture_lstm.onnx`.

## Architecture Overview

- `src/pose_estimator.py`: Extracts shoulder and wrist keypoints.
- `src/roi_extractor.py`: Calculates dynamic ROIs for hands.
- `src/hand_detector.py`: Extracts 21 hand keypoints from the ROI.
- `src/validator.py`: Filters anomalies and applies EMA smoothing.
- `src/feature_builder.py`: Normalizes and flattens keypoints into a 63D vector.
- `src/lstm_model.py`: PyTorch model definition.
- `src/inference.py`: Rolling buffer and model execution.
- `src/pipeline.py`: Main orchestration loop.
