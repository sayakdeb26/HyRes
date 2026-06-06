# HybridTestBed: Real-Time Hand Gesture & ROS 2 Pipeline

This repository contains a complete end-to-end framework for real-time hand gesture recognition, continual learning integration, and its integration into a ROS 2 autonomous robotics pipeline.

## 📂 Repository Structure

### 1. [hand_gesture_lab/](./hand_gesture_lab/)
The research, training, and continual learning environment for the gesture classification model.
- **Goal**: Trains a high-accuracy gesture classifier with continual learning capability.
- **Continual Learning**: Implements Elastic Weight Consolidation (EWC) and Replay Buffer methodologies to prevent catastrophic forgetting.
- **Key Scripts**:
  - `train.py`: Main training orchestration with advanced feature engineering.
  - `train_continual.py`: Orchestrates continual learning training tasks (e.g. L-T1 -> L-T2 -> L-T3) and handles EWC state chaining.
  - `fix_dataset.py`: Temporal carry-forward imputation for missing keypoints.
  - `ab_test.py`: Fast diagnostic training and feature comparison.

### 2. [gesture_ws/](./gesture_ws/)
The ROS 2 Humble workspace for live robot control and inference.
- **Packages**:
  - `keypoint_extractor_pkg`: Uses MediaPipe to extract coordinates from `/image`.
  - `sequence_buffer_pkg`: Maintains a rolling 30-frame window of features.
  - `lstm_inference_pkg`: Performs real-time classification, publishes `/prediction` and `/lstm/unknown` on low-confidence triggers.
  - `data_source_pkg`: Handles real-time image acquisition and injects `T0` timestamp.
  - `vlm_bridge_pkg`: Coordinates clip recording, calls the VLM service, and acts as the HITI (Human-in-the-Loop) state machine.
  - `amr_interfaces`: Declares custom message types (`InstrumentedKeypoints`, `InstrumentedSequence`) for high-precision latency instrumentation.

### 3. Extra Utilities & Monitoring
- `resource_monitor.py`: Background monitoring script tracking CPU, RAM, and GPU stats (VRAM, Power, Temperature) at 1s intervals.
- `generate_confusion_matrices.py`: Auto-generates confusion matrices (CSV and PNG) for pure LSTM and Hybrid decisions.

---

## ⚡ Continual Learning Strategy & State Persistence
To support sequential training across tasks (e.g. retraining on new gestures) without forgetting old ones, the pipeline implements a **Mixed Strategy**:
- **EWC (Elastic Weight Consolidation)**: Calculates and registers the Fisher Information Matrix and optimal parameters ($\theta^*$) for old tasks to penalize changes to important weights.
- **Replay Buffer**: Employs class-balanced prioritization for active gestures.
- **Persistence**: All state is preserved across training executions inside `hand_gesture_lab/weights/cl_state.pth`.

---

## ⏱️ Custom Timing Instrumentation (T0-T6)
To measure latency with nanosecond precision without Float32 conversion errors, the pipeline propagates timing parameters across ROS 2 nodes using custom `int64` timestamp fields:
- `T0` (Dataset Input): Recorded inside the Camera Image Header stamp.
- `T1` (Keypoint Extraction): Set in `keypoint_extractor_node`.
- `T2` (Sequence Construction): Set in `sequence_buffer_node`.
- `T3` (LSTM Prediction): Set in `lstm_inference_node`.
- `T4` (VLM Escalation Trigger): Recorded in `bridge_node` right before the asynchronous VLM call.
- `T5` (VLM Response): Recorded in `bridge_node` when the VLM client receives the result.
- `T6` (Intent Logged): Recorded when the user action is logged (either post-confirmation or directly from LSTM).

---

## 📊 Logging & Outputs
All experimental data is written to `/home/sayak/HybridTestBed/experiment_results/`:
- **Confidence logs**: Saved to `experiment_results/confidence/confidence_logs.csv`.
- **Escalation logs**: Saved to `experiment_results/escalation/hybrid_escalation_log.csv`.
- **System Resource Metrics**: Saved to `experiment_results/resource_usage/resource_log.csv`.
- **Confusion Matrices**: Generated dynamically into `experiment_results/confusion_matrices/`.

---

## 🔍 Pre-Experiment Validation & Gaps Resolved
Before running the Phase 1 experiments, a validation pass resolved the following:
- **`train_continual.py` Role**: Documented as currently operating in validation mode (with synthetic data) to test strategy functionality. For Prompt 2 production runs, it will be modified to ingest actual `X.npy`/`y.npy` partitions defined in `dataset_manifest.csv`.
- **VLM Callback Crash**: Resolved a critical `NameError` inside `bridge_node.py`'s `vlm_response_callback` by calling `resp = future.result()` before checking attributes.
- **Checkpoint Versioning**: Implemented dynamic versioned paths to avoid overwriting CL state. Checkpoints are automatically saved as `best_lstm_model_rt{1,2,3}.pth` and `cl_state_rt{1,2,3}.pth`.
- **Pre-Escalation Logging**: Configured `lstm_inference_node.py` to forward the actual top-1 LSTM predicted label and confidence instead of `"UNCERTAIN"`. This enables comparison of pre- and post-escalation choices.
- **Threshold Parameterization**: Hardcoded LSTM confidence threshold (`0.60`) converted to a ROS 2 parameter (`confidence_threshold`) allowing custom runtime tuning.

---

## 🚀 Phase 1 Continual Learning Execution Complete
The **Phase 1 LSTM Continual Learning experiment** has been successfully executed, with full training and evaluation on 18,752 samples across four dynamic splits. 
- **Model Evolution**: Trained a sequence of models (`FUSE0` -> `FUSE33` -> `FUSE66` -> `FUSE100`).
- **Minimal Forgetting**: The integration of EWC and Replay Buffer proved successful; `FUSE100` exhibited only a `-1.08%` accuracy drift compared to the `FUSE0` baseline, explicitly mitigating catastrophic forgetting.
- **Results & Deliverables**: All metrics, latency logs, confusion matrices, prediction records, and analysis reports are published inside `/experiment_results/`.

---

## 🛠️ Getting Started

### Training & Continual Retraining
1. Navigate to `hand_gesture_lab/`.
2. Ensure data is prepared in `data/processed/`.
3. Run `python3 train_continual.py` to initiate tasks retraining. EWC state and replay buffers will automatically load and save to versioned checkpoints (`weights/cl_state_rt*.pth`).

### Launching the ROS 2 Pipeline with Resource Monitor
1. Source ROS 2 Humble environment: `source /opt/ros/humble/setup.bash`.
2. Build the workspace: `colcon build`.
3. Source the overlay: `source gesture_ws/install/setup.bash`.
4. Run `./run_ros2_test.sh` to spin up all nodes, which also starts the background resource monitor automatically.
5. To plot final metrics after stopping the pipeline, run `python3 generate_confusion_matrices.py`.

---
**Maintained by Sayak Deb**  
*Project Focused on: Continual Learning, Gesture-Based Robot Control, and Temporal Motion Analysis.*