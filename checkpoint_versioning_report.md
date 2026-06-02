# Checkpoint Versioning Report

This report evaluates checkpoint management during the sequential retraining stages (L-RT1, L-RT2, L-RT3) and defines the implemented versioning strategy.

## 1. Reproducibility Risks of Using a Single Checkpoint File
In the initial implementation, a single checkpoint name (`cl_state.pth`) was used for EWC and Replay state saving and loading. This introduced several risks:
- **Destructive Overwriting**: The state of task $N-1$ is lost when task $N$ starts, preventing post-training evaluation of older tasks.
- **Pipeline Failure Recovery**: If a retraining task crashes mid-run, the checkpoint might be left corrupted, rendering it impossible to resume or restart the stage without repeating all prior tasks.
- **Traceability**: Harder to run validation or compare model predictions on different task snapshots.

## 2. Implemented Chained Versioning Strategy
To eliminate these risks, we refactored `train_continual.py` to automatically manage task-specific checkpoints based on the `task_name` parameter (`"DS1"`, `"DS2"`, `"DS3"`):

| Retraining Stage | Input Weights (`model_path`) | Input CL State (`cl_state_path`) | Output Weights (`out_model_path`) | Output CL State (`out_cl_state_path`) |
| :--- | :--- | :--- | :--- | :--- |
| **L-RT1 (DS1)** | `best_lstm_model.pth` | *None (Fresh Start)* | `best_lstm_model_rt1.pth` | `cl_state_rt1.pth` |
| **L-RT2 (DS2)** | `best_lstm_model_rt1.pth` | `cl_state_rt1.pth` | `best_lstm_model_rt2.pth` | `cl_state_rt2.pth` |
| **L-RT3 (DS3)** | `best_lstm_model_rt2.pth` | `cl_state_rt2.pth` | `best_lstm_model_rt3.pth` | `cl_state_rt3.pth` |

This chained approach ensures:
1. Complete isolation and preservation of intermediate weights and EWC matrices.
2. Direct repeatability of individual stages.
3. Full compatibility with standard ML model versioning guidelines.
