# train_continual.py Role Validation Report

This document defines the functional role of `hand_gesture_lab/train_continual.py` within the Phase 1 experimentation lifecycle.

## 1. Current State and Functional Role
Currently, `train_continual.py` serves as a **temporary strategy validation harness**. It uses:
- Synthetic random input tensors (`X_task` with shape `[100, 30, 296]`)
- Randomly generated integer labels (`y_task` between `0` and `5`)

This configuration is intended only to verify the mechanical functionality of the mixed continual learning strategy (Elastic Weight Consolidation + Replay Buffer) and its checkpoint loading and saving routines. It is not currently ready to be used as a production training runner.

## 2. Transition to Production Experiment Runner (L-RT1, L-RT2, L-RT3)
To transition this script into the actual continual-learning runner for executing L-RT1, L-RT2, and L-RT3 tasks, the following modifications will be required in Prompt 2:
1. **Load Real Dataset Features**:
   - Replace the synthetic data generation block with loading the actual preprocess outputs `X.npy` and `y.npy` from `/home/sayak/HybridTestBed/hand_gesture_lab/data/processed_full/`.
2. **Implement Task Partitioning**:
   - Query `/home/sayak/HybridTestBed/dataset_manifest.csv` to select indices corresponding to the active task split.
   - Filter `X_task` and `y_task` using those split masks (e.g., `assigned_split == "DS1"` for Task 1, `"DS2"` for Task 2, and `"DS3"` for Task 3).
3. **Configure Hyperparameters**:
   - Adjust retraining epochs, batch size, learning rates, and EWC regularization strength ($\lambda$) based on the designated Phase 1 continual learning experimental setup.
