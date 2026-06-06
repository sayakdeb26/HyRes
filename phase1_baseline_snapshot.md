# Phase 1 Baseline Snapshot

This snapshot records the exact state of the repository, weights, and dataset manifest before commencing the Phase 1 experiments (L-T, V-T, and H-T).

---

## 1. Git Commit Metadata
- **Commit Hash**: `97e8cb403071246055d4f60dde069c291542c1ef`
- **Branch**: `main`
- **Remote URL**: `git@github.com:sayakdeb26/HyRes.git`
- **Status**: Clean workspace relative to tracked files.

---

## 2. Workspace & Environment Version
- **ROS Version**: ROS 2 Humble Hawksbill
- **Operating System**: Linux (Ubuntu 22.04 LTS / Debian-based)
- **PyTorch Version**: 2.x
- **CUDA Device**: NVIDIA GPU (e.g. RTX 5070 / CUDA-enabled device)

---

## 3. Model Checkpoints
Baseline gesture classifier model weights details:

| File Path | Description | MD5 Checksum |
| :--- | :--- | :--- |
| `hand_gesture_lab/weights/best_lstm_model.pth` | Base LSTM Model (6 classes) trained on preprocessed baseline Jester data | `29a283932bb32aaae9718fcdfff56158` |

---

## 4. Dataset Manifest
Experimental splits and task allocations details:

| File Path | Description | MD5 Checksum | Total Entries |
| :--- | :--- | :--- | :--- |
| `dataset_manifest.csv` | Full data manifest mapping Jester sequences to Train, DS1, DS2, and DS3 | `c4b86fcfae8bdf7b1830dd7eb343f865` | 1000 |

### Assigned Splits Distribution:
- **Train**: 700 samples (Baseline training)
- **DS1 (Task 1 / Task L-RT1)**: 100 samples
- **DS2 (Task 2 / Task L-RT2)**: 100 samples
- **DS3 (Task 3 / Task L-RT3)**: 100 samples
