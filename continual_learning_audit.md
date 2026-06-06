# Continual Learning Implementation Audit

This document provides a detailed audit of the continual-learning codebase within the `HyRes` workspace.

---

## 1. File-by-File Audit

### `mixed_strategy.py`
- **File Location**: `/home/sayak/HyRes/mixed_strategy.py`
- **Existence**: Already existed in the workspace (placed by the user).
- **Production Readiness**: **Production-Ready** in structure. It defines the high-level math that combines Online EWC with Experience Replay. It assumes the existence of `base_strategy.py`, `ewc_strategy.py`, and `replay_strategy.py`.

### `base_strategy.py`
- **File Location**: `/home/sayak/HyRes/base_strategy.py`
- **Existence**: Created by the Agent.
- **Why Created**: The file was missing in the workspace, preventing `mixed_strategy.py` from importing `StrategyConfig` and invoking `_augment_time_series()`.
- **Assumptions Made**:
  - `StrategyConfig` maps model, optimizer, criterion, device, and training flags.
  - `BaseStrategy` provides general utilities, including a time-series sequence data augmentation method (`_augment_time_series`) applying standard Gaussian noise (std = 0.005) and short sequence cutouts.
- **Production Readiness**: **Prototype**. The augmentation strategy is a baseline heuristic. For production, the time-series noise and cutout parameters must be calibrated against real pose extraction noise distributions.

### `ewc_strategy.py`
- **File Location**: `/home/sayak/HyRes/ewc_strategy.py`
- **Existence**: Created by the Agent.
- **Why Created**: The file was missing. Without it, the subclass `MixedStrategy` could not inherit the EWC consolidation mechanics.
- **Assumptions Made**:
  - Implemented empirical Fisher Information Matrix (FIM) estimation using actual target labels rather than model-sampled distributions (empirical Fisher is standard, computationally cheaper, and numerically stable).
  - Implemented the Online EWC update rule where running Fisher information is merged using `online_alpha` (weighting factor for historical FIM vs new FIM).
  - Starred parameters ($\theta^*$) are saved at the end of each task to compute the quadratic regularization penalty during retraining.
- **Production Readiness**: **Prototype**. While mathematically correct, the FIM calculation samples a subset of the dataset (`fisher_samples = 150`) and holds the parameters in local RAM. For larger models, this state must be serialized to disk.

### `replay_strategy.py`
- **File Location**: `/home/sayak/HyRes/replay_strategy.py`
- **Existence**: Created by the Agent.
- **Why Created**: The file was missing. `mixed_strategy.py` relies on `ReplayBuffer` to sample past task inputs during training.
- **Assumptions Made**:
  - `ReplayBuffer` implements a class-balanced storage with a maximum buffer size.
  - Prioritizes preservation of weak class IDs `(3, 4, 5)` by allowing them up to $1.5\times$ the normal class quota during buffer balancing.
- **Production Readiness**: **Prototype**. Random subsampling is used to manage buffer size when it overflows. In production, a more advanced sampling strategy (like reservoir sampling or prototype selection) could prevent bias.

### `train_continual.py`
- **File Location**: `/home/sayak/HyRes/hand_gesture_lab/train_continual.py`
- **Existence**: Created by the Agent.
- **Why Created**: Created to verify that the `MixedStrategy` class connects correctly with `GestureLSTM` and PyTorch without runtime crashes.
- **Assumptions Made**:
  - Uses randomized mock data of matching dimensions (`100 x 30 x 296`) to test training steps, EWC penalty propagation, and task transitions.
  - Appends path overrides (`sys.path.append`) to handle modular execution from the workspace root.
- **Production Readiness**: **Prototype/Test Script**. It utilizes mock datasets for pipeline verification. For production, it needs to load the actual splits from `dataset_manifest.csv` and handle disk-based state loading.
