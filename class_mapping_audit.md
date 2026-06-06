# Class Mapping Audit Report

This report audits the class IDs and label configurations utilized in the continual-learning strategy.

---

## 1. Class ID Mappings

### Current Class-to-Label Mapping
The mapping configured in `hand_gesture_lab/train.py` and the database logger is:
- **`0`**: Swiping Left (source index 16)
- **`1`**: Swiping Right (source index 17)
- **`2`**: Rolling Hand Forward (source index 8)
- **`3`**: Rolling Hand Backward (source index 7)
- **`4`**: Thumb Down (source index 19)
- **`5`**: Stop Sign (source index 14)

### Original Class-to-Label Mapping
The original 27-class Jester dataset indexes are different (e.g. index 16, 17, 8, 7, 19, 14). The local `train.py` maps them into a dense subset of 6 classes (`0` to `5`) using a dictionary mapping.

### Origin of `weak_class_ids=(3, 4, 5)`
These IDs represent `Rolling Hand Backward` (3), `Thumb Down` (4), and `Stop Sign` (5). They appear to have been defined in a previous experiment setup that trained on the full 6-gesture set.

---

## 2. Phase 1 Gesture Set Alignment

The active Phase 1 gesture set evaluated for the kiosk contains **4 gestures**:
1. **Swipe Left** (Class ID `0`)
2. **Swipe Right** (Class ID `1`)
3. **Rolling Forward** (Class ID `2`)
4. **Stop** (Class ID `5`)

### Analysis of `weak_class_ids=(3, 4, 5)`
- **Class IDs `3` and `4`** (`Rolling Hand Backward` and `Thumb Down`) **do not exist** in the active Phase 1 gesture set.
- Keeping these in `weak_class_ids` will bias the `ReplayBuffer` balancing logic:
  - The buffer will attempt to reserve space for class IDs `3` and `4`. Since no new samples for classes `3` and `4` will be collected during Phase 1 experiments, the buffer space allocated for them will be wasted or underutilized.
  - The actual weak classes from the Phase 1 set (such as `2` or `5`, which are more complex than simple Swipes) will receive less priority.

---

## 3. Recommended Adjustments

1. **Modify `weak_class_ids`**: Change `weak_class_ids` in `mixed_strategy.py` to target the active weak classes in Phase 1:
   - Recommended set: `weak_class_ids = (2, 5)` (targeting `Rolling Forward` and `Stop`).
2. **Impact**: Prioritizing classes `2` and `5` in the replay buffer ensures the LSTM retains performance on complex gestures while fine-tuning on new data streams.
