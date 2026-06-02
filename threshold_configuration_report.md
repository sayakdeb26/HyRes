# Threshold Configuration Report

This report documents the parameterization of the confidence threshold inside the LSTM inference node.

## 1. Parameters Configuration
- **Initial Setup**: The threshold of `0.60` was hardcoded directly in `lstm_inference_node.py` on line 141.
- **Implemented Parameterization**:
  - Declared `confidence_threshold` as a ROS 2 node parameter in the constructor of `LSTMInferenceNode`:
    ```python
    self.declare_parameter('confidence_threshold', 0.60)
    ```
  - Updated the sequence callback logic to retrieve the parameter dynamically:
    ```python
    threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
    if confidence > threshold:
        # Pass
    else:
        # Escalate
    ```

## 2. Experimental Flexibility
This design preserves the default threshold value of `0.60` as specified while allowing operators to dynamically adjust the escalation threshold at launch or runtime:
- **Example Launch Configuration override**:
  ```bash
  ros2 run lstm_inference_pkg lstm_inference_node --ros-args -p confidence_threshold:=0.80
  ```
- No changes to the underlying model code or pipeline structural logic are required to modify uncertainty filters.
