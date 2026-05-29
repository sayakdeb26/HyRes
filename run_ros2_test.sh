#!/bin/bash
# Source the ROS 2 workspace
source /opt/ros/humble/setup.bash
source /home/sayak/HybridTestBed/gesture_ws/install/setup.bash

# Run all the nodes in the background, logging their outputs
echo "Starting ROS 2 Hybrid Test Pipeline..."
echo "1. dataset_source_node (publishes /camera/image_raw)"
ros2 run data_source_pkg dataset_source_node > /tmp/dataset.log 2>&1 &
P1=$!

echo "2. keypoint_extractor_node (extracts 144 coords -> /keypoints)"
ros2 run keypoint_extractor_pkg keypoint_extractor_node > /tmp/extractor.log 2>&1 &
P2=$!

echo "3. sequence_buffer_node (buffers 30 frames, builds 296 features -> /sequence)"
ros2 run sequence_buffer_pkg sequence_buffer_node > /tmp/buffer.log 2>&1 &
P3=$!

echo "4. lstm_inference_node (runs PyTorch model -> /gesture/prediction)"
ros2 run lstm_inference_pkg lstm_inference_node > /tmp/lstm.log 2>&1 &
P4=$!

echo "5. central_db_node"
ros2 run central_db_pkg central_db_node > /tmp/central_db.log 2>&1 &
P5=$!

echo "6. bridge_node (connects LSTM to Recorder)"
ros2 run vlm_bridge_pkg bridge_node > /tmp/bridge.log 2>&1 &
P6=$!

echo "7. recorder_node (records VLM clips)"
ros2 run vlm_recorder_pkg recorder_node --ros-args -p input_topic:=/camera/image_raw > /tmp/recorder.log 2>&1 &
P7=$!

echo "8. vlm_node (FastVLM integration)"
~/venvs/rosgpu_isolated/bin/python3 /home/sayak/HybridTestBed/gesture_ws/install/vlm_ros/lib/vlm_ros/vlm_node > /tmp/vlm.log 2>&1 &
P8=$!

echo "9. ui_kiosk_node (UI interface)"
ros2 run ui_kiosk_pkg ui_kiosk_node --ros-args --params-file /home/sayak/HybridTestBed/gesture_ws/src/ui_kiosk_pkg/config/kiosk_params.yaml > /tmp/ui_kiosk.log 2>&1 &
P9=$!

echo "10. evaluation_node (prints live predictions to your terminal!)"
echo "--------------------------------------------------------"
echo "To stop the pipeline, press CTRL+C"
echo "--------------------------------------------------------"
ros2 run evaluation_pkg evaluation_node

# Cleanup when evaluation_node exits (via CTRL+C)
kill $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8 $P9
echo "Pipeline stopped."
