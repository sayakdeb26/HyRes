#!/usr/bin/env python3
import os

files_to_edit = [
    "run_phase1_experiment.py",
    "README.md",
    "run_ros2_test.sh",
    "hand_gesture_lab/train.py",
    "dataset_manager.py",
    "run_verification.py",
    "hand_gesture_lab/train_continual.py",
    "hand_gesture_lab/preprocess_phase1.py",
    "hand_gesture_lab/fix_dataset.py",
    "hand_gesture_lab/ab_test.py",
    "phase1_baseline_snapshot.md",
    "prepare_phase1_dataset.py",
    "continual_learning_audit.md",
    "replay_buffer_validation.md",
    "task_boundary_validation.md",
    "train_continual_role_report.md",
    "gesture_ws/src/vlm_bridge_pkg/vlm_bridge_pkg/bridge_node.py",
    "gesture_ws/src/lstm_inference_pkg/lstm_inference_pkg/lstm_inference_node.py",
    "phase1_readiness_report.md",
    "generate_confusion_matrices.py",
    "pipeline_structure_and_flow.md",
    "resource_monitor.py"
]

base_dir = "/home/sayak/HybridTestBed"

for rel_path in files_to_edit:
    file_path = os.path.join(base_dir, rel_path)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()
            
        new_content = content.replace("HybridTestBed", "HyRes")
        # Also replace the README title if it's there
        if rel_path == "README.md":
            new_content = new_content.replace("# HybridTestBed: Real-Time Hand Gesture & ROS 2 Pipeline", "# HyRes - Hybrid Gesture Recognition System")
            
        if new_content != content:
            with open(file_path, "w") as f:
                f.write(new_content)
            print(f"Updated {rel_path}")

print("Replacement complete.")
