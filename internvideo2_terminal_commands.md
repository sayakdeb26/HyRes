# InternVideo2 Terminal Commands & Execution Guide

This document lists the terminal commands required to set up the environment, run the InternVideo2 smoke test, and check the generated prediction and resource utilization reports.

## 1. Environment Setup

Before running the script, activate the isolated python virtual environment containing ROS 2 Humble and the GPU-accelerated deep learning dependencies:

```bash
# Activate virtual environment
source ~/venvs/rosgpu_isolated/bin/activate

# Verify PyTorch detects the RTX 5070 GPU
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 2. Execute the Smoke Test

Run the standalone feasibility evaluation script directly from the project root directory:

```bash
# Navigate to the workspace root
cd /home/sayak/HybridTestBed

# Run the InternVideo2 smoke test
python3 run_internvideo2_smoke_test.py
```

## 3. Review Experiment Results

All outputs, predictions, confusion matrices, and resource usage statistics are saved to the results folder:

```bash
# List all generated files
ls -la /home/sayak/HybridTestBed/experiment_results/internvideo2_smoke/
```

Key output files:
- **`internvideo2_predictions.csv`**: Contains sample-by-sample classifications and execution latency.
- **`internvideo2_confusion_matrix.png`**: Heatmap visualization of predictions vs true labels.
- **`internvideo2_load_report.md`**: Tracks initialization time and VRAM usage during model loading.
- **`internvideo2_resource_report.md`**: Monitors CPU, RAM, GPU temperature, and power consumption.
- **`internvideo2_latency_report.md`**: Preprocessing and generative execution breakdown.
- **`internvideo2_smoke_test_report.md`**: Final feasibility verdict and comparison with the baseline Qwen model.
