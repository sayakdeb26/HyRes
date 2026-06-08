# Production Replay Mode — Summary

## Pipeline Specification

| Parameter | Value | Source |
|---|---|---|
| Window Size | 21 frames | recorder_node.py |
| Center Frame | N // 2 | Event-centered |
| Clip FPS | 10 | recorder_node.py |
| VLM Frames Sampled | 5 | vlm_node.py |
| Sampling Formula | `int((i+1)*total/(k+1))` | vlm_node.py |
| Codec | H.264 (libx264) | recorder_node.py |

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 | Avg Latency (ms) |
|---|---|---|---|---|---|
| fastvlm | 0.3250 | 0.2496 | 0.3250 | 0.2581 | 1669.0 |
| qwen4b | 0.4750 | 0.5222 | 0.4750 | 0.4627 | 4856.5 |
| qwen8b | 0.4000 | 0.3881 | 0.4000 | 0.3548 | 5441.4 |
| videollama3 | 0.3250 | 0.5506 | 0.3250 | 0.3112 | 629.3 |

## Per-Class Breakdown

### fastvlm

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SWIPE_LEFT | 0.2500 | 0.2000 | 0.2222 | 10 |
| SWIPE_RIGHT | 0.0000 | 0.0000 | 0.0000 | 10 |
| ROLL_FWD | 0.4286 | 0.3000 | 0.3529 | 10 |
| STOP_SIGN | 0.3200 | 0.8000 | 0.4571 | 10 |

### qwen4b

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SWIPE_LEFT | 0.2632 | 0.5000 | 0.3448 | 10 |
| SWIPE_RIGHT | 0.1667 | 0.1000 | 0.1250 | 10 |
| ROLL_FWD | 0.7500 | 0.3000 | 0.4286 | 10 |
| STOP_SIGN | 0.9091 | 1.0000 | 0.9524 | 10 |

### qwen8b

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SWIPE_LEFT | 0.2143 | 0.3000 | 0.2500 | 10 |
| SWIPE_RIGHT | 0.0000 | 0.0000 | 0.0000 | 10 |
| ROLL_FWD | 0.7500 | 0.3000 | 0.4286 | 10 |
| STOP_SIGN | 0.5882 | 1.0000 | 0.7407 | 10 |

### videollama3

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SWIPE_LEFT | 0.2857 | 0.2000 | 0.2353 | 10 |
| SWIPE_RIGHT | 0.2500 | 0.7000 | 0.3684 | 10 |
| ROLL_FWD | 0.6667 | 0.2000 | 0.3077 | 10 |
| STOP_SIGN | 1.0000 | 0.2000 | 0.3333 | 10 |

