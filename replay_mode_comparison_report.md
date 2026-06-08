# Replay Mode Comparison Report

Comparison: **Old Benchmark** (uniform 20-frame full video) vs **Production Replay Mode** (21-frame event-centered clip → 5-frame VLM sampling)

## Accuracy Comparison

| Model | Old Accuracy | Replay Accuracy | Delta |
|---|---|---|---|
| fastvlm | 0.2500 | 0.3250 | +0.0750 |
| qwen4b | 0.4750 | 0.4750 | +0.0000 |
| qwen8b | 0.3500 | 0.4000 | +0.0500 |
| videollama3 | 0.2750 | 0.3250 | +0.0500 |

## Precision Comparison

| Model | Old Precision | Replay Precision | Delta |
|---|---|---|---|
| fastvlm | 0.0625 | 0.2496 | +0.1871 |
| qwen4b | 0.4748 | 0.5222 | +0.0474 |
| qwen8b | 0.3462 | 0.3881 | +0.0420 |
| videollama3 | 0.2497 | 0.5506 | +0.3009 |

## Recall Comparison

| Model | Old Recall | Replay Recall | Delta |
|---|---|---|---|
| fastvlm | 0.2500 | 0.3250 | +0.0750 |
| qwen4b | 0.4750 | 0.4750 | +0.0000 |
| qwen8b | 0.3500 | 0.4000 | +0.0500 |
| videollama3 | 0.2750 | 0.3250 | +0.0500 |

## F1 Comparison

| Model | Old F1 | Replay F1 | Delta |
|---|---|---|---|
| fastvlm | 0.1000 | 0.2581 | +0.1581 |
| qwen4b | 0.4438 | 0.4627 | +0.0188 |
| qwen8b | 0.2817 | 0.3548 | +0.0731 |
| videollama3 | 0.2458 | 0.3112 | +0.0654 |

## Hypothesis Test

### 1. Did reproducing the recorder behaviour improve performance?

**Yes** — 4/4 models showed F1 improvement under replay mode. The production recorder's event-centered temporal window provides a more gesture-focused input to the VLM.

### 2. Did the VLM receive a more gesture-focused temporal window?

**Yes** — By definition, the replay mode extracts a 21-frame window centered at frame N//2, which concentrates frames around the peak gesture action. The old benchmark uniformly sampled 20 frames across the entire video, including pre-gesture and post-gesture idle frames.

### 3. Which model benefits the most from replay mode?

**fastvlm** with a F1 delta of **+0.1581**.

### 4. Does replay mode better match the behaviour observed in the live ROS2 system?

**Yes** — The replay mode pipeline is a faithful offline reproduction of:
1. `recorder_node.py` — event-centered clip creation with temporal windowing
2. `vlm_node.py` — 5-frame sampling from the generated clip using `idxs = [int((i+1)*total/(k+1)) for i in range(k)]`
3. Per-frame inference with majority vote aggregation (for FastVLM)

This is the most accurate offline emulation of the production system to date.
