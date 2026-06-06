# Qwen3-VL-4B Resource & VRAM Estimates

This document provides estimated VRAM requirements and inference latency projections for deploying `Qwen/Qwen3-VL-4B-Instruct` compared to `apple/FastVLM-1.5B`.

## 1. VRAM Consumption Projections

The model contains ~4.44 billion parameters. Below is the projected GPU memory footprint at different quantization and precision levels:

| Precision / Quantization | Model Weight Size | Activation & Cache Space | Projected Total VRAM | Recommended GPU Hardware |
|---|---|---|---|---|
| **BF16 / FP16** (Native) | ~8.88 GB | ~2.5 - 3.5 GB | **~11.5 - 12.5 GB** | RTX 3090 / RTX 4080 (16GB+) |
| **8-Bit Quantized** | ~4.44 GB | ~2.0 - 2.5 GB | **~6.5 - 7.0 GB** | RTX 3080 / RTX 4070 (8GB+) |
| **4-Bit Quantized** | ~2.22 GB | ~1.8 - 2.2 GB | **~4.0 - 4.5 GB** | RTX 3060 / RTX 4060 (6GB+) |

*Note: Since the GPU on the user's Omen 16 has RTX series VRAM (likely 8GB or 16GB), running in BF16/FP16 is viable if VRAM is >= 12GB. If limited to 8GB, loading the model in 8-bit or 4-bit precision via `bitsandbytes` is recommended.*

## 2. Expected Inference Latency
- **FastVLM-1.5B Baseline**: Average latency $\approx 5.4$ seconds per video clip (5 frames sampled, corresponding to $\approx 1.08$ seconds per frame).
- **Qwen3-VL-4B Projections**:
  - Qwen3-VL-4B is 3x larger than FastVLM-1.5B.
  - Expected latency per frame: $\approx 2.2 - 2.8$ seconds.
  - Expected total latency per video clip (5 independent classifications + voting): **$\approx 11.0 - 14.0$ seconds**.
  - If using the single multimodal prompt (temporal reasoning mode): $\approx 4.0 - 6.0$ seconds total, as the vision tower only processes the visual context once and processes sequential frames as a single video tensor.
