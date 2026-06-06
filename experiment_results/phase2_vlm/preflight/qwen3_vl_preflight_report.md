# Qwen3-VL-4B-Instruct Preflight Validation Report

- **Model Loaded**: `Qwen/Qwen3-VL-4B-Instruct`
- **Quantization Mode**: 8-bit
- **Model Loading Time**: 11.36 seconds

## Sample Inferences

| Video ID | True Label | True Name | Raw Output | Parsed Output | Final Label | Match? |
|---|---|---|---|---|---|---|
| 42920 | 0 | Swipe Left | `SWIPE_RIGHT` | `SWIPE_RIGHT` | 1 | NO |
| 94928 | 1 | Swipe Right | `UNKNOWN` | `UNKNOWN` | -1 | NO |
| 136106 | 2 | Rolling Hand Forward | `SWIPE_LEFT` | `SWIPE_LEFT` | 0 | NO |
| 6202 | 3 | Stop Sign | `STOP_SIGN` | `STOP_SIGN` | 3 | YES |
| 130302 | 3 | Stop Sign | `STOP_SIGN` | `STOP_SIGN` | 3 | YES |
