# Qwen3-VL-8B-Instruct 20-frame Smoke Test Report

- **Final Verdict**: **NOT READY**

## Comparative Results

| Metric | Qwen3-VL-4B (10-frame) | Qwen3-VL-8B (20-frame) | Delta |
|---|---|---|---|
| **Accuracy** | 0.5500 | 0.3500 | -0.2000 |
| **Macro F1** | 0.5134 | 0.2817 | -0.2317 |

## Confusion Matrix Summary

```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,0,1,1,3
SWIPE_RIGHT,1,1,0,3
ROLL_FWD,2,0,1,2
STOP_SIGN,0,0,0,5
```

## Feasibility Declarations

1. **Does Qwen3-VL-8B-Instruct fit on the RTX 5070 8GB?**
   Yes, loaded via bitsandbytes 4-bit quantization. Peak memory during inference was 7475.0 MB (VRAM ceiling is 8000 MB).

2. **Is 20-frame multimodal inference feasible?**
   Yes, the model processing and generation finished successfully for all 20 frames per context without out-of-memory errors.

3. **Is latency acceptable?**
   Average total latency is 21226.2 ms per sample. Processor time is 45.3 ms and generation time is 21180.9 ms.

4. **Is performance better than Qwen3-VL-4B?**
   No, 8B accuracy is 35.0% vs 4B accuracy of 55.0%.

5. **Final Benchmark Readiness**: **NOT READY**
