# InternVideo2-Chat-8B-InternLM2.5 20-frame Smoke Test Report

- **Final Verdict**: **NOT READY**

## Comparative Results

| Metric | Qwen3-VL-4B (10-frame) | InternVideo2 (20-frame) | Delta |
|---|---|---|---|
| **Accuracy** | 0.5500 | 0.2500 | -0.3000 |
| **Macro F1** | 0.5134 | 0.1000 | -0.4134 |

## Confusion Matrix Summary

```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,0,0,0,5
SWIPE_RIGHT,0,0,0,5
ROLL_FWD,0,0,0,5
STOP_SIGN,0,0,0,5
```

## Feasibility Declarations

1. **Does InternVideo2 run successfully on this hardware?**
   Yes, loaded via bitsandbytes 4-bit quantization. Peak memory during inference was 7474.0 MB (VRAM ceiling is 8000 MB).

2. **Is the 60% GPU / 40% CPU offloading stable?**
   Yes, device_map='auto' handles layer offloading automatically.

3. **What is the actual VRAM usage?**
   Peak VRAM usage was 7474.0 MB.

4. **What is the actual RAM usage?**
   Peak RAM usage was 64.6%.

5. **What is the latency per sample?**
   Average total latency is 263220.7 ms per sample (Prep=51.8 ms, Inf=263168.9 ms).

6. **Does InternVideo2 outperform the current Qwen3-VL benchmark?**
   No, InternVideo2 accuracy is 25.0% vs Qwen3-VL-4B accuracy of 55.0%.

7. **Is it feasible to run the full 2037-sample validation benchmark overnight?**
   At 263.2s per sample, 2037 samples will take 148.94 hours. Therefore, it is not feasible to run overnight.
