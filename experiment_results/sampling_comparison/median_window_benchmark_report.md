# Median Window Benchmark Report

This report evaluates whether focusing on the temporal center of the gesture (`MEDIAN_WINDOW_21`) improves VLM recognition performance compared to uniform sampling (`UNIFORM_20`).

## Performance Comparison Table

| Model | Sampling Mode | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| QWEN4B | UNIFORM_20 | 0.4750 | 0.4748 | 0.4750 | 0.4438 |
| QWEN4B | MEDIAN_WINDOW_21 | 0.5000 | 0.4915 | 0.5000 | 0.4688 |
| VIDEOLLAMA3 | UNIFORM_20 | 0.2750 | 0.2497 | 0.2750 | 0.2458 |
| VIDEOLLAMA3 | MEDIAN_WINDOW_21 | 0.4000 | 0.5622 | 0.4000 | 0.3697 |

## Key Findings & Performance Analysis

### 1. Qwen3-VL-4B Analysis
- **Uniform 20 F1**: 0.4438 (Accuracy: 0.4750)
- **Median 21 F1**: 0.4688 (Accuracy: 0.5000)
- **Delta F1**: +0.0250
- **Verdict**: Median-centered windowing successfully improves Qwen3-VL-4B gesture recognition. Concentrating frames on the high-action center of the sequence reduces the dilution caused by static/pre-gesture frames at video boundaries.

### 2. Video-LLaMA3-2B Analysis
- **Uniform 20 F1**: 0.2458 (Accuracy: 0.2750)
- **Median 21 F1**: 0.3697 (Accuracy: 0.4000)
- **Delta F1**: +0.1240
- **Verdict**: Median-centered windowing successfully improves Video-LLaMA3-2B gesture recognition. By localizing the action region, the compiled video feed contains higher gesture density.

## Scientific Conclusion
Focusing on the temporal center of the gesture generally improves performance by cropping out non-informative pre-pose and post-pose sequences. We recommend adopting `MEDIAN_WINDOW_21` as the primary frame sampling strategy for both models.
