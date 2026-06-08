# Closed-Set Qwen3-VL 8B Smoke Test Report

- **Status**: Completed
- **Quantization Mode**: 4-bit
- **Final Verdict**: **NOT READY**

## Executive Summary

This report evaluates the performance of Qwen3-VL-8B-Instruct on the closed-set validation subset under the 10-frame sequence temporal prompt config.

- **Overall Accuracy**: 0.3500
- **Macro F1**: 0.2893
- **Justification**: The closed-set classification accuracy is only 35.0%, which is below the target readiness threshold of 60%.

## Confusion Matrix Summary
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,4,8,1,12
SWIPE_RIGHT,9,2,0,14
ROLL_FWD,12,6,4,3
STOP_SIGN,0,0,0,25
```

