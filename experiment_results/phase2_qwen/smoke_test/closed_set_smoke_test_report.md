# Closed-Set Qwen3-VL Smoke Test Report

- **Status**: Completed
- **Final Verdict**: **NOT READY**

## Executive Summary

This report evaluates the performance of Qwen3-VL-4B-Instruct on the closed-set validation subset under the new 10-frame sequence temporal prompt config.

- **Overall Accuracy**: 0.5500
- **Macro F1**: 0.5134
- **Justification**: The closed-set classification accuracy is only 55.0%, which is below the target readiness threshold of 60%.

## Confusion Matrix Summary
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,12,7,1,5
SWIPE_RIGHT,16,2,1,6
ROLL_FWD,5,1,16,3
STOP_SIGN,0,0,0,25
```

