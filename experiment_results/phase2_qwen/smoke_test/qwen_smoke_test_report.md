# Qwen3-VL-4B-Instruct Balanced Smoke Test Report

- **Status**: Completed
- **Final Verdict**: **NOT READY FOR FULL BENCHMARK**

## Overall Performance Summary

- **Total Samples Processed**: 100
- **Overall Accuracy**: 0.3800 (compared to 0.0300 for FastVLM)
- **Macro F1**: 0.2943 (compared to 0.0517 for FastVLM)
- **UNKNOWN Rate**: 0.0% (compared to 95.0% for FastVLM)

## Confusion Matrix Summary
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,10,9,1,5
SWIPE_RIGHT,14,3,0,8
ROLL_FWD,11,6,0,8
STOP_SIGN,0,0,0,25
```

For more details, see the accompanying report files in this directory.
