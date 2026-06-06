# Qwen3-VL-4B-Instruct Balanced Smoke Test Report

- **Status**: Completed
- **Final Verdict**: **READY FOR FULL 2037-SAMPLE QWEN3-VL BENCHMARK**

## Overall Performance Summary

- **Total Samples Processed**: 100
- **Overall Accuracy**: 0.5700 (compared to 0.0300 for FastVLM)
- **Macro F1**: 0.4942 (compared to 0.0517 for FastVLM)
- **UNKNOWN Rate**: 0.0% (compared to 95.0% for FastVLM)

## Confusion Matrix Summary
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,2,7,11,5
SWIPE_RIGHT,4,7,8,6
ROLL_FWD,1,0,23,1
STOP_SIGN,0,0,0,25
```

For more details, see the accompanying report files in this directory.
