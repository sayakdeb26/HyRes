# Comparison: Closed-Set vs. Previous Open-Set Qwen3-VL

| Metric | Open-Set (with UNKNOWN) | Closed-Set (without UNKNOWN) | Change |
|---|---|---|---|
| **Accuracy** | 0.5700 | 0.5500 | -0.0200 |
| **Macro Precision** | 0.5023 | 0.5234 | +0.0211 |
| **Macro Recall** | 0.5700 | 0.5500 | -0.0200 |
| **Macro F1** | 0.4942 | 0.5134 | +0.0191 |

## Confusion Matrix Differences

### Previous Open-Set Heatmap Values:
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,2,7,11,5
SWIPE_RIGHT,4,7,8,6
ROLL_FWD,1,0,23,1
STOP_SIGN,0,0,0,25
```

### New Closed-Set Heatmap Values:
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,12,7,1,5
SWIPE_RIGHT,16,2,1,6
ROLL_FWD,5,1,16,3
STOP_SIGN,0,0,0,25
```
