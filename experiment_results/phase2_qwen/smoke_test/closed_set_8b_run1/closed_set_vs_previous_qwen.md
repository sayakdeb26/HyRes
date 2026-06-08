# Comparison: Closed-Set 8B vs. Previous Open-Set Qwen3-VL 4B

- Loaded mode for 8B: 4-bit

| Metric | Open-Set 4B (with UNKNOWN) | Closed-Set 8B (without UNKNOWN) | Change |
|---|---|---|---| 
| **Accuracy** | 0.5700 | 0.3500 | -0.2200 |
| **Macro Precision** | 0.5023 | 0.3870 | -0.1153 |
| **Macro Recall** | 0.5700 | 0.3500 | -0.2200 |
| **Macro F1** | 0.4942 | 0.2893 | -0.2050 |

## Confusion Matrix Differences

### Previous Open-Set 4B Heatmap Values:
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,2,7,11,5
SWIPE_RIGHT,4,7,8,6
ROLL_FWD,1,0,23,1
STOP_SIGN,0,0,0,25
```

### New Closed-Set 8B Heatmap Values:
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,4,8,1,12
SWIPE_RIGHT,9,2,0,14
ROLL_FWD,12,6,4,3
STOP_SIGN,0,0,0,25
```
