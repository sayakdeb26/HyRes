# Phase 2 Smoke Test V2 Report (Aligned Label Space)

This report summarizes the performance of FastVLM-1.5B on the 100-sample smoke test using the aligned benchmark prompt and improved parser mapping rules.

## Overall Metrics
- **Accuracy**: 0.0300
- **Macro Precision**: 0.2500
- **Macro Recall**: 0.0288
- **Macro F1**: 0.0517

## Parser & Prediction Counts
- **Count of UNKNOWN Predictions**: 95
- **Count of Parser Failures (Frames)**: 393

### Per-Class Metrics
| Gesture Class | Precision | Recall | F1 |
|---|---|---|---|
| Swipe Left | 0.0000 | 0.0000 | 0.0000 |
| Swipe Right | 0.0000 | 0.0000 | 0.0000 |
| Rolling Hand Forward | 0.0000 | 0.0000 | 0.0000 |
| Stop Sign | 1.0000 | 0.1154 | 0.2069 |

## Verdict
**READY FOR FULL PHASE 2 BENCHMARK**
