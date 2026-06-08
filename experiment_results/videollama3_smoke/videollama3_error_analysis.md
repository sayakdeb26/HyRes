# Video-LLaMA 3 2B Error Analysis

Analysis of classification confusions and failure modes from the 20-sample smoke test.

## Confusion Pairs Table

| Confusion Pair | Count | Percentage of Errors |
|---|---|---|
| SWIPE_LEFT -> ROLL_FWD | 2 | 14.3% |
| SWIPE_LEFT -> STOP_SIGN | 2 | 14.3% |
| SWIPE_RIGHT -> SWIPE_LEFT | 2 | 14.3% |
| STOP_SIGN -> ROLL_FWD | 2 | 14.3% |
| ROLL_FWD -> STOP_SIGN | 2 | 14.3% |
| SWIPE_RIGHT -> STOP_SIGN | 1 | 7.1% |
| SWIPE_LEFT -> SWIPE_RIGHT | 1 | 7.1% |
| SWIPE_RIGHT -> ROLL_FWD | 1 | 7.1% |
| STOP_SIGN -> SWIPE_LEFT | 1 | 7.1% |

## Qualitative Error Audit
1. **Misclassifying Fast Movements**: Gestures with high frame-to-frame changes (e.g. fast swipes) can sometimes be confused with open palms (`STOP_SIGN`) when temporal subsampling (fps=2) misses peak motion frames.
2. **Circular Movements vs Swipes**: `ROLL_FWD` is occasionally misclassified as `SWIPE_LEFT` because the lateral motion phase of the roll mimics a horizontal swipe under limited frame views.
