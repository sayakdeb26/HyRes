# Video-LLaMA 3 2B Latency Report

Detailed latency analysis from the 20-sample smoke test (all times in milliseconds):

| Metric | Preprocessing (ms) | Inference (ms) | Total Latency (ms) |
|---|---|---|---|
| **Average** | 188.3 | 2928.7 | 3117.0 |
| **Median** | 187.9 | 2968.2 | 3162.3 |
| **Min** | 170.6 | 1579.8 | 1756.6 |
| **Max** | 212.4 | 4176.5 | 4350.2 |

## Sample-by-Sample Breakdown

| Video ID | True Label | Predicted Label | Prep Time (ms) | Inf Time (ms) | Total Latency (ms) |
|---|---|---|---|---|---|
| 48476 | SWIPE_LEFT | ROLL_FWD | 170.6 | 2827.8 | 2998.3 |
| 26696 | SWIPE_LEFT | STOP_SIGN | 182.1 | 2327.3 | 2509.5 |
| 39995 | SWIPE_LEFT | STOP_SIGN | 188.1 | 3206.1 | 3394.2 |
| 37036 | SWIPE_LEFT | SWIPE_RIGHT | 176.8 | 1579.8 | 1756.6 |
| 46299 | SWIPE_LEFT | ROLL_FWD | 187.7 | 2587.2 | 2774.9 |
| 74390 | SWIPE_RIGHT | SWIPE_LEFT | 197.3 | 3469.3 | 3666.6 |
| 77059 | SWIPE_RIGHT | SWIPE_LEFT | 211.5 | 3015.5 | 3227.0 |
| 85662 | SWIPE_RIGHT | STOP_SIGN | 193.9 | 2751.1 | 2945.0 |
| 120211 | SWIPE_RIGHT | SWIPE_RIGHT | 193.7 | 3907.7 | 4101.4 |
| 112080 | SWIPE_RIGHT | ROLL_FWD | 192.4 | 3014.3 | 3206.7 |
| 34249 | ROLL_FWD | ROLL_FWD | 186.3 | 3320.9 | 3507.3 |
| 107172 | ROLL_FWD | STOP_SIGN | 182.1 | 3437.8 | 3619.9 |
| 115605 | ROLL_FWD | ROLL_FWD | 184.6 | 2477.8 | 2662.4 |
| 121065 | ROLL_FWD | ROLL_FWD | 193.5 | 2968.3 | 3161.8 |
| 39880 | ROLL_FWD | STOP_SIGN | 194.7 | 2968.1 | 3162.8 |
| 96617 | STOP_SIGN | SWIPE_LEFT | 176.4 | 2574.5 | 2750.9 |
| 41419 | STOP_SIGN | ROLL_FWD | 176.2 | 1921.6 | 2097.8 |
| 144061 | STOP_SIGN | STOP_SIGN | 191.5 | 3691.1 | 3882.6 |
| 2914 | STOP_SIGN | STOP_SIGN | 173.7 | 4176.5 | 4350.2 |
| 41812 | STOP_SIGN | ROLL_FWD | 212.4 | 2351.7 | 2564.1 |
