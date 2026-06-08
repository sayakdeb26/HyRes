# InternVideo2 Latency Report

Detailed latency analysis (in milliseconds):

| Metric | Preprocessing Time (ms) | Inference Time (ms) | Total Inference Time (ms) |
|---|---|---|---|
| **Average** | 51.8 | 263168.9 | 263220.7 |
| **Median** | 41.6 | 262099.9 | 262156.8 |
| **Min** | 36.3 | 261648.3 | 261686.9 |
| **Max** | 81.8 | 269904.1 | 269945.9 |

## Sample-by-Sample Breakdown

| Video ID | True Label | Predicted Label | Preprocessing Time (ms) | Inference Time (ms) | Total Latency (ms) |
|---|---|---|---|---|---|
| 48476 | SWIPE_LEFT | STOP_SIGN | 38.5 | 267423.0 | 267461.5 |
| 26696 | SWIPE_LEFT | STOP_SIGN | 41.7 | 269904.1 | 269945.9 |
| 39995 | SWIPE_LEFT | STOP_SIGN | 39.6 | 265583.9 | 265623.5 |
| 37036 | SWIPE_LEFT | STOP_SIGN | 80.5 | 264396.4 | 264477.0 |
| 46299 | SWIPE_LEFT | STOP_SIGN | 81.8 | 265327.1 | 265408.9 |
| 74390 | SWIPE_RIGHT | STOP_SIGN | 42.2 | 262962.7 | 263004.9 |
| 77059 | SWIPE_RIGHT | STOP_SIGN | 40.8 | 262295.9 | 262336.7 |
| 85662 | SWIPE_RIGHT | STOP_SIGN | 81.6 | 261784.2 | 261865.8 |
| 120211 | SWIPE_RIGHT | STOP_SIGN | 81.8 | 261899.0 | 261980.8 |
| 112080 | SWIPE_RIGHT | STOP_SIGN | 73.0 | 262086.1 | 262159.1 |
| 34249 | ROLL_FWD | STOP_SIGN | 39.8 | 261843.0 | 261882.8 |
| 107172 | ROLL_FWD | STOP_SIGN | 36.3 | 261841.3 | 261877.6 |
| 115605 | ROLL_FWD | STOP_SIGN | 38.5 | 261648.3 | 261686.9 |
| 121065 | ROLL_FWD | STOP_SIGN | 43.9 | 262000.6 | 262044.5 |
| 39880 | ROLL_FWD | STOP_SIGN | 38.0 | 261900.7 | 261938.7 |
| 96617 | STOP_SIGN | STOP_SIGN | 75.0 | 261914.4 | 261989.5 |
| 41419 | STOP_SIGN | STOP_SIGN | 41.4 | 262113.7 | 262155.1 |
| 144061 | STOP_SIGN | STOP_SIGN | 38.8 | 262424.4 | 262463.2 |
| 2914 | STOP_SIGN | STOP_SIGN | 39.9 | 261914.0 | 261953.9 |
| 41812 | STOP_SIGN | STOP_SIGN | 42.7 | 262115.9 | 262158.6 |
