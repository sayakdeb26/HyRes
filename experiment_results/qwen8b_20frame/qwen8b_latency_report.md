# Qwen3-VL-8B-Instruct Latency Report

Detailed latency analysis (in milliseconds):

| Metric | Processor Time (ms) | Generation Time (ms) | Total Inference Time (ms) |
|---|---|---|---|
| **Average** | 45.3 | 21180.9 | 21226.2 |
| **Median** | 39.8 | 21185.9 | 21226.1 |
| **Min** | 34.3 | 20461.7 | 20501.7 |
| **Max** | 103.9 | 22052.8 | 22097.2 |

## Sample-by-Sample Breakdown

| Video ID | True Label | Predicted Label | Processor Time (ms) | Generation Time (ms) | Total Latency (ms) |
|---|---|---|---|---|---|
| 48476 | SWIPE_LEFT | SWIPE_RIGHT | 103.9 | 21845.3 | 21949.2 |
| 26696 | SWIPE_LEFT | STOP_SIGN | 57.3 | 21314.1 | 21371.4 |
| 39995 | SWIPE_LEFT | STOP_SIGN | 64.6 | 21464.3 | 21529.0 |
| 37036 | SWIPE_LEFT | STOP_SIGN | 49.9 | 21525.5 | 21575.4 |
| 46299 | SWIPE_LEFT | ROLL_FWD | 44.3 | 22052.8 | 22097.2 |
| 74390 | SWIPE_RIGHT | SWIPE_LEFT | 38.5 | 21392.0 | 21430.5 |
| 77059 | SWIPE_RIGHT | STOP_SIGN | 38.7 | 21324.5 | 21363.2 |
| 85662 | SWIPE_RIGHT | STOP_SIGN | 44.5 | 21064.2 | 21108.7 |
| 120211 | SWIPE_RIGHT | SWIPE_RIGHT | 37.9 | 21114.0 | 21151.9 |
| 112080 | SWIPE_RIGHT | STOP_SIGN | 44.8 | 21128.1 | 21172.9 |
| 34249 | ROLL_FWD | ROLL_FWD | 37.8 | 21460.4 | 21498.2 |
| 107172 | ROLL_FWD | SWIPE_LEFT | 45.7 | 21681.7 | 21727.4 |
| 115605 | ROLL_FWD | STOP_SIGN | 37.1 | 20676.1 | 20713.2 |
| 121065 | ROLL_FWD | SWIPE_LEFT | 34.3 | 20609.2 | 20643.5 |
| 39880 | ROLL_FWD | STOP_SIGN | 34.9 | 20513.7 | 20548.6 |
| 96617 | STOP_SIGN | STOP_SIGN | 40.6 | 21191.1 | 21231.7 |
| 41419 | STOP_SIGN | STOP_SIGN | 36.2 | 21096.4 | 21132.6 |
| 144061 | STOP_SIGN | STOP_SIGN | 35.4 | 20521.4 | 20556.8 |
| 2914 | STOP_SIGN | STOP_SIGN | 39.7 | 21180.8 | 21220.4 |
| 41812 | STOP_SIGN | STOP_SIGN | 40.0 | 20461.7 | 20501.7 |
