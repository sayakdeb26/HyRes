# Sampling Strategy Comparison

This document compares the **UNIFORM_20** and **MEDIAN_WINDOW_21** sampling strategies across all gesture classes.

## Gesture Class: SWIPE_LEFT (Video 48476)
- **Total Frames in Video**: 33
- **UNIFORM_20 Indices**: `[0, 1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24, 26, 28, 29, 31]` (Indices span full video length)
- **MEDIAN_WINDOW_21 Indices**: `[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]` (Indices centered around M=16)

| UNIFORM_20 Contact Sheet | MEDIAN_WINDOW_21 Contact Sheet |
|---|---|
| ![Uniform 20](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_uniform20_48476_SWIPE_LEFT.png) | ![Median 21](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_median21_48476_SWIPE_LEFT.png) |

## Gesture Class: SWIPE_RIGHT (Video 74390)
- **Total Frames in Video**: 37
- **UNIFORM_20 Indices**: `[0, 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35]` (Indices span full video length)
- **MEDIAN_WINDOW_21 Indices**: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` (Indices centered around M=18)

| UNIFORM_20 Contact Sheet | MEDIAN_WINDOW_21 Contact Sheet |
|---|---|
| ![Uniform 20](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_uniform20_74390_SWIPE_RIGHT.png) | ![Median 21](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_median21_74390_SWIPE_RIGHT.png) |

## Gesture Class: ROLL_FWD (Video 34249)
- **Total Frames in Video**: 32
- **UNIFORM_20 Indices**: `[0, 1, 3, 4, 6, 8, 9, 11, 12, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 30]` (Indices span full video length)
- **MEDIAN_WINDOW_21 Indices**: `[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]` (Indices centered around M=16)

| UNIFORM_20 Contact Sheet | MEDIAN_WINDOW_21 Contact Sheet |
|---|---|
| ![Uniform 20](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_uniform20_34249_ROLL_FWD.png) | ![Median 21](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_median21_34249_ROLL_FWD.png) |

## Gesture Class: STOP_SIGN (Video 96617)
- **Total Frames in Video**: 37
- **UNIFORM_20 Indices**: `[0, 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35]` (Indices span full video length)
- **MEDIAN_WINDOW_21 Indices**: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` (Indices centered around M=18)

| UNIFORM_20 Contact Sheet | MEDIAN_WINDOW_21 Contact Sheet |
|---|---|
| ![Uniform 20](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_uniform20_96617_STOP_SIGN.png) | ![Median 21](file:///home/sayak/HybridTestBed/experiment_results/sampling_comparison/contact_median21_96617_STOP_SIGN.png) |

