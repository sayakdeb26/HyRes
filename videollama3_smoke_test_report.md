# Video-LLaMA 3 2B Smoke Test Report

- **Status**: Completed
- **Target Model**: DAMO-NLP-SG/VideoLLaMA3-2B
- **Quantization Mode**: 8-bit Quantized
- **Hardware Platform**: RTX 5070 Laptop (8 GB VRAM), Ryzen 9 HX CPU, 32 GB RAM

## Overall Performance Summary

- **Total Samples Processed**: 20
- **Overall Accuracy**: 0.3000
- **Macro F1-Score**: 0.2701
- **Mean Latency per Sample**: 3117.0 ms
- **Peak VRAM Consumption**: 2852.0 MB

## Confusion Matrix Summary
```csv
,SWIPE_LEFT,SWIPE_RIGHT,ROLL_FWD,STOP_SIGN
SWIPE_LEFT,0,1,2,2
SWIPE_RIGHT,2,1,1,1
ROLL_FWD,0,0,3,2
STOP_SIGN,1,0,2,2
```
