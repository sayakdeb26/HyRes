# VLM Configuration Report

- **Exact Model Name**: `apple/FastVLM-1.5B`
- **Model Version**: Latest (HuggingFace Hub)
- **Prompt Template**:
```
You will be given frames from a short video.
Each frame contains a single person performing exactly one hand gesture.
Focus only on the hands and ignore the face, background, or other objects.

Choose exactly ONE label from the following list that best describes the hand gesture:

- SWIPE_LEFT
- SWIPE_RIGHT
- ROLL_FWD
- STOP_SIGN
- UNKNOWN

If none of the labels fits, answer UNKNOWN.
Respond with ONLY the label text, nothing else.

```
- **Inference Parameters**:
  - Max New Tokens: 24
  - Temperature: 0.0
  - Top-P: 1.0
  - Sampled Frames Per Video: 5
- **Quantization Settings**: fp16 (half precision)
- **GPU Usage Configuration**: cuda
- **Output Post-Processing Logic**: Exact string match mapped via `VLM_TO_CLASS` dictionary. Any unmapped or UNKNOWN response resolves to label `-1`.
