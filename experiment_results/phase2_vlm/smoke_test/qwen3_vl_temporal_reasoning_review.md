# Qwen3-VL Temporal Reasoning Review: Multi-Frame vs. Frame-wise Voting

This document compares the current frame-wise voting method against a single multimodal prompt approach, evaluating their alignment with Qwen3-VL-4B capabilities.

## 1. Current Method: Frame-wise Voting
- **Workflow**: Sample 5 frames. Run 5 independent VLM generation calls. Canonicalize outputs. Take the majority vote.
- **Pros**:
  - Resilient to individual frame noise/refusals (if one frame returns a safety refusal, the other 4 can still form a majority).
  - Simple, state-free logic.
- **Cons**:
  - **Temporal Blindness**: The model cannot see motion direction. It cannot distinguish between a hand moving left (`SWIPE_LEFT`) or right (`SWIPE_RIGHT`) from static postures, as the sequence order of frames is lost.
  - **High Latency**: Requires 5 complete visual encoding and text generation runs per video sample.

## 2. Proposed Method: Single Multimodal Prompt (Temporal Reasoning Mode)
- **Workflow**: Format the 5 sampled frames as a video sequence or interleaved images in a single prompt:
  ```python
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "image": frame1},
              {"type": "image", "image": frame2},
              {"type": "image", "image": frame3},
              {"type": "image", "image": frame4},
              {"type": "image", "image": frame5},
              {"type": "text", "text": "Analyze these sequential frames of a hand gesture in chronological order. Choose one label: SWIPE_LEFT, SWIPE_RIGHT, ROLL_FWD, STOP_SIGN."}
          ]
      }
  ]
  ```
- **Pros**:
  - **Motion Awareness**: Qwen3-VL's Interleaved Multimodal Rotary Position Embedding (Interleaved-MRoPE) models spatial-temporal relations across frames. It can easily detect the sweep direction and rolling motion.
  - **Lower Latency**: The vision tower encodes all frames in a single forward pass, and the text decoder only runs once, saving significant auto-regressive processing time.
- **Cons**:
  - Increased context length per prompt (higher initial VRAM peak during attention computation).

## 3. Recommendation
**The Single Multimodal Prompt (Temporal Reasoning Mode) is strongly recommended for Qwen3-VL.**
Because the validation set (Jester dataset) contains motion-based gestures (`SWIPE_LEFT`, `SWIPE_RIGHT`, `ROLL_FWD`), static frame-wise classification is fundamentally limited. Transitioning to Qwen3-VL's native multi-frame processing will significantly increase accuracy and cut down latency.
