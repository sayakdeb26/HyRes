# Qwen3-VL Temporal Validation Evidence

## Evidence of Single Multimodal Context Processing

1. **Visual Packaging**: The 5 sampled frames are sent as a sequence of `type: image` elements in the content list of a single user message block.
2. **Processor packaging**: `qwen_vl_utils.process_vision_info` processes the 5 distinct PIL images together. The `AutoProcessor` creates a unified `pixel_values` tensor with batch/frame dimension matching the inputs.
3. **Generation execution**: A single forward generation call `model.generate(**inputs)` performs autoregressive decoding on the entire spatial-temporal interleaved sequence at once.

### Verification of Inputs Package Shapes
- Video `42920`: input shape `torch.Size([1440, 1536])` containing all 5 sequential frames packaged together.
- Video `94928`: input shape `torch.Size([1400, 1536])` containing all 5 sequential frames packaged together.
- Video `136106`: input shape `torch.Size([1440, 1536])` containing all 5 sequential frames packaged together.
- Video `6202`: input shape `torch.Size([1400, 1536])` containing all 5 sequential frames packaged together.
- Video `130302`: input shape `torch.Size([1400, 1536])` containing all 5 sequential frames packaged together.
