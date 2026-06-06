# Qwen3-VL Migration Report: FastVLM Architecture Audit

This report details the audit of the current FastVLM-1.5B evaluation pipeline and maps out the structural changes required to support Qwen3-VL-4B.

## 1. Current FastVLM Architecture
- **Model**: `apple/FastVLM-1.5B`
- **Vision Tower**: SigLIP (loaded via `trust_remote_code=True`).
- **Language Backbone**: Llama-based autoregressive decoder.
- **Input Type**: Single image per inference call.

## 2. Current Inference Path
For each gesture video clip:
1. **Frame Extraction**: Extract 5 frames from the video directory.
2. **Sequential Inference**: Run the model 5 times (once for each frame) using `model.generate()`.
3. **Canonicalization**: Parse each frame output using the substring parser.
4. **Majority Vote Aggregation**: Perform majority voting over the 5 predictions using `collections.Counter`.

## 3. Current Prompt Injection Method
FastVLM uses custom token stitching:
1. Prepare message list: `messages = [{"role": "user", "content": f"<image>\n{PROMPT}"}]`.
2. Apply chat template via tokenizer to render text.
3. Split the rendered text at the `<image>` placeholder.
4. Tokenize the prefix and suffix text independently.
5. Concatenate prefix tokens, the specific visual token (`IMAGE_TOKEN_INDEX = -200`), and suffix tokens.
6. Feed the stitched token tensor and the processed image pixel values into `model.generate()`.

## 4. Current Frame Sampling Method
- **Logic**: Uniform sampling across the video clip.
- **Frame Index Calculation**:
  ```python
  total = len(images)
  idxs = [int((i + 1) * total / (k + 1)) for i in range(k)]  # Where k = 5
  ```
- **File Retrieval**: Retrieves sorted frame files (`*.jpg`) and converts them to RGB using PIL.

## 5. Current Parser & Normalization Rules
The `_canonical_label` method in `vlm_node.py` performs substring matching:
- Strips whitespace and punctuation, converting to lowercase.
- Checks lowercase string against specific patterns:
  - **SWIPE_LEFT**: `["swipe left", "swiping left", "swipe_left", "swiping_left"]`
  - **SWIPE_RIGHT**: `["swipe right", "swiping right", "swipe_right", "swiping_right"]`
  - **ROLL_FWD**: `["rolling hand forward", "roll forward", "roll fwd", "rolling forward", "roll_fwd", "rollumont", "rollplayable", "rollversed", "roll"]`
  - **STOP_SIGN**: `["stop sign", "stop hand", "stop gesture", "open palm", "stop signal", "stop_sign", "stop signing", "stop drawing", "stop signifies", "stop"]`
- If a match is found in the active taxonomy, returns that label. Otherwise, returns `UNKNOWN`.

## 6. Current GPU Loading Strategy
- Loaded in half-precision `float16` on `cuda` if available:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID,
      torch_dtype=torch.float16,
      device_map=None,
      trust_remote_code=True
  ).to(device)
  ```
- No advanced quantization (like 4-bit/8-bit GPTQ or AWQ) or model parallelism is configured.
