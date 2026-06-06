# Qwen3-VL Integration Code Change Plan

This document outlines the minimal non-destructive changes required to integrate `Qwen/Qwen3-VL-4B-Instruct` into the existing benchmark and production files while keeping `apple/FastVLM-1.5B` fully operational and switchable.

## 1. Model Selection Control
We introduce the `VLM_MODEL` environment variable (default: `fastvlm`):
- `VLM_MODEL=fastvlm`: Loads FastVLM-1.5B (standard path).
- `VLM_MODEL=qwen3vl`: Loads Qwen3-VL-4B-Instruct (new path).

---

## 2. Modifications to `vlm_node.py`

### Model Initialization & Loading
Add a conditional block to load the appropriate model class:
```python
VLM_MODEL_TYPE = os.getenv("VLM_MODEL", "fastvlm").lower()

if VLM_MODEL_TYPE == "qwen3vl":
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    # Load model and processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
else:
    # Existing FastVLM loading code...
```

### Inference Logic (`_process_clip` or frame-wise loops)
Provide conditional branching inside the inference methods:
```python
if VLM_MODEL_TYPE == "qwen3vl":
    # Format messages for Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": PROMPT}]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)
    
    out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # Trim inputs and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids)
    ]
    text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
else:
    # Existing FastVLM custom token stitching and generate call...
```

---

## 3. Modifications to `run_phase2_vlm_experiment.py` & Smoke Test Scripts
Incorporate the same conditional loading and execution logic based on `os.getenv("VLM_MODEL", "fastvlm")` in:
- `run_phase2_vlm_experiment.py`
- `run_phase2_vlm_smoke_test_v2.py`

This ensures that the metric logging, CSV exports, GPU utilization logs, and comparisons remain completely identical across both model evaluations.
