# Qwen3-VL Dependency & Compatibility Report

This document outlines the dependencies and configuration required to support `Qwen/Qwen3-VL-4B-Instruct` in the `rosgpu_isolated` virtual environment.

## 1. Virtual Environment Status Audit
The active virtual environment (`/home/sayak/venvs/rosgpu_isolated`) was audited:
- **`torch`**: `2.9.1` (Compatible)
- **`torchvision`**: `0.24.1` (Compatible)
- **`transformers`**: `4.57.1` (Compatible; Qwen3-VL requires `transformers >= 4.57.0`)
- **`pillow`**: `12.0.0` (Compatible)
- **`qwen-vl-utils`**: Not installed (Requires installation)

## 2. Required Setup Steps
To load and run Qwen3-VL-4B, `qwen-vl-utils` must be installed:
```bash
pip install qwen-vl-utils==0.0.14
```

## 3. Library & API Mapping

| Requirement | FastVLM-1.5B (Current) | Qwen3-VL-4B-Instruct (Proposed) |
|---|---|---|
| **Transformers Model Class** | `AutoModelForCausalLM` | `Qwen3VLForConditionalGeneration` |
| **Processor Class** | custom wrapper around tokenizer | `AutoProcessor` |
| **Model Loader API** | `AutoModelForCausalLM.from_pretrained` | `Qwen3VLForConditionalGeneration.from_pretrained` |
| **Inputs Helper** | tokenizer token splicing | `qwen_vl_utils.process_vision_info` |
| **Chat Template** | manual formatting | `processor.apply_chat_template` |
| **Inference Call** | `model.generate(inputs, images, ...)` | `model.generate(**inputs, ...)` |

## 4. Input & Prompt Formatting
Qwen3-VL requires a standardized chat template input format:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": BENCHMARK_PROMPT}
        ]
    }
]
```
Using `qwen_vl_utils.process_vision_info(messages)` transforms these inputs into appropriate formats for the processor to create PyTorch tensors.
