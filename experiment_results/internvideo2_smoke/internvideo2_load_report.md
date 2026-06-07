# InternVideo2-Chat-8B-InternLM2.5 Load Report

- **Load Status**: SUCCESS
- **Mode Used**: bfloat16 GPU-priority (device_map=auto)
- **Load Time**: 104.43 seconds
- **VRAM Before Load**: 182.0 MB
- **VRAM After Load**: 5902.0 MB
- **Delta VRAM**: 5720.0 MB
- **RAM Before Load**: 26.2%
- **RAM After Load**: 61.5%

## Device Map Layer Placement
```python
{'query_tokens': 0, 'extra_query_tokens': 0, 'vision_encoder': 0, 'vision_layernorm': 0, 'lm.base_model.model.model.tok_embeddings': 0, 'lm.base_model.model.model.layers.0': 0, 'lm.base_model.model.model.layers.1': 0, 'lm.base_model.model.model.layers.2': 0, 'lm.base_model.model.model.layers.3': 0, 'lm.base_model.model.model.layers.4': 0, 'lm.base_model.model.model.layers.5': 0, 'lm.base_model.model.model.layers.6': 0, 'lm.base_model.model.model.layers.7': 'cpu', 'lm.base_model.model.model.layers.8': 'cpu', 'lm.base_model.model.model.layers.9': 'cpu', 'lm.base_model.model.model.layers.10': 'cpu', 'lm.base_model.model.model.layers.11': 'cpu', 'lm.base_model.model.model.layers.12': 'cpu', 'lm.base_model.model.model.layers.13': 'cpu', 'lm.base_model.model.model.layers.14': 'cpu', 'lm.base_model.model.model.layers.15': 'cpu', 'lm.base_model.model.model.layers.16': 'cpu', 'lm.base_model.model.model.layers.17': 'cpu', 'lm.base_model.model.model.layers.18': 'cpu', 'lm.base_model.model.model.layers.19': 'cpu', 'lm.base_model.model.model.layers.20': 'cpu', 'lm.base_model.model.model.layers.21': 'cpu', 'lm.base_model.model.model.layers.22': 'cpu', 'lm.base_model.model.model.layers.23': 'cpu', 'lm.base_model.model.model.layers.24': 'cpu', 'lm.base_model.model.model.layers.25': 'cpu', 'lm.base_model.model.model.layers.26': 'cpu', 'lm.base_model.model.model.layers.27': 'cpu', 'lm.base_model.model.model.layers.28': 'cpu', 'lm.base_model.model.model.layers.29': 'cpu', 'lm.base_model.model.model.layers.30': 'cpu', 'lm.base_model.model.model.layers.31': 'cpu', 'lm.base_model.model.model.norm': 'cpu', 'lm.base_model.model.output': 'cpu', 'project_up': 'cpu', 'project_down': 'cpu', 'qformer': 'cpu'}
```

