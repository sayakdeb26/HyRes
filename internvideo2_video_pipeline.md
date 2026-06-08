# InternVideo2 Video Input Pipeline Architecture

This document details the design of the video input pipeline for InternVideo2, confirming the native video-oriented inference path and the frame loading process.

## 1. Inference Path Selection

InternVideo2-Chat-8B-InternLM2.5 supports a **native video-oriented inference path**.
Unlike image-based VLMs that process frames as separate images, InternVideo2 utilizes a unified video spatio-temporal modeling pipeline.
- It aggregates multi-frame embeddings through its Vision Transformer (ViT) and Q-Former.
- In `modeling_videochat2.py`, it constructs a sequential prompt with video tokens `<vid>video_token</vid>` representing the temporal segments, passing them directly to the LLM.

## 2. Frame-Sampling Strategy

Since the dataset is stored as individual `.jpg` frames, we implement a uniform frame-sampling pipeline to select exactly **20 frames** per video without requiring the overhead of `decord` video parsing.

### Uniform Index Generation
```python
def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets
```
For a clip with 30 frames and `num_segments=20`, the indices are spread evenly to cover the entire duration.

## 3. Image Preprocessing & Multi-Patch Scaling

Once frames are selected, they undergo the following transforms:
1. **Resize & Aspect Ratio Handling**: Frames are resized using bicubic interpolation to fit the aspect ratio (2:1 by default, resulting in a target size of 448x224).
2. **Sub-Image Splitting**: The resized frame is split into two `224x224` local patches and one global `224x224` patch.
3. **Normalization**: Colors are normalized using ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
4. **Tensor Assembly**: The sub-images and global images are concatenated into a 6D tensor of shape:
   `[1, 3, num_segments, 3, 224, 224]`
   where `3` represents the `num_blocks + 1` patches per frame.
