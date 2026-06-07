# InternVideo2 Checkpoint Audit Report

This report provides a hardware-aware audit of the InternVideo2 model family to select the most capable checkpoint that fits within the hardware limits of the **NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)**.

## 1. Candidate Checkpoints from `OpenGVLab`

The following checkpoints were queried from Hugging Face:

| Checkpoint Name | Parameter Count | Gated Status | Base LLM | Suitable for 8GB VRAM? |
|---|---|---|---|---|
| `OpenGVLab/InternVideo2-Chat-8B` | ~8.5B | **Gated** (Restricted Access) | Mistral-7B | No (unless access token provided) |
| `OpenGVLab/InternVideo2_chat_8B_HD` | ~8.5B | **Gated** | Mistral-7B | No (unless access token provided) |
| `OpenGVLab/InternVideo2_Chat_8B_InternLM2_5` | ~8.5B | **Public** (Open) | InternLM2.5-7B | **Yes** (via 4-bit quantization) |
| `OpenGVLab/InternVideo2_5_Chat_8B` | ~8.5B | **Public** (Open) | InternLM2.5-7B | No (Internal configuration bugs) |
| `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | ~1.1B | Public | None (Encoder only) | Yes (Inference only, no LLM chat) |
| `OpenGVLab/InternVideo2-Stage2_6B` | ~6B | Public | None (Encoder only) | Yes (Inference only, no LLM chat) |

## 2. Selection & Rationale

We selected **`OpenGVLab/InternVideo2_Chat_8B_InternLM2_5`** for the following reasons:
1. **Public Accessibility**: Unlike the gated `InternVideo2-Chat-8B`, this model is public and does not block loading due to authorization.
2. **Generative Chat Interface**: It integrates the high-performance InternVideo2 Vision Encoder with the InternLM2.5-7B language model backbone, allowing conversational reasoning and Chain-of-Thought gesture identification.
3. **Optimized Multi-Block Processing**: It natively splits high-resolution inputs and aggregates temporal context, making it ideal for dynamic motion analysis.

## 3. Hardware Alignment & VRAM Estimations

The RTX 5070 Laptop GPU has **8 GB of VRAM**.
- **BF16 Precision**: Weights consume ~16.9 GB. Loading this directly leads to immediate Out-of-Memory (OOM).
- **8-bit Quantization**: Weights consume ~8.5 GB, which exceeds the VRAM budget before inference activations.
- **4-bit Quantization**: Weights consume **~4.5 GB**. This leaves ~3.5 GB of VRAM for intermediate activations, kv-cache, and system processes, which is safe for 20-frame context lengths.
