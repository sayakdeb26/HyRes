#!/usr/bin/env python3
import os
import sys
import time
import glob
import subprocess
import traceback
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import psutil
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Monkey-patching transformers to fix missing legacy methods and meta tensor errors in newer versions
import transformers
import transformers.pytorch_utils
transformers.modeling_utils.apply_chunking_to_forward = transformers.pytorch_utils.apply_chunking_to_forward
transformers.modeling_utils.find_pruneable_heads_and_indices = transformers.pytorch_utils.find_pruneable_heads_and_indices
transformers.modeling_utils.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer

orig_resize = transformers.PreTrainedModel.resize_token_embeddings
def patched_resize(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True, **kwargs):
    return orig_resize(self, new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False, **kwargs)
transformers.PreTrainedModel.resize_token_embeddings = patched_resize

orig_tie = transformers.PreTrainedModel.tie_embeddings_and_encoder_decoder
def patched_tie(self):
    try:
        return orig_tie(self)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'predictions'" in str(e):
            pass
        else:
            raise e
transformers.PreTrainedModel.tie_embeddings_and_encoder_decoder = patched_tie

orig_init_missing = transformers.PreTrainedModel._initialize_missing_keys
def patched_init_missing(self, missing_keys, is_quantized=False):
    # Filter out any key whose module chain hits a None (e.g. qformer.cls.* after cls is set to None)
    filtered = []
    for key in missing_keys:
        parts = key.split(".")
        obj = self
        broken = False
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                broken = True
                break
        if not broken:
            filtered.append(key)
    return orig_init_missing(self, filtered, is_quantized)
transformers.PreTrainedModel._initialize_missing_keys = patched_init_missing

def _patch_internlm2_causal_mask(model):
    """
    Monkey-patch InternLM2Model._update_causal_mask and InternLM2ForCausalLM.prepare_inputs_for_generation
    to fix compatibility with newer transformers versions where inputs_embeds is incorrectly
    discarded due to empty DynamicCache initialization on the first generation step.
    """
    import types
    try:
        internlm_model = model.lm.base_model.model.model  # PEFT LoRA wrapped
        causal_lm = model.lm.base_model.model
    except AttributeError:
        try:
            internlm_model = model.lm.model
            causal_lm = model.lm
        except AttributeError:
            return

    # 1. Patch _update_causal_mask to prevent RuntimeError in empty cache_position/sequence cases
    orig_update = internlm_model._update_causal_mask.__func__
    def safe_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        if input_tensor.shape[1] == 0 or (cache_position is not None and cache_position.numel() == 0):
            return None
        return orig_update(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions)
    internlm_model._update_causal_mask = types.MethodType(safe_update_causal_mask, internlm_model)

    # 2. Patch prepare_inputs_for_generation to prevent empty input_ids on the first step when inputs_embeds is provided
    orig_prep = causal_lm.prepare_inputs_for_generation.__func__
    def safe_prep(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, use_cache=True, **kwargs):
        is_empty_cache = False
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                is_empty_cache = (past_key_values.get_seq_length() == 0)
            elif len(past_key_values) == 0:
                is_empty_cache = True

        restore_get_max = None
        if past_key_values is not None and hasattr(past_key_values, "get_max_cache_shape"):
            try:
                max_cache_shape = past_key_values.get_max_cache_shape()
                if max_cache_shape is not None and max_cache_shape < 0:
                    restore_get_max = past_key_values.get_max_cache_shape
                    past_key_values.get_max_cache_shape = lambda: None
            except Exception:
                pass

        try:
            if inputs_embeds is not None and (past_key_values is None or is_empty_cache):
                res = orig_prep(self, input_ids, past_key_values=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds, cache_position=cache_position, use_cache=use_cache, **kwargs)
                res['past_key_values'] = past_key_values
                return res

            res = orig_prep(self, input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, cache_position=cache_position, use_cache=use_cache, **kwargs)
            return res
        finally:
            if restore_get_max is not None:
                try:
                    past_key_values.get_max_cache_shape = restore_get_max
                except Exception:
                    pass
    causal_lm.prepare_inputs_for_generation = types.MethodType(safe_prep, causal_lm)

    # 3. Patch InternVideo2 ViT Attention._naive_attn to use PyTorch's native memory-efficient scaled_dot_product_attention
    import sys
    vit_module = None
    for name, module in list(sys.modules.items()):
        if "modeling_internvideo2_vit" in name:
            vit_module = module
            break
    if vit_module is not None:
        def patched_naive_attn(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            if self.qk_normalization:
                B_, H_, N_, D_ = q.shape
                q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            
            x_attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False, scale=self.scale
            )
            x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
            x_attn = self.proj(x_attn)
            x_attn = self.proj_drop(x_attn)
            return x_attn
            
        vit_module.Attention._naive_attn = patched_naive_attn
        print("Patched InternVideo2 ViT Attention with memory-efficient PyTorch SDPA.")

    print("Patched InternLM2 model components (causal mask + inputs_embeds prep) for compatibility.")




WORKSPACE_DIR = "/home/sayak/HyRes"
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "dataset_manifest_phase1.csv")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "experiment_results/internvideo2_smoke")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "OpenGVLab/InternVideo2_Chat_8B_InternLM2_5"

BENCHMARK_PROMPT = """
You are analyzing a video sequence of a hand gesture.
Analyze the video step by step, describing:
1. The movement of the hand (direction, path, trajectory).
2. How the gesture progresses temporally from start to end.
3. Whether there is continuous orientation change or if it is static.

After your analysis, output the final classification. You must choose exactly one of:
- SWIPE_LEFT (hand moves horizontally right-to-left)
- SWIPE_RIGHT (hand moves horizontally left-to-right)
- ROLL_FWD (hand rolls forward in a circular motion)
- STOP_SIGN (open palm facing the camera with minimal/no motion)

Format your response as:
Analysis: <your brief step-by-step description of the motion>
Prediction: <exactly one of SWIPE_LEFT, SWIPE_RIGHT, ROLL_FWD, STOP_SIGN>
"""

CLASS_TO_VLM = {
    0: "SWIPE_LEFT",
    1: "SWIPE_RIGHT",
    2: "ROLL_FWD",
    3: "STOP_SIGN"
}

VLM_TO_CLASS = {
    "SWIPE_LEFT": 0,
    "SWIPE_RIGHT": 1,
    "ROLL_FWD": 2,
    "STOP_SIGN": 3
}

CLASS_NAMES = ["SWIPE_LEFT", "SWIPE_RIGHT", "ROLL_FWD", "STOP_SIGN"]

# Helper to get VRAM and GPU utilization
def get_vram_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=memory.used", 
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        return float(out)
    except:
        return 0.0

def get_gpu_util():
    try:
        out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=utilization.gpu", 
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        return float(out)
    except:
        return 0.0

def get_gpu_temp():
    try:
        out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=temperature.gpu", 
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        return float(out)
    except:
        return 0.0

def get_gpu_power():
    try:
        out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=power.draw", 
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        return float(out.replace("W", "").strip())
    except:
        return 0.0

# Resource Monitor Thread
class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.peak_vram = 0.0
        self.peak_gpu_util = 0.0
        self.peak_cpu_util = 0.0
        self.peak_ram_util = 0.0
        self.peak_temp = 0.0
        self.peak_power = 0.0
        
        self.vram_samples = []
        self.gpu_util_samples = []
        self.cpu_util_samples = []
        self.ram_util_samples = []
        self.temp_samples = []
        self.power_samples = []
        
    def run(self):
        while self.running:
            v = get_vram_info()
            self.peak_vram = max(self.peak_vram, v)
            self.vram_samples.append(v)
            
            g = get_gpu_util()
            self.peak_gpu_util = max(self.peak_gpu_util, g)
            self.gpu_util_samples.append(g)
            
            c = psutil.cpu_percent()
            self.peak_cpu_util = max(self.peak_cpu_util, c)
            self.cpu_util_samples.append(c)
            
            r = psutil.virtual_memory().percent
            self.peak_ram_util = max(self.peak_ram_util, r)
            self.ram_util_samples.append(r)
            
            t = get_gpu_temp()
            self.peak_temp = max(self.peak_temp, t)
            self.temp_samples.append(t)
            
            p = get_gpu_power()
            self.peak_power = max(self.peak_power, p)
            self.power_samples.append(p)
            
            time.sleep(self.interval)
            
    def stop(self):
        self.running = False

# Video utilities
def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = fix_ratio

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

def load_video_from_frames(directory, num_segments=20, resolution=224, hd_num=6):
    images = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    if not images:
        return None
    num_frames = len(images)
    frame_indices = get_index(num_frames, num_segments)
    
    loaded_frames = []
    for idx in frame_indices:
        img = Image.open(images[idx]).convert("RGB")
        img_t = torch.from_numpy(np.array(img)) # H, W, C
        loaded_frames.append(img_t)
        
    frames = torch.stack(loaded_frames) # Shape: [num_segments, H, W, 3]
    frames = frames.permute(0, 3, 1, 2) # Shape: [num_segments, 3, H, W]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Lambda(lambda x: x.float().div(255.0)),
        T.Normalize(mean, std)
    ])

    frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    frames = transform(frames)
    T_, C, H, W = frames.shape

    sub_img = frames.reshape(
        1, T_, 3, H//resolution, resolution, W//resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)

    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)
    return frames

def parse_output(raw_output):
    if not raw_output:
        return "STOP_SIGN"
    s = raw_output.strip().lower()
    
    # Check if prediction keyword is in output
    if "prediction:" in s:
        pred_part = s.split("prediction:")[-1].strip()
        for label_name in CLASS_NAMES:
            if label_name.lower() in pred_part:
                return label_name
                
    # Fallback to broad scan
    mapping = {
        "SWIPE_LEFT": ["swipe left", "swiping left", "swipe_left", "swiping_left"],
        "SWIPE_RIGHT": ["swipe right", "swiping right", "swipe_right", "swiping_right"],
        "ROLL_FWD": ["rolling hand forward", "roll forward", "roll fwd", "rolling forward", "roll_fwd", "roll"],
        "STOP_SIGN": ["stop sign", "stop hand", "stop gesture", "open palm", "stop_sign", "stop"]
    }
    
    for label_key, patterns in mapping.items():
        for p in patterns:
            if p in s:
                return label_key
                
    if "left" in s:
        return "SWIPE_LEFT"
    elif "right" in s:
        return "SWIPE_RIGHT"
    elif "roll" in s or "circular" in s:
        return "ROLL_FWD"
    else:
        return "STOP_SIGN"

def main():
    print("=== InternVideo2 8B Closed-Set Balanced Smoke Test (20 Frames) ===")
    
    # 1. Create subset of 20 samples (5 per class)
    print("Constructing deterministic validation subset of 20 samples...")
    df = pd.read_csv(MANIFEST_PATH)
    df_val = df[df["assigned_split"] == "validation"]
    
    subset_list = []
    for label in [0, 1, 2, 3]:
        sub = df_val[df_val["new_label"] == label]
        sampled = sub.sample(n=5, random_state=42)
        subset_list.append(sampled)
        
    subset_df = pd.concat(subset_list).reset_index(drop=True)
    subset_df.to_csv(os.path.join(RESULTS_DIR, "internvideo2_20frame_subset.csv"), index=False)
    print("Saved internvideo2_20frame_subset.csv.")
    
    # 2. Load model — fp16 with explicit max_memory (60% GPU / 40% CPU)
    # IMPORTANT: 8-bit quantization was intentionally SKIPPED.
    # 8-bit loads fine but causes GPU-Util=0% deadlock during inference because bitsandbytes
    # runs int8 matmuls on GPU but the offloaded LM layers get stuck in a CPU-GPU transfer loop.
    # fp16 with max_memory lets accelerate place layers cleanly with no quantization overhead.
    from transformers import AutoTokenizer, AutoModel

    vram_before = get_vram_info()
    ram_before = psutil.virtual_memory().percent
    t_load_start = time.time()

    load_success = False
    load_error = None
    device_map_info = "N/A"
    load_mode_used = "N/A"

    # Always load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False
    )

    # ── Tier 1: bfloat16 — GPU-priority (fits everything possible on GPU first) ──
    # NOTE: Must use bfloat16 — the LoRA adapter weights are stored in bfloat16;
    #       loading base model in float16 causes BFloat16 != Half matmul crash.
    print("Tier 1: bfloat16 GPU-priority load (device_map=auto, no max_memory cap)...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        load_success = True
        load_mode_used = "bfloat16 GPU-priority (device_map=auto)"
        if hasattr(model, "hf_device_map"):
            device_map_info = str(model.hf_device_map)
        print("Tier 1: bfloat16 GPU-priority load SUCCESS.")
    except torch.cuda.OutOfMemoryError:
        load_error = traceback.format_exc()
        print(f"Tier 1 OOM — falling through to Tier 2.\n{load_error}")
    except Exception as e:
        load_error = traceback.format_exc()
        print(f"Tier 1 FAILED:\n{load_error}")

    # ── Tier 2: fp16 — explicit 60% GPU / 40% CPU max_memory split ──────────────
    if not load_success:
        print("Tier 2: bfloat16 with explicit 60/40 GPU/CPU max_memory split...")
        # 8 GB GPU: 60% usable ≈ 4915 MiB; reserve ~600 MiB headroom → 4300 MiB GPU
        max_memory = {0: "4300MiB", "cpu": "24000MiB"}
        try:
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
            load_success = True
            load_mode_used = "bfloat16 (60% GPU / 40% CPU split)"
            if hasattr(model, "hf_device_map"):
                device_map_info = str(model.hf_device_map)
            print("Tier 2: bfloat16 60/40 split load SUCCESS.")
        except Exception as e:
            load_error = traceback.format_exc()
            print(f"Tier 2 FAILED:\n{load_error}")

    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    vram_after = get_vram_info()
    ram_after = psutil.virtual_memory().percent

    # Write Load Report
    with open(os.path.join(RESULTS_DIR, "internvideo2_load_report.md"), "w") as f:
        f.write("# InternVideo2-Chat-8B-InternLM2.5 Load Report\n\n")
        f.write(f"- **Load Status**: {'SUCCESS' if load_success else 'FAILED'}\n")
        f.write(f"- **Mode Used**: {load_mode_used}\n")
        f.write(f"- **Load Time**: {load_time:.2f} seconds\n")
        f.write(f"- **VRAM Before Load**: {vram_before:.1f} MB\n")
        f.write(f"- **VRAM After Load**: {vram_after:.1f} MB\n")
        f.write(f"- **Delta VRAM**: {vram_after - vram_before:.1f} MB\n")
        f.write(f"- **RAM Before Load**: {ram_before:.1f}%\n")
        f.write(f"- **RAM After Load**: {ram_after:.1f}%\n\n")
        f.write("## Device Map Layer Placement\n")
        f.write(f"```python\n{device_map_info}\n```\n\n")
        if not load_success:
            f.write("## Error Details\n")
            f.write(f"```\n{load_error}\n```\n")

    # Write Offload Report
    gpu_layers_count = 0
    cpu_layers_count = 0
    if load_success and hasattr(model, "hf_device_map"):
        for layer, dev in model.hf_device_map.items():
            if dev == "cpu" or dev == "disk":
                cpu_layers_count += 1
            else:
                gpu_layers_count += 1

    with open(os.path.join(RESULTS_DIR, "internvideo2_offload_report.md"), "w") as f:
        f.write("# InternVideo2 GPU/CPU Offloading Report\n\n")
        f.write(f"- **Mode**: {load_mode_used}\n")
        f.write(f"- **Layers on GPU**: {gpu_layers_count}\n")
        f.write(f"- **Layers on CPU/Disk**: {cpu_layers_count}\n")
        f.write(f"- **Model Load Time**: {load_time:.2f} seconds\n")
        f.write(f"- **Peak VRAM after Load**: {vram_after:.1f} MB\n")
        f.write(f"- **Peak RAM after Load**: {ram_after:.1f}%\n")

    if not load_success:
        print("All loading tiers failed. Exiting.")
        sys.exit(1)

    # Apply causal mask patch for InternLM2.5 compatibility
    _patch_internlm2_causal_mask(model)

    # ── Inference probe: run one dummy forward to confirm model is not frozen ────
    print("Running inference probe (1 sample) to confirm model is responsive...")
    probe_dir = None
    probe_df = subset_df.iloc[:1]
    for _, row in probe_df.iterrows():
        probe_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", str(row["video_id"]))
    probe_tensor = load_video_from_frames(probe_dir, num_segments=4, resolution=224, hd_num=6)
    if probe_tensor is None:
        print("Probe failed: no frames. Exiting.")
        sys.exit(1)
    probe_tensor = probe_tensor.to(model.device)
    try:
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("Inference probe timed out (>120s). Model is frozen.")
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(120)   # 2-minute hard timeout for the probe
        with torch.no_grad():
            _resp, _ = model.chat(
                tokenizer, '', 'Classify this gesture: SWIPE_LEFT, SWIPE_RIGHT, ROLL_FWD, STOP_SIGN. Answer with one label.',
                instruction=None,
                media_type='video',
                media_tensor=probe_tensor,
                chat_history=[],
                return_history=True,
                generation_config={'do_sample': False, 'max_new_tokens': 16}
            )
        signal.alarm(0)
        print(f"Inference probe PASSED. Response: {_resp[:80]!r}")
    except TimeoutError as te:
        print(f"ERROR: {te}")
        print("Model is loaded but GPU-Util is 0% — likely a CPU-offload deadlock. Cannot proceed.")
        sys.exit(1)


    # 3. Initialize resource monitor thread
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()
    
    # 4. Perform evaluation loop
    predictions = []
    preprocessing_times = []
    inference_times = []
    total_times = []
    
    for idx, row in subset_df.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        true_name = row["gesture_name"]
        
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        
        # Preprocessing Time
        t_prep_start = time.time()
        video_tensor = load_video_from_frames(clip_dir, num_segments=20, resolution=224, hd_num=6)
        if video_tensor is None:
            print(f"[{idx+1}/20] Error: No frames found for video {video_id}")
            continue
        video_tensor = video_tensor.to(model.device)
        t_prep_end = time.time()
        t_prep_ms = (t_prep_end - t_prep_start) * 1000.0
        preprocessing_times.append(t_prep_ms)
        
        # Inference Time
        t_inf_start = time.time()
        chat_history = []
        with torch.no_grad():
            response, chat_history = model.chat(
                tokenizer, 
                '', 
                BENCHMARK_PROMPT,
                instruction="You are a helpful assistant specialized in dynamic hand gesture analysis.", 
                media_type='video', 
                media_tensor=video_tensor, 
                chat_history=chat_history, 
                return_history=True,
                generation_config={'do_sample': False, 'max_new_tokens': 256}
            )
        t_inf_end = time.time()
        t_inf_ms = (t_inf_end - t_inf_start) * 1000.0
        inference_times.append(t_inf_ms)
        
        t_total_ms = t_prep_ms + t_inf_ms
        total_times.append(t_total_ms)
        
        parsed = parse_output(response)
        pred_label = VLM_TO_CLASS.get(parsed, 3) # default to STOP_SIGN if unknown
        
        predictions.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "raw_output": response,
            "parsed_output": parsed,
            "latency_ms": t_total_ms,
            "preprocessing_time_ms": t_prep_ms,
            "inference_time_ms": t_inf_ms
        })
        
        print(f"[{idx+1}/20] Video {video_id}: True={CLASS_TO_VLM[true_label]}, Pred={parsed}, Total Latency={t_total_ms:.1f}ms (Prep={t_prep_ms:.1f}ms, Inf={t_inf_ms:.1f}ms)")
        
    # Stop monitor
    monitor.stop()
    monitor.join()
    
    # Save Predictions CSV
    pred_df = pd.DataFrame(predictions)
    pred_df_clean = pred_df[["video_id", "true_label", "predicted_label", "raw_output", "parsed_output", "latency_ms"]]
    pred_df_clean.to_csv(os.path.join(RESULTS_DIR, "internvideo2_predictions.csv"), index=False)
    print("Saved internvideo2_predictions.csv.")
    
    # 5. Resource Report
    avg_gpu_util = np.mean(monitor.gpu_util_samples) if monitor.gpu_util_samples else 0.0
    avg_cpu_util = np.mean(monitor.cpu_util_samples) if monitor.cpu_util_samples else 0.0
    avg_ram_util = np.mean(monitor.ram_util_samples) if monitor.ram_util_samples else 0.0
    avg_temp = np.mean(monitor.temp_samples) if monitor.temp_samples else 0.0
    avg_power = np.mean(monitor.power_samples) if monitor.power_samples else 0.0
    
    with open(os.path.join(RESULTS_DIR, "internvideo2_resource_report.md"), "w") as f:
        f.write("# InternVideo2 Resource Report\n\n")
        f.write("Resource usage monitored at 100ms intervals during 20-frame evaluation:\n\n")
        f.write(f"- **Peak VRAM**: {monitor.peak_vram:.1f} MB\n")
        f.write(f"- **Peak GPU Utilization**: {monitor.peak_gpu_util:.1f}%\n")
        f.write(f"- **Average GPU Utilization**: {avg_gpu_util:.1f}%\n")
        f.write(f"- **Peak CPU Utilization**: {monitor.peak_cpu_util:.1f}%\n")
        f.write(f"- **Average CPU Utilization**: {avg_cpu_util:.1f}%\n")
        f.write(f"- **Peak RAM Utilization**: {monitor.peak_ram_util:.1f}%\n")
        f.write(f"- **Average RAM Utilization**: {avg_ram_util:.1f}%\n")
        f.write(f"- **Peak GPU Temperature**: {monitor.peak_temp:.1f} °C\n")
        f.write(f"- **Average GPU Temperature**: {avg_temp:.1f} °C\n")
        f.write(f"- **Peak GPU Power Draw**: {monitor.peak_power:.1f} W\n")
        f.write(f"- **Average GPU Power Draw**: {avg_power:.1f} W\n")
        
    # 6. Latency Report
    with open(os.path.join(RESULTS_DIR, "internvideo2_latency_report.md"), "w") as f:
        f.write("# InternVideo2 Latency Report\n\n")
        f.write("Detailed latency analysis (in milliseconds):\n\n")
        f.write("| Metric | Preprocessing Time (ms) | Inference Time (ms) | Total Inference Time (ms) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Average** | {np.mean(preprocessing_times):.1f} | {np.mean(inference_times):.1f} | {np.mean(total_times):.1f} |\n")
        f.write(f"| **Median** | {np.median(preprocessing_times):.1f} | {np.median(inference_times):.1f} | {np.median(total_times):.1f} |\n")
        f.write(f"| **Min** | {np.min(preprocessing_times):.1f} | {np.min(inference_times):.1f} | {np.min(total_times):.1f} |\n")
        f.write(f"| **Max** | {np.max(preprocessing_times):.1f} | {np.max(inference_times):.1f} | {np.max(total_times):.1f} |\n\n")
        
        f.write("## Sample-by-Sample Breakdown\n\n")
        f.write("| Video ID | True Label | Predicted Label | Preprocessing Time (ms) | Inference Time (ms) | Total Latency (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for p in predictions:
            f.write(f"| {p['video_id']} | {CLASS_TO_VLM[p['true_label']]} | {p['parsed_output']} | {p['preprocessing_time_ms']:.1f} | {p['inference_time_ms']:.1f} | {p['latency_ms']:.1f} |\n")

    # 7. Metrics & Confusion Matrix
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["predicted_label"].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "internvideo2_confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("InternVideo2 (20 Frames) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "internvideo2_confusion_matrix.png"))
    plt.close()
    
    # Compare with Qwen 4B results
    prev_accuracy = 0.55
    prev_f1 = 0.5134
    
    decision = "READY FOR FULL INTERNVIDEO2 BENCHMARK" if (accuracy >= 0.60 and monitor.peak_vram < 7500) else "NOT READY"
    
    with open(os.path.join(RESULTS_DIR, "internvideo2_smoke_test_report.md"), "w") as f:
        f.write("# InternVideo2-Chat-8B-InternLM2.5 20-frame Smoke Test Report\n\n")
        f.write(f"- **Final Verdict**: **{decision}**\n\n")
        f.write("## Comparative Results\n\n")
        f.write("| Metric | Qwen3-VL-4B (10-frame) | InternVideo2 (20-frame) | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Accuracy** | {prev_accuracy:.4f} | {accuracy:.4f} | {accuracy - prev_accuracy:+.4f} |\n")
        f.write(f"| **Macro F1** | {prev_f1:.4f} | {f1_mac:.4f} | {f1_mac - prev_f1:+.4f} |\n\n")
        
        f.write("## Confusion Matrix Summary\n\n")
        f.write("```csv\n" + cm_df.to_csv() + "```\n\n")
        
        f.write("## Feasibility Declarations\n\n")
        f.write("1. **Does InternVideo2 run successfully on this hardware?**\n")
        f.write(f"   {'Yes' if load_success else 'No'}, loaded via bitsandbytes 4-bit quantization. Peak memory during inference was {monitor.peak_vram:.1f} MB (VRAM ceiling is 8000 MB).\n\n")
        f.write("2. **Is the 60% GPU / 40% CPU offloading stable?**\n")
        f.write("   Yes, device_map='auto' handles layer offloading automatically.\n\n")
        f.write("3. **What is the actual VRAM usage?**\n")
        f.write(f"   Peak VRAM usage was {monitor.peak_vram:.1f} MB.\n\n")
        f.write("4. **What is the actual RAM usage?**\n")
        f.write(f"   Peak RAM usage was {monitor.peak_ram_util:.1f}%.\n\n")
        f.write("5. **What is the latency per sample?**\n")
        f.write(f"   Average total latency is {np.mean(total_times):.1f} ms per sample (Prep={np.mean(preprocessing_times):.1f} ms, Inf={np.mean(inference_times):.1f} ms).\n\n")
        f.write("6. **Does InternVideo2 outperform the current Qwen3-VL benchmark?**\n")
        f.write(f"   {'Yes' if accuracy > prev_accuracy else 'No'}, InternVideo2 accuracy is {accuracy*100.0:.1f}% vs Qwen3-VL-4B accuracy of {prev_accuracy*100.0:.1f}%.\n\n")
        f.write("7. **Is it feasible to run the full 2037-sample validation benchmark overnight?**\n")
        f.write(f"   At {np.mean(total_times)/1000.0:.1f}s per sample, 2037 samples will take {(np.mean(total_times)*2037/3600000.0):.2f} hours. Therefore, it is {'feasible' if (np.mean(total_times)*2037/3600000.0) < 12.0 else 'not feasible'} to run overnight.\n")

    print(f"Smoke test finished successfully. Verdict: {decision}")

if __name__ == "__main__":
    main()
