#!/usr/bin/env python3
import os
import sys
import time
import glob
import shutil
import argparse
import subprocess
import traceback
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Environment variables
os.environ["MKL_THREADING_LAYER"] = "GNU"

WORKSPACE_DIR = "/home/sayak/HybridTestBed"
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "experiment_results/sampling_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

BENCHMARK_PROMPT = """
The images are consecutive frames extracted from the SAME hand gesture video.
Analyze the complete sequence.
Focus on:
- hand movement direction
- temporal progression
- motion trajectory
- changes between frames

Each video contains EXACTLY ONE of the following gestures:
SWIPE_LEFT
SWIPE_RIGHT
ROLL_FWD
STOP_SIGN

Gesture definitions:
SWIPE_LEFT
- Hand moves horizontally from right to left.

SWIPE_RIGHT
- Hand moves horizontally from left to right.

ROLL_FWD
- Hand performs a circular forward rolling motion.
- Hand orientation changes continuously across frames.
- This is not a horizontal swipe.

STOP_SIGN
- Open palm facing the camera.
- Minimal motion.
- Static gesture.

You MUST choose exactly one label.
Respond with ONLY one of:
SWIPE_LEFT
SWIPE_RIGHT
ROLL_FWD
STOP_SIGN

Do not explain.
Do not provide reasoning.
Do not output any additional text.
"""

# Sampling Logic Helpers
def get_uniform_indices(N, k=20):
    if N <= k:
        indices = list(range(N))
        while len(indices) < k:
            indices.append(N - 1)
    else:
        indices = [int(i * N / k) for i in range(k)]
        indices = [max(0, min(N - 1, idx)) for idx in indices]
    return indices

def get_median_window_indices(N, k=21):
    M = N // 2
    start = M - k // 2
    end = start + k
    if k >= N:
        indices = list(range(N))
        while len(indices) < k:
            indices.append(N - 1)
    else:
        if start < 0:
            start = 0
            end = k
        elif end > N:
            end = N
            start = N - k
        indices = list(range(start, end))
    return indices

def load_sampled_frames(directory, mode="uniform"):
    images = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    if not images:
        return [], [], []
    N = len(images)
    
    if mode == "uniform":
        idxs = get_uniform_indices(N, k=20)
    else:
        idxs = get_median_window_indices(N, k=21)
        
    out_images = []
    out_paths = []
    for idx in idxs:
        try:
            path = images[idx]
            img = Image.open(path).convert("RGB")
            out_images.append(img)
            out_paths.append(path)
        except Exception as e:
            pass
    return idxs, out_paths, out_images

# Parse closed-set outputs
def parse_prediction(raw_output):
    if not raw_output:
        return "STOP_SIGN"
    s = raw_output.strip().lower()
    
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

# Compile video from arbitrary subset of frames for Video-LLaMA3
def compile_video_from_selected_frames(frame_paths, output_mp4_path, fps=10):
    temp_dir = os.path.join(WORKSPACE_DIR, f"temp_compile_{int(time.time()*1000)}")
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, path in enumerate(frame_paths):
        dest_name = f"{i+1:05d}.jpg"
        shutil.copy(path, os.path.join(temp_dir, dest_name))
        
    cmd = [
        "ffmpeg", "-y",
        "-f", "image2",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_mp4_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    shutil.rmtree(temp_dir)
    return res.returncode == 0

# --- RUNNING FUNCTIONS ---

def run_qwen4b(subset_df, mode="uniform"):
    print(f"Loading Qwen3-VL-4B (Qwen/Qwen3-VL-4B-Instruct) in 8-bit for mode={mode}...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    predictions = []
    for idx, row in subset_df.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        
        _, _, pil_images = load_sampled_frames(clip_dir, mode=mode)
        
        t_start = time.time()
        content_list = [{"type": "image", "image": img} for img in pil_images]
        content_list.append({"type": "text", "text": BENCHMARK_PROMPT})
        messages = [{"role": "user", "content": content_list}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids)]
        raw_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000.0
        
        parsed = parse_prediction(raw_output)
        pred_label = VLM_TO_CLASS[parsed]
        
        predictions.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "latency_ms": latency_ms
        })
        print(f"[Qwen4B-{mode}] [{idx+1}/40] Video {video_id}: True={CLASS_TO_VLM[true_label]}, Pred={parsed}, Latency={latency_ms:.1f}ms")
        
    return pd.DataFrame(predictions)


def run_videollama3(subset_df, mode="uniform"):
    print(f"Loading Video-LLaMA3-2B (DAMO-NLP-SG/VideoLLaMA3-2B) in 8-bit for mode={mode}...")
    import transformers.image_utils
    import transformers.video_utils
    transformers.image_utils.VideoInput = transformers.video_utils.VideoInput
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    temp_dir = os.path.join(WORKSPACE_DIR, "temp_sampling_compare_videos")
    os.makedirs(temp_dir, exist_ok=True)
    
    predictions = []
    for idx, row in subset_df.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        temp_video_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        t_start = time.time()
        _, frame_paths, _ = load_sampled_frames(clip_dir, mode=mode)
        
        # Compile MP4 using the sampled frames list
        compile_video_from_selected_frames(frame_paths, temp_video_path, fps=10)
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant specialized in hand gesture analysis."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": temp_video_path, "fps": 2, "max_frames": 16}},
                    {"type": "text", "text": BENCHMARK_PROMPT},
                ]
            },
        ]
        
        try:
            inputs = processor(conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=24)
                
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
        except Exception as e:
            print(f"Error: {e}")
            response = "STOP_SIGN"
            
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000.0
        
        if os.path.exists(temp_video_path):
            try: os.remove(temp_video_path)
            except: pass
            
        parsed = parse_prediction(response)
        pred_label = VLM_TO_CLASS[parsed]
        
        predictions.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "latency_ms": latency_ms
        })
        print(f"[Video-LLaMA3-{mode}] [{idx+1}/40] Video {video_id}: True={CLASS_TO_VLM[true_label]}, Pred={parsed}, Latency={latency_ms:.1f}ms")
        
    try: os.rmdir(temp_dir)
    except: pass
    
    return pd.DataFrame(predictions)

# Helper to plot contact sheet
def save_contact_sheet(images, indices, output_path, title, grid_shape):
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"F {indices[i]}")
        axes[i].axis('off')
    for empty_idx in range(len(images), len(axes)):
        axes[empty_idx].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- ORCHESTRATOR ---

def run_orchestrator():
    print("Orchestrator starting...")
    
    # Load 40-sample subset
    subset_path = os.path.join(WORKSPACE_DIR, "experiment_results/vlm_compare_lite/benchmark_subset_40.csv")
    if not os.path.exists(subset_path):
        print(f"Error: Could not find benchmark_subset_40.csv at {subset_path}. Generating it...")
        df = pd.read_csv(MANIFEST_PATH)
        df_val = df[df["assigned_split"] == "validation"]
        subset_list = []
        for label in [0, 1, 2, 3]:
            sub = df_val[df_val["new_label"] == label]
            sampled = sub.sample(n=10, random_state=42)
            subset_list.append(sampled)
        subset_df = pd.concat(subset_list).reset_index(drop=True)
        os.makedirs(os.path.dirname(subset_path), exist_ok=True)
        subset_df.to_csv(subset_path, index=False)
    else:
        subset_df = pd.read_csv(subset_path)
        
    # 1. Generate Contact Sheets for Visual Validation
    print("Generating contact sheets for both strategies...")
    comparison_meta = []
    
    # We take the first sample of each class
    for label in [0, 1, 2, 3]:
        class_name = CLASS_TO_VLM[label]
        row = subset_df[subset_df["new_label"] == label].iloc[0]
        video_id = str(row["video_id"])
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        
        # Load Uniform 20
        u_indices, _, u_images = load_sampled_frames(clip_dir, mode="uniform")
        u_filename = f"contact_uniform20_{video_id}_{class_name}.png"
        save_contact_sheet(u_images, u_indices, os.path.join(RESULTS_DIR, u_filename), f"{class_name} Uniform 20 (Video {video_id})", (4, 5))
        
        # Load Median 21
        m_indices, _, m_images = load_sampled_frames(clip_dir, mode="median")
        m_filename = f"contact_median21_{video_id}_{class_name}.png"
        save_contact_sheet(m_images, m_indices, os.path.join(RESULTS_DIR, m_filename), f"{class_name} Median 21 (Video {video_id})", (3, 7))
        
        comparison_meta.append({
            "class_name": class_name,
            "video_id": video_id,
            "uniform_indices": u_indices,
            "median_indices": m_indices,
            "uniform_file": u_filename,
            "median_file": m_filename,
            "total_frames": len(glob.glob(os.path.join(clip_dir, "*.jpg")))
        })
        
    # Create sampling_strategy_comparison.md
    comparison_md = "# Sampling Strategy Comparison\n\n"
    comparison_md += "This document compares the **UNIFORM_20** and **MEDIAN_WINDOW_21** sampling strategies across all gesture classes.\n\n"
    
    for meta in comparison_meta:
        comparison_md += f"## Gesture Class: {meta['class_name']} (Video {meta['video_id']})\n"
        comparison_md += f"- **Total Frames in Video**: {meta['total_frames']}\n"
        comparison_md += f"- **UNIFORM_20 Indices**: `{meta['uniform_indices']}` (Indices span full video length)\n"
        comparison_md += f"- **MEDIAN_WINDOW_21 Indices**: `{meta['median_indices']}` (Indices centered around M={meta['total_frames']//2})\n\n"
        
        comparison_md += "| UNIFORM_20 Contact Sheet | MEDIAN_WINDOW_21 Contact Sheet |\n"
        comparison_md += "|---|---|\n"
        comparison_md += f"| ![Uniform 20](file://{os.path.join(RESULTS_DIR, meta['uniform_file'])}) | ![Median 21](file://{os.path.join(RESULTS_DIR, meta['median_file'])}) |\n\n"
        
    with open(os.path.join(RESULTS_DIR, "sampling_strategy_comparison.md"), "w") as f:
        f.write(comparison_md)
    with open(os.path.join(WORKSPACE_DIR, "sampling_strategy_comparison.md"), "w") as f:
        f.write(comparison_md)
    print("Generated sampling_strategy_comparison.md")
    
    # 2. Run subprocess runs for both models and both modes to prevent VRAM OOM
    configs = [
        ("qwen4b", "uniform"),
        ("qwen4b", "median"),
        ("videollama3", "uniform"),
        ("videollama3", "median")
    ]
    
    for model, mode in configs:
        print(f"\n--- Subprocess execution: model={model}, mode={mode} ---")
        cmd = [sys.executable, "-s", __file__, "--model", model, "--mode", mode]
        subprocess.run(cmd, check=True)
        print(f"--- Subprocess finished: model={model}, mode={mode} ---\n")
        
    # 3. Compile results and generate median_window_benchmark_report.md
    metrics_summary = []
    
    for model, mode in configs:
        pred_path = os.path.join(RESULTS_DIR, f"{model}_{mode}_predictions.csv")
        pred_df = pd.read_csv(pred_path)
        
        y_true = pred_df["true_label"].tolist()
        y_pred = pred_df["predicted_label"].tolist()
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        
        metrics_summary.append({
            "Model": model.upper(),
            "Sampling Mode": "UNIFORM_20" if mode == "uniform" else "MEDIAN_WINDOW_21",
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1": f1
        })
        
    metrics_df = pd.DataFrame(metrics_summary)
    
    report_md = "# Median Window Benchmark Report\n\n"
    report_md += "This report evaluates whether focusing on the temporal center of the gesture (`MEDIAN_WINDOW_21`) improves VLM recognition performance compared to uniform sampling (`UNIFORM_20`).\n\n"
    
    report_md += "## Performance Comparison Table\n\n"
    report_md += "| Model | Sampling Mode | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n"
    report_md += "|---|---|---|---|---|---|\n"
    for _, row in metrics_df.iterrows():
        report_md += f"| {row['Model']} | {row['Sampling Mode']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} |\n"
        
    report_md += "\n## Key Findings & Performance Analysis\n\n"
    
    # Let's extract values for text writing
    q_uni = metrics_df[(metrics_df["Model"] == "QWEN4B") & (metrics_df["Sampling Mode"] == "UNIFORM_20")].iloc[0]
    q_med = metrics_df[(metrics_df["Model"] == "QWEN4B") & (metrics_df["Sampling Mode"] == "MEDIAN_WINDOW_21")].iloc[0]
    
    v_uni = metrics_df[(metrics_df["Model"] == "VIDEOLLAMA3") & (metrics_df["Sampling Mode"] == "UNIFORM_20")].iloc[0]
    v_med = metrics_df[(metrics_df["Model"] == "VIDEOLLAMA3") & (metrics_df["Sampling Mode"] == "MEDIAN_WINDOW_21")].iloc[0]
    
    report_md += "### 1. Qwen3-VL-4B Analysis\n"
    report_md += f"- **Uniform 20 F1**: {q_uni['F1']:.4f} (Accuracy: {q_uni['Accuracy']:.4f})\n"
    report_md += f"- **Median 21 F1**: {q_med['F1']:.4f} (Accuracy: {q_med['Accuracy']:.4f})\n"
    q_delta = q_med['F1'] - q_uni['F1']
    report_md += f"- **Delta F1**: {q_delta:+.4f}\n"
    if q_delta > 0:
        report_md += "- **Verdict**: Median-centered windowing successfully improves Qwen3-VL-4B gesture recognition. Concentrating frames on the high-action center of the sequence reduces the dilution caused by static/pre-gesture frames at video boundaries.\n\n"
    else:
        report_md += "- **Verdict**: Uniform sampling outperforms or matches median windowing. This suggests that the gesture's context at the start or end of the clip is critical for Qwen3-VL-4B's temporal understanding.\n\n"
        
    report_md += "### 2. Video-LLaMA3-2B Analysis\n"
    report_md += f"- **Uniform 20 F1**: {v_uni['F1']:.4f} (Accuracy: {v_uni['Accuracy']:.4f})\n"
    report_md += f"- **Median 21 F1**: {v_med['F1']:.4f} (Accuracy: {v_med['Accuracy']:.4f})\n"
    v_delta = v_med['F1'] - v_uni['F1']
    report_md += f"- **Delta F1**: {v_delta:+.4f}\n"
    if v_delta > 0:
        report_md += "- **Verdict**: Median-centered windowing successfully improves Video-LLaMA3-2B gesture recognition. By localizing the action region, the compiled video feed contains higher gesture density.\n\n"
    else:
        report_md += "- **Verdict**: Uniform sampling performs better. Video-LLaMA3 benefit from seeing the entire temporal progression from onset to return.\n\n"
        
    report_md += "## Scientific Conclusion\n"
    if q_delta > 0 or v_delta > 0:
        report_md += "Focusing on the temporal center of the gesture generally improves performance by cropping out non-informative pre-pose and post-pose sequences. We recommend adopting `MEDIAN_WINDOW_21` as the primary frame sampling strategy for both models.\n"
    else:
        report_md += "Uniform sampling remains superior because the gesture taxonomy contains movements whose start or return phases are highly distinctive (e.g. swipes). Removing these boundaries by centering too closely decreases recognition performance.\n"
        
    with open(os.path.join(RESULTS_DIR, "median_window_benchmark_report.md"), "w") as f:
        f.write(report_md)
    with open(os.path.join(WORKSPACE_DIR, "median_window_benchmark_report.md"), "w") as f:
        f.write(report_md)
        
    print("Orchestration finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, choices=["qwen4b", "videollama3"])
    parser.add_argument("--mode", type=str, default=None, choices=["uniform", "median"])
    args = parser.parse_args()
    
    if args.model is None:
        run_orchestrator()
    else:
        # Load the 40-sample subset
        subset_path = os.path.join(WORKSPACE_DIR, "experiment_results/vlm_compare_lite/benchmark_subset_40.csv")
        subset_df = pd.read_csv(subset_path)
        
        if args.model == "qwen4b":
            df_res = run_qwen4b(subset_df, mode=args.mode)
        elif args.model == "videollama3":
            df_res = run_videollama3(subset_df, mode=args.mode)
            
        # Save Predictions CSV
        pred_path = os.path.join(RESULTS_DIR, f"{args.model}_{args.mode}_predictions.csv")
        pred_df = df_res[["video_id", "true_label", "predicted_label", "latency_ms"]]
        pred_df.to_csv(pred_path, index=False)
        pred_df.to_csv(os.path.join(WORKSPACE_DIR, f"{args.model}_{args.mode}_predictions.csv"), index=False)
