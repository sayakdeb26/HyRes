#!/usr/bin/env python3
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
import time
import subprocess
import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Force paths
sys.path.append('/home/sayak')
sys.path.append('/home/sayak/HyRes')

from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKSPACE_DIR = "/home/sayak/HyRes"
DATA_DIR = os.path.join(WORKSPACE_DIR, "hand_gesture_lab/data/phase1")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "experiment_results/phase2_vlm")
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "dataset_manifest_phase1.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "apple/FastVLM-1.5B"
FRAMES_TO_SAMPLE = 5
MAX_NEW_TOKENS = 24
TEMPERATURE = 0.0
TOP_P = 1.0
IMAGE_TOKEN_INDEX = -200

GESTURE_MAPPING = {
    "Swipe Left": 0,
    "Swipe Right": 1,
    "Rolling Hand Forward": 2,
    "Stop Sign": 3
}
REVERSE_MAPPING = {v: k for k, v in GESTURE_MAPPING.items()}
ALLOWED_CLASSES = [0, 1, 2, 3]

DEFAULT_PROMPT = """You will be given frames from a short video.
Each frame contains a single person performing exactly one hand gesture.
Focus only on the hands and ignore the face, background, or other objects.

Choose exactly ONE label from the following list that best describes the hand gesture:

- SWIPE_LEFT
- SWIPE_RIGHT
- ROLL_FWD
- STOP_SIGN
- UNKNOWN

If none of the labels fits, answer UNKNOWN.
Respond with ONLY the label text, nothing else.
"""

VLM_TO_CLASS = {
    "SWIPE_LEFT": 0,
    "SWIPE_RIGHT": 1,
    "ROLL_FWD": 2,
    "STOP_SIGN": 3
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def run_pre_flight_checks():
    print("Running Pre-Execution Validation...")
    
    val_x_path = os.path.join(DATA_DIR, "X_validation.npy")
    val_y_path = os.path.join(DATA_DIR, "y_validation.npy")
    
    assert os.path.exists(val_x_path), "X_validation.npy not found!"
    assert os.path.exists(val_y_path), "y_validation.npy not found!"
    
    y_val = np.load(val_y_path)
    unique_labels = np.unique(y_val)
    
    assert all(lbl in ALLOWED_CLASSES for lbl in unique_labels), f"Invalid labels found! {unique_labels}"
    
    df = pd.read_csv(MANIFEST_PATH)
    df_val = df[df["assigned_split"] == "validation"]
    assert len(df_val) == len(y_val), f"Manifest mismatch! Manifest: {len(df_val)} vs Tensor: {len(y_val)}"
    
    report_path = os.path.join(RESULTS_DIR, "phase2_pre_execution_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase 2 Pre-Execution Validation Report\n\n")
        f.write("- Validation Tensor Exists: YES\n")
        f.write("- Validation Labels Exist: YES\n")
        f.write(f"- Validation Label Classes: {list(unique_labels)}\n")
        f.write(f"- Manifest Validation Samples: {len(df_val)}\n")
        f.write(f"- Tensor Validation Samples: {len(y_val)}\n")
        f.write("- Mapping Check: PASSED\n")
        
    print("Pre-Execution Validation PASSED.")
    return df_val

def generate_config_audit(device):
    print("Generating VLM Configuration Audit...")
    report_path = os.path.join(RESULTS_DIR, "vlm_configuration_report.md")
    with open(report_path, "w") as f:
        f.write("# VLM Configuration Report\n\n")
        f.write(f"- **Exact Model Name**: `{MODEL_ID}`\n")
        f.write(f"- **Model Version**: Latest (HuggingFace Hub)\n")
        f.write(f"- **Prompt Template**:\n```\n{DEFAULT_PROMPT}\n```\n")
        f.write(f"- **Inference Parameters**:\n")
        f.write(f"  - Max New Tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"  - Temperature: {TEMPERATURE}\n")
        f.write(f"  - Top-P: {TOP_P}\n")
        f.write(f"  - Sampled Frames Per Video: {FRAMES_TO_SAMPLE}\n")
        f.write(f"- **Quantization Settings**: fp16 (half precision)\n")
        f.write(f"- **GPU Usage Configuration**: {device}\n")
        f.write(f"- **Output Post-Processing Logic**: Exact string match mapped via `VLM_TO_CLASS` dictionary. Any unmapped or UNKNOWN response resolves to label `-1`.\n")

def sample_frames_from_dir(directory, k):
    images = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    if not images:
        return []
        
    total = len(images)
    if total <= k:
        idxs = list(range(total))
    else:
        idxs = [int((i+1)*total/(k+1)) for i in range(k)]
        
    out = []
    for i in idxs:
        try:
            img = Image.open(images[i]).convert("RGB")
            out.append(img)
        except:
            pass
    return out

def canonicalize_prediction(raw_text):
    if not raw_text:
        return -1
        
    s = raw_text.strip().splitlines()[0].strip().strip(".").upper()
    s = s.replace(" ", "_")
    
    if s in VLM_TO_CLASS:
        return VLM_TO_CLASS[s]
        
    for key, val in VLM_TO_CLASS.items():
        if key in s:
            return val
            
    return -1

def aggregate_predictions(preds):
    from collections import Counter
    if not preds:
        return -1, 0.0
        
    c = Counter(preds)
    best_label, votes = c.most_common(1)[0]
    conf = votes / len(preds)
    conf = 0.98 * conf + 0.01
    return best_label, conf

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("Starting Phase 2 VLM-Only Evaluation...")
    
    df_val = run_pre_flight_checks()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading FastVLM model on {device}...")
    generate_config_audit(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    img_proc = model.get_vision_tower().image_processor

    # Start Resource Monitor
    monitor_proc = subprocess.Popen(["python3", "/home/sayak/HyRes/resource_monitor.py"])
    
    results = []
    latencies = []
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    print("Beginning validation inference...")
    total_samples = len(df_val)
    
    for idx, row in df_val.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        
        # Adjust path to the original symlinked frame directory
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        
        frames = sample_frames_from_dir(clip_dir, FRAMES_TO_SAMPLE)
        
        if not frames:
            print(f"Warning: No frames found for {video_id}")
            results.append({
                "video_id": video_id,
                "true_label": true_label,
                "predicted_label": -1,
                "correctness": 0,
                "confidence": 0.0,
                "latency_ms": 0.0
            })
            continue
            
        frame_preds = []
        
        start_time = time.time()
        
        with torch.inference_mode():
            for pil_img in frames:
                messages = [{"role": "user", "content": f"<image>\n{DEFAULT_PROMPT}"}]
                rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                pre, post = rendered.split("<image>", 1)
                
                pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
                
                img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
                
                px = img_proc(images=pil_img, return_tensors="pt")["pixel_values"].to(device, dtype=model.dtype)
                
                out_ids = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
                label_val = canonicalize_prediction(text)
                frame_preds.append(label_val)
                
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0
        
        final_label, conf = aggregate_predictions(frame_preds)
        is_correct = 1 if final_label == true_label else 0
        
        results.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": final_label,
            "correctness": is_correct,
            "confidence": conf,
            "latency_ms": latency_ms
        })
        
        latencies.append(latency_ms)
        confidences.append(conf)
        if is_correct:
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_samples} samples...")
            
    # Terminate Monitor
    monitor_proc.terminate()
    print("Inference completed. Terminating resource monitor.")
    time.sleep(2)
    
    # Copy resource log to phase2_vlm
    src_resource_log = "/home/sayak/HyRes/experiment_results/resource_usage/resource_log.csv"
    if os.path.exists(src_resource_log):
        df_res = pd.read_csv(src_resource_log)
        df_res.to_csv(os.path.join(RESULTS_DIR, "vlm_resource_report.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "vlm_resource_report.md"), "w") as f:
            f.write("# VLM Resource Report\n\n")
            f.write(f"- CPU Utilization Avg: {df_res['cpu_utilization'].mean():.2f}%\n")
            f.write(f"- RAM Used Avg: {df_res['ram_used_mb'].mean():.2f} MB\n")
            f.write(f"- GPU Utilization Avg: {df_res['gpu_utilization'].mean():.2f}%\n")
            f.write(f"- VRAM Used Avg: {df_res['vram_used_mb'].mean():.2f} MB\n")

    # Generate Predictions Log
    df_preds = pd.DataFrame(results)
    df_preds.to_csv(os.path.join(RESULTS_DIR, "VLM_predictions.csv"), index=False)
    
    # Latency Report
    with open(os.path.join(RESULTS_DIR, "vlm_latency_report.md"), "w") as f:
        f.write("# VLM Latency Report\n\n")
        f.write(f"- Mean Latency: {np.mean(latencies):.2f} ms\n")
        f.write(f"- Median Latency: {np.median(latencies):.2f} ms\n")
        f.write(f"- P95 Latency: {np.percentile(latencies, 95):.2f} ms\n")
        f.write(f"- Maximum Latency: {np.max(latencies):.2f} ms\n")
        
    # Confidence Report
    with open(os.path.join(RESULTS_DIR, "vlm_confidence_report.md"), "w") as f:
        f.write("# VLM Confidence Report\n\n")
        f.write(f"- Mean Confidence: {np.mean(confidences):.4f}\n")
        f.write(f"- Correct Prediction Confidence Avg: {np.mean(correct_confidences) if correct_confidences else 0:.4f}\n")
        f.write(f"- Incorrect Prediction Confidence Avg: {np.mean(incorrect_confidences) if incorrect_confidences else 0:.4f}\n")
        
    # Error Analysis
    with open(os.path.join(RESULTS_DIR, "vlm_error_analysis.md"), "w") as f:
        f.write("# VLM Error Analysis\n\n")
        df_errors = df_preds[df_preds["correctness"] == 0]
        f.write(f"Total Errors: {len(df_errors)} ({len(df_errors)/len(df_preds)*100:.2f}%)\n\n")
        
        f.write("### Top Misclassifications\n")
        f.write("| True Label | Predicted Label | Count |\n")
        f.write("|---|---|---|\n")
        
        error_pairs = df_errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
        error_pairs = error_pairs.sort_values(by="count", ascending=False)
        
        for _, row in error_pairs.iterrows():
            t_str = REVERSE_MAPPING.get(row['true_label'], "UNKNOWN")
            p_str = REVERSE_MAPPING.get(row['predicted_label'], "UNKNOWN")
            f.write(f"| {t_str} | {p_str} | {row['count']} |\n")

    # Metrics
    y_true = df_preds["true_label"].values
    y_pred = df_preds["predicted_label"].values
    
    # Filter out unknowns for precision/recall strict mapping if needed, but we keep them to penalize accuracy
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average="macro", zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    df_cm = pd.DataFrame(cm, index=[REVERSE_MAPPING[i] for i in [0,1,2,3]], columns=[REVERSE_MAPPING[i] for i in [0,1,2,3]])
    df_cm.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_vlm.csv"))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("VLM Standalone Confusion Matrix")
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_vlm.png"))
    
    # Load LSTM metrics to compare
    lstm_metrics_path = os.path.join(WORKSPACE_DIR, "experiment_results/reports/evaluation/evaluation_metrics_FUSE0.csv")
    lstm_acc, lstm_p, lstm_r, lstm_f1 = "N/A", "N/A", "N/A", "N/A"
    if os.path.exists(lstm_metrics_path):
        df_lstm = pd.read_csv(lstm_metrics_path)
        lstm_acc = f"{df_lstm['Accuracy'].values[0]:.4f}"
        lstm_p = f"{df_lstm['Macro_Precision'].values[0]:.4f}"
        lstm_r = f"{df_lstm['Macro_Recall'].values[0]:.4f}"
        lstm_f1 = f"{df_lstm['Macro_F1'].values[0]:.4f}"
        
    with open(os.path.join(RESULTS_DIR, "phase2_comparison_report.md"), "w") as f:
        f.write("# Phase 2 Comparison Report: LSTM FUSE0 vs VLM\n\n")
        f.write("| Metric | LSTM FUSE0 | VLM (Standalone) |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Accuracy | {lstm_acc} | {acc:.4f} |\n")
        f.write(f"| Precision | {lstm_p} | {p:.4f} |\n")
        f.write(f"| Recall | {lstm_r} | {r:.4f} |\n")
        f.write(f"| F1 Score | {lstm_f1} | {f1:.4f} |\n")
        f.write(f"| Mean Latency | ~2.5 ms | {np.mean(latencies):.2f} ms |\n")
        
    print("Phase 2 Execution Complete! Artifacts saved to experiment_results/phase2_vlm/")

if __name__ == "__main__":
    main()
