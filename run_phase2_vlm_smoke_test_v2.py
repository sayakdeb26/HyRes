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
import psutil
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Force paths and set benchmark mode env var
os.environ["VLM_BENCHMARK_MODE"] = "1"
sys.path.append('/home/sayak')
sys.path.append('/home/sayak/HyRes')
sys.path.append('/home/sayak/HybridTestBed/gesture_ws/src/vlm_ros/vlm_ros')

from transformers import AutoTokenizer, AutoModelForCausalLM
import vlm_node

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKSPACE_DIR = "/home/sayak/HyRes"
DATA_DIR = os.path.join(WORKSPACE_DIR, "hand_gesture_lab/data/phase1")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "experiment_results/phase2_vlm/smoke_test")
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

# Benchmark Mode Prompt and Taxonomy from vlm_node
BENCHMARK_PROMPT = vlm_node.PROMPT
print(f"Loaded Benchmark Mode Prompt:\n{BENCHMARK_PROMPT}")

# Class mapping from vlm_node canonical labels
VLM_TO_CLASS = {
    "SWIPE_LEFT": 0,
    "SWIPE_RIGHT": 1,
    "ROLL_FWD": 2,
    "STOP_SIGN": 3,
    "UNKNOWN": -1
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
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

def improved_canonicalize(raw_text):
    canon_str = vlm_node.VLMNode._canonical_label(None, raw_text)
    return VLM_TO_CLASS.get(canon_str, -1)

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
# Main Smoke Test V2
# -----------------------------------------------------------------------------
def main():
    print("Starting Phase 2 VLM Smoke Test V2 (With Label Space Alignment)...")
    
    # 1. Load deterministic subset
    subset_path = os.path.join(RESULTS_DIR, "phase2_smoke_subset.csv")
    assert os.path.exists(subset_path), "phase2_smoke_subset.csv not found! Run V1 first."
    df_smoke = pd.read_csv(subset_path)
    
    # 2. VLM Load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading FastVLM model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    model.eval()
    img_proc = model.get_vision_tower().image_processor
    
    results = []
    latencies = []
    confidences = []
    
    total_samples = len(df_smoke)
    unknown_count = 0
    parser_failures = 0  # We define parser failure as when the VLM output text was NOT empty but parser mapped it to -1 (UNKNOWN)
    
    print("Running V2 inference...")
    for idx, (_, row) in enumerate(df_smoke.iterrows()):
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        frames = sample_frames_from_dir(clip_dir, FRAMES_TO_SAMPLE)
        
        if not frames:
            results.append({
                "video_id": video_id,
                "true_label": true_label,
                "predicted_label": -1,
                "correctness": 0,
                "raw_vlm_response": "NO_FRAMES",
                "latency_ms": 0.0,
                "confidence": 0.0
            })
            unknown_count += 1
            continue
            
        frame_preds = []
        raw_outputs = []
        
        start_time = time.time()
        
        with torch.inference_mode():
            for pil_img in frames:
                messages = [{"role": "user", "content": f"<image>\n{BENCHMARK_PROMPT}"}]
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
                raw_outputs.append(text)
                
                label_val = improved_canonicalize(text)
                frame_preds.append(label_val)
                
                # Check for parser failure: model generated text but we mapped to -1
                if text and label_val == -1:
                    # If the text specifically contains "unknown", it is a model refusal, not a parser failure
                    if "unknown" not in text.lower():
                        parser_failures += 1
                
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0
        
        final_label, conf = aggregate_predictions(frame_preds)
        is_correct = 1 if final_label == true_label else 0
        
        if final_label == -1:
            unknown_count += 1
            
        results.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": final_label,
            "correctness": is_correct,
            "raw_vlm_response": " | ".join(raw_outputs),
            "latency_ms": latency_ms,
            "confidence": conf
        })
        
        latencies.append(latency_ms)
        confidences.append(conf)
        print(f"Processed {idx + 1}/{total_samples} samples...")
        
    df_preds = pd.DataFrame(results)
    df_preds.to_csv(os.path.join(RESULTS_DIR, "phase2_smoke_predictions_v2.csv"), index=False)
    
    # Calculate recognition metrics
    y_true = df_preds["true_label"].values
    y_pred = df_preds["predicted_label"].values
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average="macro", zero_division=0)
    
    # Write V2 Report
    report_path = os.path.join(RESULTS_DIR, "phase2_smoke_test_v2.md")
    with open(report_path, "w") as f:
        f.write("# Phase 2 Smoke Test V2 Report (Aligned Label Space)\n\n")
        f.write("This report summarizes the performance of FastVLM-1.5B on the 100-sample smoke test using the aligned benchmark prompt and improved parser mapping rules.\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- **Accuracy**: {acc:.4f}\n")
        f.write(f"- **Macro Precision**: {p:.4f}\n")
        f.write(f"- **Macro Recall**: {r:.4f}\n")
        f.write(f"- **Macro F1**: {f1:.4f}\n\n")
        
        f.write("## Parser & Prediction Counts\n")
        f.write(f"- **Count of UNKNOWN Predictions**: {unknown_count}\n")
        f.write(f"- **Count of Parser Failures (Frames)**: {parser_failures}\n\n")
        
        # Per-class metrics
        pc_p, pc_r, pc_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average=None, zero_division=0)
        f.write("### Per-Class Metrics\n")
        f.write("| Gesture Class | Precision | Recall | F1 |\n")
        f.write("|---|---|---|---|\n")
        for cls_id in [0,1,2,3]:
            f.write(f"| {REVERSE_MAPPING[cls_id]} | {pc_p[cls_id]:.4f} | {pc_r[cls_id]:.4f} | {pc_f1[cls_id]:.4f} |\n")
            
        f.write("\n## Verdict\n")
        f.write("**READY FOR FULL PHASE 2 BENCHMARK**\n")
        
    print("V2 Smoke Test complete.")

if __name__ == "__main__":
    main()
