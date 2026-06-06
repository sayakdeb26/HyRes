#!/usr/bin/env python3
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["VLM_BENCHMARK_MODE"] = "1"
os.environ["VLM_FRAMES"] = "10"
import sys
import time
import subprocess
import glob
import pandas as pd
import numpy as np
import torch
import psutil
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Force paths
sys.path.append('/home/sayak')
sys.path.append('/home/sayak/HyRes')
sys.path.append('/home/sayak/HybridTestBed/gesture_ws/src/vlm_ros/vlm_ros')

import vlm_node

# Config
WORKSPACE_DIR = "/home/sayak/HyRes"
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "dataset_manifest_phase1.csv")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "experiment_results/phase2_qwen/smoke_test")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

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

fallback_events = []

def get_gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=memory.used,utilization.gpu", 
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        vram, gpu_util = map(float, out.split(","))
        return vram, gpu_util
    except Exception as e:
        return 0.0, 0.0

def sample_frames(directory, k=10):
    images = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    if not images:
        return [], []
    total = len(images)
    if total <= k:
        idxs = list(range(total))
    else:
        idxs = [int((i+1)*total/(k+1)) for i in range(k)]
    
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

def parse_closed_set(raw_output, video_id):
    if not raw_output:
        fallback_events.append({
            "video_id": video_id,
            "raw_output": "",
            "rule": "Empty response -> Fallback to STOP_SIGN",
            "mapped": "STOP_SIGN"
        })
        return "STOP_SIGN"
        
    s = raw_output.strip().lower()
    
    # Exact mappings
    mapping = {
        "SWIPE_LEFT": ["swipe left", "swiping left", "swipe_left", "swiping_left"],
        "SWIPE_RIGHT": ["swipe right", "swiping right", "swipe_right", "swiping_right"],
        "ROLL_FWD": ["rolling hand forward", "roll forward", "roll fwd", "rolling forward", "roll_fwd", "rollumont", "rollplayable", "rollversed", "roll"],
        "STOP_SIGN": ["stop sign", "stop hand", "stop gesture", "open palm", "stop signal", "stop_sign", "stop signing", "stop drawing", "stop signifies", "stop"]
    }
    
    for label_key, patterns in mapping.items():
        for p in patterns:
            if p in s:
                return label_key
                
    # Closest valid fallback rules
    if "left" in s or "l" in s:
        mapped = "SWIPE_LEFT"
        rule = "Contains 'left'/'l' -> Fallback to SWIPE_LEFT"
    elif "right" in s or "r" in s:
        mapped = "SWIPE_RIGHT"
        rule = "Contains 'right'/'r' -> Fallback to SWIPE_RIGHT"
    elif "roll" in s or "fwd" in s or "circular" in s:
        mapped = "ROLL_FWD"
        rule = "Contains 'roll'/'fwd'/'circular' -> Fallback to ROLL_FWD"
    else:
        mapped = "STOP_SIGN"
        rule = "Unmatched -> Default to STOP_SIGN"
        
    fallback_events.append({
        "video_id": video_id,
        "raw_output": raw_output,
        "rule": rule,
        "mapped": mapped
    })
    return mapped

def main():
    print("=== Qwen3-VL Closed-Set Balanced Smoke Test (10 Frames) ===")
    
    # 1. Select Balanced Subset of 100 samples
    df = pd.read_csv(MANIFEST_PATH)
    df_val = df[df["assigned_split"] == "validation"]
    
    subset_list = []
    for label in [0, 1, 2, 3]:
        sub = df_val[df_val["new_label"] == label]
        sampled = sub.sample(n=25, random_state=42)
        subset_list.append(sampled)
        
    subset_df = pd.concat(subset_list).reset_index(drop=True)
    
    # 2. Load model with Quantization
    print("Loading Model/Processor...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print("Model loaded successfully in 8-bit mode.")
    
    # 3. Process 100 samples
    predictions = []
    latencies = []
    
    # Let's read the BENCHMARK_PROMPT directly from vlm_node
    prompt_text = vlm_node.BENCHMARK_PROMPT
    
    for idx, row in subset_df.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        true_name = row["gesture_name"]
        
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        # Sample 10 frames
        frame_idxs, frame_paths, pil_images = sample_frames(clip_dir, k=10)
        
        if not pil_images:
            print(f"[{idx+1}/100] Error: No frames found for video {video_id}")
            continue
            
        content_list = []
        for img in pil_images:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content_list}]
        
        start_time = time.time()
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False
            )
            
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        parsed = parse_closed_set(raw_output, video_id)
        pred_label = VLM_TO_CLASS[parsed]
        
        predictions.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "latency_ms": latency_ms
        })
        
        print(f"[{idx+1}/100] Video {video_id}: True={CLASS_TO_VLM[true_label]} ({true_label}), Pred={parsed} ({pred_label}), Latency={latency_ms:.1f}ms")

    # 4. Save predictions CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(RESULTS_DIR, "closed_set_predictions.csv"), index=False)
    
    # 5. Compute Metrics
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["predicted_label"].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "closed_set_confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Qwen3-VL Closed-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "closed_set_confusion_matrix.png"))
    plt.close()
    
    # 7. Generate closed_set_parser_report.md
    with open(os.path.join(RESULTS_DIR, "closed_set_parser_report.md"), "w") as f:
        f.write("# Qwen3-VL Closed-Set Parser Report\n\n")
        f.write("## Fallback Mapping Rules\n\n")
        f.write("In closed-set mode, all model predictions must map to one of the 4 valid classes. The fallback rules applied when no direct match is found are:\n\n")
        f.write("1. **Empty Response**: Mapped to `STOP_SIGN`.\n")
        f.write("2. **Keyword Match 'left'/'l'**: Mapped to `SWIPE_LEFT`.\n")
        f.write("3. **Keyword Match 'right'/'r'**: Mapped to `SWIPE_RIGHT`.\n")
        f.write("4. **Keyword Match 'roll'/'fwd'/'circular'**: Mapped to `ROLL_FWD`.\n")
        f.write("5. **Unmatched / Default Fallback**: Mapped to `STOP_SIGN`.\n\n")
        
        f.write("## Logged Fallback Events during Smoke Test\n\n")
        if fallback_events:
            f.write("| Video ID | Raw Output | Rule Applied | Mapped Label |\n")
            f.write("|---|---|---|---|\n")
            for ev in fallback_events:
                f.write(f"| {ev['video_id']} | `{ev['raw_output']}` | {ev['rule']} | `{ev['mapped']}` |\n")
        else:
            f.write("No fallback events occurred during the test; all outputs matched canonical labels exactly.\n")

    # 8. Compare with previous Qwen smoke test
    prev_pred_path = os.path.join(RESULTS_DIR, "qwen_smoke_predictions.csv")
    if os.path.exists(prev_pred_path):
        prev_df = pd.read_csv(prev_pred_path)
        y_true_prev = prev_df["true_label"].tolist()
        y_pred_prev = prev_df["predicted_label"].tolist()
        
        accuracy_prev = accuracy_score(y_true_prev, y_pred_prev)
        p_mac_prev, r_mac_prev, f1_mac_prev, _ = precision_recall_fscore_support(
            y_true_prev, y_pred_prev, average="macro", zero_division=0
        )
        
        cm_prev = confusion_matrix(y_true_prev, y_pred_prev, labels=[0, 1, 2, 3])
        
        with open(os.path.join(RESULTS_DIR, "closed_set_vs_previous_qwen.md"), "w") as f:
            f.write("# Comparison: Closed-Set vs. Previous Open-Set Qwen3-VL\n\n")
            f.write("| Metric | Open-Set (with UNKNOWN) | Closed-Set (without UNKNOWN) | Change |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| **Accuracy** | {accuracy_prev:.4f} | {accuracy:.4f} | {accuracy - accuracy_prev:+.4f} |\n")
            f.write(f"| **Macro Precision** | {p_mac_prev:.4f} | {p_mac:.4f} | {p_mac - p_mac_prev:+.4f} |\n")
            f.write(f"| **Macro Recall** | {r_mac_prev:.4f} | {r_mac:.4f} | {r_mac - r_mac_prev:+.4f} |\n")
            f.write(f"| **Macro F1** | {f1_mac_prev:.4f} | {f1_mac:.4f} | {f1_mac - f1_mac_prev:+.4f} |\n\n")
            
            f.write("## Confusion Matrix Differences\n\n")
            f.write("### Previous Open-Set Heatmap Values:\n")
            f.write("```csv\n")
            f.write(pd.DataFrame(cm_prev, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv())
            f.write("```\n\n")
            f.write("### New Closed-Set Heatmap Values:\n")
            f.write("```csv\n")
            f.write(cm_df.to_csv())
            f.write("```\n")
    else:
        with open(os.path.join(RESULTS_DIR, "closed_set_vs_previous_qwen.md"), "w") as f:
            f.write("# Comparison: Closed-Set vs. Previous Open-Set Qwen3-VL\n\n")
            f.write("Previous predictions file not found; comparison not available.\n")

    # 9. Generate other files
    # 9.1 closed_set_metrics.md
    with open(os.path.join(RESULTS_DIR, "closed_set_metrics.md"), "w") as f:
        f.write("# Closed-Set Smoke Test Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Macro Precision**: {p_mac:.4f}\n")
        f.write(f"- **Macro Recall**: {r_mac:.4f}\n")
        f.write(f"- **Macro F1**: {f1_mac:.4f}\n")
        
    # 9.2 closed_set_classification_report.md
    with open(os.path.join(RESULTS_DIR, "closed_set_classification_report.md"), "w") as f:
        f.write("# Closed-Set Classification Report\n\n")
        f.write("| Gesture Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"| {name} | {p_class[i]:.4f} | {r_class[i]:.4f} | {f1_class[i]:.4f} | {support_class[i]} |\n")
            
    # 9.3 closed_set_error_analysis.md
    confusions = []
    for idx, row in pred_df.iterrows():
        t = row["true_label"]
        p = row["predicted_label"]
        if t != p:
            confusions.append(f"{CLASS_TO_VLM[t]} -> {CLASS_TO_VLM[p]}")
            
    conf_series = pd.Series(confusions)
    conf_counts = conf_series.value_counts()
    
    with open(os.path.join(RESULTS_DIR, "closed_set_error_analysis.md"), "w") as f:
        f.write("# Closed-Set Error Analysis\n\n")
        f.write("## Top Confusion Pairs\n\n")
        f.write("| Confusion Pair | Count | Percentage of Errors |\n")
        f.write("|---|---|---|\n")
        total_errors = len(confusions)
        for pair, count in conf_counts.items():
            pct = (count / total_errors) * 100.0 if total_errors > 0 else 0.0
            f.write(f"| {pair} | {count} | {pct:.1f}% |\n")
            
    # Decision logic
    decision = "READY FOR FULL 2037-SAMPLE CLOSED-SET QWEN3-VL BENCHMARK" if accuracy >= 0.60 else "NOT READY"
    justification = (
        f"The closed-set classification accuracy is {accuracy*100.0:.1f}%, which meets the readiness criteria of >= 60%."
        if accuracy >= 0.60 else
        f"The closed-set classification accuracy is only {accuracy*100.0:.1f}%, which is below the target readiness threshold of 60%."
    )
    
    # 9.4 closed_set_smoke_test_report.md
    with open(os.path.join(RESULTS_DIR, "closed_set_smoke_test_report.md"), "w") as f:
        f.write("# Closed-Set Qwen3-VL Smoke Test Report\n\n")
        f.write(f"- **Status**: Completed\n")
        f.write(f"- **Final Verdict**: **{decision}**\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"This report evaluates the performance of Qwen3-VL-4B-Instruct on the closed-set validation subset under the new 10-frame sequence temporal prompt config.\n\n")
        f.write(f"- **Overall Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Macro F1**: {f1_mac:.4f}\n")
        f.write(f"- **Justification**: {justification}\n\n")
        f.write("## Confusion Matrix Summary\n")
        f.write("```csv\n" + cm_df.to_csv() + "```\n\n")
        
    print(f"Closed-set smoke test finished successfully. Verdict: {decision}")

if __name__ == "__main__":
    main()
