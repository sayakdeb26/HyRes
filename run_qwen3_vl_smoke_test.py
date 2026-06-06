#!/usr/bin/env python3
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["VLM_BENCHMARK_MODE"] = "1"
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
    "STOP_SIGN": 3,
    "UNKNOWN": -1
}

CLASS_NAMES = ["SWIPE_LEFT", "SWIPE_RIGHT", "ROLL_FWD", "STOP_SIGN"]

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

def sample_frames(directory, k=5):
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

def parse_prediction(raw_output):
    s = raw_output.strip().lower()
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
    return "UNKNOWN"

def main():
    print("=== Qwen3-VL Balanced Smoke Test ===")
    
    # 1. Select Balanced Subset of 100 samples
    df = pd.read_csv(MANIFEST_PATH)
    df_val = df[df["assigned_split"] == "validation"]
    
    subset_list = []
    for label in [0, 1, 2, 3]:
        sub = df_val[df_val["new_label"] == label]
        sampled = sub.sample(n=25, random_state=42)
        subset_list.append(sampled)
        
    subset_df = pd.concat(subset_list).reset_index(drop=True)
    subset_df.to_csv(os.path.join(RESULTS_DIR, "phase2_qwen_smoke_subset.csv"), index=False)
    print(f"Created subset CSV with {len(subset_df)} balanced samples.")
    
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
    resource_stats = []
    
    prompt_text = (
        "The five images are consecutive frames from the same hand gesture video.\n"
        "Analyze the motion occurring across the sequence of frames.\n"
        "Focus on:\n"
        "- movement direction\n"
        "- temporal progression\n"
        "- hand trajectory\n"
        "- changes between consecutive frames\n\n"
        "Possible gestures:\n"
        "SWIPE_LEFT\n"
        "SWIPE_RIGHT\n"
        "ROLL_FWD\n"
        "STOP_SIGN\n\n"
        "Gesture descriptions:\n"
        "SWIPE_LEFT\n"
        "- Hand moves horizontally from right to left.\n\n"
        "SWIPE_RIGHT\n"
        "- Hand moves horizontally from left to right.\n\n"
        "ROLL_FWD\n"
        "- Hand performs a circular rolling motion forward.\n\n"
        "STOP_SIGN\n"
        "- Static open palm facing the camera.\n"
        "- Minimal motion across frames.\n\n"
        "Determine which gesture best matches the complete sequence.\n"
        "Respond with ONLY ONE label:\n"
        "SWIPE_LEFT\n"
        "SWIPE_RIGHT\n"
        "ROLL_FWD\n"
        "STOP_SIGN\n"
        "UNKNOWN\n\n"
        "Do not explain.\n"
        "Do not reason.\n"
        "Do not output any additional text."
    )
    
    for idx, row in subset_df.iterrows():
        video_id = str(row["video_id"])
        true_label = int(row["new_label"])
        true_name = row["gesture_name"]
        
        clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
        frame_idxs, frame_paths, pil_images = sample_frames(clip_dir, k=5)
        
        if not pil_images:
            print(f"[{idx+1}/100] Error: No frames found for video {video_id}")
            continue
            
        content_list = []
        for img in pil_images:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content_list}]
        
        # Resource stats before
        vram_start, gpu_start = get_gpu_stats()
        cpu_start = psutil.cpu_percent(interval=None)
        ram_start = psutil.virtual_memory().used / (1024 * 1024)
        
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
        
        # Resource stats after
        vram_end, gpu_end = get_gpu_stats()
        cpu_end = psutil.cpu_percent(interval=None)
        ram_end = psutil.virtual_memory().used / (1024 * 1024)
        
        resource_stats.append({
            "vram_peak": max(vram_start, vram_end),
            "vram_mean": (vram_start + vram_end) / 2.0,
            "gpu_peak": max(gpu_start, gpu_end),
            "gpu_mean": (gpu_start + gpu_end) / 2.0,
            "cpu_peak": max(cpu_start, cpu_end),
            "cpu_mean": (cpu_start + cpu_end) / 2.0,
            "ram_peak": max(ram_start, ram_end),
            "ram_mean": (ram_start + ram_end) / 2.0
        })
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        parsed = parse_prediction(raw_output)
        pred_label = VLM_TO_CLASS.get(parsed, -1)
        
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
    pred_df.to_csv(os.path.join(RESULTS_DIR, "qwen_smoke_predictions.csv"), index=False)
    
    # 5. Compute Metrics
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["predicted_label"].tolist()
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # UNKNOWN prediction count
    unknown_count = sum(1 for p in y_pred if p == -1)
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "qwen_smoke_confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Qwen3-VL Smoke Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qwen_smoke_confusion_matrix.png"))
    plt.close()
    
    # 7. Latency Analysis
    lat_mean = np.mean(latencies)
    lat_med = np.median(latencies)
    lat_p95 = np.percentile(latencies, 95)
    lat_max = np.max(latencies)
    lat_min = np.min(latencies)
    
    # 8. Resource Analysis
    res_df = pd.DataFrame(resource_stats)
    mean_gpu = res_df["gpu_mean"].mean()
    peak_gpu = res_df["gpu_peak"].max()
    mean_vram = res_df["vram_mean"].mean()
    peak_vram = res_df["vram_peak"].max()
    mean_cpu = res_df["cpu_mean"].mean()
    peak_cpu = res_df["cpu_peak"].max()
    
    # 9. Error Analysis
    confusions = []
    for idx, row in pred_df.iterrows():
        t = row["true_label"]
        p = row["predicted_label"]
        if t != p:
            t_name = CLASS_TO_VLM[t]
            p_name = CLASS_TO_VLM[p] if p in CLASS_TO_VLM else "UNKNOWN"
            confusions.append(f"{t_name} -> {p_name}")
            
    conf_series = pd.Series(confusions)
    conf_counts = conf_series.value_counts()
    
    # 10. Generate deliverables
    # 10.1 qwen_smoke_metrics.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_metrics.md"), "w") as f:
        f.write("# Qwen3-VL Smoke Test Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Macro Precision**: {precision_mac:.4f}\n")
        f.write(f"- **Macro Recall**: {recall_mac:.4f}\n")
        f.write(f"- **Macro F1**: {f1_mac:.4f}\n")
        f.write(f"- **UNKNOWN Predictions**: {unknown_count} / {len(pred_df)} ({unknown_count/len(pred_df)*100.1:.1f}%)\n")
        
    # 10.2 qwen_smoke_latency.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_latency.md"), "w") as f:
        f.write("# Qwen3-VL Latency Analysis\n\n")
        f.write(f"- **Mean Latency**: {lat_mean:.2f} ms\n")
        f.write(f"- **Median Latency**: {lat_med:.2f} ms\n")
        f.write(f"- **P95 Latency**: {lat_p95:.2f} ms\n")
        f.write(f"- **Maximum Latency**: {lat_max:.2f} ms\n")
        f.write(f"- **Minimum Latency**: {lat_min:.2f} ms\n")
        
    # 10.3 qwen_smoke_resources.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_resources.md"), "w") as f:
        f.write("# Qwen3-VL Resource Consumption Analysis\n\n")
        f.write(f"- **Mean GPU Utilization**: {mean_gpu:.1f}%\n")
        f.write(f"- **Peak GPU Utilization**: {peak_gpu:.1f}%\n")
        f.write(f"- **Mean VRAM Usage**: {mean_vram:.1f} MB\n")
        f.write(f"- **Peak VRAM Usage**: {peak_vram:.1f} MB\n")
        f.write(f"- **Mean CPU Usage**: {mean_cpu:.1f}%\n")
        f.write(f"- **Peak CPU Usage**: {peak_cpu:.1f}%\n")
        
    # 10.4 qwen_smoke_classification_report.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_classification_report.md"), "w") as f:
        f.write("# Qwen3-VL Smoke Test Classification Report\n\n")
        f.write("| Gesture Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"| {name} | {p_class[i]:.4f} | {r_class[i]:.4f} | {f1_class[i]:.4f} | {support_class[i]} |\n")
            
    # 10.5 qwen_smoke_error_analysis.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_error_analysis.md"), "w") as f:
        f.write("# Qwen3-VL Smoke Test Error Analysis\n\n")
        f.write("## Top Confusion Pairs\n\n")
        f.write("| Confusion Pair | Count | Percentage of Errors |\n")
        f.write("|---|---|---|\n")
        total_errors = len(confusions)
        for pair, count in conf_counts.items():
            pct = (count / total_errors) * 100.0 if total_errors > 0 else 0.0
            f.write(f"| {pair} | {count} | {pct:.1f}% |\n")
            
    # 10.6 qwen_smoke_vs_fastvlm.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_vs_fastvlm.md"), "w") as f:
        f.write("# Comparison: Qwen3-VL vs. FastVLM Smoke Test V2\n\n")
        f.write("| Metric | FastVLM-1.5B (Frame Voting) | Qwen3-VL-4B (Temporal Reasoning) | Change |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Accuracy** | 0.0300 | {accuracy:.4f} | {accuracy - 0.0300:+.4f} |\n")
        f.write(f"| **Macro F1** | 0.0517 | {f1_mac:.4f} | {f1_mac - 0.0517:+.4f} |\n")
        f.write(f"| **UNKNOWN Rate** | 95.0% | {unknown_count/len(pred_df)*100.0:.1f}% | {(unknown_count/len(pred_df) - 0.95)*100.0:+.1f}% |\n")
        f.write(f"| **Mean Latency (ms)** | 2325.15 | {lat_mean:.2f} | {lat_mean - 2325.15:+.2f} |\n")
        f.write(f"| **P95 Latency (ms)** | 2359.88 | {lat_p95:.2f} | {lat_p95 - 2359.88:+.2f} |\n")
        
    # Decision Logic:
    decision = "READY FOR FULL 2037-SAMPLE QWEN3-VL BENCHMARK" if accuracy >= 0.50 else "NOT READY FOR FULL BENCHMARK"
    
    # 10.7 qwen_smoke_test_report.md
    with open(os.path.join(RESULTS_DIR, "qwen_smoke_test_report.md"), "w") as f:
        f.write("# Qwen3-VL-4B-Instruct Balanced Smoke Test Report\n\n")
        f.write(f"- **Status**: Completed\n")
        f.write(f"- **Final Verdict**: **{decision}**\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write(f"- **Total Samples Processed**: {len(pred_df)}\n")
        f.write(f"- **Overall Accuracy**: {accuracy:.4f} (compared to 0.0300 for FastVLM)\n")
        f.write(f"- **Macro F1**: {f1_mac:.4f} (compared to 0.0517 for FastVLM)\n")
        f.write(f"- **UNKNOWN Rate**: {unknown_count/len(pred_df)*100.0:.1f}% (compared to 95.0% for FastVLM)\n\n")
        f.write("## Confusion Matrix Summary\n")
        f.write("```csv\n" + cm_df.to_csv() + "```\n\n")
        f.write("For more details, see the accompanying report files in this directory.\n")
        
    print(f"Balanced smoke test finished successfully. Verdict: {decision}")

if __name__ == "__main__":
    main()
