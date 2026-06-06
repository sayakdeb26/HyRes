#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

LABELS = ["Swipe Left", "Swipe Right", "Rolling Forward", "Rolling Backward", "Thumb Down", "Stop Sign"]

def generate_matrix(true_labels, pred_labels, title, output_prefix):
    # Standard classes plus possible extra prediction outputs
    unique_true = LABELS
    extra_preds = ["UNCERTAIN", "TIMEOUT", "REJECTED"]
    
    all_possible_preds = LABELS + extra_preds
    
    # Filter to only predictions that actually occurred in the dataset
    present_preds = [l for l in all_possible_preds if l in pred_labels or l in true_labels]
    if not present_preds:
        present_preds = unique_true
        
    cm = confusion_matrix(true_labels, pred_labels, labels=all_possible_preds)
    
    # Slice rows corresponding to true classes, and columns corresponding to present predictions
    row_indices = [all_possible_preds.index(l) for l in unique_true]
    col_indices = [all_possible_preds.index(l) for l in present_preds]
    
    cm_sliced = cm[row_indices, :][:, col_indices]
    
    df_cm = pd.DataFrame(cm_sliced, index=unique_true, columns=present_preds)
    
    # Save CSV
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    df_cm.to_csv(output_prefix + ".csv")
    
    # Plot PNG
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_prefix + ".png", dpi=300)
    plt.close()
    print(f"Generated {output_prefix}.csv and {output_prefix}.png")

def main():
    log_path = '/home/sayak/HyRes/experiment_results/escalation/hybrid_escalation_log.csv'
    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}. Run experiments first to generate logs.")
        return
        
    df = pd.read_csv(log_path)
    if df.empty:
        print("Log file is empty.")
        return
        
    # Map predictions/true labels to string to avoid comparison issues
    df['true_label'] = df['true_label'].astype(str)
    df['lstm_prediction'] = df['lstm_prediction'].astype(str)
    df['final_prediction'] = df['final_prediction'].astype(str)
    
    out_dir = '/home/sayak/HyRes/experiment_results/confusion_matrices'
    
    # Generate LSTM Confusion Matrix
    generate_matrix(
        df['true_label'].tolist(),
        df['lstm_prediction'].tolist(),
        "LSTM Model Confusion Matrix (Including UNCERTAIN)",
        os.path.join(out_dir, "lstm_confusion_matrix")
    )
    
    # Generate Hybrid Confusion Matrix
    generate_matrix(
        df['true_label'].tolist(),
        df['final_prediction'].tolist(),
        "Hybrid LSTM+VLM Confusion Matrix",
        os.path.join(out_dir, "hybrid_confusion_matrix")
    )

if __name__ == '__main__':
    main()
