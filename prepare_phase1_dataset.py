#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    print("Starting Phase-1 Dataset Preparation...")
    
    # Paths
    workspace_dir = "/home/sayak/HybridTestBed"
    dataset_full_dir = os.path.join(workspace_dir, "DataSet_Full")
    train_src_dir = os.path.join(dataset_full_dir, "train")
    val_src_dir = os.path.join(dataset_full_dir, "val")
    
    annotations_dir = "/home/sayak/Downloads/CVND---Gesture-Recognition-master/20bn-jester-v1/annotations"
    train_csv_path = os.path.join(annotations_dir, "jester-v1-train.csv")
    val_csv_path = os.path.join(annotations_dir, "jester-v1-validation.csv")
    
    phase1_dir = os.path.join(dataset_full_dir, "phase1")
    
    # 4 target gestures and mapping
    gesture_mapping = {
        "Swiping Left": ("Swipe Left", 0),
        "Swiping Right": ("Swipe Right", 1),
        "Rolling Hand Forward": ("Rolling Hand Forward", 2),
        "Stop Sign": ("Stop Sign", 3)
    }
    
    target_original_gestures = list(gesture_mapping.keys())
    
    # Load and filter Train CSV
    print(f"Loading train CSV from {train_csv_path}...")
    df_train = pd.read_csv(train_csv_path, sep=";", names=["video_id", "gesture"])
    df_train_filtered = df_train[df_train["gesture"].isin(target_original_gestures)].copy()
    df_train_filtered["gesture_name"] = df_train_filtered["gesture"].map(lambda x: gesture_mapping[x][0])
    df_train_filtered["new_label"] = df_train_filtered["gesture"].map(lambda x: gesture_mapping[x][1])
    df_train_filtered["source_split"] = "train"
    
    # Load and filter Val CSV
    print(f"Loading val CSV from {val_csv_path}...")
    df_val = pd.read_csv(val_csv_path, sep=";", names=["video_id", "gesture"])
    df_val_filtered = df_val[df_val["gesture"].isin(target_original_gestures)].copy()
    df_val_filtered["gesture_name"] = df_val_filtered["gesture"].map(lambda x: gesture_mapping[x][0])
    df_val_filtered["new_label"] = df_val_filtered["gesture"].map(lambda x: gesture_mapping[x][1])
    df_val_filtered["source_split"] = "val"
    df_val_filtered["assigned_split"] = "validation"
    
    # Perform deterministic stratified splitting on Train using seed 42
    print("Splitting train set into 70/10/10/10...")
    # 70% train, 30% rest
    df_train_70, df_rest = train_test_split(
        df_train_filtered,
        test_size=0.30,
        random_state=42,
        stratify=df_train_filtered["new_label"]
    )
    
    # From 30% rest, split 1/3 (10% of total) for inc10_a and 2/3 (20% of total) for rest2
    df_inc10_a, df_rest2 = train_test_split(
        df_rest,
        test_size=2.0/3.0,
        random_state=42,
        stratify=df_rest["new_label"]
    )
    
    # From 20% rest2, split 50% for inc10_b and 50% for inc10_c
    df_inc10_b, df_inc10_c = train_test_split(
        df_rest2,
        test_size=0.5,
        random_state=42,
        stratify=df_rest2["new_label"]
    )
    
    # Assign splits
    df_train_70 = df_train_70.copy()
    df_train_70["assigned_split"] = "train_70"
    
    df_inc10_a = df_inc10_a.copy()
    df_inc10_a["assigned_split"] = "inc10_a"
    
    df_inc10_b = df_inc10_b.copy()
    df_inc10_b["assigned_split"] = "inc10_b"
    
    df_inc10_c = df_inc10_c.copy()
    df_inc10_c["assigned_split"] = "inc10_c"
    
    # Combine all into a single manifest
    df_manifest = pd.concat([df_train_70, df_inc10_a, df_inc10_b, df_inc10_c, df_val_filtered], ignore_index=True)
    # Reorder columns
    df_manifest = df_manifest[["video_id", "gesture_name", "new_label", "source_split", "assigned_split"]]
    
    # Save manifest
    manifest_path = os.path.join(workspace_dir, "dataset_manifest_phase1.csv")
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")
    
    # Generate split report
    split_report_data = []
    splits = ["train_70", "inc10_a", "inc10_b", "inc10_c", "validation"]
    gestures_ordered = ["Swipe Left", "Swipe Right", "Rolling Hand Forward", "Stop Sign"]
    
    for split in splits:
        df_split = df_manifest[df_manifest["assigned_split"] == split]
        for gesture in gestures_ordered:
            count = len(df_split[df_split["gesture_name"] == gesture])
            split_report_data.append({
                "split": split,
                "gesture": gesture,
                "count": count
            })
            
    df_split_report = pd.DataFrame(split_report_data)
    split_report_path = os.path.join(workspace_dir, "split_report.csv")
    df_split_report.to_csv(split_report_path, index=False)
    print(f"Saved split report to {split_report_path}")
    
    # Create directory structure and symlink folders
    print("Creating target directories and setting up symlinks...")
    for split in splits:
        split_dir = os.path.join(phase1_dir, split)
        if os.path.exists(split_dir):
            print(f"Removing existing directory {split_dir}...")
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        
    # Symlinking videos
    missing_folders = []
    created_links_count = 0
    
    for idx, row in df_manifest.iterrows():
        video_id = str(row["video_id"])
        source_split = row["source_split"]
        assigned_split = row["assigned_split"]
        
        # Source directory
        src_parent = train_src_dir if source_split == "train" else val_src_dir
        src_video_dir = os.path.join(src_parent, video_id)
        
        # Target directory
        dst_video_dir = os.path.join(phase1_dir, assigned_split, video_id)
        
        # Check source existence
        if not os.path.exists(src_video_dir):
            missing_folders.append((video_id, src_video_dir))
            continue
            
        # Create symbolic link
        os.symlink(src_video_dir, dst_video_dir)
        created_links_count += 1
        
    print(f"Created {created_links_count} symbolic links under {phase1_dir}")
    if missing_folders:
        print(f"WARNING: {len(missing_folders)} source folders were missing!")
        for vid, p in missing_folders[:5]:
            print(f"  Missing: {vid} at {p}")
            
    # Perform Integrity Checks
    print("Running Integrity Checks...")
    s_train_70 = set(df_manifest[df_manifest["assigned_split"] == "train_70"]["video_id"])
    s_inc10_a = set(df_manifest[df_manifest["assigned_split"] == "inc10_a"]["video_id"])
    s_inc10_b = set(df_manifest[df_manifest["assigned_split"] == "inc10_b"]["video_id"])
    s_inc10_c = set(df_manifest[df_manifest["assigned_split"] == "inc10_c"]["video_id"])
    s_validation = set(df_manifest[df_manifest["assigned_split"] == "validation"]["video_id"])
    
    # Overlap checks
    overlaps = {
        "train_70 vs inc10_a": s_train_70.intersection(s_inc10_a),
        "train_70 vs inc10_b": s_train_70.intersection(s_inc10_b),
        "train_70 vs inc10_c": s_train_70.intersection(s_inc10_c),
        "train_70 vs validation": s_train_70.intersection(s_validation),
        "inc10_a vs inc10_b": s_inc10_a.intersection(s_inc10_b),
        "inc10_b vs inc10_c": s_inc10_b.intersection(s_inc10_c),
        "inc10_a vs inc10_c": s_inc10_a.intersection(s_inc10_c),
        "inc10_a vs validation": s_inc10_a.intersection(s_validation),
        "inc10_b vs validation": s_inc10_b.intersection(s_validation),
        "inc10_c vs validation": s_inc10_c.intersection(s_validation)
    }
    
    overlap_detected = False
    for pair, intersection in overlaps.items():
        if len(intersection) > 0:
            print(f"CRITICAL ERROR: Overlap detected in {pair}! Size: {len(intersection)}")
            overlap_detected = True
            
    # Verification of created symlinks
    actual_symlinks_count = 0
    broken_symlinks = []
    for split in splits:
        split_dir = os.path.join(phase1_dir, split)
        items = os.listdir(split_dir)
        for item in items:
            item_path = os.path.join(split_dir, item)
            if os.path.islink(item_path):
                actual_symlinks_count += 1
                if not os.path.exists(item_path):
                    broken_symlinks.append(item_path)
                    
    print(f"Verified symlinks: Found {actual_symlinks_count} symlinks.")
    if broken_symlinks:
        print(f"CRITICAL ERROR: {len(broken_symlinks)} broken symlinks found!")
        
    # Write dataset integrity report
    integrity_report_path = os.path.join(workspace_dir, "dataset_integrity_report.txt")
    with open(integrity_report_path, "w") as f:
        f.write("=========================================================\n")
        f.write("Phase-1 Dataset Integrity Verification Audit\n")
        f.write("=========================================================\n\n")
        f.write(f"Total Source Videos Found & Symlinked: {created_links_count}\n")
        f.write(f"Total Missing Folders: {len(missing_folders)}\n")
        f.write(f"Total Symlinks Verified in Folders: {actual_symlinks_count}\n")
        f.write(f"Broken Symlinks Detected: {len(broken_symlinks)}\n\n")
        
        f.write("Split Sizes (Video Counts):\n")
        f.write(f"  train_70:   {len(s_train_70)}\n")
        f.write(f"  inc10_a:    {len(s_inc10_a)}\n")
        f.write(f"  inc10_b:    {len(s_inc10_b)}\n")
        f.write(f"  inc10_c:    {len(s_inc10_c)}\n")
        f.write(f"  validation: {len(s_validation)}\n")
        f.write(f"  Total:      {len(df_manifest)}\n\n")
        
        f.write("Overlap Verification:\n")
        for pair, intersection in overlaps.items():
            f.write(f"  {pair}: {'FAILED' if len(intersection) > 0 else 'PASSED (0 overlap)'}\n")
            
        f.write("\nOverall Integrity Status: ")
        if overlap_detected or len(broken_symlinks) > 0 or len(missing_folders) > 0:
            f.write("FAIL\n")
        else:
            f.write("PASS\n")
            
    print(f"Saved integrity report to {integrity_report_path}")
    
    # Write summary markdown report
    summary_md_path = os.path.join(workspace_dir, "phase1_dataset_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("# Phase-1 Dataset Summary Report\n\n")
        f.write("This report provides a summary of the curated Phase-1 dataset for the gesture recognition pipeline.\n\n")
        
        f.write("## Target Gestures & Label Mapping\n")
        f.write("| Original Gesture Name | Mapped Name | Label |\n")
        f.write("| --- | --- | --- |\n")
        for orig, (mapped, val) in gesture_mapping.items():
            f.write(f"| {orig} | {mapped} | {val} |\n")
        f.write("\n")
        
        f.write("## Split Distributions\n")
        f.write("| Split | Swipe Left | Swipe Right | Rolling Hand Forward | Stop Sign | Total |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for split in splits:
            df_split = df_manifest[df_manifest["assigned_split"] == split]
            c_left = len(df_split[df_split["gesture_name"] == "Swipe Left"])
            c_right = len(df_split[df_split["gesture_name"] == "Swipe Right"])
            c_roll = len(df_split[df_split["gesture_name"] == "Rolling Hand Forward"])
            c_stop = len(df_split[df_split["gesture_name"] == "Stop Sign"])
            tot = len(df_split)
            f.write(f"| {split} | {c_left} | {c_right} | {c_roll} | {c_stop} | {tot} |\n")
        
        total_left = len(df_manifest[df_manifest["gesture_name"] == "Swipe Left"])
        total_right = len(df_manifest[df_manifest["gesture_name"] == "Swipe Right"])
        total_roll = len(df_manifest[df_manifest["gesture_name"] == "Rolling Hand Forward"])
        total_stop = len(df_manifest[df_manifest["gesture_name"] == "Stop Sign"])
        total_tot = len(df_manifest)
        f.write(f"| **Total** | **{total_left}** | **{total_right}** | **{total_roll}** | **{total_stop}** | **{total_tot}** |\n\n")
        
        f.write("## Integrity Verification Status\n")
        f.write("- **Overlap Check**: " + ("FAILED" if overlap_detected else "PASSED (No overlapping video IDs across splits)") + "\n")
        f.write("- **Link Verification Check**: " + ("FAILED" if len(broken_symlinks) > 0 else "PASSED (All symbolic links resolved successfully)") + "\n")
        f.write("- **Overall Status**: " + ("FAIL" if (overlap_detected or len(broken_symlinks) > 0 or len(missing_folders) > 0) else "PASS") + "\n\n")
        
        f.write("## File Artifacts Generated\n")
        f.write("- **Dataset Manifest**: [dataset_manifest_phase1.csv](dataset_manifest_phase1.csv)\n")
        f.write("- **Split Class Distribution Report**: [split_report.csv](split_report.csv)\n")
        f.write("- **Detailed Integrity Audit**: [dataset_integrity_report.txt](dataset_integrity_report.txt)\n\n")
        
        f.write("---  \n*Note: Video folders under `DataSet_Full/phase1` are symlinked to conserve disk space.*  \n")
        
    print(f"Saved summary report to {summary_md_path}")
    print("Phase-1 Dataset Preparation Complete successfully!")

if __name__ == "__main__":
    main()
