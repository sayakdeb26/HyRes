import os
import glob
import shutil
import pandas as pd
from PIL import Image

WORKSPACE_DIR = "/home/sayak/HyRes"
RESULTS_DIR = "/home/sayak/HybridTestBed/experiment_results/phase2_qwen/smoke_test"

pred_csv_path = os.path.join(RESULTS_DIR, "closed_set_predictions.csv")
df = pd.read_csv(pred_csv_path)

# 1. Create copied_videos folder
copied_videos_dir = os.path.join(RESULTS_DIR, "copied_videos")
os.makedirs(copied_videos_dir, exist_ok=True)

# 2. Make annotations.csv
CLASS_TO_VLM = {
    0: "SWIPE_LEFT",
    1: "SWIPE_RIGHT",
    2: "ROLL_FWD",
    3: "STOP_SIGN"
}
annotations_df = pd.DataFrame({
    "video_id": df["video_id"],
    "actual_label": df["true_label"].map(CLASS_TO_VLM),
    "predicted_label": df["parsed_output"]
})
annotations_df.to_csv(os.path.join(copied_videos_dir, "annotations.csv"), index=False)
print("Saved annotations.csv")

# 3. Copy all video folders
for idx, row in df.iterrows():
    video_id = str(row["video_id"])
    clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
    dest_dir = os.path.join(copied_videos_dir, video_id)
    shutil.copytree(clip_dir, dest_dir, dirs_exist_ok=True)

print("Copied all video frame directories.")

# 4. Generate contact sheets for 20 failed SWIPE_RIGHT samples
# SWIPE_RIGHT is 1. Failed means true_label == 1 and predicted_label != 1.
failed_swipe_right = df[(df["true_label"] == 1) & (df["predicted_label"] != 1)]
print(f"Found {len(failed_swipe_right)} failed SWIPE_RIGHT samples in 4B.")

failed_dir = os.path.join(RESULTS_DIR, "failed_swipe_right")
os.makedirs(failed_dir, exist_ok=True)

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
    for idx in idxs:
        try:
            path = images[idx]
            img = Image.open(path).convert("RGB")
            out_images.append(img)
        except Exception as e:
            pass
    return idxs, out_images

def make_contact_sheet(pil_images, save_path):
    w, h = 160, 120
    resized_imgs = [img.resize((w, h)) for img in pil_images]
    grid_w = 5 * w
    grid_h = 2 * h
    contact_sheet = Image.new("RGB", (grid_w, grid_h))
    for idx, img in enumerate(resized_imgs):
        col = idx % 5
        row = idx // 5
        contact_sheet.paste(img, (col * w, row * h))
    contact_sheet.save(save_path)

failures_info = []
for count, (idx, row) in enumerate(failed_swipe_right.head(20).iterrows()):
    video_id = str(row["video_id"])
    parsed_out = row["parsed_output"]
    raw_out = row["raw_output"]
    
    clip_dir = os.path.join(WORKSPACE_DIR, "DataSet_Full", "phase1", "validation", video_id)
    idxs, pil_imgs = sample_frames(clip_dir, k=10)
    
    contact_sheet_path = os.path.join(failed_dir, f"contact_sheet_{video_id}.png")
    make_contact_sheet(pil_imgs, contact_sheet_path)
    
    failures_info.append({
        "video_id": video_id,
        "predicted": parsed_out,
        "raw_output": raw_out,
        "sampled_indices": str(idxs),
        "contact_sheet_rel_path": f"failed_swipe_right/contact_sheet_{video_id}.png"
    })

# Write failed_swipe_right_inspection.md
with open(os.path.join(RESULTS_DIR, "failed_swipe_right_inspection.md"), "w") as f:
    f.write("# Visual Inspection Report: 20 Failed SWIPE_RIGHT Samples (4B Model)\n\n")
    f.write("This report documents the frame sequences and predictions for up to 20 failed SWIPE_RIGHT samples to analyze directional and motion confusion.\n\n")
    f.write("| Video ID | True Label | Predicted Label | Raw Output | Sampled Indices | Contact Sheet |\n")
    f.write("|---|---|---|---|---|---|\n")
    for fail in failures_info:
        f.write(f"| {fail['video_id']} | `SWIPE_RIGHT` | `{fail['predicted']}` | `{fail['raw_output']}` | `{fail['sampled_indices']}` | [View Contact Sheet]({fail['contact_sheet_rel_path']}) |\n")

print("Generated failed_swipe_right_inspection.md and contact sheets.")
