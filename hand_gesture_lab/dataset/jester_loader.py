import os
import pandas as pd
import json

class JesterLoader:
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.csv_path = csv_path
        
        # Jester CSV usually doesn't have headers and is separated by semicolons
        self.df = pd.read_csv(csv_path, sep=';', header=None, names=['video_id', 'label_str'])
        
        # Create label mapping
        unique_labels = sorted(self.df['label_str'].unique())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def get_video_paths_and_labels(self):
        """
        Returns a list of tuples (video_path, int_label)
        """
        items = []
        for _, row in self.df.iterrows():
            vid_id = str(row['video_id'])
            # Assuming videos are stored as .webm or MP4. 
            # If they are directories of frames, the path logic remains the same.
            vid_path_webm = os.path.join(self.data_dir, f"{vid_id}.webm")
            vid_path_mp4 = os.path.join(self.data_dir, f"{vid_id}.mp4")
            dir_path = os.path.join(self.data_dir, vid_id)
            
            label_id = self.label_to_id[row['label_str']]
            
            if os.path.exists(vid_path_webm):
                items.append((vid_path_webm, label_id))
            elif os.path.exists(vid_path_mp4):
                items.append((vid_path_mp4, label_id))
            elif os.path.isdir(dir_path):
                items.append((dir_path, label_id))
            else:
                # Assuming standard format and adding path anyway (will fail during reading if not found)
                # But it's better to warn or skip.
                pass
                
        return items
        
    def save_label_map(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.id_to_label, f, indent=4)
