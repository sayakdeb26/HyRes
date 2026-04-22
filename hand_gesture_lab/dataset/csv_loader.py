import pandas as pd
import json

class CSVLoader:
    def __init__(self, csv_path):
        """
        Parses Jester CSV format: video_id;label
        """
        self.csv_path = csv_path
        # Jester CSV uses comma separator and has a header
        self.df = pd.read_csv(csv_path, sep=',', dtype={'video_id': str, 'label': str})
        self.df = self.df.dropna(subset=['video_id', 'label'])
        
        # Create label mapping
        unique_labels = sorted(self.df['label'].unique())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Create video_id to label_idx mapping
        self.vid_to_label = {}
        for _, row in self.df.iterrows():
            vid_id = str(row['video_id']).strip()
            label_name = str(row['label']).strip()
            self.vid_to_label[vid_id] = self.label_to_id[label_name]

    def get_vid_to_label_map(self):
        return self.vid_to_label
        
    def get_label_map(self):
        return self.id_to_label
        
    def save_label_map(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.id_to_label, f, indent=4)
