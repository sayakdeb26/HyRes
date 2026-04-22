import os
import json
import logging
import numpy as np

def setup_logger(log_file="preprocessing.log"):
    """Sets up the global logger."""
    logger = logging.getLogger("JesterPreprocessor")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

def build_label_map(input_dir):
    """
    Scans the input directory for subdirectories, treating them as class labels.
    Returns a dictionary mapping label_string -> label_index.
    """
    labels = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    label_map = {label: idx for idx, label in enumerate(labels)}
    return label_map

def save_label_map(label_map, output_dir):
    """Saves the label map to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "label_map.json")
    with open(path, 'w') as f:
        json.dump(label_map, f, indent=4)

def sliding_window(features, seq_len=30, stride=5):
    """
    Generates overlapping sequences from a list of features.
    If the features list is shorter than seq_len, it pads it once and returns.
    
    Args:
        features: List of numpy arrays or a 2D numpy array (num_frames, feature_dim)
        seq_len: Target length of each sequence
        stride: Step size between sequences
        
    Returns:
        A list of 2D numpy arrays, each of shape (seq_len, feature_dim).
    """
    if len(features) == 0:
        return []
        
    features_arr = np.array(features)
    num_frames, feature_dim = features_arr.shape
    
    sequences = []
    
    if num_frames < seq_len:
        # Pad with zeros at the beginning
        pad_len = seq_len - num_frames
        padding = np.zeros((pad_len, feature_dim))
        padded_seq = np.vstack((padding, features_arr))
        sequences.append(padded_seq)
    else:
        for i in range(0, num_frames - seq_len + 1, stride):
            seq = features_arr[i : i + seq_len]
            sequences.append(seq)
            
    return sequences
