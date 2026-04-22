import torch
import numpy as np
from collections import deque
from . import config

class InferenceEngine:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        self.sequence_length = config.SEQUENCE_LENGTH
        self.threshold = config.PREDICTION_THRESHOLD
        self.debounce_frames = config.DEBOUNCE_FRAMES
        
        # Buffer for sequence features
        self.buffer = {'left': deque(maxlen=self.sequence_length), 
                       'right': deque(maxlen=self.sequence_length)}
        
        # Debounce counters
        self.cooldown = {'left': 0, 'right': 0}
        
        # Last predicted
        self.last_prediction = {'left': "None", 'right': "None"}

    def process(self, hand_label, feature_vector):
        """
        Adds a feature vector to the buffer and runs inference if the buffer is full.
        """
        # Decrease cooldown
        if self.cooldown[hand_label] > 0:
            self.cooldown[hand_label] -= 1
            
        if feature_vector is None:
            # If tracking lost, add zeros to let the sequence flush out
            self.buffer[hand_label].append(np.zeros(config.FEATURE_DIM))
        else:
            self.buffer[hand_label].append(feature_vector)
            
        if len(self.buffer[hand_label]) == self.sequence_length:
            if self.cooldown[hand_label] == 0:
                # Prepare tensor
                seq = np.array(self.buffer[hand_label])
                # Shape: (1, seq_len, feature_dim)
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(seq_tensor)
                    probabilities = torch.softmax(output, dim=1).squeeze()
                    
                    max_prob, predicted_class = torch.max(probabilities, dim=0)
                    
                    if max_prob.item() > self.threshold:
                        pred_idx = predicted_class.item()
                        self.last_prediction[hand_label] = config.GESTURE_CLASSES.get(pred_idx, "Unknown")
                        self.cooldown[hand_label] = self.debounce_frames
                        
        return self.last_prediction[hand_label]
        
    def load_weights(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Loaded weights from {path}")
        except Exception as e:
            print(f"Could not load weights: {e}. Using random weights for now.")
