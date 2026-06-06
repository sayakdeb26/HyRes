import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

class GestureDataset(Dataset):
    def __init__(self, x_path, y_path, use_velocity=True):
        X_raw = np.load(x_path).astype(np.float32)
        y_raw = np.load(y_path)
        
        if len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
            self.y = np.argmax(y_raw, axis=1)
        else:
            self.y = y_raw
        self.y = self.y.astype(np.int64)
        
        if use_velocity:
            velocities = np.zeros_like(X_raw)
            velocities[:, 1:, :] = X_raw[:, 1:, :] - X_raw[:, :-1, :]
            self.X = np.concatenate([X_raw, velocities], axis=2)
        else:
            self.X = X_raw
            
        seq_mean = np.mean(self.X, axis=1, keepdims=True)
        seq_std = np.std(self.X, axis=1, keepdims=True)
        seq_std[seq_std == 0] = 1e-6
        self.X = (self.X - seq_mean) / seq_std
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class GestureLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        last_out = out[:, -1, :]
        return self.fc2(self.relu(self.fc1(last_out)))

def run_experiment(use_velocity):
    print(f"\n--- Running Experiment: {'WITH' if use_velocity else 'WITHOUT'} Velocity ---")
    data_dir = '/home/sayak/HyRes/hand_gesture_lab/data/processed/train'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = False
    
    dataset = GestureDataset(os.path.join(data_dir, 'X_fixed.npy'), os.path.join(data_dir, 'y.npy'), use_velocity)
    input_dim = dataset.X.shape[2]
    num_classes = len(np.unique(dataset.y))
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    y_train = dataset.y[train_dataset.indices]
    cw = compute_class_weight('balanced', classes=np.unique(dataset.y), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    
    model = GestureLSTM(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val = 0.0
    for epoch in range(8):  # Train for 8 epochs
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = torch.max(model(batch_X.to(device)), 1)[1]
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.numpy())
                
        val_acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}/8 | Val Acc: {val_acc:.4f}")
        best_val = max(best_val, val_acc)
        
    return best_val

if __name__ == '__main__':
    acc_no_vel = run_experiment(use_velocity=False)
    acc_vel = run_experiment(use_velocity=True)
    
    print("\n=== A/B TEST RESULTS ===")
    print(f"A (No Velocity)  : {acc_no_vel:.4f}")
    print(f"B (With Velocity): {acc_vel:.4f}")
    print(f"Improvement      : {acc_vel - acc_no_vel:.4f}")
