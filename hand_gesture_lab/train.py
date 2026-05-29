import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class GestureDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        
        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01
            # Time warp
            if torch.rand(1).item() < 0.3:
                T = x.shape[0]
                warp = torch.linspace(0, T-1, T) + torch.randn(T) * 0.5
                warp = warp.clamp(0, T-1).long()
                x = x[warp]
            # Time cutout
            if torch.rand(1).item() < 0.2:
                feat_dim = x.shape[1]
                start = torch.randint(0, max(1, feat_dim - 18), (1,)).item()
                x[:, start:start+18] = 0
            # Flip
            if torch.rand(1).item() < 0.3:
                x = x.flip(0)
                
        return x, y

class GestureLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GestureLSTM, self).__init__()
        
        # LSTM(128 -> 64) with Dropout 0.3
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        B, T, _ = x.size()
        
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        out = self.dropout2(out)
        
        # Frame-level classification
        out_flat = out.contiguous().view(B * T, -1)
        logits_flat = self.classifier(out_flat)
        logits = logits_flat.view(B, T, -1)
        
        # Temporal smoothing
        smoothed = torch.zeros_like(logits)
        smoothed[:, 0, :] = logits[:, 0, :]
        alpha = 0.6
        for t in range(1, T):
            smoothed[:, t, :] = alpha * logits[:, t, :] + (1 - alpha) * smoothed[:, t-1, :]
            
        return smoothed[:, -1, :]

def train_model():
    data_dir = '/home/sayak/HybridTestBed/hand_gesture_lab/data/processed_full'
    x_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')
    
    # 1. GPU Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Force disable cuDNN to fix local mismatch (9.21.1 vs 9.19.0)
    torch.backends.cudnn.enabled = False
    
    # 2. Filter Dataset
    print(f"Loading data from {x_path}...")
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    
    if len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
        y_raw = np.argmax(y_raw, axis=1)
        
    # Re-map Labels
    target_classes = {
        16: 0,  # Swiping Left
        17: 1,  # Swiping Right
        8: 2,   # Rolling Hand Forward
        7: 3,   # Rolling Hand Backward
        19: 4,  # Thumb Down
        14: 5   # Stop Sign
    }
    
    mask = np.isin(y_raw, list(target_classes.keys()))
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]
    
    y_mapped = np.vectorize(target_classes.get)(y_filtered)
    
    unique, counts = np.unique(y_mapped, return_counts=True)
    print("\nClass distribution (6 Classes):")
    
    # Names for the accuracy table specifically requested by the user
    short_names = ["Swipe Left", "Swipe Right", "Rolling Forward", 
                   "Rolling Backward", "Thumb Down", "Stop Sign"]
                   
    for u, c in zip(unique, counts):
        print(f"  {short_names[u]:<22}: {c} samples")
        
    # 3. Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered, y_mapped,
        test_size=0.3,
        stratify=y_mapped,
        random_state=42
    )
    
    print(f"\nTrain size: {len(X_train)} | Val size: {len(X_val)}")
    
    train_dataset = GestureDataset(X_train, y_train, augment=True)
    val_dataset = GestureDataset(X_val, y_val, augment=False)
    
    input_dim = X_train.shape[2]
    num_classes = 6
    
    # 4. DataLoader Optimization
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    
    # 5. Model
    model = GestureLSTM(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 150
    patience = 10
    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs('weights', exist_ok=True)
    
    best_all_preds = []
    best_all_targets = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
            
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        all_probs_np = np.array(all_probs)
        all_targets_np = np.array(all_targets)
        top2_preds = np.argsort(all_probs_np, axis=1)[:, -2:]
        top2_correct = np.array([t in p for t, p in zip(all_targets_np, top2_preds)])
        top2_acc = np.mean(top2_correct)
        
        scheduler.step(val_acc)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Top-2: {top2_acc:.4f} | LR: {lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top2 = top2_acc
            best_all_preds = all_preds
            best_all_targets = all_targets
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'weights/best_lstm_model.pth')
            print("  --> Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.4f} | Best Top-2: {best_top2:.4f}")
    
    # --- EVALUATION ---
    print("\nClassification Report:")
    print(classification_report(best_all_targets, best_all_preds, target_names=short_names))
    
    # Per-Class Accuracy Table
    print("\nGesture                     Accuracy")
    print("------------------------------------")
    cm = confusion_matrix(best_all_targets, best_all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for name, acc in zip(short_names, per_class_acc):
        print(f"{name:<27} {acc*100:.2f}%")
        
    # Generate Confusion Matrix Figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=short_names, yticklabels=short_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (Val Acc: {best_val_acc:.4f})')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=150)
    print("\nSaved confusion matrix to confusion_matrix.png")

if __name__ == '__main__':
    train_model()
