import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.lstm_model import GestureLSTM
from training.utils import get_dataloaders, export_to_onnx

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/processed", help="Directory with .npy data")
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument('--model_dir', type=str, default="models", help="Output directory for weights")
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Data files not found in {args.data_dir}. Run preprocess.py first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(x_path, y_path, batch_size=args.batch_size)
    
    num_classes = len(np.unique(np.load(y_path)))
    print(f"Detected {num_classes} classes.")
    
    model = GestureLSTM(
        input_size=config.FEATURE_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    weights_path = os.path.join(args.model_dir, "gesture_lstm.pth")
    
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, weights_path)
    
    # Load best weights for ONNX export
    model.load_state_dict(torch.load(weights_path))
    onnx_path = os.path.join(args.model_dir, "gesture_lstm.onnx")
    export_to_onnx(model, onnx_path, input_shape=(1, config.SEQUENCE_LENGTH, config.FEATURE_DIM))

if __name__ == "__main__":
    main()
