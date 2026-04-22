import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GestureDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        # Convert to float32 and long respectively
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def get_dataloaders(x_path, y_path, batch_size=32, val_split=0.2):
    dataset = GestureDataset(x_path, y_path)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def export_to_onnx(model, save_path, input_shape=(1, 30, 63)):
    model.eval()
    dummy_input = torch.randn(*input_shape, device=next(model.parameters()).device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to ONNX format at {save_path}")
