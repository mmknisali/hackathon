import torch
import numpy as np
from data_generator import prepare_data
from model import SmartStarTrackerDenoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Preparing data...")
train_loader, val_loader, test_loader = prepare_data(num_samples=50000, sequence_length=100, batch_size=64)

print("Loading model...")
model = SmartStarTrackerDenoiser(input_size=7, hidden_size=128, num_layers=2, num_heads=4).to(device)
model.load_state_dict(torch.load('../models/star_tracker_model.pth', map_location=device))
model.eval()

def calculate_metrics(model, test_loader, device):
    mse_before = []
    mse_after = []
    mae_before = []
    mae_after = []
    
    with torch.no_grad():
        for corrupted, clean in test_loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            
            output = model(corrupted)
            
            mse_before.append(((corrupted - clean) ** 2).mean().item())
            mse_after.append(((output - clean) ** 2).mean().item())
            
            mae_before.append(torch.abs(corrupted - clean).mean().item())
            mae_after.append(torch.abs(output - clean).mean().item())
    
    metrics = {
        'mse_before': np.mean(mse_before),
        'mse_after': np.mean(mse_after),
        'mae_before': np.mean(mae_before),
        'mae_after': np.mean(mae_after),
        'mse_reduction': (1 - np.mean(mse_after) / np.mean(mse_before)) * 100,
        'mae_reduction': (1 - np.mean(mae_after) / np.mean(mae_before)) * 100
    }
    
    return metrics

print("\nCalculating final metrics...")
metrics = calculate_metrics(model, test_loader, device)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"MSE Before correction:  {metrics['mse_before']:.6f}")
print(f"MSE After correction:   {metrics['mse_after']:.6f}")
print(f"MSE Reduction:           {metrics['mse_reduction']:.2f}%")
print(f"MAE Before correction:   {metrics['mae_before']:.6f}")
print(f"MAE After correction:    {metrics['mae_after']:.6f}")
print(f"MAE Reduction:           {metrics['mae_reduction']:.2f}%")
print("="*50)
