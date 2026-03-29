import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
from data_generator import prepare_data
from model import SmartStarTrackerDenoiser

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for corrupted, clean in train_loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        output = model(corrupted)
        loss = criterion(output, clean)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for corrupted, clean in val_loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            
            output = model(corrupted)
            loss = criterion(output, clean)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def calculate_metrics(model, test_loader, device):
    model.eval()
    
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(
        num_samples=50000,
        sequence_length=100,
        batch_size=64
    )
    
    print("Creating model...")
    model = SmartStarTrackerDenoiser(
        input_size=7,
        hidden_size=128,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 50
    patience = 8
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '../models/star_tracker_model.pth')
            print(f"  -> Model saved! Best val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('../models/star_tracker_model.pth'))
    
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
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, '../models/star_tracker_full.pth')
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
