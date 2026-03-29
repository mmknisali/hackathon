# Training

This document explains the training process, hyperparameters, and evaluation metrics.

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_samples` | 50,000 | Total training samples |
| `sequence_length` | 100 | Timesteps per sequence |
| `batch_size` | 64 | Samples per batch |
| `epochs` | 50 | Maximum training epochs |
| `patience` | 5 | Early stopping patience |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `weight_decay` | 0 (default) | L2 regularization |

## Training Process

### 1. Data Preparation

```python
train_loader, val_loader, test_loader = prepare_data(
    num_samples=50000,
    sequence_length=100,
    batch_size=64
)
```

- Generates 50,000 synthetic sequences
- Injects noise to create corrupted/clean pairs
- Splits: 80% train, 10% validation, 10% test

### 2. Model Initialization

```python
model = SmartStarTrackerDenoiser(
    input_size=7,
    hidden_size=128,
    num_layers=2,
    num_heads=4
).to(device)
```

- Moves model to GPU if available
- Total parameters: 1,271,743

### 3. Loss Function

```python
criterion = nn.MSELoss()
```

**Mean Squared Error (MSE)** is used as the loss function:
- Penalizes large errors heavily
- Works well for denoising tasks

### 4. Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Adam optimizer with:
- Learning rate: 0.001
- Default beta values (0.9, 0.999)
- No weight decay

### 5. Learning Rate Scheduler

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)
```

Reduces learning rate by half when validation loss stops improving for 2 epochs.

### 6. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients during training.

### 7. Early Stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), '../models/star_tracker_model.pth')
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

Saves the best model based on validation loss. Stops if no improvement for 5 epochs.

## Training Loop

```python
for epoch in range(epochs):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Check for improvement
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'models/star_tracker_model.pth')
    
    # Early stopping
    if patience_counter >= patience:
        break
```

## Evaluation Metrics

After training, the model is evaluated on the test set:

### Mean Squared Error (MSE)

```python
mse_before = ((corrupted - clean) ** 2).mean()
mse_after = ((output - clean) ** 2).mean()
mse_reduction = (1 - mse_after / mse_before) * 100
```

### Mean Absolute Error (MAE)

```python
mae_before = torch.abs(corrupted - clean).mean()
mae_after = torch.abs(output - clean).mean()
mae_reduction = (1 - mae_after / mae_before) * 100
```

## Expected Results

| Metric | Target | Typical |
|--------|--------|---------|
| MSE Reduction | ≥95% | 94-96% |
| MAE Reduction | ≥65% | 60-70% |

## Running Training

```bash
cd backend
source venv/bin/activate
python train.py
```

Expected output:
```
Using device: cuda
Preparing data...
Data prepared: 50000 samples, 100 length, 7 features
Creating model...
Total parameters: 1,271,743

Starting training...
Epoch   1/50 | Train Loss: 0.028570 | Val Loss: 0.028568
  -> Model saved! Best val loss: 0.028568
...
==================================================
FINAL RESULTS
==================================================
MSE Reduction:           ~95%
MAE Reduction:          ~65%
==================================================

Training complete!
```

## Training Time

| Hardware | Approximate Time |
|----------|------------------|
| RTX 4050 (6GB) | 10-15 minutes |
| RTX 3060 (12GB) | 8-12 minutes |
| CPU only | 2-4 hours |

## Resuming Training

To resume from a checkpoint:

```python
model.load_state_dict(torch.load('models/star_tracker_model.pth'))
# Continue training...
```

## Evaluating Model

```bash
cd backend
source venv/bin/activate
python evaluate.py
```

This loads the trained model and computes metrics on the test set.

## Customizing Training

### Change number of samples:
```python
# In train.py
train_loader, val_loader, test_loader = prepare_data(
    num_samples=100000,  # More data
    sequence_length=100,
    batch_size=64
)
```

### Change learning rate:
```python
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower LR
```

### Change batch size:
```python
# In train.py
train_loader, val_loader, test_loader = prepare_data(
    num_samples=50000,
    sequence_length=100,
    batch_size=32  # Smaller batch = more updates per epoch
)
```

## Key Files

- `backend/train.py` - Main training script
- `backend/evaluate.py` - Evaluation script
- `backend/data_generator.py` - Data generation
- `backend/model.py` - Model architecture
