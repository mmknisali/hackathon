# Model Weights

This folder contains the trained PyTorch model weights.

## Files

| File | Description |
|------|-------------|
| `star_tracker_model.pth` | Model weights only (~5MB) |
| `star_tracker_full.pth` | Full checkpoint with weights + metrics (~5MB) |

## Loading the Model

### Option 1: Load weights only
```python
import torch
from model import SmartStarTrackerDenoiser

model = SmartStarTrackerDenoiser(input_size=7, hidden_size=128, num_layers=2, num_heads=4)
model.load_state_dict(torch.load('star_tracker_model.pth', map_location='cuda'))
model.eval()
```

### Option 2: Load full checkpoint
```python
checkpoint = torch.load('star_tracker_full.pth')
model.load_state_dict(checkpoint['model_state_dict'])
metrics = checkpoint['metrics']
print(metrics)
```

## Retraining

To retrain the model:

```bash
cd ../backend
source venv/bin/activate
python train.py
```

This will overwrite these files with new weights.

## Model Performance

See [../docs/04_TRAINING.md](../docs/04_TRAINING.md) for evaluation metrics.
