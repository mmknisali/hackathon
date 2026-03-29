# Contributing Guide

This guide explains how to contribute to the project and maintain code quality.

## Code Style

### Python

We follow PEP 8 with some modifications:

- **Line length**: Maximum 120 characters
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Classes: `CamelCase` (e.g., `SmartStarTrackerDenoiser`)
  - Functions: `snake_case` (e.g., `train_epoch`)
  - Variables: `snake_case` (e.g., `learning_rate`)
- **Imports**: Group in order: standard library, third-party, local

```python
# Correct example
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from model import SmartStarTrackerDenoiser
from data_generator import prepare_data
```

### Comments

- Use comments to explain **why**, not **what**
- Keep comments up to date with code
- Use docstrings for functions and classes

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss for the epoch
    """
```

---

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add batch prediction endpoint

- Added /batch-predict endpoint to API
- Supports multiple sequences at once
- Returns array of corrected outputs
```

### Pull Request Process

1. Create a new branch
2. Make your changes
3. Test locally
4. Push and create PR
5. Request review

---

## Project Structure

```
backend/
├── api.py              # API endpoints (FastAPI)
├── model.py            # Model architecture
├── train.py            # Training script
├── data_generator.py   # Data generation
├── evaluate.py         # Evaluation
└── requirements.txt    # Dependencies
```

### Which file to modify?

| Want to... | Modify this file |
|-----------|-----------------|
| Add new API endpoint | `api.py` |
| Change model architecture | `model.py` |
| Change training process | `train.py` |
| Change data generation | `data_generator.py` |
| Add evaluation metrics | `evaluate.py` |
| Add dependencies | `requirements.txt` |

---

## Testing

### Manual Testing

Before submitting:

1. **Test training**:
   ```bash
   cd backend
   source venv/bin/activate
   python train.py
   ```

2. **Test API**:
   ```bash
   cd backend
   source venv/bin/activate
   python api.py
   
   # In another terminal:
   curl http://localhost:8000/health
   ```

3. **Test frontend**:
   - Open `frontend/index.html`
   - Click "Generate Sample"
   - Click "Correct Errors"

### Testing Checklist

- [ ] Code runs without errors
- [ ] Model trains successfully
- [ ] API endpoints work
- [ ] Frontend displays correctly
- [ ] No console errors in browser

---

## Adding New Features

### Adding a new API endpoint

1. Add endpoint to `api.py`:

```python
@app.post("/new-endpoint")
def new_endpoint(data: YourModel):
    # Your code here
    return {"result": "success"}
```

2. Test it:

```bash
curl -X POST http://localhost:8000/new-endpoint \
  -H "Content-Type: application/json" \
  -d '{"your": "data"}'
```

### Adding new noise type

1. Edit `data_generator.py`:

```python
def inject_noise(clean_data):
    # Existing noise types...
    
    # Add new noise type
    for i in range(num_samples):
        for t in range(seq_len):
            if random() < new_noise_prob:
                # Apply new noise
                ...
    return corrupted
```

2. Update documentation in `docs/03_DATA_PIPELINE.md`

### Modifying model architecture

1. Edit `model.py`:

```python
class SmartStarTrackerDenoiser(nn.Module):
    def __init__(self, ...):
        # Add/remove layers
        ...
    
    def forward(self, x):
        # Modify forward pass
        ...
```

2. Update documentation in `docs/02_ARCHITECTURE.md`

---

## Documentation

When making changes, update relevant documentation:

- **API changes** → `docs/05_API.md`
- **Model changes** → `docs/02_ARCHITECTURE.md`
- **Data changes** → `docs/03_DATA_PIPELINE.md`
- **Training changes** → `docs/04_TRAINING.md`

---

## Common Tasks

### Update dependencies

```bash
cd backend
source venv/bin/activate
pip install new-package
pip freeze > requirements.txt
```

### Train with different parameters

```python
# In train.py, modify:
model = SmartStarTrackerDenoiser(
    hidden_size=256,  # Larger
    num_layers=3,     # Deeper
    ...
)
```

### Add new evaluation metric

```python
# In train.py or evaluate.py
def calculate_metrics(model, test_loader, device):
    # Add new metric
    new_metric = ...
    
    return {
        'mse_reduction': ...,
        'mae_reduction': ...,
        'new_metric': ...
    }
```

---

## Questions?

If you have questions:
1. Check existing documentation
2. Check code comments
3. Ask team members
4. Create an issue

---

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
