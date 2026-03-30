# AGENTS.md - Development Guidelines for AI Agents

This file provides guidelines for AI agents working on this codebase.

---

## Project Overview

**Star Tracker Sensor Error Correction** - LSTM Autoencoder neural network for correcting errors in star tracker sensor data used for spacecraft attitude determination.

### Tech Stack
- **Backend**: Python, PyTorch, FastAPI, NumPy
- **Frontend**: HTML, JavaScript, Chart.js
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended)

### Directory Structure
```
.
├── AGENTS.md                    # This file
├── README.md                    # Project overview
├── .gitignore                   # Git ignore rules
├── backend/
│   ├── api.py                   # FastAPI server
│   ├── model.py                 # LSTM Autoencoder model
│   ├── train.py                 # Training script
│   ├── data_generator.py        # Synthetic data generation
│   ├── evaluate.py              # Model evaluation
│   └── requirements.txt         # Python dependencies
├── frontend/
│   └── index.html               # Demo UI with Chart.js
├── models/
│   ├── star_tracker_model.pth  # Trained model weights
│   └── star_tracker_full.pth    # Full checkpoint with metrics
└── docs/                        # Documentation
```

---

## Build, Run & Test Commands

### Setting Up Environment

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate

# Unix/MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Running the Application

**Train the Model:**
```bash
cd backend
# Activate venv first
python train.py
```

**Run the API Server:**
```bash
cd backend
# Activate venv first
python api.py
```
Server runs at `http://localhost:8000`

**Open the Frontend:**
Open `frontend/index.html` in a web browser.

### Testing

**Note:** This project does NOT have a formal test framework (no pytest). Testing is done manually:

1. **Test Training:**
   ```bash
   python train.py
   ```

2. **Test API:**
   ```bash
   python api.py
   
   # In another terminal:
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sequence": [[0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12]]}'
   ```

3. **Test Model Architecture:**
   ```bash
   cd backend
   python model.py
   ```

4. **Test Data Generation:**
   ```bash
   cd backend
   python data_generator.py
   ```

5. **Evaluate Model:**
   ```bash
   cd backend
   python evaluate.py
   ```

---

## Code Style Guidelines

### Python Style (PEP 8 with modifications)

**Line Length:** Maximum 120 characters

**Indentation:** 4 spaces (no tabs)

**Naming Conventions:**
| Type | Convention | Example |
|------|------------|---------|
| Classes | CamelCase | `SmartStarTrackerDenoiser` |
| Functions | snake_case | `train_epoch()`, `calculate_metrics()` |
| Variables | snake_case | `learning_rate`, `batch_size` |
| Constants | UPPER_SNAKE | `MAX_EPOCHS` |

### Import Order

Group imports in the following order (separate each group with a blank line):

```python
# 1. Standard library
import os
import sys
import math

# 2. Third-party packages
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 3. Local application
from model import SmartStarTrackerDenoiser
from data_generator import prepare_data
```

### Docstrings

Use docstrings for all public functions and classes. Follow this format:

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer instance
        device: Device to train on (cuda/cpu)
    
    Returns:
        Average training loss for the epoch
    """
```

### Comments

- Use comments to explain **WHY**, not **WHAT**
- Keep comments up to date with code
- Remove outdated comments when modifying code

### Error Handling

- Use specific exception types when possible
- Always log or return meaningful error messages
- For FastAPI endpoints, use `HTTPException` for user-facing errors

```python
# Good example from api.py
@app.post("/predict")
def predict(data: SensorData):
    try:
        sequence = np.array(data.sequence, dtype=np.float32)
        # ... processing
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Type Hints

While not strictly enforced, add type hints for function parameters and return values:

```python
def calculate_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
```

---

## Frontend Guidelines

### HTML/JS Conventions

- Use semantic HTML5 elements
- Keep JavaScript minimal and inline for simplicity
- CSS uses CSS custom properties (variables) for theming

### External Dependencies (CDN)

- Chart.js: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`
- Google Fonts: Share Tech Mono, Rajdhani, Orbitron

---

## Git Workflow

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages
```
feat: Add batch prediction endpoint

- Added /batch-predict endpoint to API
- Supports multiple sequences at once
```

---

## Model Configuration

Default hyperparameters in `train.py`:
```python
model = SmartStarTrackerDenoiser(
    input_size=7,
    hidden_size=128,
    num_layers=2,
    num_heads=4
)

# Training
batch_size = 64
epochs = 50
learning_rate = 0.001
patience = 8
```

Input/output shape: `(batch, sequence_length, features)` = `(batch, 100, 7)`

---

## API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Correct single sequence |
| `/batch-predict` | POST | Correct batch of sequences |

---

## Common Tasks

### Add a new API endpoint in `api.py`:
```python
@app.post("/new-endpoint")
def new_endpoint(data: YourModel):
    return {"result": "success"}
```

### Modify model architecture in `model.py`:
```python
class SmartStarTrackerDenoiser(nn.Module):
    def __init__(self, ...):
        # Add/remove layers
    
    def forward(self, x):
        # Modify forward pass
```

### Add new noise type in `data_generator.py`:
Edit the `inject_noise()` function to add new noise patterns.

---

## Documentation Updates

When making changes, update relevant documentation:
- **API changes** → `docs/05_API.md`
- **Model changes** → `docs/02_ARCHITECTURE.md`
- **Data changes** → `docs/03_DATA_PIPELINE.md`
- **Training changes** → `docs/04_TRAINING.md`

---

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
