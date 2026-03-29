# Getting Started

This guide will help you set up the development environment and run the project.

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 4050 or similar)
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2

## Directory Structure

```
star-tracker-ml/
├── backend/                 # All backend code
│   ├── api.py              # FastAPI server - serves predictions
│   ├── model.py            # PyTorch LSTM model definition
│   ├── train.py            # Training script
│   ├── data_generator.py   # Synthetic data generation
│   ├── evaluate.py         # Evaluation script
│   ├── requirements.txt    # Python dependencies
│   └── venv/               # Virtual environment (created by you)
├── frontend/               # Frontend code
│   └── index.html          # Single-page demo with Plotly
├── models/                 # Trained model weights
│   ├── star_tracker_model.pth
│   └── star_tracker_full.pth
└── docs/                   # Documentation
```

## Installation Steps

### 1. Navigate to Backend Directory

```bash
cd /home/ali/workspace/mmknisali/backend
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install torch numpy fastapi uvicorn pydantic python-multipart
```

For GPU support (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Running the Project

### Option 1: Train the Model (if not already trained)

```bash
source venv/bin/activate
python train.py
```

Expected output:
```
Using device: cuda
Preparing data...
Generating clean star sequences...
Injecting noise...
Data prepared: 50000 samples, 100 length, 7 features
Creating model...
Total parameters: 1,271,743

Starting training...
Epoch   1/50 | Train Loss: 0.028570 | Val Loss: 0.028568
...
==================================================
FINAL RESULTS
==================================================
MSE Reduction:           ~95%
MAE Reduction:           ~65%
```

### Option 2: Run API Server (use existing model)

```bash
source venv/bin/activate
python api.py
```

Expected output:
```
Model loaded from ../models/star_tracker_model.pth
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Option 3: Use Pre-trained Model

The model is already trained. Just start the API:

```bash
cd backend
source venv/bin/activate
python api.py
```

### Option 4: Evaluate Model

```bash
source venv/bin/activate
python evaluate.py
```

## Running the Frontend

Simply open the HTML file in a browser:

```bash
# If on Linux with Chrome
google-chrome frontend/index.html

# Or just open in any browser
firefox frontend/index.html
```

## Quick Test

Test the API is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","device":"cuda","model_loaded":true}
```

## Next Steps

- Read [ARCHITECTURE.md](02_ARCHITECTURE.md) to understand the model
- Read [DATA_PIPELINE.md](03_DATA_PIPELINE.md) to understand data generation
- Read [API.md](05_API.md) to learn about endpoints
- Read [TRAINING.md](04_TRAINING.md) to customize training
