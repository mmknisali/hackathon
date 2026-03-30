# Star Tracker Sensor Error Correction

LSTM Autoencoder neural network for correcting errors in star tracker sensor data used for spacecraft attitude determination.

## What is a Star Tracker?

A star tracker is a spacecraft sensor that determines orientation/attitude by photographing stars and matching them against a catalog. Common errors include:
- Gaussian noise (random jitter)
- Quaternion drift (slow orientation error accumulation)
- Cosmic ray spikes (sudden massive reading spikes)
- Star misidentification (wrong star matched)
- Thermal distortion (heat warping readings)

## Performance

| Metric | Value |
|--------|-------|
| MSE Reduction | ~95% |
| MAE Reduction | ~65% |
| Model Parameters | 1.27M |
| Training Time (RTX 4050) | ~10-15 min |

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install torch numpy fastapi uvicorn pydantic python-multipart
```

### 2. Train the Model
#### *only if you want to, there is a model there thats already trained*

```bash
cd backend
source venv/bin/activate
python train.py
```

### 3. Run the API

```bash
cd backend
source venv/bin/activate
python api.py
```

### 4. Open the Frontend

Open `frontend/index.html` in a web browser.

## Project Structure

```
star-tracker-ml/
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── backend/
│   ├── api.py                   # FastAPI server
│   ├── model.py                # LSTM Autoencoder model
│   ├── train.py                 # Training script
│   ├── data_generator.py       # Synthetic data generation
│   ├── evaluate.py             # Model evaluation
│   ├── requirements.txt         # Python dependencies
│   └── venv/                   # Virtual environment
├── frontend/
│   └── index.html              # Demo UI with Plotly charts
├── models/
│   ├── star_tracker_model.pth  # Trained model weights
│   └── star_tracker_full.pth   # Full checkpoint with metrics
└── docs/
    ├── 01_GETTING_STARTED.md    # Setup guide
    ├── 02_ARCHITECTURE.md       # Model architecture
    ├── 03_DATA_PIPELINE.md     # Data generation
    ├── 04_TRAINING.md          # Training procedures
    ├── 05_API.md               # API documentation
    ├── 06_FRONTEND.md          # Frontend guide
    ├── 07_TROUBLESHOOTING.md   # Common issues
    └── 08_CONTRIBUTING.md      # Contribution guide
```

## Architecture

```
Input (batch, 100, 7)
    ↓
Input Projection (7 → 128)
    ↓
BiLSTM Encoder (bidirectional, 2 layers)
    ↓
Multi-Head Attention (4 heads)
    ↓
Projection (256 → 128)
    ↓
LSTM Decoder (2 layers)
    ↓
Residual Blocks (2x)
    ↓
Noise Predictor (128 → 7)
    ↓
Output: Corrected sequence
```

## Tech Stack

- **Backend**: Python, PyTorch, FastAPI, NumPy
- **Frontend**: HTML, JavaScript, Plotly.js
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Correct single sequence |
| `/batch-predict` | POST | Correct batch of sequences |

## Example Usage

### Using curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12]]}'
```

### Using Python:

```python
import requests

data = {
    "sequence": [[0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12]]
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## License

MIT License

## Team

Hackathon Project - Star Tracker ML Pipeline
