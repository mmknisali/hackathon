# Troubleshooting

This guide covers common issues and their solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Issues](#training-issues)
3. [API Issues](#api-issues)
4. [Frontend Issues](#frontend-issues)
5. [Model Performance Issues](#model-performance-issues)

---

## Installation Issues

### "python: command not found"

**Problem**: `python` command not found

**Solution**: Use `python3` instead
```bash
python3 -m venv venv
python3 train.py
```

---

### "No module named 'torch'"

**Problem**: PyTorch not installed

**Solution**: Install PyTorch
```bash
source venv/bin/activate
pip install torch numpy
```

---

### "Permission denied" during pip install

**Problem**: Permission issues

**Solution**: Use `--user` flag or create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir torch numpy
```

---

## Training Issues

### "CUDA out of memory"

**Problem**: Not enough GPU memory

**Solutions**:
1. Reduce batch size in `train.py`:
   ```python
   batch_size=32  # Was 64
   ```
2. Reduce sequence length in `data_generator.py`:
   ```python
   sequence_length=50  # Was 100
   ```

---

### Training too slow on CPU

**Problem**: Training takes hours on CPU

**Solutions**:
1. Use GPU if available
2. Reduce dataset size:
   ```python
   # In train.py
   train_loader, val_loader, test_loader = prepare_data(
       num_samples=10000,  # Was 50000
       sequence_length=100,
       batch_size=64
   )
   ```

---

### Model not converging

**Problem**: Loss doesn't decrease

**Solutions**:
1. Check learning rate:
   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Try lower
   ```
2. Increase noise in data generation
3. Check data normalization

---

### Early stopping triggers immediately

**Problem**: Training stops after few epochs

**Solution**: Increase patience
```python
patience = 10  # Was 5
```

---

## API Issues

### "Model not found"

**Problem**: API can't find model file

**Solution**: Train the model first or ensure correct path:
```bash
ls -la models/
# Should show star_tracker_model.pth
```

---

### "Connection refused" when calling API

**Problem**: API server not running

**Solution**: Start the API server:
```bash
cd backend
source venv/bin/activate
python api.py
```

Then test:
```bash
curl http://localhost:8000/health
```

---

### API returns 500 error

**Problem**: Server error

**Solution**: Check server logs for details. Common causes:
- Model not loaded correctly
- Invalid input format
- CUDA out of memory

---

### Wrong input/output shape

**Problem**: API expects different data format

**Solution**: Ensure input has correct shape:
- Must be 7 features per timestep
- Must be list of lists

```python
# Correct format
sequence = [
    [x, y, z, q1, q2, q3, q4],  # timestep 1
    [x, y, z, q1, q2, q3, q4],  # timestep 2
]
```

---

## Frontend Issues

### "API Not Connected" status

**Problem**: Frontend can't reach API

**Solutions**:
1. Ensure API is running: `python api.py`
2. Check URL in JavaScript (default: `http://localhost:8000`)
3. Try different browser or disable ad-blocker

---

### Charts not displaying

**Problem**: Plotly charts empty

**Solutions**:
1. Check browser console (F12)
2. Verify Plotly CDN is loading
3. Check data format in textarea

---

### "Invalid JSON" error

**Problem**: Input textarea has malformed JSON

**Solution**: 
1. Click "Generate Sample" to get valid JSON
2. Or validate your JSON at https://jsonlint.com

---

### CORS error in browser

**Problem**: Can't call API from frontend

**Solution**: Add CORS headers to API (temporary fix for development):

```python
# In api.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Model Performance Issues

### Low MSE reduction (<80%)

**Problem**: Model not correcting errors well

**Solutions**:
1. Train longer (more epochs)
2. Increase noise in data generation
3. Check model architecture
4. Verify training data is correct

---

### Low MAE reduction (<50%)

**Problem**: Small errors not being corrected

**Solutions**:
1. Use hybrid loss (MSE + MAE)
2. Increase drift in noise injection
3. Train longer

---

### Model overfitting

**Problem**: Train loss low, test loss high

**Solutions**:
1. Increase dropout
2. Reduce model size
3. Add regularization
4. Use more training data

---

### Inconsistent results

**Problem**: Different results each run

**Solutions**:
1. Set random seeds:
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   ```
2. Use deterministic PyTorch:
   ```python
   torch.backends.cudnn.deterministic = True
   ```

---

## Getting Help

If you encounter issues not listed here:

1. Check the logs for error messages
2. Verify all dependencies are installed
3. Check API health endpoint: `curl http://localhost:8000/health`
4. Try running with debug mode

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU VRAM full | Reduce batch size |
| `Model not found` | No trained model | Run `python train.py` |
| `Connection refused` | API not running | Start API with `python api.py` |
| `Invalid JSON` | Malformed input | Use valid JSON format |
| `Shape mismatch` | Wrong input dimensions | Use 7 features per timestep |
