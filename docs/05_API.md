# API Documentation

This document describes the FastAPI endpoints for the star tracker error correction service.

## Starting the API

```bash
cd backend
source venv/bin/activate
python api.py
```

The API will start on `http://localhost:8000`

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint

**URL**: `/`

**Method**: `GET`

**Description**: Returns API information

**Response**:
```json
{
  "message": "Star Tracker Sensor Error Correction API",
  "status": "running"
}
```

---

### 2. Health Check

**URL**: `/health`

**Method**: `GET`

**Description**: Check if API is running and model is loaded

**Response**:
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true
}
```

---

### 3. Predict (Single Sequence)

**URL**: `/predict`

**Method**: `POST`

**Description**: Correct a single star tracker sequence

**Request Body**:
```json
{
  "sequence": [
    [x1, y1, z1, q1, q2, q3, q4],
    [x2, y2, z2, q1, q2, q3, q4],
    ...
  ]
}
```

**Response**:
```json
{
  "input": [[...]],
  "corrected": [[...]],
  "error_reduction": "computed"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12],
      [0.86, 0.11, 0.50, 0.70, 0.15, 0.14, 0.13]
    ]
  }'
```

---

### 4. Batch Predict

**URL**: `/batch-predict`

**Method**: `POST`

**Description**: Correct multiple sequences at once

**Request Body**:
```json
{
  "sequences": [
    [[x1, y1, z1, q1, q2, q3, q4], ...],
    [[x1, y1, z1, q1, q2, q3, q4], ...]
  ]
}
```

**Response**:
```json
{
  "predictions": [
    [[x1, y1, z1, q1, q2, q3, q4], ...],
    [[x1, y1, z1, q1, q2, q3, q4], ...]
  ]
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      [[0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12]],
      [[0.86, 0.11, 0.50, 0.70, 0.15, 0.14, 0.13]]
    ]
  }'
```

---

## Using the API in Python

### Single Prediction

```python
import requests

# Prepare data
sequence = [
    [0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12],
    [0.86, 0.11, 0.50, 0.70, 0.15, 0.14, 0.13],
    [1.20, 0.15, 0.52, 0.69, 0.16, 0.15, 0.14]
]

# Make request
response = requests.post(
    "http://localhost:8000/predict",
    json={"sequence": sequence}
)

# Get result
result = response.json()
corrected = result["corrected"]
print(corrected)
```

### Batch Prediction

```python
import requests

sequences = [
    [[0.85, 0.12, 0.51, 0.71, 0.14, 0.15, 0.12]],
    [[0.86, 0.11, 0.50, 0.70, 0.15, 0.14, 0.13]]
]

response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"sequences": sequences}
)

result = response.json()
predictions = result["predictions"]
```

---

## Data Format

### Input Sequence Format

Each timestep should have **7 values**:

| Index | Value | Description |
|-------|-------|-------------|
| 0 | x | Star X position |
| 1 | y | Star Y position |
| 2 | z | Star Z position |
| 3 | q1 | Quaternion component 1 |
| 4 | q2 | Quaternion component 2 |
| 5 | q3 | Quaternion component 3 |
| 6 | q4 | Quaternion component 4 |

### Sequence Length

- Minimum: 1 timestep
- Maximum: 100 timesteps
- Recommended: 50-100 timesteps for best results

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid JSON"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "Error message here"
}
```

---

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can use these to test endpoints directly from the browser.

---

## Key Files

- `backend/api.py` - FastAPI application
- `backend/model.py` - Model definition
- `docs/06_FRONTEND.md` - How frontend uses API
