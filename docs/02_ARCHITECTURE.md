# Model Architecture

This document explains the neural network architecture used for star tracker error correction.

## Overview

The model is a **Smart LSTM Autoencoder** with:
- Bidirectional LSTM encoder
- Multi-Head Attention mechanism
- LSTM decoder
- Residual blocks for better gradient flow

## Architecture Diagram

```
Input Tensor: (batch_size, sequence_length, features)
                     ↓
              Input Projection
            (7 features → 128 hidden)
                     ↓
         ┌─────────────────────────┐
         │   BiLSTM Encoder       │
         │  (2 layers, bidirect.) │
         │  hidden_size=128       │
         └─────────────────────────┘
                     ↓
         ┌─────────────────────────┐
         │  Multi-Head Attention   │
         │       (4 heads)         │
         └─────────────────────────┘
                     ↓
              Attention Projection
            (256 hidden → 128)
                     ↓
         ┌─────────────────────────┐
         │     LSTM Decoder        │
         │   (2 layers, forward)   │
         └─────────────────────────┘
                     ↓
         ┌─────────────────────────┐
         │    Residual Blocks      │
         │        (2x)             │
         └─────────────────────────┘
                     ↓
         ┌─────────────────────────┐
         │    Noise Predictor      │
         │   (128 → 7 features)   │
         └─────────────────────────┘
                     ↓
Output: Corrected sequence
```

## Input/Output Specifications

### Input
- **Shape**: `(batch_size, 100, 7)`
- **Features**: 7 values per timestep
  - Position: x, y, z (3 values)
  - Quaternion: q1, q2, q3, q4 (4 values)

### Output
- **Shape**: `(batch_size, 100, 7)`
- Same format as input
- Represents the **corrected** sensor readings

## Component Details

### 1. Input Projection
```python
nn.Linear(7, 128)  # Projects 7 features to 128 hidden dimensions
```

### 2. BiLSTM Encoder
- **Type**: Bidirectional LSTM
- **Layers**: 2
- **Hidden Size**: 128 (per direction)
- **Bidirectional**: Yes (256 total output)
- **Dropout**: 0.2

The bidirectional encoder reads the sequence in both directions, capturing context from past and future timesteps.

### 3. Multi-Head Attention
- **Heads**: 4
- **Hidden Size**: 256 (after bidirectional LSTM)
- **Head Dimension**: 256 / 4 = 64

Attention allows the model to focus on the most relevant timesteps when making predictions. This helps identify:
- Cosmic ray spikes (isolated anomalies)
- Drift patterns (gradual changes over time)

### 4. Attention Projection
```python
nn.Linear(256, 128)  # Reduce back to 128 for decoder
```

### 5. LSTM Decoder
- **Type**: Forward (unidirectional) LSTM
- **Layers**: 2
- **Hidden Size**: 128
- **Dropout**: 0.2

### 6. Residual Blocks
- **Count**: 2
- **Structure**:
  ```python
  x → LayerNorm → Linear(128→128) → GELU → Add to x
  ```

Residual (skip) connections help with gradient flow and allow the network to learn identity mappings when needed.

### 7. Noise Predictor
```python
nn.Sequential(
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, 7)
)
```

Instead of predicting the clean signal directly, the model predicts the **noise** and subtracts it from the input. This is called **residual denoising** and is more effective for denoising tasks.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 7 | Number of features |
| `hidden_size` | 128 | LSTM hidden dimension |
| `num_layers` | 2 | Number of LSTM layers |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.2 | Dropout rate |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 50 | Training epochs |
| `patience` | 5 | Early stopping patience |

## Model Size

- **Total Parameters**: 1,271,743
- **Model File Size**: ~5MB
- **GPU Memory (Training)**: ~2-3GB

## Why This Architecture?

### 1. Bidirectional LSTM
Star tracker errors often have temporal patterns. Bidirectional encoding captures both past and future context.

### 2. Attention Mechanism
- Helps identify **cosmic ray spikes** (isolated high errors)
- Helps smooth out **drift errors** (gradual accumulation)
- Allows model to "focus" on problematic timesteps

### 3. Residual Denoising
Instead of learning `f(x) = clean`, we learn `f(x) = noise` and compute `clean = input - noise`.
- Easier optimization
- Better generalization
- Proven effective for denoising tasks

### 4. Skip Connections
Residual blocks ensure gradient flow through deep networks and allow the model to preserve information.

## Key Files

- `backend/model.py` - Full model implementation
- `backend/train.py` - Training loop
- `docs/04_TRAINING.md` - Training details
