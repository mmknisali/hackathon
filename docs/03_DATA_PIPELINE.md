# Data Pipeline

This document explains how synthetic star tracker data is generated and how noise is injected.

## Overview

Since real star tracker data with labeled errors is hard to obtain, we generate synthetic data:
1. Generate clean star position + quaternion sequences
2. Inject various types of synthetic noise
3. Use corrupted input + clean labels for training

## Data Generation Process

### Step 1: Generate Clean Sequences

```python
# For each sample:
base_position = random_unit_vector()  # x, y, z on unit sphere
base_quaternion = random_quaternion()  # normalized q1,q2,q3,q4

# For each timestep in sequence:
angle = t * 0.01 + noise
rotation_matrix = rotate_by_angle(angle)
position = rotation_matrix @ base_position
quaternion = base_quaternion + small_noise
```

### Step 2: Inject Noise

Four types of synthetic noise are injected:

```
Corrupted = Clean + Gaussian + Drift + CosmicRay + Misidentification
```

## Feature Vector

Each timestep has **7 features**:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x | Star position X (normalized) |
| 1 | y | Star position Y (normalized) |
| 2 | z | Star position Z (normalized) |
| 3 | q1 | Quaternion component 1 |
| 4 | q2 | Quaternion component 2 |
| 5 | q3 | Quaternion component 3 |
| 6 | q4 | Quaternion component 4 |

## Noise Types

### 1. Gaussian Noise (Random Jitter)

```python
corrupted[i, t, :3] += np.random.normal(0, gaussian_std, 3)
```

- **Purpose**: Simulates random sensor reading jitter
- **Default std**: 0.1
- **Effect**: Small random errors on all readings

### 2. Quaternion Drift (Slow Accumulation)

```python
cumulative_drift = np.zeros(7)
for t in range(seq_len):
    cumulative_drift += np.random.randn(7) * drift_factor
    corrupted[i, t, :] += cumulative_drift
```

- **Purpose**: Simulates slow orientation error accumulation
- **Default factor**: 0.002
- **Effect**: Errors that grow over time

### 3. Cosmic Ray Spikes

```python
if random() < cosmic_ray_prob:
    spike_idx = random(0, 7)
    corrupted[i, t, spike_idx] += random.choice([-1, 1]) * magnitude
```

- **Purpose**: Simulates cosmic ray hits causing massive outliers
- **Default probability**: 0.02 (2%)
- **Default magnitude**: 2.0
- **Effect**: Sudden large spikes in readings

### 4. Star Misidentification

```python
if random() < misid_prob:
    misid_idx = random(0, 3)  # Only position
    corrupted[i, t, misid_idx] += random.randn() * magnitude
```

- **Purpose**: Simulates wrong star matching
- **Default probability**: 0.01 (1%)
- **Default magnitude**: 1.5
- **Effect**: Position jumps to wrong values

## Configuration

The noise parameters are defined in `data_generator.py`:

```python
noise_config = {
    'gaussian_std': 0.1,        # Random jitter
    'drift_factor': 0.002,      # Cumulative drift
    'cosmic_ray_prob': 0.02,    # Spike probability
    'cosmic_ray_magnitude': 2.0, # Spike size
    'misid_prob': 0.01,         # Mis-ID probability
    'misid_magnitude': 1.5      # Mis-ID size
}
```

## Data Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 40,000 | 80% |
| Validation | 5,000 | 10% |
| Test | 5,000 | 10% |

## Sequence Dimensions

| Dimension | Size | Description |
|-----------|------|-------------|
| Batch | 64 | Number of sequences per batch |
| Sequence | 100 | Timesteps per sequence |
| Features | 7 | Values per timestep |

## PyTorch Dataset

The data is wrapped in a PyTorch Dataset:

```python
class StarTrackerDataset(Dataset):
    def __init__(self, corrupted, clean):
        self.corrupted = torch.FloatTensor(corrupted)
        self.clean = torch.FloatTensor(clean)
    
    def __getitem__(self, idx):
        return self.corrupted[idx], self.clean[idx]
```

## DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

## Usage

```python
from data_generator import prepare_data

train_loader, val_loader, test_loader = prepare_data(
    num_samples=50000,
    sequence_length=100,
    batch_size=64
)

# Get a batch
corrupted, clean = next(iter(train_loader))
print(corrupted.shape)  # (64, 100, 7)
```

## Key Files

- `backend/data_generator.py` - Data generation and noise injection
- `docs/04_TRAINING.md` - How training uses this data
