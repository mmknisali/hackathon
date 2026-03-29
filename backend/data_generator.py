import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

np.random.seed(42)
torch.manual_seed(42)

def normalize_quaternion(q):
    """Normalize quaternion to unit length"""
    return q / (np.linalg.norm(q) + 1e-8)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    q = normalize_quaternion(q)
    q0, q1, q2, q3 = q[0], q[1], q2 = q[2], q[3]
    
    R = np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])
    return R

def generate_clean_star_sequences(num_samples=50000, sequence_length=100):
    """Generate clean star tracker data sequences"""
    clean_data = []
    
    for _ in range(num_samples):
        sequence = []
        
        base_position = np.random.randn(3)
        base_position = base_position / np.linalg.norm(base_position)
        
        base_quaternion = np.random.randn(4)
        base_quaternion = normalize_quaternion(base_quaternion)
        
        for t in range(sequence_length):
            angle = t * 0.01 + np.random.randn() * 0.1
            rotation = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            rotated_position = rotation @ base_position
            rotated_position += np.random.randn(3) * 0.001
            
            q_change = np.random.randn(4) * 0.002
            quaternion = normalize_quaternion(base_quaternion + q_change)
            
            feature_vector = np.concatenate([rotated_position, quaternion])
            sequence.append(feature_vector)
        
        clean_data.append(sequence)
    
    return np.array(clean_data, dtype=np.float32)

def inject_noise(clean_data, noise_config=None):
    """Inject various types of noise into clean data"""
    if noise_config is None:
        noise_config = {
            'gaussian_std': 0.1,
            'drift_factor': 0.002,
            'cosmic_ray_prob': 0.02,
            'cosmic_ray_magnitude': 2.0,
            'misid_prob': 0.01,
            'misid_magnitude': 1.5
        }
    
    corrupted = clean_data.copy()
    num_samples, seq_len, features = corrupted.shape
    
    for i in range(num_samples):
        cumulative_drift = np.zeros(7)
        
        for t in range(seq_len):
            corrupted[i, t, :3] += np.random.normal(0, noise_config['gaussian_std'], 3)
            
            cumulative_drift += np.random.randn(7) * noise_config['drift_factor']
            corrupted[i, t, :] += cumulative_drift
            
            if np.random.rand() < noise_config['cosmic_ray_prob']:
                spike_idx = np.random.randint(0, 7)
                corrupted[i, t, spike_idx] += np.random.choice([-1, 1]) * noise_config['cosmic_ray_magnitude']
            
            if np.random.rand() < noise_config['misid_prob']:
                misid_idx = np.random.randint(0, 3)
                corrupted[i, t, misid_idx] += np.random.randn() * noise_config['misid_magnitude']
        
        corrupted[i, :, 3:] = corrupted[i, :, 3:] / (np.linalg.norm(corrupted[i, :, 3:], axis=1, keepdims=True) + 1e-8)
    
    return corrupted

class StarTrackerDataset(Dataset):
    def __init__(self, corrupted, clean):
        self.corrupted = torch.FloatTensor(corrupted)
        self.clean = torch.FloatTensor(clean)
    
    def __len__(self):
        return len(self.corrupted)
    
    def __getitem__(self, idx):
        return self.corrupted[idx], self.clean[idx]

def prepare_data(num_samples=50000, sequence_length=100, batch_size=64):
    """Prepare train/val/test datasets"""
    print("Generating clean star sequences...")
    clean_data = generate_clean_star_sequences(num_samples, sequence_length)
    
    print("Injecting noise...")
    corrupted_data = inject_noise(clean_data)
    
    split_idx = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)
    
    train_corrupted = corrupted_data[:split_idx]
    train_clean = clean_data[:split_idx]
    
    val_corrupted = corrupted_data[split_idx:val_split]
    val_clean = clean_data[split_idx:val_split]
    
    test_corrupted = corrupted_data[val_split:]
    test_clean = clean_data[val_split:]
    
    train_dataset = StarTrackerDataset(train_corrupted, train_clean)
    val_dataset = StarTrackerDataset(val_corrupted, val_clean)
    test_dataset = StarTrackerDataset(test_corrupted, test_clean)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data prepared: {num_samples} samples, {sequence_length} length, {clean_data.shape[-1]} features")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_data()
    print("Sample batch shape:", next(iter(train_loader))[0].shape)
