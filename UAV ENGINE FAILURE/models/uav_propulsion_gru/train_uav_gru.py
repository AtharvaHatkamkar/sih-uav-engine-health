"""
Train Primary UAV Engine/Propulsion Failure Prediction GRU Model
21 Telemetry Features, Sequence Length = 20, Threshold = 0.30
Software-Only UAV Engine Health & Early Warning Platform
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

FEATURE_NAMES = [
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_magnitude",
    "angular_velocity_magnitude",
    "velocity_magnitude",
    "acceleration_change",
    "angular_velocity_change",
    "velocity_change",
    "acceleration_mean",
    "acceleration_std",
    "angular_velocity_mean",
    "angular_velocity_std",
    "velocity_mean",
    "velocity_std"
]

def generate_flight_trajectory(flight_type="healthy", length=400, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, length / 10.0, length)
    
    # Base kinematics
    # Healthy nominal flight has steady velocity (10-15 m/s), small accelerations, near-zero mean angular rates
    vx = 12.0 + 1.5 * np.sin(0.1 * t) + np.random.normal(0, 0.2, length)
    vy = 0.5 * np.cos(0.08 * t) + np.random.normal(0, 0.15, length)
    vz = -0.1 * np.sin(0.05 * t) + np.random.normal(0, 0.1, length) # slight altitude trim
    
    ax = np.gradient(vx, 0.1) + np.random.normal(0, 0.1, length)
    ay = np.gradient(vy, 0.1) + np.random.normal(0, 0.1, length)
    az = np.gradient(vz, 0.1) + 9.81 + np.random.normal(0, 0.15, length) # gravity offset
    
    wx = np.random.normal(0, 0.05, length)
    wy = np.random.normal(0, 0.05, length)
    wz = 0.02 * np.sin(0.08 * t) + np.random.normal(0, 0.03, length)
    
    labels = np.zeros(length, dtype=np.float32)
    
    if flight_type == "developing_anomaly":
        # At t ~ 50% length, motor bearing begins degrading -> vibration rises, periodic oscillation
        onset = int(length * 0.45)
        for i in range(onset, length):
            severity = (i - onset) / (length - onset)
            # High frequency jitter & gyro oscillation
            wx[i] += 0.25 * severity * np.sin(15.0 * t[i]) + np.random.normal(0, 0.15 * severity)
            wy[i] += 0.30 * severity * np.cos(15.0 * t[i]) + np.random.normal(0, 0.18 * severity)
            az[i] += 1.8 * severity * np.sin(25.0 * t[i]) + np.random.normal(0, 0.6 * severity)
            ax[i] += np.random.normal(0, 0.35 * severity)
            labels[i] = min(1.0, 0.2 + 0.6 * severity)
            
    elif flight_type == "critical_failure":
        # Sudden or rapid progressive failure -> ESC loss / motor stall / rotor delamination
        onset = int(length * 0.35)
        for i in range(onset, length):
            severity = (i - onset) / (length - onset)
            if severity > 0.5:
                # Motor failing, vehicle struggles for pitch/roll authority, velocity drops
                vx[i] -= 4.0 * (severity - 0.5)
                az[i] += np.random.normal(-2.5 * severity, 1.2 * severity)
                wz[i] += 1.2 * severity * np.sin(8.0 * t[i])
            wx[i] += 0.8 * severity * np.sin(18.0 * t[i]) + np.random.normal(0, 0.4 * severity)
            wy[i] += 0.9 * severity * np.cos(18.0 * t[i]) + np.random.normal(0, 0.4 * severity)
            ax[i] += np.random.normal(0, 0.8 * severity)
            ay[i] += np.random.normal(0, 0.8 * severity)
            az[i] += 3.5 * severity * np.sin(30.0 * t[i]) + np.random.normal(0, 1.2 * severity)
            labels[i] = min(1.0, 0.3 + 0.7 * (severity ** 0.8))
            
    # Compute the 12 engineered features
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    
    acc_change = np.zeros(length)
    acc_change[1:] = np.abs(np.diff(acc_mag))
    
    gyro_change = np.zeros(length)
    gyro_change[1:] = np.abs(np.diff(gyro_mag))
    
    vel_change = np.zeros(length)
    vel_change[1:] = np.abs(np.diff(vel_mag))
    
    # Rolling stats with window=5
    def rolling_mean(arr, w=5):
        res = np.zeros_like(arr)
        for i in range(len(arr)):
            res[i] = np.mean(arr[max(0, i-w+1):i+1])
        return res
        
    def rolling_std(arr, w=5):
        res = np.zeros_like(arr)
        for i in range(len(arr)):
            sub = arr[max(0, i-w+1):i+1]
            res[i] = np.std(sub) if len(sub) > 1 else 0.0
        return res

    acc_mean = rolling_mean(acc_mag)
    acc_std = rolling_std(acc_mag)
    gyro_mean = rolling_mean(gyro_mag)
    gyro_std = rolling_std(gyro_mag)
    vel_mean = rolling_mean(vel_mag)
    vel_std = rolling_std(vel_mag)
    
    features = np.column_stack([
        wx, wy, wz,
        ax, ay, az,
        vx, vy, vz,
        acc_mag, gyro_mag, vel_mag,
        acc_change, gyro_change, vel_change,
        acc_mean, acc_std,
        gyro_mean, gyro_std,
        vel_mean, vel_std
    ])
    
    return features, labels

class UAVGRUModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use last time step output
        last_out = out[:, -1, :]
        h = self.relu(self.fc1(last_out))
        logits = self.fc2(h)
        return torch.sigmoid(logits).squeeze(-1)

def main():
    print("Synthesizing flight trajectories for UAV 21-feature model training...")
    all_sequences = []
    all_targets = []
    
    seq_len = 20
    
    # Generate 150 flights of each category
    np.random.seed(42)
    for i in range(120):
        feats, labs = generate_flight_trajectory("healthy", length=300, seed=1000 + i)
        for s in range(len(feats) - seq_len + 1):
            if s % 3 == 0:
                all_sequences.append(feats[s:s+seq_len])
                all_targets.append(labs[s+seq_len-1])
                
    for i in range(100):
        feats, labs = generate_flight_trajectory("developing_anomaly", length=350, seed=2000 + i)
        for s in range(len(feats) - seq_len + 1):
            if s % 2 == 0:
                all_sequences.append(feats[s:s+seq_len])
                all_targets.append(labs[s+seq_len-1])
                
    for i in range(100):
        feats, labs = generate_flight_trajectory("critical_failure", length=350, seed=3000 + i)
        for s in range(len(feats) - seq_len + 1):
            if s % 2 == 0:
                all_sequences.append(feats[s:s+seq_len])
                all_targets.append(labs[s+seq_len-1])
                
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    
    print(f"Total sequences generated: {len(X)}, Shape: {X.shape}")
    
    # Compute feature-wise normalization parameters (Z-score mean & std)
    flat_X = X.reshape(-1, 21)
    mean_vals = np.mean(flat_X, axis=0)
    std_vals = np.std(flat_X, axis=0)
    std_vals[std_vals < 1e-6] = 1.0 # prevent div zero
    
    # Also compute MinMax for compatibility
    min_vals = np.min(flat_X, axis=0)
    max_vals = np.max(flat_X, axis=0)
    
    # Normalize X
    X_norm = (X - mean_vals) / std_vals
    
    # Train / Val split
    indices = np.random.permutation(len(X_norm))
    split = int(0.8 * len(X_norm))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = torch.tensor(X_norm[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X_norm[val_idx]), torch.tensor(y[val_idx])
    
    print(f"Train set: {len(X_train)}, Val set: {len(X_val)}")
    
    model = UAVGRUModel(input_size=21, hidden_size=64)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    batch_size = 128
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Training GRU Model over 12 epochs...")
    model.train()
    for epoch in range(12):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        avg_loss = total_loss / len(X_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            binary_preds = (val_preds >= 0.30).float()
            binary_targets = (y_val >= 0.30).float()
            acc = (binary_preds == binary_targets).float().mean().item()
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy @ 0.30: {acc*100:.2f}%")
        model.train()
        
    print("\nTraining completed successfully! Evaluating metrics...")
    model.eval()
    with torch.no_grad():
        val_preds_np = model(X_val).numpy()
        bin_preds = (val_preds_np >= 0.30).astype(int)
        bin_true = (y_val.numpy() >= 0.30).astype(int)
        
        tp = np.sum((bin_preds == 1) & (bin_true == 1))
        fp = np.sum((bin_preds == 1) & (bin_true == 0))
        fn = np.sum((bin_preds == 0) & (bin_true == 1))
        tn = np.sum((bin_preds == 0) & (bin_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
    output_dir = "models/uav_propulsion_gru"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save PyTorch model
    torch.save(model.state_dict(), os.path.join(output_dir, "uav_gru_21.pt"))
    print("Saved PyTorch model weights to uav_gru_21.pt")
    
    # 2. Extract and save pure NumPy weights for lightweight, ultra-fast, zero-dependency inference
    gru_w_ih = model.gru.weight_ih_l0.detach().numpy() # (3*64, 21)
    gru_w_hh = model.gru.weight_hh_l0.detach().numpy() # (3*64, 64)
    gru_b_ih = model.gru.bias_ih_l0.detach().numpy()   # (3*64,)
    gru_b_hh = model.gru.bias_hh_l0.detach().numpy()   # (3*64,)
    
    fc1_w = model.fc1.weight.detach().numpy() # (32, 64)
    fc1_b = model.fc1.bias.detach().numpy()   # (32,)
    fc2_w = model.fc2.weight.detach().numpy() # (1, 32)
    fc2_b = model.fc2.bias.detach().numpy()   # (1,)
    
    np.savez_compressed(
        os.path.join(output_dir, "uav_gru_numpy_weights.npz"),
        gru_w_ih=gru_w_ih,
        gru_w_hh=gru_w_hh,
        gru_b_ih=gru_b_ih,
        gru_b_hh=gru_b_hh,
        fc1_w=fc1_w,
        fc1_b=fc1_b,
        fc2_w=fc2_w,
        fc2_b=fc2_b
    )
    print("Saved NumPy weights to uav_gru_numpy_weights.npz")
    
    # 3. Save scaler parameters
    scaler_dict = {
        "scaler_type": "standard_and_minmax",
        "feature_names": FEATURE_NAMES,
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist(),
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
    }
    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print("Saved scaler parameters to scaler_params.json")
    
    # 4. Save model configuration
    config_dict = {
        "model_name": "UAV Propulsion Health & Failure Prediction GRU",
        "version": "2.1.0-AeroVigil",
        "architecture": "Gated Recurrent Unit (GRU) + Dense Multi-Layer Perceptron",
        "input_features": FEATURE_NAMES,
        "feature_count": 21,
        "sequence_length": 20,
        "operating_threshold": 0.30,
        "risk_levels": {
            "NORMAL": [0.0, 0.20],
            "ATTENTION": [0.20, 0.30],
            "WARNING": [0.30, 0.60],
            "HIGH_RISK": [0.60, 1.0]
        },
        "evaluation_metrics": {
            "validation_accuracy": round(acc * 100, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "description": "Primary time-series deep learning model for real-time UAV motor, rotor, and propulsion health monitoring. Converts 20-sample sliding windows of 21 kinematic and dynamic indicators into time-dependent failure risk probabilities."
    }
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print("Saved model configuration to model_config.json")
    print("Primary UAV GRU Pipeline generation completed successfully!")

if __name__ == "__main__":
    main()
