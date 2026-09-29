"""
Generate Realistic UAV Flight Telemetry Datasets for Demonstration & Testing
Includes both full 21-feature representations and raw 9-channel telemetry representations.
"""

import os
import csv
import numpy as np

OUTPUT_DIR = "data/demo_flights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "timestamp",
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

RAW_NAMES = [
    "timestamp",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "velocity_x",
    "velocity_y",
    "velocity_z"
]

def generate_flight(profile="healthy", length=500, dt=0.1, seed=42):
    np.random.seed(seed)
    t = np.arange(0, length * dt, dt)
    
    # Kinematic trajectories
    if profile == "healthy":
        vx = 14.0 + 1.2 * np.sin(0.08 * t) + np.random.normal(0, 0.12, length)
        vy = 0.4 * np.cos(0.06 * t) + np.random.normal(0, 0.10, length)
        vz = -0.05 * np.sin(0.04 * t) + np.random.normal(0, 0.08, length)
        
        ax = np.gradient(vx, dt) + np.random.normal(0, 0.08, length)
        ay = np.gradient(vy, dt) + np.random.normal(0, 0.08, length)
        az = np.gradient(vz, dt) + 9.81 + np.random.normal(0, 0.12, length)
        
        wx = np.random.normal(0, 0.03, length)
        wy = np.random.normal(0, 0.03, length)
        wz = 0.015 * np.sin(0.05 * t) + np.random.normal(0, 0.02, length)
        
    elif profile == "developing_anomaly":
        # First 40% is healthy, then motor bearing wear develops vibration & roll jitter
        vx = 14.0 + 1.2 * np.sin(0.08 * t) + np.random.normal(0, 0.12, length)
        vy = 0.4 * np.cos(0.06 * t) + np.random.normal(0, 0.10, length)
        vz = -0.05 * np.sin(0.04 * t) + np.random.normal(0, 0.08, length)
        
        ax = np.gradient(vx, dt) + np.random.normal(0, 0.08, length)
        ay = np.gradient(vy, dt) + np.random.normal(0, 0.08, length)
        az = np.gradient(vz, dt) + 9.81 + np.random.normal(0, 0.12, length)
        
        wx = np.random.normal(0, 0.03, length)
        wy = np.random.normal(0, 0.03, length)
        wz = 0.015 * np.sin(0.05 * t) + np.random.normal(0, 0.02, length)
        
        onset = int(0.40 * length)
        for i in range(onset, length):
            s = (i - onset) / (length - onset)
            # Motor 2 bearing friction -> micro-vibrations
            az[i] += 1.6 * s * np.sin(28.0 * t[i]) + np.random.normal(0, 0.5 * s)
            ax[i] += 0.8 * s * np.sin(20.0 * t[i]) + np.random.normal(0, 0.3 * s)
            wx[i] += 0.22 * s * np.sin(14.0 * t[i]) + np.random.normal(0, 0.12 * s)
            wy[i] += 0.26 * s * np.cos(14.0 * t[i]) + np.random.normal(0, 0.14 * s)
            # Slight velocity trim loss
            vx[i] -= 1.2 * (s ** 1.5)
            
    elif profile == "critical_failure":
        # Progressive rotor delamination & ESC phase desync
        vx = 14.0 + 1.2 * np.sin(0.08 * t) + np.random.normal(0, 0.12, length)
        vy = 0.4 * np.cos(0.06 * t) + np.random.normal(0, 0.10, length)
        vz = -0.05 * np.sin(0.04 * t) + np.random.normal(0, 0.08, length)
        
        ax = np.gradient(vx, dt) + np.random.normal(0, 0.08, length)
        ay = np.gradient(vy, dt) + np.random.normal(0, 0.08, length)
        az = np.gradient(vz, dt) + 9.81 + np.random.normal(0, 0.12, length)
        
        wx = np.random.normal(0, 0.03, length)
        wy = np.random.normal(0, 0.03, length)
        wz = 0.015 * np.sin(0.05 * t) + np.random.normal(0, 0.02, length)
        
        onset = int(0.30 * length)
        critical_point = int(0.65 * length)
        for i in range(onset, length):
            s = (i - onset) / (length - onset)
            if i >= critical_point:
                # Motor stalls / ESC cuts out periodically
                az[i] += 4.5 * np.sin(35.0 * t[i]) + np.random.normal(0, 1.8)
                ax[i] += 2.2 * np.cos(22.0 * t[i]) + np.random.normal(0, 1.1)
                ay[i] += 2.4 * np.sin(22.0 * t[i]) + np.random.normal(0, 1.2)
                wx[i] += 0.95 * np.sin(16.0 * t[i]) + np.random.normal(0, 0.5)
                wy[i] += 1.10 * np.cos(16.0 * t[i]) + np.random.normal(0, 0.6)
                wz[i] += 1.40 * np.sin(9.0 * t[i]) + np.random.normal(0, 0.4)
                vx[i] -= 5.5 * s
                vz[i] -= 2.0 * s
            else:
                az[i] += 2.0 * s * np.sin(30.0 * t[i]) + np.random.normal(0, 0.6 * s)
                wx[i] += 0.4 * s * np.sin(15.0 * t[i]) + np.random.normal(0, 0.2 * s)
                wy[i] += 0.45 * s * np.cos(15.0 * t[i]) + np.random.normal(0, 0.2 * s)

    # Magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Deltas
    acc_change = np.zeros(length)
    acc_change[1:] = np.abs(np.diff(acc_mag))
    
    gyro_change = np.zeros(length)
    gyro_change[1:] = np.abs(np.diff(gyro_mag))
    
    vel_change = np.zeros(length)
    vel_change[1:] = np.abs(np.diff(vel_mag))
    
    # Rolling statistics (window=5)
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
    
    # Full 21-feature records
    full_rows = []
    raw_rows = []
    
    for i in range(length):
        ts = round(t[i], 2)
        raw_rows.append([
            ts,
            round(wx[i], 5), round(wy[i], 5), round(wz[i], 5),
            round(ax[i], 5), round(ay[i], 5), round(az[i], 5),
            round(vx[i], 5), round(vy[i], 5), round(vz[i], 5)
        ])
        
        full_rows.append([
            ts,
            round(wx[i], 5), round(wy[i], 5), round(wz[i], 5),
            round(ax[i], 5), round(ay[i], 5), round(az[i], 5),
            round(vx[i], 5), round(vy[i], 5), round(vz[i], 5),
            round(acc_mag[i], 5), round(gyro_mag[i], 5), round(vel_mag[i], 5),
            round(acc_change[i], 5), round(gyro_change[i], 5), round(vel_change[i], 5),
            round(acc_mean[i], 5), round(acc_std[i], 5),
            round(gyro_mean[i], 5), round(gyro_std[i], 5),
            round(vel_mean[i], 5), round(vel_std[i], 5)
        ])
        
    return raw_rows, full_rows

def save_csv(filepath, headers, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Generated {filepath} ({len(rows)} samples)")

def main():
    print("Generating official demo flight telemetry datasets...")
    
    # 1. Healthy Flight (600 samples, ~60 seconds)
    raw1, full1 = generate_flight("healthy", length=600, dt=0.1, seed=101)
    save_csv(os.path.join(OUTPUT_DIR, "flight_01_healthy_nominal.csv"), FEATURE_NAMES, full1)
    save_csv(os.path.join(OUTPUT_DIR, "flight_01_raw_healthy.csv"), RAW_NAMES, raw1)
    
    # 2. Developing Propulsion Anomaly (750 samples, ~75 seconds)
    raw2, full2 = generate_flight("developing_anomaly", length=750, dt=0.1, seed=202)
    save_csv(os.path.join(OUTPUT_DIR, "flight_02_developing_propulsion_anomaly.csv"), FEATURE_NAMES, full2)
    save_csv(os.path.join(OUTPUT_DIR, "flight_02_raw_developing_anomaly.csv"), RAW_NAMES, raw2)
    
    # 3. Critical Pre-Failure (900 samples, ~90 seconds)
    raw3, full3 = generate_flight("critical_failure", length=900, dt=0.1, seed=303)
    save_csv(os.path.join(OUTPUT_DIR, "flight_03_critical_pre_failure.csv"), FEATURE_NAMES, full3)
    save_csv(os.path.join(OUTPUT_DIR, "flight_03_raw_critical.csv"), RAW_NAMES, raw3)
    
    print("All demo flight datasets generated successfully!")

if __name__ == "__main__":
    main()
