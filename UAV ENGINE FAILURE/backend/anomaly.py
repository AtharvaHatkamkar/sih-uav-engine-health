"""
Telemetry Anomaly Detection Engine
Multi-layer statistical and kinematic anomaly detection independent of neural failure prediction.
Analyzes acceleration variance, angular rate jitter, rate-of-change spikes, and kinematic mismatch.
"""

from typing import Dict, Any, List
import numpy as np

class TelemetryAnomalyDetector:
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        
    def detect_anomalies(self, data_matrix: np.ndarray, feature_names: List[str], timestamps: List[float]) -> Dict[str, Any]:
        """
        data_matrix: (N, 21) or (N, D)
        timestamps: (N,)
        Returns structured telemetry anomaly profile.
        """
        N, D = data_matrix.shape
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        
        anomalies: List[Dict[str, Any]] = []
        anomaly_scores = np.zeros(N, dtype=np.float32)
        
        # 1. Baseline statistics across initial calm window (first 15% or 30 samples)
        calm_len = max(20, min(50, int(0.15 * N)))
        baseline = data_matrix[:calm_len]
        base_means = np.mean(baseline, axis=0)
        base_stds = np.std(baseline, axis=0)
        base_stds[base_stds < 1e-4] = 1.0
        
        # 2. Channel-wise Z-score calculation
        z_scores = np.abs((data_matrix - base_means) / base_stds)
        
        # Specific channel indices
        az_idx = name_to_idx.get("acceleration_z", -1)
        ax_idx = name_to_idx.get("acceleration_x", -1)
        ay_idx = name_to_idx.get("acceleration_y", -1)
        wx_idx = name_to_idx.get("angular_velocity_x", -1)
        wy_idx = name_to_idx.get("angular_velocity_y", -1)
        wz_idx = name_to_idx.get("angular_velocity_z", -1)
        acc_std_idx = name_to_idx.get("acceleration_std", -1)
        acc_change_idx = name_to_idx.get("acceleration_change", -1)
        gyro_change_idx = name_to_idx.get("angular_velocity_change", -1)
        vel_mag_idx = name_to_idx.get("velocity_magnitude", -1)
        
        for i in range(calm_len, N):
            ts = round(float(timestamps[i]), 2)
            pt_anomalies = []
            
            # Check A: Severe High-Frequency Acceleration Vibration
            if acc_std_idx != -1 and z_scores[i, acc_std_idx] > self.z_threshold:
                z_val = z_scores[i, acc_std_idx]
                sev = "HIGH" if z_val > 5.0 else ("MEDIUM" if z_val > 3.8 else "LOW")
                score = min(1.0, float(z_val / 6.0))
                pt_anomalies.append({
                    "type": "Resonant Vibration Anomaly",
                    "channel": "acceleration_std",
                    "severity": sev,
                    "score": round(score, 3),
                    "description": f"Acceleration variance {z_val:.1f}x standard deviations above nominal flight baseline."
                })
                
            # Check B: Sudden Angular Rate Jitter / Attitude Instability
            if gyro_change_idx != -1 and z_scores[i, gyro_change_idx] > (self.z_threshold * 1.1):
                z_val = z_scores[i, gyro_change_idx]
                sev = "HIGH" if z_val > 5.2 else "MEDIUM"
                score = min(1.0, float(z_val / 6.0))
                pt_anomalies.append({
                    "type": "Attitude Rate Oscillation",
                    "channel": "angular_velocity_change",
                    "severity": sev,
                    "score": round(score, 3),
                    "description": f"Abrupt angular velocity delta spike ({z_val:.1f}σ) indicating motor torque perturbation."
                })
                
            # Check C: Severe Jerk / Acceleration Shock
            if acc_change_idx != -1 and z_scores[i, acc_change_idx] > 3.8:
                z_val = z_scores[i, acc_change_idx]
                pt_anomalies.append({
                    "type": "Kinematic Jerk Shock",
                    "channel": "acceleration_change",
                    "severity": "HIGH" if z_val > 5.5 else "MEDIUM",
                    "score": round(min(1.0, float(z_val / 6.5)), 3),
                    "description": f"Sudden acceleration step change of {z_val:.1f}σ detected."
                })
                
            # Check D: Deceleration & Motor Drag (Velocity dropping despite high dynamic activity)
            if vel_mag_idx != -1 and acc_std_idx != -1:
                # If vibration is high and velocity has dropped > 20% from baseline
                base_v = base_means[vel_mag_idx]
                curr_v = data_matrix[i, vel_mag_idx]
                if base_v > 5.0 and (curr_v < 0.75 * base_v) and z_scores[i, acc_std_idx] > 2.5:
                    pt_anomalies.append({
                        "type": "Propulsion Thrust Degradation",
                        "channel": "velocity_magnitude",
                        "severity": "HIGH",
                        "score": 0.88,
                        "description": "Forward speed degradation co-occurring with elevated propulsion vibration signature."
                    })
                    
            if pt_anomalies:
                # Choose the highest severity anomaly for the point
                highest = max(pt_anomalies, key=lambda a: a["score"])
                anomaly_scores[i] = highest["score"]
                anomalies.append({
                    "sample_index": i,
                    "timestamp": ts,
                    "anomaly_type": highest["type"],
                    "affected_channel": highest["channel"],
                    "severity": highest["severity"],
                    "anomaly_score": highest["score"],
                    "details": highest["description"]
                })
                
        # Group contiguous anomaly points into consolidated anomaly burst events
        consolidated_events = []
        if anomalies:
            start_a = anomalies[0]
            curr_cluster = [start_a]
            for a in anomalies[1:]:
                # If within 5 samples (0.5s), merge
                if a["sample_index"] - curr_cluster[-1]["sample_index"] <= 6:
                    curr_cluster.append(a)
                else:
                    consolidated_events.append(self._summarize_cluster(curr_cluster))
                    curr_cluster = [a]
            if curr_cluster:
                consolidated_events.append(self._summarize_cluster(curr_cluster))
                
        total_anomaly_samples = len(anomalies)
        anomaly_ratio = total_anomaly_samples / max(1, N)
        
        return {
            "total_anomaly_points": total_anomaly_samples,
            "anomaly_rate_percent": round(anomaly_ratio * 100.0, 2),
            "anomaly_event_clusters": consolidated_events,
            "anomaly_timeline_scores": [round(float(s), 3) for s in anomaly_scores],
            "layer_disclaimer": "Telemetry anomaly detection identifies statistical outliers and dynamic irregularities in sensor channels. It is decoupled from the neural failure prediction model."
        }
        
    def _summarize_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_ts = cluster[0]["timestamp"]
        end_ts = cluster[-1]["timestamp"]
        max_sev = "LOW"
        scores = [a["anomaly_score"] for a in cluster]
        types = set(a["anomaly_type"] for a in cluster)
        channels = set(a["affected_channel"] for a in cluster)
        
        for a in cluster:
            if a["severity"] == "HIGH":
                max_sev = "HIGH"
                break
            elif a["severity"] == "MEDIUM":
                max_sev = "MEDIUM"
                
        return {
            "start_time": start_ts,
            "end_time": end_ts,
            "duration_seconds": round(end_ts - start_ts, 2),
            "sample_count": len(cluster),
            "max_severity": max_sev,
            "peak_anomaly_score": round(max(scores), 3),
            "anomaly_types": list(types),
            "primary_channels": list(channels),
            "description": f"Detected {len(cluster)} anomalous telemetry points between {start_ts:.1f}s and {end_ts:.1f}s primarily in {', '.join(channels)}."
        }
