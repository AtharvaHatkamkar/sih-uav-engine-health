"""
AI / ML Time-Series Prediction Engine
Supports Primary UAV GRU 21-Feature Model and Turbofan Benchmark Model.
Fast sliding-window inference with configurable operating thresholds.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np

DEFAULT_THRESHOLD = 0.30

class UAVGRUNumPyEngine:
    """
    Ultra-fast, zero-overhead pure NumPy inference engine for UAV GRU model.
    Runs identically to PyTorch / Keras with zero framework latency.
    """
    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w_ih = data["gru_w_ih"] # (192, 21)
        self.w_hh = data["gru_w_hh"] # (192, 64)
        self.b_ih = data["gru_b_ih"] # (192,)
        self.b_hh = data["gru_b_hh"] # (192,)
        self.fc1_w = data["fc1_w"]   # (32, 64)
        self.fc1_b = data["fc1_b"]   # (32,)
        self.fc2_w = data["fc2_w"]   # (1, 32)
        self.fc2_b = data["fc2_b"]   # (1,)
        self.hidden_size = 64

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def predict_window(self, seq: np.ndarray) -> float:
        """
        seq: shape (seq_len=20, 21)
        PyTorch GRU gate ordering: [r, z, n] where:
        r: reset gate
        z: update gate
        n: new candidate state
        """
        h = np.zeros(self.hidden_size, dtype=np.float32)
        for x_t in seq:
            # (192,) = (192, 21) @ (21,) + (192,)
            gates_x = self.w_ih @ x_t + self.b_ih
            gates_h = self.w_hh @ h + self.b_hh
            
            # Split into r, z, n (each of size 64)
            r_x, z_x, n_x = np.split(gates_x, 3)
            r_h, z_h, n_h = np.split(gates_h, 3)
            
            r = self._sigmoid(r_x + r_h)
            z = self._sigmoid(z_x + z_h)
            n = np.tanh(n_x + r * n_h)
            h = (1.0 - z) * n + z * h
            
        # MLP layers
        d1 = np.maximum(0.0, self.fc1_w @ h + self.fc1_b)
        logit_arr = self.fc2_w @ d1 + self.fc2_b
        logit = float(logit_arr.item() if hasattr(logit_arr, 'item') else logit_arr[0])
        prob = float(self._sigmoid(logit))
        return prob

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        windows: (num_windows, 20, 21)
        Returns: (num_windows,) probabilities
        """
        probs = np.zeros(len(windows), dtype=np.float32)
        for i in range(len(windows)):
            probs[i] = self.predict_window(windows[i])
        return probs

class TurbofanGRUNumPyEngine:
    """
    Pure NumPy inference engine for NASA C-MAPSS 17-feature Turbofan GRU benchmark model.
    Predicts Remaining Useful Life (RUL) and maps to calibrated failure probability.
    """
    def __init__(self, keras_path: str):
        import zipfile, h5py, io
        with zipfile.ZipFile(keras_path, 'r') as z:
            w_bytes = z.read('model.weights.h5')
        f = h5py.File(io.BytesIO(w_bytes), 'r')
        self.W_z, self.W_r, self.W_h = np.split(f['layers/gru/cell/vars/0'][:], 3, axis=1) # (17, 64)
        self.U_z, self.U_r, self.U_h = np.split(f['layers/gru/cell/vars/1'][:], 3, axis=1) # (64, 64)
        bias = f['layers/gru/cell/vars/2'][:]
        self.b_z = bias[0, :64] + bias[1, :64]
        self.b_r = bias[0, 64:128] + bias[1, 64:128]
        self.b_h = bias[0, 128:] + bias[1, 128:]
        self.fc1_w = f['layers/dense/vars/0'][:] # (64, 32)
        self.fc1_b = f['layers/dense/vars/1'][:] # (32,)
        self.fc2_w = f['layers/dense_1/vars/0'][:] # (32, 1)
        self.fc2_b = f['layers/dense_1/vars/1'][:] # (1,)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def predict_window(self, seq: np.ndarray) -> float:
        h = np.zeros(64, dtype=np.float32)
        for x_t in seq:
            z = self._sigmoid(x_t @ self.W_z + h @ self.U_z + self.b_z)
            r = self._sigmoid(x_t @ self.W_r + h @ self.U_r + self.b_r)
            h_cand = np.tanh(x_t @ self.W_h + (r * h) @ self.U_h + self.b_h)
            h = (1.0 - z) * h + z * h_cand
        d1 = np.maximum(0.0, h @ self.fc1_w + self.fc1_b)
        rul = float((d1 @ self.fc2_w + self.fc2_b)[0])
        # Map RUL to failure probability: RUL <= 30 is critical failure zone
        prob = float(np.clip(1.0 - (rul / 125.0), 0.02, 0.98))
        return prob

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        probs = np.zeros(len(windows), dtype=np.float32)
        for i in range(len(windows)):
            probs[i] = self.predict_window(windows[i])
        return probs

class PredictionPipeline:
    def __init__(self, model_dir: Optional[str] = None, threshold: float = DEFAULT_THRESHOLD):
        if not model_dir:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "uav_propulsion_gru")
        self.model_dir = model_dir
        self.threshold = threshold
        self.model_name = "Primary UAV GRU (21 Features)"
        self.engine = None
        self.turbofan_engine = None
        self._load_model()
        
    def _load_model(self):
        weights_path = os.path.join(self.model_dir, "uav_gru_numpy_weights.npz")
        if os.path.exists(weights_path):
            self.engine = UAVGRUNumPyEngine(weights_path)
            
        turbofan_path = os.path.join(os.path.dirname(__file__), "..", "models", "turbofan_c_mapss", "uav_gru_model.keras")
        if os.path.exists(turbofan_path):
            try:
                self.turbofan_engine = TurbofanGRUNumPyEngine(turbofan_path)
            except Exception as e:
                print(f"Turbofan engine init error: {e}")
            
    def set_threshold(self, threshold: float):
        if 0.01 <= threshold <= 0.99:
            self.threshold = threshold

    def get_risk_level(self, prob: float) -> str:
        """
        Risk classification calibrated against operating threshold.
        Default threshold: 0.30
        """
        warn_th = self.threshold
        att_th = warn_th * 0.66  # ~0.20
        high_th = min(0.95, warn_th * 2.0) # ~0.60
        
        if prob < att_th:
            return "NORMAL"
        elif prob < warn_th:
            return "ATTENTION"
        elif prob < high_th:
            return "WARNING"
        else:
            return "HIGH RISK"

    def calculate_health_score(self, peak_risk: float, avg_risk: float, anomaly_ratio: float = 0.0) -> int:
        """
        Computes the 0-100 overall UAV Engine Health Score.
        Health formula: 100 - (70 * peak_risk + 25 * avg_risk + 5 * anomaly_penalty)
        """
        penalty = 70.0 * peak_risk + 25.0 * avg_risk + 10.0 * min(1.0, anomaly_ratio)
        score = int(round(max(0.0, min(100.0, 100.0 - penalty))))
        return score

    def predict_flight_risk(self, scaled_windows: np.ndarray, timestamps: List[float]) -> Dict[str, Any]:
        """
        Performs full sliding-window analysis on telemetry windows.
        Auto-dispatches to 21-feature UAV model or 17-feature Turbofan model based on window dimension.
        """
        num_windows = len(scaled_windows)
        feat_dim = scaled_windows.shape[-1] if len(scaled_windows.shape) == 3 else 21
        
        if feat_dim == 17 and self.turbofan_engine is not None:
            probabilities = self.turbofan_engine.predict_batch(scaled_windows)
        elif self.engine is not None:
            probabilities = self.engine.predict_batch(scaled_windows)
        else:
            raise RuntimeError("Inference engine is not initialized.")
        
        # Smooth slight jitter with causal EMA (alpha=0.35)
        smoothed_probs = np.zeros_like(probabilities)
        smoothed_probs[0] = probabilities[0]
        for i in range(1, num_windows):
            smoothed_probs[i] = 0.35 * probabilities[i] + 0.65 * smoothed_probs[i - 1]
            
        peak_prob = float(np.max(smoothed_probs))
        avg_prob = float(np.mean(smoothed_probs))
        curr_prob = float(smoothed_probs[-1])
        
        # Risk timeline
        risk_timeline = []
        high_risk_windows = 0
        abnormal_windows = 0
        
        for idx in range(num_windows):
            p = float(smoothed_probs[idx])
            ts = timestamps[idx + 19] if (idx + 19) < len(timestamps) else timestamps[-1]
            lvl = self.get_risk_level(p)
            if lvl in ("WARNING", "HIGH RISK"):
                abnormal_windows += 1
            if lvl == "HIGH RISK":
                high_risk_windows += 1
                
            risk_timeline.append({
                "window_index": idx,
                "timestamp": round(float(ts), 2),
                "failure_probability": round(p, 4),
                "risk_level": lvl
            })
            
        current_status = self.get_risk_level(curr_prob)
        # If flight has experienced high risk at end, prioritize the worst recent state
        if peak_prob >= min(0.95, self.threshold * 2.0) and curr_prob >= self.threshold:
            overall_status = "HIGH RISK"
        elif peak_prob >= self.threshold or curr_prob >= self.threshold:
            overall_status = "WARNING"
        elif peak_prob >= self.threshold * 0.66:
            overall_status = "ATTENTION"
        else:
            overall_status = "NORMAL"
            
        anomaly_ratio = abnormal_windows / max(1, num_windows)
        health_score = self.calculate_health_score(peak_prob, avg_prob, anomaly_ratio)
        
        return {
            "current_failure_probability": round(curr_prob, 4),
            "peak_failure_probability": round(peak_prob, 4),
            "average_failure_probability": round(avg_prob, 4),
            "health_score": health_score,
            "status": overall_status,
            "current_risk_level": current_status,
            "operating_threshold": self.threshold,
            "high_risk_windows_count": high_risk_windows,
            "abnormal_windows_count": abnormal_windows,
            "total_windows": num_windows,
            "risk_timeline": risk_timeline,
            "raw_probabilities": smoothed_probs
        }
