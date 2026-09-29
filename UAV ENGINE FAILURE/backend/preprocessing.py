"""
Telemetry Data Preprocessing, Multi-Format Validation (CSV, TXT, LOG, TSV),
Feature Engineering, Physical Units Conversion, and Window Generation.
UAV Engine Health & Failure Prediction Platform
"""

import io
import re
import csv
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

UAV_21_FEATURES = [
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

RAW_9_KINEMATICS = [
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

TURBOFAN_26_COLS = [
    "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

TURBOFAN_17_FEATURES = [
    "op_setting_1",
    "op_setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

class PreprocessingError(Exception):
    pass

class TelemetryPreprocessor:
    def __init__(self, scaler_path: Optional[str] = None):
        self.scaler_params = self._load_scaler(scaler_path)
        
    def _load_scaler(self, path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            default_path = os.path.join(os.path.dirname(__file__), "..", "models", "uav_propulsion_gru", "scaler_params.json")
            if os.path.exists(default_path):
                path = default_path
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _split_line(self, line: str) -> List[str]:
        line = line.strip()
        if not line:
            return []
        if "," in line:
            # Comma-separated
            return [x.strip() for x in line.split(",")]
        elif "\t" in line:
            # Tab-separated
            return [x.strip() for x in line.split("\t")]
        elif ";" in line:
            # Semicolon-separated
            return [x.strip() for x in line.split(";")]
        else:
            # Space or multi-space separated (common in .txt flight logs and NASA C-MAPSS)
            return [x.strip() for x in re.split(r"\s+", line) if x.strip()]

    def validate_and_parse_file(self, file_content: str, filename: str = "telemetry.csv") -> Dict[str, Any]:
        """
        Validates CSV, TXT, LOG, TSV, DAT structure, column availability, data integrity, and numeric values.
        Supports comma, tab, whitespace, and C-MAPSS 26-column formats.
        """
        lines = [l.strip() for l in file_content.strip().split("\n") if l.strip()]
        if not lines:
            raise PreprocessingError("The uploaded telemetry file is empty.")

        first_tokens = self._split_line(lines[0])
        if not first_tokens:
            raise PreprocessingError("The file does not contain readable data tokens.")

        # Check if first line is a text header or numeric data
        first_line_is_header = False
        try:
            float(first_tokens[0])
            first_line_is_header = False
        except ValueError:
            first_line_is_header = True

        header = []
        data_lines = []

        if first_line_is_header:
            header = [col.lower() for col in first_tokens]
            data_lines = lines[1:]
        else:
            data_lines = lines
            # Infer schema by column count
            col_count = len(first_tokens)
            if col_count == 26:
                # NASA C-MAPSS turbofan dataset format
                header = TURBOFAN_26_COLS
            elif col_count == 21:
                header = UAV_21_FEATURES
            elif col_count == 22: # with timestamp
                header = ["timestamp"] + UAV_21_FEATURES
            elif col_count == 9:
                header = RAW_9_KINEMATICS
            elif col_count == 10: # with timestamp
                header = ["timestamp"] + RAW_9_KINEMATICS
            else:
                # Default generic names
                header = [f"sensor_{i+1}" for i in range(col_count)]

        if len(data_lines) < 20:
            raise PreprocessingError(
                f"Insufficient telemetry samples: Time-series analysis requires a sequence length of at least 20 samples. "
                f"The uploaded file '{filename}' contains only {len(data_lines)} valid rows."
            )

        # Detect schema: Full 21-feature, Raw 9-kinematics, or C-MAPSS Turbofan
        has_full_21 = all(f in header for f in UAV_21_FEATURES)
        has_raw_9 = all(f in header for f in RAW_9_KINEMATICS)
        has_turbofan = all(f in header for f in TURBOFAN_17_FEATURES)

        # If it's a 26-col C-MAPSS file without explicit header, it has TURBOFAN_17_FEATURES in TURBOFAN_26_COLS
        if not (has_full_21 or has_raw_9 or has_turbofan):
            # Check if 9 kinematics can be mapped from common flight log abbreviations
            alias_map = {
                "gyro_x": "angular_velocity_x", "gx": "angular_velocity_x", "roll_rate": "angular_velocity_x", "p": "angular_velocity_x",
                "gyro_y": "angular_velocity_y", "gy": "angular_velocity_y", "pitch_rate": "angular_velocity_y", "q": "angular_velocity_y",
                "gyro_z": "angular_velocity_z", "gz": "angular_velocity_z", "yaw_rate": "angular_velocity_z", "r": "angular_velocity_z",
                "acc_x": "acceleration_x", "ax": "acceleration_x",
                "acc_y": "acceleration_y", "ay": "acceleration_y",
                "acc_z": "acceleration_z", "az": "acceleration_z",
                "vel_x": "velocity_x", "vx": "velocity_x", "groundspeed_x": "velocity_x",
                "vel_y": "velocity_y", "vy": "velocity_y", "groundspeed_y": "velocity_y",
                "vel_z": "velocity_z", "vz": "velocity_z", "climb_rate": "velocity_z",
                "time": "timestamp", "t": "timestamp"
            }
            mapped_header = [alias_map.get(col, col) for col in header]
            has_full_21 = all(f in mapped_header for f in UAV_21_FEATURES)
            has_raw_9 = all(f in mapped_header for f in RAW_9_KINEMATICS)
            if has_full_21 or has_raw_9:
                header = mapped_header

        if not (has_full_21 or has_raw_9 or has_turbofan):
            missing_9 = [f for f in RAW_9_KINEMATICS if f not in header]
            raise PreprocessingError(
                f"Unrecognized telemetry structure in '{filename}'.\n"
                f"Expected either 9 raw kinematic channels (gyro X/Y/Z, accel X/Y/Z, vel X/Y/Z) or full 21 engineered features.\n"
                f"Missing channels: {', '.join(missing_9[:6])}."
            )

        # Timestamps
        has_timestamp = "timestamp" in header or "time" in header
        ts_idx = header.index("timestamp") if "timestamp" in header else (header.index("time") if "time" in header else -1)

        parsed_data = []
        timestamps = []
        corrupted_row_count = 0

        for r_idx, line in enumerate(data_lines):
            tokens = self._split_line(line)
            if not tokens or len(tokens) != len(header):
                corrupted_row_count += 1
                continue

            row_dict = {}
            valid_row = True
            for col_idx, col_name in enumerate(header):
                val_str = tokens[col_idx]
                if val_str == "" or val_str.lower() in ("nan", "null", "none"):
                    row_dict[col_name] = np.nan
                else:
                    try:
                        row_dict[col_name] = float(val_str)
                    except ValueError:
                        valid_row = False
                        break

            if not valid_row:
                corrupted_row_count += 1
                continue

            if has_timestamp and ts_idx != -1:
                timestamps.append(row_dict[header[ts_idx]])
            elif "cycle" in header:
                # Turbofan operational cycle as time
                timestamps.append(row_dict["cycle"])
            else:
                timestamps.append(round(r_idx * 0.1, 2)) # 10Hz sampling default

            parsed_data.append(row_dict)

        if len(parsed_data) < 20:
            raise PreprocessingError(
                f"Too many corrupted or non-numeric rows in '{filename}'. Only {len(parsed_data)} valid rows recovered (minimum 20 required)."
            )

        # For multi-engine C-MAPSS datasets (which pack 100 engines into one file), isolate Unit 1 for a continuous single-engine flight mission
        if "unit" in header or "unit_number" in header:
            u_key = "unit" if "unit" in header else "unit_number"
            unit_1_records = [r for r in parsed_data if r.get(u_key) == 1]
            if len(unit_1_records) >= 20:
                parsed_data = unit_1_records
                timestamps = [float(r.get("cycle", i+1)) for i, r in enumerate(parsed_data)]

        schema_type = "uav_21" if has_full_21 else ("uav_raw_9" if has_raw_9 else "turbofan_17")

        return {
            "filename": filename,
            "schema_type": schema_type,
            "raw_row_count": len(data_lines),
            "valid_sample_count": len(parsed_data),
            "corrupted_rows": corrupted_row_count,
            "timestamps": timestamps,
            "records": parsed_data
        }

    # Backward compatibility alias
    def validate_and_parse_csv(self, file_content: str, filename: str = "telemetry.csv") -> Dict[str, Any]:
        return self.validate_and_parse_file(file_content, filename)

    def engineer_features(self, parsed_result: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Extracts or computes the 21 telemetry features, ensuring exact ordering and missing value imputation.
        """
        records = parsed_result["records"]
        timestamps = parsed_result["timestamps"]
        schema = parsed_result["schema_type"]
        N = len(records)

        if schema == "turbofan_17":
            data_matrix = np.zeros((N, len(TURBOFAN_17_FEATURES)), dtype=np.float32)
            for j, feat in enumerate(TURBOFAN_17_FEATURES):
                vals = [r[feat] for r in records]
                series = np.array(vals, dtype=np.float32)
                nans = np.isnan(series)
                if np.any(nans):
                    x = np.arange(N)
                    series[nans] = np.interp(x[nans], x[~nans], series[~nans])
                data_matrix[:, j] = series
            return data_matrix, TURBOFAN_17_FEATURES, timestamps

        elif schema == "uav_21":
            data_matrix = np.zeros((N, 21), dtype=np.float32)
            for j, feat in enumerate(UAV_21_FEATURES):
                vals = [r[feat] for r in records]
                series = np.array(vals, dtype=np.float32)
                nans = np.isnan(series)
                if np.any(nans):
                    x = np.arange(N)
                    series[nans] = np.interp(x[nans], x[~nans], series[~nans])
                data_matrix[:, j] = series
            return data_matrix, UAV_21_FEATURES, timestamps

        else: # uav_raw_9
            wx = np.array([r["angular_velocity_x"] for r in records], dtype=np.float32)
            wy = np.array([r["angular_velocity_y"] for r in records], dtype=np.float32)
            wz = np.array([r["angular_velocity_z"] for r in records], dtype=np.float32)

            ax = np.array([r["acceleration_x"] for r in records], dtype=np.float32)
            ay = np.array([r["acceleration_y"] for r in records], dtype=np.float32)
            az = np.array([r["acceleration_z"] for r in records], dtype=np.float32)

            vx = np.array([r["velocity_x"] for r in records], dtype=np.float32)
            vy = np.array([r["velocity_y"] for r in records], dtype=np.float32)
            vz = np.array([r["velocity_z"] for r in records], dtype=np.float32)

            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
            gyro_mag = np.sqrt(wx**2 + wy**2 + wz**2)
            vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

            acc_change = np.zeros(N, dtype=np.float32)
            acc_change[1:] = np.abs(np.diff(acc_mag))

            gyro_change = np.zeros(N, dtype=np.float32)
            gyro_change[1:] = np.abs(np.diff(gyro_mag))

            vel_change = np.zeros(N, dtype=np.float32)
            vel_change[1:] = np.abs(np.diff(vel_mag))

            def rolling_stats(arr: np.ndarray, w: int = 5):
                m = np.zeros_like(arr)
                s = np.zeros_like(arr)
                for i in range(len(arr)):
                    window = arr[max(0, i - w + 1):i + 1]
                    m[i] = np.mean(window)
                    s[i] = np.std(window) if len(window) > 1 else 0.0
                return m, s

            acc_mean, acc_std = rolling_stats(acc_mag, 5)
            gyro_mean, gyro_std = rolling_stats(gyro_mag, 5)
            vel_mean, vel_std = rolling_stats(vel_mag, 5)

            data_matrix = np.column_stack([
                wx, wy, wz,
                ax, ay, az,
                vx, vy, vz,
                acc_mag, gyro_mag, vel_mag,
                acc_change, gyro_change, vel_change,
                acc_mean, acc_std,
                gyro_mean, gyro_std,
                vel_mean, vel_std
            ]).astype(np.float32)

            return data_matrix, UAV_21_FEATURES, timestamps

    def scale_features(self, data_matrix: np.ndarray) -> np.ndarray:
        N, D = data_matrix.shape
        if D == 21 and self.scaler_params and "mean" in self.scaler_params and "std" in self.scaler_params:
            mean = np.array(self.scaler_params["mean"], dtype=np.float32)
            std = np.array(self.scaler_params["std"], dtype=np.float32)
            std[std < 1e-6] = 1.0
            return (data_matrix - mean) / std
        elif D == 17:
            min_v = np.min(data_matrix, axis=0)
            max_v = np.max(data_matrix, axis=0)
            rng = max_v - min_v
            rng[rng < 1e-6] = 1.0
            return (data_matrix - min_v) / rng
        else:
            m = np.mean(data_matrix, axis=0)
            s = np.std(data_matrix, axis=0)
            s[s < 1e-6] = 1.0
            return (data_matrix - m) / s

    def generate_sliding_windows(self, data_matrix: np.ndarray, seq_len: int = 20) -> np.ndarray:
        N, D = data_matrix.shape
        if N < seq_len:
            raise PreprocessingError(f"Data length ({N}) is smaller than sequence length ({seq_len}).")
        num_windows = N - seq_len + 1
        windows = np.zeros((num_windows, seq_len, D), dtype=np.float32)
        for i in range(num_windows):
            windows[i] = data_matrix[i:i + seq_len]
        return windows
