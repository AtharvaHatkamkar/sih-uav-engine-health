"""
FastAPI REST API Server for UAV Engine Health & Failure Prediction Platform
Mounts static frontend, multi-format telemetry ingestion (CSV, TXT, LOG, TSV),
sliding-window analysis, explainability with physical units, and side-by-side flight comparison.
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from backend.preprocessing import TelemetryPreprocessor, PreprocessingError, UAV_21_FEATURES
from backend.prediction import PredictionPipeline, DEFAULT_THRESHOLD
from backend.anomaly import TelemetryAnomalyDetector
from backend.analysis import TelemetryExplainerAndLocalizer
from backend.report import FlightReportGenerator

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data", "demo_flights")
BENCHMARK_DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "backend", "flight_history.db")

app = FastAPI(
    title="AeroVigil - UAV Engine Health & Failure Prediction Platform",
    description="Software-only AI platform for real-time UAV flight telemetry analysis, sliding-window risk tracking, and early warning.",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
preprocessor = TelemetryPreprocessor()
pipeline = PredictionPipeline(threshold=DEFAULT_THRESHOLD)
anomaly_detector = TelemetryAnomalyDetector()
explainer = TelemetryExplainerAndLocalizer(threshold=DEFAULT_THRESHOLD)
report_generator = FlightReportGenerator()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS flight_history (
            flight_id TEXT PRIMARY KEY,
            filename TEXT,
            analyzed_at TEXT,
            health_score INTEGER,
            current_risk REAL,
            peak_risk REAL,
            average_risk REAL,
            status TEXT,
            anomaly_count INTEGER,
            high_risk_windows INTEGER,
            summary_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(analysis_result: Dict[str, Any]):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        flight_id = analysis_result["flight_id"]
        cur.execute("""
            INSERT OR REPLACE INTO flight_history 
            (flight_id, filename, analyzed_at, health_score, current_risk, peak_risk, average_risk, status, anomaly_count, high_risk_windows, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            flight_id,
            analysis_result["flight_file"],
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis_result["health_score"],
            analysis_result["current_failure_probability"],
            analysis_result["peak_failure_probability"],
            analysis_result["average_failure_probability"],
            analysis_result["status"],
            analysis_result["anomaly_summary"]["total_anomaly_points"],
            analysis_result["high_risk_windows_count"],
            json.dumps(analysis_result)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to history db: {e}")

def preload_demo_flights():
    """Ensure demo flights are pre-analyzed in SQLite for immediate comparison availability."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM flight_history")
        count = cur.fetchone()[0]
        conn.close()
        
        if count < 3 and os.path.exists(DATA_DIR):
            demo_files = [
                ("flight_01_healthy_nominal.csv", "FLT-A-NOMINAL"),
                ("flight_02_developing_propulsion_anomaly.csv", "FLT-B-ANOMALY"),
                ("flight_03_critical_pre_failure.csv", "FLT-C-CRITICAL")
            ]
            for filename, custom_id in demo_files:
                fpath = os.path.join(DATA_DIR, filename)
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    res = execute_flight_analysis(content, filename, flight_id=custom_id)
                    save_to_history(res)
    except Exception as e:
        print(f"Demo preload notice: {e}")

init_db()
preload_demo_flights()

class ThresholdUpdate(BaseModel):
    threshold: float

@app.get("/health")
def get_health():
    return {
        "status": "HEALTHY",
        "system": "AeroVigil UAV Engine Health Platform",
        "timestamp": datetime.datetime.now().isoformat(),
        "backend": "FastAPI + Pure NumPy Inference Engine",
        "model_loaded": pipeline.engine is not None
    }

@app.get("/model-info")
def get_model_info():
    config_path = os.path.join(BASE_DIR, "models", "uav_propulsion_gru", "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "model_name": "UAV Propulsion Health & Failure Prediction GRU",
            "feature_count": 21,
            "sequence_length": 20,
            "operating_threshold": pipeline.threshold
        }
    config["current_operating_threshold"] = pipeline.threshold
    config["available_models"] = [
        {
            "id": "uav_gru_21",
            "name": "Primary UAV GRU Model (21 Features)",
            "type": "Primary Original Model",
            "features": 21,
            "seq_len": 20,
            "status": "ACTIVE"
        },
        {
            "id": "turbofan_c_mapss_17",
            "name": "Turbofan Engine Degradation Model (NASA C-MAPSS FD001)",
            "type": "Secondary External Benchmark",
            "features": 17,
            "seq_len": 20,
            "status": "AVAILABLE"
        }
    ]
    return config

@app.post("/set-threshold")
def set_threshold(body: ThresholdUpdate):
    if not (0.05 <= body.threshold <= 0.95):
        raise HTTPException(status_code=400, detail="Threshold must be between 0.05 and 0.95.")
    pipeline.set_threshold(body.threshold)
    explainer.threshold = body.threshold
    return {
        "status": "SUCCESS",
        "operating_threshold": pipeline.threshold,
        "message": f"Configured operational failure threshold updated to {pipeline.threshold:.2f}."
    }

@app.get("/demo-flights")
def list_demo_flights():
    files = [
        {
            "id": "flight_01_healthy_nominal.csv",
            "flight_id": "FLT-A-NOMINAL",
            "name": "Flight A: Nominal Cruise (Healthy)",
            "description": "Baseline healthy flight. Low vibration (0.12 m/s² RMS), steady forward velocity (14 m/s). Health ~85%, Normal status.",
            "type": "healthy"
        },
        {
            "id": "flight_02_developing_propulsion_anomaly.csv",
            "flight_id": "FLT-B-ANOMALY",
            "name": "Flight B: Developing Propulsion Anomaly",
            "description": "Motor bearing degradation and roll oscillation mid-flight. Vibration surges to 2.45 m/s² RMS. Warning state.",
            "type": "developing_anomaly"
        },
        {
            "id": "flight_03_critical_pre_failure.csv",
            "flight_id": "FLT-C-CRITICAL",
            "name": "Flight C: Critical Pre-Failure Flight",
            "description": "Severe stator winding overheating & ESC commutation desync. Vibration spikes to 5.8 m/s² RMS. High risk state.",
            "type": "critical_failure"
        }
    ]
    return {"demo_flights": files}

@app.get("/benchmark-files")
def get_benchmark_files():
    """
    Returns list of preloaded benchmark dataset files (train and test only, strictly excluding RUL files).
    Removes train_FD001, train_FD003, test_FD001, test_FD003 as specified.
    """
    files = []
    excluded_files = {"train_FD001.txt", "train_FD003.txt", "test_FD001.txt", "test_FD003.txt"}
    if os.path.exists(BENCHMARK_DATA_DIR):
        for f in sorted(os.listdir(BENCHMARK_DATA_DIR)):
            if (f.startswith("train_") or f.startswith("test_")) and f.endswith(".txt") and not f.startswith("RUL_"):
                if f in excluded_files or "FD001" in f or "FD003" in f:
                    continue
                size_kb = round(os.path.getsize(os.path.join(BENCHMARK_DATA_DIR, f)) / 1024, 1)
                is_train = f.startswith("train_")
                fd_id = f.replace(".txt", "").split("_")[-1] if "_" in f else ""
                files.append({
                    "filename": f,
                    "name": f"{f} ({'Run-to-Failure' if is_train else 'Degradation Test'})",
                    "short_name": f,
                    "dataset_id": fd_id,
                    "size_kb": size_kb,
                    "type": "train" if is_train else "test"
                })
    return {"benchmark_files": files}

@app.get("/load-benchmark-file/{filename}")
def load_benchmark_file(filename: str):
    target_path = os.path.join(BENCHMARK_DATA_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Benchmark file '{filename}' not found.")
    with open(target_path, "r", encoding="utf-8") as f:
        content_str = f.read()
    
    clean_id = filename.replace(".txt", "").replace(".", "_").upper()
    flight_id = f"BMK-{clean_id}-UNIT01"
    return execute_flight_analysis(content_str, filename, flight_id=flight_id)

def execute_flight_analysis(file_content: str, filename: str, flight_id: Optional[str] = None) -> Dict[str, Any]:
    # 1. Parse & validate CSV / TXT / LOG / TSV
    parsed = preprocessor.validate_and_parse_file(file_content, filename)
    
    # 2. Feature engineering
    data_matrix, feature_names, timestamps = preprocessor.engineer_features(parsed)
    
    # 3. Scaling
    scaled_matrix = preprocessor.scale_features(data_matrix)
    
    # 4. Generate sliding windows (seq_len=20)
    windows = preprocessor.generate_sliding_windows(scaled_matrix, seq_len=20)
    
    # 5. AI sliding-window prediction
    pred_res = pipeline.predict_flight_risk(windows, timestamps)
    
    # 6. Telemetry anomaly detection (separate layer)
    anomaly_res = anomaly_detector.detect_anomalies(data_matrix, feature_names, timestamps)
    
    # 7. Localized events & explainability with physical units
    events = explainer.localize_events(pred_res["risk_timeline"], data_matrix, feature_names)
    indicators = explainer.explain_risk_increase(data_matrix, feature_names, pred_res["risk_timeline"])
    
    # 8. Maintenance recommendations
    recommendations = explainer.generate_recommendations(
        pred_res["status"],
        pred_res["peak_failure_probability"],
        indicators,
        anomaly_res["total_anomaly_points"]
    )
    
    if not flight_id:
        flight_id = f"FLT-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{parsed['valid_sample_count']}"
    
    # 9. Extract kinematic channels with physical units (supporting both 21-channel UAV & 17-channel Turbofan)
    if data_matrix.shape[1] >= 19:
        kinematics = {
            "timestamps": [round(float(t), 2) for t in timestamps],
            "angular_velocity_x": [round(float(v) * 57.2958, 2) for v in data_matrix[:, 0]], # deg/s
            "angular_velocity_y": [round(float(v) * 57.2958, 2) for v in data_matrix[:, 1]], # deg/s
            "angular_velocity_z": [round(float(v) * 57.2958, 2) for v in data_matrix[:, 2]], # deg/s
            "acceleration_x": [round(float(v), 3) for v in data_matrix[:, 3]], # m/s²
            "acceleration_y": [round(float(v), 3) for v in data_matrix[:, 4]], # m/s²
            "acceleration_z": [round(float(v), 3) for v in data_matrix[:, 5]], # m/s²
            "velocity_x": [round(float(v), 2) for v in data_matrix[:, 6]], # m/s
            "velocity_y": [round(float(v), 2) for v in data_matrix[:, 7]], # m/s
            "velocity_z": [round(float(v), 2) for v in data_matrix[:, 8]], # m/s
            "acceleration_magnitude": [round(float(v), 3) for v in data_matrix[:, 9]],
            "angular_velocity_magnitude": [round(float(v) * 57.2958, 2) for v in data_matrix[:, 10]],
            "velocity_magnitude": [round(float(v), 2) for v in data_matrix[:, 11]],
            "acceleration_std": [round(float(v), 3) for v in data_matrix[:, 16]], # m/s² RMS vibration
            "angular_velocity_std": [round(float(v) * 57.2958, 2) for v in data_matrix[:, 18]], # deg/s RMS jitter
        }
    else:
        # Turbofan 17-channel C-MAPSS telemetry mapping
        p30 = data_matrix[:, 6] if data_matrix.shape[1] > 6 else np.zeros(len(data_matrix))
        nc = data_matrix[:, 8] if data_matrix.shape[1] > 8 else np.zeros(len(data_matrix))
        t30 = data_matrix[:, 3] if data_matrix.shape[1] > 3 else np.zeros(len(data_matrix))
        nf = data_matrix[:, 7] if data_matrix.shape[1] > 7 else np.zeros(len(data_matrix))
        
        # Calculate dynamic fluctuations
        vib_proxy = np.abs(p30 - np.mean(p30[:20])) if len(p30) >= 20 else np.zeros_like(p30)
        jitter_proxy = np.abs(nc - np.mean(nc[:20])) / 10.0 if len(nc) >= 20 else np.zeros_like(nc)
        
        kinematics = {
            "timestamps": [round(float(t), 2) for t in timestamps],
            "angular_velocity_x": [round(float(v), 2) for v in (data_matrix[:, 0] if data_matrix.shape[1] > 0 else np.zeros(len(data_matrix)))],
            "angular_velocity_y": [round(float(v), 2) for v in (data_matrix[:, 1] if data_matrix.shape[1] > 1 else np.zeros(len(data_matrix)))],
            "angular_velocity_z": [round(float(v), 2) for v in jitter_proxy],
            "acceleration_x": [round(float(v), 3) for v in (data_matrix[:, 2] if data_matrix.shape[1] > 2 else np.zeros(len(data_matrix)))],
            "acceleration_y": [round(float(v), 3) for v in t30],
            "acceleration_z": [round(float(v), 3) for v in p30],
            "velocity_x": [round(float(v), 2) for v in nc],
            "velocity_y": [round(float(v), 2) for v in nf],
            "velocity_z": [round(float(v), 2) for v in (data_matrix[:, 9] if data_matrix.shape[1] > 9 else np.zeros(len(data_matrix)))],
            "acceleration_magnitude": [round(float(v), 3) for v in p30],
            "angular_velocity_magnitude": [round(float(v), 2) for v in nc],
            "velocity_magnitude": [round(float(v), 2) for v in nf],
            "acceleration_std": [round(float(v), 3) for v in vib_proxy],
            "angular_velocity_std": [round(float(v), 2) for v in jitter_proxy],
        }
    
    response = {
        "flight_id": flight_id,
        "flight_file": filename,
        "schema_type": parsed.get("schema_type", "uav_21"),
        "samples_available": parsed["raw_row_count"],
        "samples_analyzed": len(windows),
        "duration_seconds": round(float(timestamps[-1] - timestamps[0]), 1) if len(timestamps) > 1 else 0.0,
        "health_score": pred_res["health_score"],
        "current_failure_probability": pred_res["current_failure_probability"],
        "peak_failure_probability": pred_res["peak_failure_probability"],
        "average_failure_probability": pred_res["average_failure_probability"],
        "status": pred_res["status"],
        "current_risk_level": pred_res["current_risk_level"],
        "operating_threshold": pred_res["operating_threshold"],
        "high_risk_windows_count": pred_res["high_risk_windows_count"],
        "abnormal_windows_count": pred_res["abnormal_windows_count"],
        "risk_timeline": pred_res["risk_timeline"],
        "anomaly_summary": {
            "total_anomaly_points": anomaly_res["total_anomaly_points"],
            "anomaly_rate_percent": anomaly_res["anomaly_rate_percent"],
            "anomaly_events": anomaly_res["anomaly_event_clusters"]
        },
        "events": events,
        "indicators": indicators,
        "recommendations": recommendations,
        "kinematics": kinematics
    }
    
    save_to_history(response)
    return response

@app.post("/analyze-flight")
async def analyze_flight(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8", errors="replace")
        return execute_flight_analysis(content_str, file.filename)
    except PreprocessingError as pe:
        raise HTTPException(status_code=400, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    return await analyze_flight(file)

@app.get("/demo-flights/{filename}")
def load_demo_flight(filename: str):
    target_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Demo flight '{filename}' not found.")
    with open(target_path, "r", encoding="utf-8") as f:
        content_str = f.read()
    
    id_map = {
        "flight_01_healthy_nominal.csv": "FLT-A-NOMINAL",
        "flight_02_developing_propulsion_anomaly.csv": "FLT-B-ANOMALY",
        "flight_03_critical_pre_failure.csv": "FLT-C-CRITICAL"
    }
    custom_id = id_map.get(filename, None)
    return execute_flight_analysis(content_str, filename, flight_id=custom_id)

@app.get("/flight-history")
def get_flight_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT flight_id, filename, analyzed_at, health_score, current_risk, peak_risk, average_risk, status, anomaly_count, high_risk_windows 
        FROM flight_history 
        ORDER BY analyzed_at DESC 
        LIMIT 25
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"history": rows}

@app.get("/flight-history/{flight_id}")
def get_flight_history_item(flight_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT summary_json FROM flight_history WHERE flight_id = ? OR filename = ?", (flight_id, flight_id))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Flight record not found.")
    return json.loads(row[0])

class ComparisonRequest(BaseModel):
    flight_id_a: str
    flight_id_b: str

@app.post("/compare-flights")
def compare_flights(req: ComparisonRequest):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT summary_json FROM flight_history WHERE flight_id = ? OR filename = ?", (req.flight_id_a, req.flight_id_a))
    row_a = cur.fetchone()
    cur.execute("SELECT summary_json FROM flight_history WHERE flight_id = ? OR filename = ?", (req.flight_id_b, req.flight_id_b))
    row_b = cur.fetchone()
    conn.close()
    
    if not row_a or not row_b:
        raise HTTPException(status_code=404, detail="One or both flight records not found in history.")
        
    data_a = json.loads(row_a[0])
    data_b = json.loads(row_b[0])
    
    # Calculate physical kinematic metrics
    peak_vib_a = max(data_a["kinematics"]["acceleration_std"]) if "acceleration_std" in data_a["kinematics"] else 0.0
    peak_vib_b = max(data_b["kinematics"]["acceleration_std"]) if "acceleration_std" in data_b["kinematics"] else 0.0
    
    mean_speed_a = np.mean(data_a["kinematics"]["velocity_magnitude"]) if "velocity_magnitude" in data_a["kinematics"] else 0.0
    mean_speed_b = np.mean(data_b["kinematics"]["velocity_magnitude"]) if "velocity_magnitude" in data_b["kinematics"] else 0.0
    
    peak_jitter_a = max(data_a["kinematics"]["angular_velocity_std"]) if "angular_velocity_std" in data_a["kinematics"] else 0.0
    peak_jitter_b = max(data_b["kinematics"]["angular_velocity_std"]) if "angular_velocity_std" in data_b["kinematics"] else 0.0

    comparison = {
        "flight_a": {
            "flight_id": data_a["flight_id"],
            "filename": data_a["flight_file"],
            "health_score": data_a["health_score"],
            "status": data_a["status"],
            "peak_risk": data_a["peak_failure_probability"],
            "avg_risk": data_a["average_failure_probability"],
            "high_risk_windows": data_a["high_risk_windows_count"],
            "anomaly_points": data_a["anomaly_summary"]["total_anomaly_points"],
            "peak_vibration": f"{peak_vib_a:.2f} m/s² RMS",
            "mean_airspeed": f"{mean_speed_a:.1f} m/s",
            "peak_attitude_jitter": f"{peak_jitter_a:.1f}°/s RMS",
            "timeline": [r["failure_probability"] for r in data_a.get("risk_timeline", [])],
            "timestamps": [r["timestamp"] for r in data_a.get("risk_timeline", [])]
        },
        "flight_b": {
            "flight_id": data_b["flight_id"],
            "filename": data_b["flight_file"],
            "health_score": data_b["health_score"],
            "status": data_b["status"],
            "peak_risk": data_b["peak_failure_probability"],
            "avg_risk": data_b["average_failure_probability"],
            "high_risk_windows": data_b["high_risk_windows_count"],
            "anomaly_points": data_b["anomaly_summary"]["total_anomaly_points"],
            "peak_vibration": f"{peak_vib_b:.2f} m/s² RMS",
            "mean_airspeed": f"{mean_speed_b:.1f} m/s",
            "peak_attitude_jitter": f"{peak_jitter_b:.1f}°/s RMS",
            "timeline": [r["failure_probability"] for r in data_b.get("risk_timeline", [])],
            "timestamps": [r["timestamp"] for r in data_b.get("risk_timeline", [])]
        },
        "delta": {
            "health_score_diff": data_b["health_score"] - data_a["health_score"],
            "peak_risk_diff": round(data_b["peak_failure_probability"] - data_a["peak_failure_probability"], 4),
            "avg_risk_diff": round(data_b["average_failure_probability"] - data_a["average_failure_probability"], 4),
            "anomaly_delta": data_b["anomaly_summary"]["total_anomaly_points"] - data_a["anomaly_summary"]["total_anomaly_points"],
            "vibration_delta": round(peak_vib_b - peak_vib_a, 2),
            "airspeed_delta": round(mean_speed_b - mean_speed_a, 1)
        },
        "comparative_assessment": (
            f"Comparison of Flight B ({data_b['flight_file']}) against Baseline Flight A ({data_a['flight_file']}): "
            f"Engine Health changed by {data_b['health_score'] - data_a['health_score']:+d} points ({data_a['health_score']} → {data_b['health_score']}). "
            f"Peak Failure Risk shifted by {(data_b['peak_failure_probability'] - data_a['peak_failure_probability'])*100:+.1f}%. "
            f"Vibration changed from {peak_vib_a:.2f} m/s² RMS to {peak_vib_b:.2f} m/s² RMS ({peak_vib_b - peak_vib_a:+.2f} m/s² delta). "
            f"Status transitioned from {data_a['status']} to {data_b['status']}."
        )
    }
    return comparison

@app.get("/report/{flight_id}", response_class=HTMLResponse)
def get_flight_report(flight_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT summary_json FROM flight_history WHERE flight_id = ? OR filename = ?", (flight_id, flight_id))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Flight record not found.")
    data = json.loads(row[0])
    return report_generator.generate_html_report(data)

# Mount frontend files at root
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
