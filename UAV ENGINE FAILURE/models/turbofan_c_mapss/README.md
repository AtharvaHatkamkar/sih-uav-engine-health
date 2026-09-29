# Turbofan Benchmark Time-Series Model (C-MAPSS FD001)

## 1. Model Source & Provenance
- **Original Source**: NASA Prognostics Center of Excellence (PCoE)
- **Dataset**: Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Turbofan Engine Degradation Simulation Dataset (`FD001`)
- **Originator**: A. Saxena, K. Goebel, D. Simon, N. Eklund (NASA Ames Research Center)
- **License**: Public Domain (NASA Open Source Data Policy)

## 2. Architecture & Input/Output Specification
- **Model Architecture**: GRU (Gated Recurrent Unit, 64 units) -> Dropout (0.2) -> Dense (32, ReLU) -> Dense (1, Linear)
- **Input Dimensions**: `(batch_size, sequence_length=20, feature_count=17)`
- **Input Normalization**: Fitted `MinMaxScaler` across 17 selected operational settings and engine sensor degradation channels (`minmax_scaler.joblib`)
- **Included Features (17 channels)**:
  - `op_setting_1`, `op_setting_2`
  - `sensor_2` (LPC outlet temperature)
  - `sensor_3` (HPC outlet temperature)
  - `sensor_4` (LPT outlet temperature)
  - `sensor_6` (Interstage pressure)
  - `sensor_7` (HPC outlet pressure)
  - `sensor_8` (Physical fan speed)
  - `sensor_9` (Physical core speed)
  - `sensor_11` (Static pressure at HPC outlet)
  - `sensor_12` (Ratio of fuel flow to static pressure)
  - `sensor_13` (Corrected fan speed)
  - `sensor_14` (Corrected core speed)
  - `sensor_15` (Bypass ratio)
  - `sensor_17` (Bleed enthalpy)
  - `sensor_20` (HPT coolant bleed)
  - `sensor_21` (LPT coolant bleed)
- **Model Output**: Predicted Remaining Useful Life (RUL) in operational cycles.
- **Critical Risk Threshold**: Cycles $\le 30$ denotes critical failure zone (mapped to high failure probability).

## 3. Anomaly Detection Layer
- **Model**: `IsolationForest` (`isolation_forest.joblib`) with 100 estimators, contamination ~0.05.
- **Function**: Multivariate out-of-distribution engine anomaly scoring.

## 4. Attribution & Limitations
- This model is strictly an external turbofan engine benchmark model and is preserved for technical comparability.
- It is NOT the primary rotary-wing/fixed-wing UAV kinematic telemetry model.
- Aircraft turbofan thermodynamics differ in physics from brushless DC (BLDC) electric motors and small internal combustion UAV engines.
