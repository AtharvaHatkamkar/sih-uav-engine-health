"""
Explainability, Event Localization, Subsystem Attribution, and Maintenance Action Recommendations
UAV Engine Health & Failure Prediction Platform
With Real Aerospace Physical Units (m/s², deg/s, m/s, RPM, kPa, °C)
"""

from typing import Dict, Any, List, Optional
import numpy as np

TELEMETRY_METRICS_META = {
    # UAV 21-Feature Channels
    "angular_velocity_x": {"name": "Roll Angular Rate (p)", "unit": "deg/s", "scale": 57.2958, "subsystem": "Rotor Dynamic Balance & Roll Authority"},
    "angular_velocity_y": {"name": "Pitch Angular Rate (q)", "unit": "deg/s", "scale": 57.2958, "subsystem": "Pitch Tilt Mechanism & Dynamic Balance"},
    "angular_velocity_z": {"name": "Yaw Angular Rate (r)", "unit": "deg/s", "scale": 57.2958, "subsystem": "Yaw Counter-Torque & Motor Balance"},
    "acceleration_x": {"name": "Surge Acceleration (Ax)", "unit": "m/s²", "scale": 1.0, "subsystem": "Forward Thrust & Drag Dynamics"},
    "acceleration_y": {"name": "Sway Acceleration (Ay)", "unit": "m/s²", "scale": 1.0, "subsystem": "Lateral Rotor Force Imbalance"},
    "acceleration_z": {"name": "Heave Acceleration (Az)", "unit": "m/s²", "scale": 1.0, "subsystem": "Lift / Vertical Thrust Reserve"},
    "velocity_x": {"name": "Longitudinal Velocity (Vx)", "unit": "m/s", "scale": 1.0, "subsystem": "Propulsion Airspeed Maintenance"},
    "velocity_y": {"name": "Lateral Velocity (Vy)", "unit": "m/s", "scale": 1.0, "subsystem": "Crosswind Drift Dynamics"},
    "velocity_z": {"name": "Vertical Rate (Vz)", "unit": "m/s", "scale": 1.0, "subsystem": "Climb / Descent Thrust Reserve"},
    "acceleration_magnitude": {"name": "Total Dynamic Acceleration", "unit": "m/s²", "scale": 1.0, "subsystem": "Airframe Dynamic Stress Load"},
    "angular_velocity_magnitude": {"name": "Total Angular Rate", "unit": "deg/s", "scale": 57.2958, "subsystem": "Attitude Stability & Commutation"},
    "velocity_magnitude": {"name": "Resultant Airspeed / Groundspeed", "unit": "m/s", "scale": 1.0, "subsystem": "Propulsion Thrust Generation"},
    "acceleration_change": {"name": "Kinematic Jerk Shock", "unit": "m/s³", "scale": 10.0, "subsystem": "Motor Mount Structural Integrity"},
    "angular_velocity_change": {"name": "Attitude Rate Perturbation", "unit": "deg/s²", "scale": 572.958, "subsystem": "ESC Torque Delivery & Phase Commutation"},
    "velocity_change": {"name": "Thrust Surge / Drag Step", "unit": "m/s²", "scale": 10.0, "subsystem": "Motor Thrust Continuity"},
    "acceleration_mean": {"name": "Rolling Mean Acceleration", "unit": "m/s²", "scale": 1.0, "subsystem": "Sustained Thrust Baseline"},
    "acceleration_std": {"name": "High-Frequency Vibration (RMS)", "unit": "m/s² RMS", "scale": 1.0, "subsystem": "Motor Bearing & Shaft Balance"},
    "angular_velocity_mean": {"name": "Mean Rotational Rate Drift", "unit": "deg/s", "scale": 57.2958, "subsystem": "Trim Torque Compensation"},
    "angular_velocity_std": {"name": "Rotational Rate Jitter (RMS)", "unit": "deg/s RMS", "scale": 57.2958, "subsystem": "Rotor Dynamic Resonance"},
    "velocity_mean": {"name": "Sustained Velocity Baseline", "unit": "m/s", "scale": 1.0, "subsystem": "Cruise Velocity Envelope"},
    "velocity_std": {"name": "Airspeed Turbulence Deviation", "unit": "m/s", "scale": 1.0, "subsystem": "Aerodynamic Stability"},

    # Turbofan Channels (NASA C-MAPSS)
    "op_setting_1": {"name": "Operational Altitude / Mach", "unit": "Mach", "scale": 1.0, "subsystem": "Flight Condition Regime"},
    "op_setting_2": {"name": "Engine Throttle Setting", "unit": "TRA", "scale": 1.0, "subsystem": "Engine Throttle Command"},
    "sensor_2": {"name": "LPC Outlet Temperature (T24)", "unit": "°C", "scale": 1.0, "subsystem": "Low Pressure Compressor (LPC)"},
    "sensor_3": {"name": "HPC Outlet Temperature (T30)", "unit": "°C", "scale": 1.0, "subsystem": "High Pressure Compressor (HPC)"},
    "sensor_4": {"name": "LPT Outlet Temperature (T50)", "unit": "°C", "scale": 1.0, "subsystem": "Low Pressure Turbine (LPT)"},
    "sensor_6": {"name": "Interstage Pressure (P15)", "unit": "psia", "scale": 1.0, "subsystem": "Bypass Duct Pressure"},
    "sensor_7": {"name": "HPC Outlet Pressure (P30)", "unit": "psia", "scale": 1.0, "subsystem": "High Pressure Compressor (HPC)"},
    "sensor_8": {"name": "Physical Fan Speed (Nf)", "unit": "RPM", "scale": 1.0, "subsystem": "Low Pressure Spool (Fan)"},
    "sensor_9": {"name": "Physical Core Speed (Nc)", "unit": "RPM", "scale": 1.0, "subsystem": "High Pressure Core Spool"},
    "sensor_11": {"name": "Static HPC Outlet Pressure (Ps30)", "unit": "psia", "scale": 1.0, "subsystem": "Combustor Diffuser Pressure"},
    "sensor_12": {"name": "Ratio of Fuel Flow to Ps30", "unit": "pps/psi", "scale": 1.0, "subsystem": "Fuel Metering & Injection"},
    "sensor_13": {"name": "Corrected Fan Speed (NRf)", "unit": "RPM", "scale": 1.0, "subsystem": "Fan Rotor Aerodynamics"},
    "sensor_14": {"name": "Corrected Core Speed (NRc)", "unit": "RPM", "scale": 1.0, "subsystem": "Core Spool Aerodynamics"},
    "sensor_15": {"name": "Engine Bypass Ratio (BPR)", "unit": "ratio", "scale": 1.0, "subsystem": "Thermodynamic Bypass Circuit"},
    "sensor_17": {"name": "Bleed Enthalpy (htble)", "unit": "Btu/lb", "scale": 1.0, "subsystem": "Pneumatic Bleed System"},
    "sensor_20": {"name": "HPT Coolant Bleed Flow", "unit": "lbm/s", "scale": 1.0, "subsystem": "High Pressure Turbine Cooling"},
    "sensor_21": {"name": "LPT Coolant Bleed Flow", "unit": "lbm/s", "scale": 1.0, "subsystem": "Low Pressure Turbine Cooling"}
}

class TelemetryExplainerAndLocalizer:
    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def localize_events(
        self,
        risk_timeline: List[Dict[str, Any]],
        data_matrix: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Identifies discrete time intervals of degradation and extracts physical telemetry indicators.
        """
        events = []
        in_event = False
        event_start_idx = 0
        event_start_risk = 0.0
        event_start_ts = 0.0
        
        warn_th = self.threshold * 0.75
        
        for i, item in enumerate(risk_timeline):
            prob = item["failure_probability"]
            ts = item["timestamp"]
            
            if not in_event and prob >= warn_th:
                in_event = True
                event_start_idx = i
                event_start_risk = risk_timeline[max(0, i - 1)]["failure_probability"]
                event_start_ts = ts
                
            elif in_event and (prob < warn_th or i == len(risk_timeline) - 1):
                in_event = False
                event_end_idx = i
                event_end_ts = ts
                
                sub_probs = [r["failure_probability"] for r in risk_timeline[event_start_idx:event_end_idx+1]]
                max_risk = max(sub_probs)
                
                if (event_end_ts - event_start_ts) >= 1.0 or max_risk >= self.threshold:
                    s_row = event_start_idx
                    e_row = min(len(data_matrix), event_end_idx + 20)
                    window_data = data_matrix[s_row:e_row]
                    
                    indicators = self._calculate_top_indicators(window_data, data_matrix, feature_names)
                    
                    def to_mmss(seconds):
                        m = int(seconds // 60)
                        s = int(seconds % 60)
                        return f"{m:02d}:{s:02d}"
                        
                    events.append({
                        "event_id": f"EVT-{len(events)+1:02d}",
                        "start_timestamp": round(event_start_ts, 2),
                        "end_timestamp": round(event_end_ts, 2),
                        "start_time_str": to_mmss(event_start_ts),
                        "end_time_str": to_mmss(event_end_ts),
                        "duration_seconds": round(event_end_ts - event_start_ts, 1),
                        "initial_risk_percent": round(event_start_risk * 100.0, 1),
                        "peak_risk_percent": round(max_risk * 100.0, 1),
                        "risk_delta_percent": round((max_risk - event_start_risk) * 100.0, 1),
                        "severity": "CRITICAL" if max_risk >= 0.60 else ("WARNING" if max_risk >= self.threshold else "ATTENTION"),
                        "contributing_indicators": indicators[:3],
                        "summary": (
                            f"Risk escalated from {event_start_risk*100:.0f}% to {max_risk*100:.0f}% "
                            f"between {to_mmss(event_start_ts)} and {to_mmss(event_end_ts)}."
                        )
                    })
                    
        return events

    def explain_risk_increase(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        risk_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Determines overall flight contributing indicators with actual physical units.
        """
        high_risk_indices = [
            i for i, r in enumerate(risk_timeline)
            if r["failure_probability"] >= (self.threshold * 0.70)
        ]
        
        if not high_risk_indices:
            peak_idx = max(range(len(risk_timeline)), key=lambda i: risk_timeline[i]["failure_probability"])
            high_risk_indices = [peak_idx]
            
        high_risk_rows = []
        for idx in high_risk_indices:
            high_risk_rows.extend(range(idx, min(len(data_matrix), idx + 20)))
        high_risk_rows = sorted(list(set(high_risk_rows)))
        
        target_data = data_matrix[high_risk_rows]
        return self._calculate_top_indicators(target_data, data_matrix, feature_names, top_n=6)

    def _calculate_top_indicators(
        self,
        target_data: np.ndarray,
        full_data: np.ndarray,
        feature_names: List[str],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Computes standardized shift, percentage attribution, and physical values with real aerospace units.
        """
        calm_len = max(20, min(50, int(0.20 * len(full_data))))
        baseline = full_data[:calm_len]
        base_means = np.mean(baseline, axis=0)
        base_stds = np.std(baseline, axis=0)
        base_stds[base_stds < 1e-4] = 1.0
        
        target_means = np.mean(target_data, axis=0)
        shifts = np.abs(target_means - base_means) / base_stds
        
        diagnostic_weights = {
            "acceleration_std": 1.50,
            "angular_velocity_std": 1.45,
            "angular_velocity_change": 1.35,
            "acceleration_change": 1.30,
            "velocity_change": 1.25,
            "velocity_magnitude": 1.20,
            "acceleration_magnitude": 1.15,
            "sensor_4": 1.40,
            "sensor_9": 1.40,
            "sensor_14": 1.35,
            "sensor_11": 1.30
        }
        
        weighted_scores = np.zeros(len(feature_names))
        for j, feat in enumerate(feature_names):
            w = diagnostic_weights.get(feat, 1.0)
            weighted_scores[j] = shifts[j] * w
            
        total_score = np.sum(weighted_scores)
        if total_score < 1e-5:
            total_score = 1.0
            
        ranked_indices = np.argsort(weighted_scores)[::-1]
        
        indicators = []
        for rank, idx in enumerate(ranked_indices[:top_n]):
            feat = feature_names[idx]
            meta = TELEMETRY_METRICS_META.get(feat, {
                "name": feat.replace("_", " ").title(),
                "unit": "units",
                "scale": 1.0,
                "subsystem": "General Airframe / Dynamics"
            })
            
            unit = meta["unit"]
            scale = meta["scale"]
            subsystem = meta["subsystem"]
            human_name = meta["name"]
            
            b_val = float(base_means[idx]) * scale
            o_val = float(target_means[idx]) * scale
            diff = o_val - b_val
            pct_contrib = round((weighted_scores[idx] / total_score) * 100.0, 1)
            
            # Physical formatting
            if "RPM" in unit:
                b_str = f"{b_val:.0f} RPM"
                o_str = f"{o_val:.0f} RPM"
                d_str = f"{diff:+.0f} RPM"
            elif "°C" in unit:
                b_str = f"{b_val:.1f} °C"
                o_str = f"{o_val:.1f} °C"
                d_str = f"{diff:+.1f} °C"
            elif "deg/s" in unit:
                b_str = f"{abs(b_val):.1f}°/s"
                o_str = f"{abs(o_val):.1f}°/s"
                d_str = f"{diff:+.1f}°/s"
            else:
                b_str = f"{b_val:.2f} {unit}"
                o_str = f"{o_val:.2f} {unit}"
                d_str = f"{diff:+.2f} {unit}"
                
            # Plain language explanation with physical units
            if "vibration" in human_name.lower():
                desc = f"Vibration elevated from {b_str} to {o_str} ({d_str} increase above cruise)."
            elif "angular rate" in human_name.lower() or "jitter" in human_name.lower():
                desc = f"Attitude rate oscillation surged from {b_str} to {o_str} ({d_str} perturbation)."
            elif "velocity" in human_name.lower() and diff < 0:
                desc = f"Forward airspeed degraded from {b_str} to {o_str} ({d_str} thrust loss)."
            elif "temperature" in human_name.lower():
                desc = f"Thermal sensor increased from {b_str} to {o_str} ({d_str} thermal drift)."
            elif "speed" in human_name.lower() or "rpm" in unit.lower():
                desc = f"Spool speed drifted from {b_str} to {o_str} ({d_str} deviation)."
            else:
                desc = f"{human_name} shifted from {b_str} to {o_str} ({d_str} deviation)."
                
            indicators.append({
                "feature_name": feat,
                "display_name": human_name,
                "unit": unit,
                "subsystem": subsystem,
                "baseline_val": b_str,
                "observed_val": o_str,
                "physical_delta": d_str,
                "contribution_percent": pct_contrib,
                "description": desc
            })
            
        return indicators

    def generate_recommendations(
        self,
        overall_status: str,
        peak_risk: float,
        indicators: List[Dict[str, Any]],
        anomaly_count: int
    ) -> Dict[str, Any]:
        top_subsystems = list(set(ind["subsystem"] for ind in indicators[:3]))
        
        if overall_status == "NORMAL":
            urgency = "ROUTINE"
            summary_statement = "Telemetry parameters remain within nominal flight envelope. Continue standard operational monitoring."
            actions = [
                "Proceed with scheduled flight mission without intervention.",
                "Verify standard pre-arm rotor balance and battery connector security.",
                "Maintain periodic flight telemetry archiving for fleet trend monitoring."
            ]
            checkpoints = [
                "Visual inspection of propeller leading edge for micro-nicks.",
                "Routine motor temperature check post-flight."
            ]
            airworthiness_status = "CLEARED FOR NORMAL OPERATIONS"
            
        elif overall_status == "ATTENTION":
            urgency = "ADVISORY"
            summary_statement = "Minor telemetry dynamic perturbations detected. Subsystem inspection recommended prior to next sortie."
            actions = [
                "Review flight logs focusing on timestamps with elevated rate-of-change.",
                "Perform physical hand-rotation check of all propulsion motors to detect bearing friction.",
                "Verify propeller balance and hub tightening torques before subsequent flights."
            ]
            checkpoints = [
                f"Inspect {top_subsystems[0]} for mechanical play or resonant vibration.",
                "Inspect ESC signal leads and power distribution solder joints."
            ]
            airworthiness_status = "PRE-FLIGHT SUBSYSTEM REVIEW ADVISED"
            
        elif overall_status == "WARNING":
            urgency = "ELEVATED"
            summary_statement = "Significant propulsion risk detected. Subsystem inspection recommended prior to next flight."
            actions = [
                "Ground UAV for dedicated mechanical inspection of propulsion subsystem.",
                "Inspect motor bearings for axial play, radial runout, or thermal discoloration.",
                "Perform static bench thrust test to measure current draw and vibration spectrum."
            ]
            checkpoints = [
                f"Primary Focus: {', '.join(top_subsystems)}.",
                "Check for stator winding phase imbalance and ESC thermal throttling logs.",
                "Check propeller dynamic balance using digital vibration analyzer."
            ]
            airworthiness_status = "GROUNDED PENDING SUBSYSTEM INSPECTION"
            
        else: # HIGH RISK
            urgency = "CRITICAL"
            summary_statement = "Severe propulsion instability detected. Mandatory engineering inspection required before continued operation."
            actions = [
                "Ground aircraft immediately. Do not attempt armed takeoff.",
                "Perform full teardown and bench diagnostic of affected propulsion units.",
                "Examine motor stator windings for partial insulation breakdown or phase short.",
                "Check Electronic Speed Controller (ESC) logs for commutation desynchronization or over-current faults."
            ]
            checkpoints = [
                "Full motor disassembly and stator resistance measurement across all phases.",
                "Digital dynamic balancing of all rotor hubs and replacement of fatigued bearings.",
                "Airframe motor-mount spar integrity and fastener torque verification."
            ]
            airworthiness_status = "GROUNDED - MANDATORY ENGINEERING DISASSEMBLY"
            
        return {
            "urgency": urgency,
            "airworthiness_status": airworthiness_status,
            "summary_statement": summary_statement,
            "recommended_actions": actions,
            "target_checkpoints": checkpoints,
            "affected_subsystems": top_subsystems,
            "disclaimer": (
                "These recommendations represent software-assisted engineering guidance based on telemetry analysis. "
                "They do not constitute certified airworthiness approval or a definitive physical root-cause warranty."
            )
        }
