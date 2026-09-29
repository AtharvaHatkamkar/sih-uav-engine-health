"""
Digital Flight Health & Early Warning Report Generator
Generates comprehensive aerospace-grade inspection reports with print/PDF styling.
"""

from typing import Dict, Any
import datetime

DISCLAIMER_TEXT = (
    "NOTICE: This system provides an AI-assisted risk assessment based on available telemetry "
    "and should not be treated as a certified aviation safety system or a definitive diagnosis "
    "of physical component failure. All maintenance actions must comply with manufacturer manuals "
    "and certified airworthiness inspections."
)

class FlightReportGenerator:
    def __init__(self):
        pass

    def generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        flight_name = analysis_result.get("flight_file", "UAV_TELEMETRY_LOG")
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        health = analysis_result.get("health_score", 100)
        status = analysis_result.get("status", "NORMAL")
        peak_risk = analysis_result.get("peak_failure_probability", 0.0) * 100.0
        avg_risk = analysis_result.get("average_failure_probability", 0.0) * 100.0
        curr_risk = analysis_result.get("current_failure_probability", 0.0) * 100.0
        total_samples = analysis_result.get("samples_available", 0)
        analyzed_samples = analysis_result.get("samples_analyzed", 0)
        operating_threshold = analysis_result.get("operating_threshold", 0.30) * 100.0
        
        events = analysis_result.get("events", [])
        indicators = analysis_result.get("indicators", [])
        recommendations = analysis_result.get("recommendations", {})
        actions = recommendations.get("recommended_actions", [])
        checkpoints = recommendations.get("target_checkpoints", [])
        
        # Color badge
        color_map = {
            "NORMAL": "#10b981",
            "ATTENTION": "#f59e0b",
            "WARNING": "#f97316",
            "HIGH RISK": "#ef4444"
        }
        status_color = color_map.get(status, "#06b6d4")
        
        # Events HTML rows
        events_html = ""
        if events:
            for ev in events:
                ev_color = "#ef4444" if ev.get("severity") == "CRITICAL" else ("#f97316" if ev.get("severity") == "WARNING" else "#f59e0b")
                top_ind = ", ".join([ind["display_name"] for ind in ev.get("contributing_indicators", [])])
                events_html += f"""
                <tr>
                    <td style="font-family: monospace; font-weight: bold;">{ev.get('event_id')}</td>
                    <td>{ev.get('start_time_str')} – {ev.get('end_time_str')} ({ev.get('duration_seconds')}s)</td>
                    <td>{ev.get('initial_risk_percent')}% → <strong style="color: {ev_color};">{ev.get('peak_risk_percent')}%</strong></td>
                    <td><span style="background: {ev_color}22; color: {ev_color}; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{ev.get('severity')}</span></td>
                    <td style="font-size: 12px; color: #475569;">{top_ind if top_ind else 'Dynamic jitter variance'}</td>
                </tr>
                """
        else:
            events_html = """<tr><td colspan="5" style="text-align: center; color: #64748b; padding: 15px;">No critical abnormal event intervals detected during flight.</td></tr>"""

        # Indicators HTML rows
        ind_html = ""
        for ind in indicators:
            ind_html += f"""
            <tr>
                <td style="font-weight: 600;">{ind.get('display_name')}</td>
                <td style="font-family: monospace; color: #64748b;">{ind.get('feature_name')}</td>
                <td>{ind.get('subsystem')}</td>
                <td style="font-weight: bold; color: #0284c7;">{ind.get('contribution_percent')}%</td>
                <td style="font-family: monospace; font-weight: 600; color: #0f172a;">{ind.get('baseline_val')} &rarr; {ind.get('observed_val')} ({ind.get('physical_delta')})</td>
            </tr>
            """

        # Actions HTML
        actions_html = "".join([f"<li style='margin-bottom: 6px;'>{act}</li>" for act in actions])
        checkpoints_html = "".join([f"<li style='margin-bottom: 6px;'>{chk}</li>" for chk in checkpoints])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AeroVigil Flight Health Report - {flight_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            background: #ffffff;
            margin: 0;
            padding: 30px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .report-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0f172a;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .brand-title {{
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #0f172a;
            margin: 0;
        }}
        .brand-sub {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }}
        .meta-box {{
            text-align: right;
            font-size: 11px;
            color: #475569;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }}
        .kpi-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
        }}
        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .kpi-val {{
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            font-family: "SF Mono", Monaco, Consolas, monospace;
        }}
        .section-title {{
            font-size: 15px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #1e293b;
            border-left: 4px solid #0284c7;
            padding-left: 10px;
            margin-top: 25px;
            margin-bottom: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f1f5f9;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #475569;
            font-weight: 700;
        }}
        .guidance-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 20px;
        }}
        .disclaimer-box {{
            background: #fffbeb;
            border: 1px solid #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 12px 15px;
            font-size: 11px;
            color: #92400e;
            border-radius: 4px;
            margin-top: 30px;
        }}
        @media print {{
            body {{ padding: 15px; font-size: 12px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <div>
            <h1 class="brand-title">AEROVIGIL // UAV PROPULSION HEALTH REPORT</h1>
            <div class="brand-sub">Autonomous Telemetry Diagnostics & Early-Warning Platform</div>
        </div>
        <div class="meta-box">
            <div><strong>Report Ref:</strong> REP-{flight_name.replace('.csv', '')}</div>
            <div><strong>Generated:</strong> {generated_at}</div>
            <div><strong>Flight Source:</strong> {flight_name}</div>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Engine Health Index</div>
            <div class="kpi-val" style="color: {status_color};">{health} <span style="font-size: 14px; font-weight: normal; color: #64748b;">/ 100</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Failure Risk</div>
            <div class="kpi-val">{peak_risk:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Risk</div>
            <div class="kpi-val">{avg_risk:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Current Status</div>
            <div style="margin-top: 4px;">
                <span class="badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color};">{status}</span>
            </div>
        </div>
    </div>

    <div class="section-title">Flight Telemetry Scope & Configuration</div>
    <table>
        <tr>
            <th>Telemetry Records</th>
            <th>Analyzed Windows</th>
            <th>Time-Series Model</th>
            <th>Configured Threshold</th>
            <th>Airworthiness Status</th>
        </tr>
        <tr>
            <td>{total_samples} samples</td>
            <td>{analyzed_samples} windows (seq_len=20)</td>
            <td>Primary UAV GRU (21 Features)</td>
            <td>{operating_threshold:.1f}%</td>
            <td style="font-weight: bold; color: {status_color};">{recommendations.get('airworthiness_status', 'STATUS VERIFIED')}</td>
        </tr>
    </table>

    <div class="section-title">Localized Anomaly / Degradation Events</div>
    <table>
        <thead>
            <tr>
                <th>Event ID</th>
                <th>Time Window</th>
                <th>Risk Progression</th>
                <th>Severity</th>
                <th>Dominant Telemetry Signatures</th>
            </tr>
        </thead>
        <tbody>
            {events_html}
        </tbody>
    </table>

    <div class="section-title">Key Contributing Telemetry Indicators (Attribution)</div>
    <table>
        <thead>
            <tr>
                <th>Telemetry Parameter</th>
                <th>Signal Channel</th>
                <th>Affected Subsystem</th>
                <th>Contribution %</th>
                <th>Physical Baseline &rarr; Observed (Delta)</th>
            </tr>
        </thead>
        <tbody>
            {ind_html}
        </tbody>
    </table>

    <div class="section-title">Recommended Engineering Actions & Pre-Flight Guidance</div>
    <div class="guidance-box">
        <div style="font-weight: 700; color: #1e293b; margin-bottom: 8px;">{recommendations.get('summary_statement', '')}</div>
        <div style="font-weight: 600; margin-top: 10px; color: #334155;">Required Maintenance Directives:</div>
        <ul style="padding-left: 20px; color: #475569;">
            {actions_html}
        </ul>
        <div style="font-weight: 600; margin-top: 10px; color: #334155;">Subsystem Checkpoints:</div>
        <ul style="padding-left: 20px; color: #475569;">
            {checkpoints_html}
        </ul>
    </div>

    <div class="disclaimer-box">
        <strong>HONEST TECHNICAL LIMITATIONS & DISCLAIMER:</strong><br>
        {DISCLAIMER_TEXT}
    </div>

    <div class="no-print" style="margin-top: 30px; text-align: center;">
        <button onclick="window.print()" style="background: #0284c7; color: white; border: none; padding: 10px 24px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px;">Print / Save as PDF</button>
    </div>
</body>
</html>
        """
        return html
