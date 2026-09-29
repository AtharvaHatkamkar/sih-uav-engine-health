/**
 * AeroVigil - UAV Propulsion Health & Early Warning Ground Station
 * Cockpit Flight Data Recorders, Dual-Trace Comparison, and Real Physical Units
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentAnalysis = null;
    let selectedFile = null;
    let operatingThreshold = 0.30;
    
    // Replay State
    let isReplaying = false;
    let replayInterval = null;
    let replayCurrentIndex = 0;
    let replaySpeed = 1;
    let totalSamples = 0;
    let currentTelemTab = "accel";
    
    // DOM Elements
    const dropzone = document.getElementById("csvDropzone");
    const fileInput = document.getElementById("fileInput");
    const fileStatusStrip = document.getElementById("fileStatusStrip");
    const dispFilename = document.getElementById("dispFilename");
    const dispSampleCount = document.getElementById("dispSampleCount");
    const dispSchemaBadge = document.getElementById("dispSchemaBadge");
    const btnStartAnalysis = document.getElementById("btnStartAnalysis");
    const analysisSpinner = document.getElementById("analysisSpinner");
    const btnStartText = document.getElementById("btnStartText");
    
    // Presets
    const btnDemoHealthy = document.getElementById("btnDemoHealthy");
    const btnDemoAnomaly = document.getElementById("btnDemoAnomaly");
    const btnDemoCritical = document.getElementById("btnDemoCritical");
    
    // Gauges & Status
    const gaugeHealthScore = document.getElementById("gaugeHealthScore");
    const gaugeHealthFill = document.getElementById("gaugeHealthFill");
    const gaugeRiskScore = document.getElementById("gaugeRiskScore");
    const gaugeRiskFill = document.getElementById("gaugeRiskFill");
    const dispMainStatus = document.getElementById("dispMainStatus");
    const dispStatusText = document.getElementById("dispStatusText");
    const dispStatusExplanation = document.getElementById("dispStatusExplanation");
    const dispThresholdVal = document.getElementById("dispThresholdVal");
    const dispThresholdRef = document.getElementById("dispThresholdRef");
    const dispFlightRef = document.getElementById("dispFlightRef");
    const dispSysState = document.getElementById("dispSysState");
    
    // Summary
    const dispMinHealth = document.getElementById("dispMinHealth");
    const dispPeakRisk = document.getElementById("dispPeakRisk");
    const dispAvgRisk = document.getElementById("dispAvgRisk");
    const dispRiskTrend = document.getElementById("dispRiskTrend");
    const dispAbnormalWindows = document.getElementById("dispAbnormalWindows");
    const dispHighRiskWindows = document.getElementById("dispHighRiskWindows");
    
    // Timeline Canvas
    const timelineCanvas = document.getElementById("riskTimelineCanvas");
    const timelineContainer = document.getElementById("timelineChartContainer");
    const timelineCursorLine = document.getElementById("timelineCursorLine");
    const timelineTooltip = document.getElementById("timelineTooltip");
    const dispTimelineEndLabel = document.getElementById("dispTimelineEndLabel");
    
    // Telemetry Canvas
    const telemetryCanvas = document.getElementById("telemetryCanvas");
    const telemLegendStrip = document.getElementById("telemLegendStrip");
    
    // Explainability & Events
    const indicatorList = document.getElementById("indicatorList");
    const eventList = document.getElementById("eventList");
    
    // Maintenance
    const maintCallout = document.getElementById("maintCallout");
    const subsysTagContainer = document.getElementById("subsysTagContainer");
    const recommendedActionsList = document.getElementById("recommendedActionsList");
    const targetCheckpointsList = document.getElementById("targetCheckpointsList");
    const airworthinessBadge = document.getElementById("airworthinessBadge");
    
    // Replay
    const btnReplayPlay = document.getElementById("btnReplayPlay");
    const btnReplayRestart = document.getElementById("btnReplayRestart");
    const btnReplayStop = document.getElementById("btnReplayStop");
    const replayScrubber = document.getElementById("replayScrubber");
    const replayCurrentTime = document.getElementById("replayCurrentTime");
    const replayTotalTime = document.getElementById("replayTotalTime");
    const replayStateBadge = document.getElementById("replayStateBadge");
    const progNodeNormal = document.getElementById("progNodeNormal");
    const progNodeAttn = document.getElementById("progNodeAttn");
    const progNodeWarn = document.getElementById("progNodeWarn");
    const progNodeHigh = document.getElementById("progNodeHigh");
    
    // Action Buttons
    const btnOpenReport = document.getElementById("btnOpenReport");
    const btnOpenHistory = document.getElementById("btnOpenHistory");
    const btnOpenCompare = document.getElementById("btnOpenCompare");
    
    // Modals
    const thresholdModal = document.getElementById("thresholdModal");
    const btnEditThreshold = document.getElementById("btnEditThreshold");
    const btnCloseThresholdModal = document.getElementById("btnCloseThresholdModal");
    const btnCancelThreshold = document.getElementById("btnCancelThreshold");
    const btnSaveThreshold = document.getElementById("btnSaveThreshold");
    const thresholdInput = document.getElementById("thresholdInput");
    const thresholdSliderValue = document.getElementById("thresholdSliderValue");
    
    const historyModal = document.getElementById("historyModal");
    const btnCloseHistoryModal = document.getElementById("btnCloseHistoryModal");
    const historyTableBody = document.getElementById("historyTableBody");
    
    const compareModal = document.getElementById("compareModal");
    const btnCloseCompareModal = document.getElementById("btnCloseCompareModal");
    const selectFlightA = document.getElementById("selectFlightA");
    const selectFlightB = document.getElementById("selectFlightB");
    const btnExecuteCompare = document.getElementById("btnExecuteCompare");
    const comparisonResults = document.getElementById("comparisonResults");
    const compareTimelineCanvas = document.getElementById("compareTimelineCanvas");
    const compareMatrixTbody = document.getElementById("compareMatrixTbody");
    const comparativeTextBox = document.getElementById("comparativeTextBox");
    
    const hudToast = document.getElementById("hudToast");

    function showToast(msg) {
        hudToast.textContent = msg;
        hudToast.classList.add("show");
        setTimeout(() => hudToast.classList.remove("show"), 3500);
    }

    function resizeCanvases() {
        if (timelineCanvas && timelineContainer) {
            timelineCanvas.width = timelineContainer.clientWidth;
            timelineCanvas.height = timelineContainer.clientHeight;
        }
        if (telemetryCanvas) {
            const parent = telemetryCanvas.parentElement;
            telemetryCanvas.width = parent.clientWidth;
            telemetryCanvas.height = parent.clientHeight;
        }
        if (currentAnalysis) {
            renderTimelineChart();
            renderTelemetryChart();
        }
    }
    window.addEventListener("resize", resizeCanvases);
    setTimeout(resizeCanvases, 100);

    // FETCH THRESHOLD & MODEL INFO
    async function loadModelInfo() {
        try {
            const res = await fetch("/model-info");
            const data = await res.json();
            operatingThreshold = data.current_operating_threshold || 0.30;
            updateThresholdDisplay();
        } catch (e) {
            console.warn("Could not fetch model info:", e);
        }
    }
    loadModelInfo();

    function updateThresholdDisplay() {
        dispThresholdVal.textContent = operatingThreshold.toFixed(2);
        dispThresholdRef.textContent = `Operating Threshold: ${(operatingThreshold * 100).toFixed(0)}%`;
        thresholdSliderValue.textContent = operatingThreshold.toFixed(2);
        thresholdInput.value = operatingThreshold;
    }

    // THRESHOLD MODAL EVENTS
    btnEditThreshold.addEventListener("click", () => thresholdModal.style.display = "flex");
    btnCloseThresholdModal.addEventListener("click", () => thresholdModal.style.display = "none");
    btnCancelThreshold.addEventListener("click", () => thresholdModal.style.display = "none");
    thresholdInput.addEventListener("input", (e) => thresholdSliderValue.textContent = parseFloat(e.target.value).toFixed(2));
    
    btnSaveThreshold.addEventListener("click", async () => {
        const newTh = parseFloat(thresholdInput.value);
        try {
            const res = await fetch("/set-threshold", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ threshold: newTh })
            });
            if (res.ok) {
                operatingThreshold = newTh;
                updateThresholdDisplay();
                thresholdModal.style.display = "none";
                showToast(`Operating threshold set to ${(newTh * 100).toFixed(0)}%`);
                if (selectedFile) {
                    analyzeFlightFile(selectedFile);
                }
            }
        } catch (err) {
            showToast("Failed to update threshold.");
        }
    });

    // 1-CLICK DEMO BUTTONS
    btnDemoHealthy.addEventListener("click", () => loadDemoFlightPreset("flight_01_healthy_nominal.csv", "FLIGHT A: Nominal Cruise"));
    btnDemoAnomaly.addEventListener("click", () => loadDemoFlightPreset("flight_02_developing_propulsion_anomaly.csv", "FLIGHT B: Motor Anomaly"));
    btnDemoCritical.addEventListener("click", () => loadDemoFlightPreset("flight_03_critical_pre_failure.csv", "FLIGHT C: Critical Pre-Failure"));

    async function loadDemoFlightPreset(filename, label) {
        setLoadingState(true, `INGESTING ${label.toUpperCase()}...`);
        try {
            const res = await fetch(`/demo-flights/${filename}`);
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Demo flight load failed.");
            }
            const data = await res.json();
            dispFilename.textContent = filename;
            dispSampleCount.textContent = `${data.samples_available} samples`;
            dispSchemaBadge.textContent = `SCHEMA: ${data.schema_type.toUpperCase()}`;
            fileStatusStrip.style.display = "block";
            btnStartAnalysis.disabled = false;
            
            processAnalysisResult(data);
            showToast(`Loaded ${label}`);
        } catch (e) {
            showToast(`Error: ${e.message}`);
        } finally {
            setLoadingState(false);
        }
    }

    // PRELOADED BENCHMARK FILE SELECTOR (FOR JUDGES & EVALUATION)
    const benchmarkSelect = document.getElementById("benchmarkSelect");
    const btnLoadBenchmark = document.getElementById("btnLoadBenchmark");

    async function initBenchmarkSelector() {
        if (!benchmarkSelect) return;
        try {
            const res = await fetch("/benchmark-files");
            const data = await res.json();
            if (data.benchmark_files && data.benchmark_files.length > 0) {
                benchmarkSelect.innerHTML = `<option value="" disabled selected>-- Select Benchmark File (FD002 / FD004) --</option>` + 
                    data.benchmark_files.map(bf => `
                        <option value="${bf.filename}">
                            ${bf.name}
                        </option>
                    `).join("");
            }
        } catch (e) {
            console.warn("Could not load benchmark files list:", e);
        }
    }
    initBenchmarkSelector();

    async function loadBenchmarkFilePreset(filename) {
        if (!filename) return;
        setLoadingState(true, `ANALYZING: ${filename}...`);
        try {
            const res = await fetch(`/load-benchmark-file/${filename}`);
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Benchmark file load failed.");
            }
            const data = await res.json();
            dispFilename.textContent = filename;
            dispSampleCount.textContent = `${data.samples_available} samples`;
            dispSchemaBadge.textContent = `SCHEMA: ${data.schema_type.toUpperCase()}`;
            fileStatusStrip.style.display = "block";
            btnStartAnalysis.disabled = false;
            
            processAnalysisResult(data);
            showToast(`Loaded and analyzed benchmark: ${filename}`);
        } catch (e) {
            showToast(`Error: ${e.message}`);
        } finally {
            setLoadingState(false);
        }
    }

    if (btnLoadBenchmark) {
        btnLoadBenchmark.addEventListener("click", () => {
            const chosen = benchmarkSelect.value;
            if (chosen) loadBenchmarkFilePreset(chosen);
            else showToast("Please select a benchmark file from the dropdown.");
        });
    }

    if (benchmarkSelect) {
        benchmarkSelect.addEventListener("change", (e) => {
            if (e.target.value) {
                loadBenchmarkFilePreset(e.target.value);
            }
        });
    }

    // MULTI-FORMAT DRAG AND DROP & FILE SELECTION (CSV, TXT, LOG, TSV, DAT)
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        const validExts = [".csv", ".txt", ".log", ".tsv", ".dat"];
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        if (!validExts.includes(ext)) {
            showToast("Supported file types: .csv, .txt, .log, .tsv, .dat");
            return;
        }
        selectedFile = file;
        dispFilename.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const lines = e.target.result.split("\n").filter(l => l.trim().length > 0);
            const count = Math.max(0, lines.length - 1);
            dispSampleCount.textContent = `${count} samples`;
            dispSchemaBadge.textContent = "INSPECTING FORMAT...";
            fileStatusStrip.style.display = "block";
            btnStartAnalysis.disabled = false;
            showToast(`Telemetry file loaded (${count} samples). Click 'Execute Telemetry Inference' to analyze.`);
        };
        reader.readAsText(file.slice(0, 100000));
    }

    btnStartAnalysis.addEventListener("click", () => {
        if (selectedFile) {
            analyzeFlightFile(selectedFile);
        }
    });

    async function analyzeFlightFile(file) {
        setLoadingState(true, "PROCESSING SLIDING WINDOWS...");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/analyze-flight", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Analysis failed.");
            }

            const data = await res.json();
            dispSchemaBadge.textContent = `SCHEMA: ${data.schema_type.toUpperCase()}`;
            processAnalysisResult(data);
            showToast("Flight telemetry analysis completed successfully.");
        } catch (e) {
            showToast(`Analysis Error: ${e.message}`);
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(loading, text = "EXECUTE TELEMETRY INFERENCE") {
        if (loading) {
            analysisSpinner.style.display = "inline-block";
            btnStartText.textContent = text;
            btnStartAnalysis.disabled = true;
            dispSysState.textContent = "ANALYZING";
            dispSysState.className = "state-badge status-active";
        } else {
            analysisSpinner.style.display = "none";
            btnStartText.textContent = "EXECUTE TELEMETRY INFERENCE";
            btnStartAnalysis.disabled = false;
        }
    }

    // PROCESS COMPLETE ANALYSIS RESULT
    function processAnalysisResult(data) {
        currentAnalysis = data;
        stopReplay();
        
        // Update Metadata
        dispFlightRef.textContent = data.flight_id;
        dispSysState.textContent = "SYNCED";
        dispSysState.className = "state-badge status-active";
        btnOpenReport.disabled = false;
        
        // Gauges & Status
        updateHealthGauges(data.health_score, data.current_failure_probability, data.status);
        
        // Flight Summary
        dispMinHealth.textContent = data.health_score;
        dispPeakRisk.textContent = `${(data.peak_failure_probability * 100).toFixed(1)}%`;
        dispAvgRisk.textContent = `${(data.average_failure_probability * 100).toFixed(1)}%`;
        dispAbnormalWindows.textContent = data.abnormal_windows_count;
        dispHighRiskWindows.textContent = data.high_risk_windows_count;
        
        const delta = data.peak_failure_probability - data.average_failure_probability;
        if (delta > 0.35) {
            dispRiskTrend.textContent = "SURGE ESCALATION";
            dispRiskTrend.style.color = "#ef4444";
        } else if (delta > 0.15) {
            dispRiskTrend.textContent = "ELEVATED DRIFT";
            dispRiskTrend.style.color = "#f97316";
        } else {
            dispRiskTrend.textContent = "STABLE";
            dispRiskTrend.style.color = "#10b981";
        }
        
        // Replay Bar limits
        totalSamples = data.kinematics.timestamps.length;
        replayScrubber.max = totalSamples - 1;
        replayScrubber.value = 0;
        const totalDuration = data.duration_seconds;
        replayTotalTime.textContent = `/ ${formatTime(totalDuration)}`;
        replayCurrentTime.textContent = `T+ 00:00.0`;

        // Render Charts
        resizeCanvases();
        
        // Render Explainability & Events
        renderIndicators(data.indicators);
        renderEvents(data.events);
        
        // Render Maintenance Guidance
        renderMaintenance(data.recommendations);
    }

    // UPDATE GAUGES AND STATUS
    function updateHealthGauges(health, riskProb, status) {
        const circ = 427.25;
        
        // Health Fill (Higher is better)
        const healthPct = Math.max(0, Math.min(100, health)) / 100.0;
        const healthOffset = circ - (circ * healthPct);
        gaugeHealthFill.style.strokeDashoffset = healthOffset;
        gaugeHealthScore.textContent = health;
        
        if (health >= 80) gaugeHealthFill.style.stroke = "#10b981";
        else if (health >= 60) gaugeHealthFill.style.stroke = "#f59e0b";
        else if (health >= 35) gaugeHealthFill.style.stroke = "#f97316";
        else gaugeHealthFill.style.stroke = "#ef4444";
        
        // Risk Fill (Lower is better)
        const riskPct = Math.max(0, Math.min(1.0, riskProb));
        const riskOffset = circ - (circ * riskPct);
        gaugeRiskFill.style.strokeDashoffset = riskOffset;
        gaugeRiskScore.textContent = `${(riskProb * 100).toFixed(0)}%`;
        
        if (riskProb >= 0.60) gaugeRiskFill.style.stroke = "#ef4444";
        else if (riskProb >= operatingThreshold) gaugeRiskFill.style.stroke = "#f97316";
        else if (riskProb >= (operatingThreshold * 0.66)) gaugeRiskFill.style.stroke = "#f59e0b";
        else gaugeRiskFill.style.stroke = "#10b981";
        
        // Status Annunciator
        dispStatusText.textContent = status;
        dispMainStatus.className = "status-banner-badge";
        
        if (status === "NORMAL") {
            dispMainStatus.classList.add("status-normal");
            dispStatusExplanation.textContent = "Telemetry parameters remain within nominal operating envelope.";
            highlightProgNode("NORMAL");
        } else if (status === "ATTENTION") {
            dispMainStatus.classList.add("status-attention");
            dispStatusExplanation.textContent = "Telemetry exhibiting localized dynamic variance or bearing vibration.";
            highlightProgNode("ATTENTION");
        } else if (status === "WARNING") {
            dispMainStatus.classList.add("status-warning-state");
            dispStatusExplanation.textContent = "Configured failure threshold exceeded. Pre-flight subsystem inspection advised.";
            highlightProgNode("WARNING");
        } else {
            dispMainStatus.classList.add("status-high-risk");
            dispStatusExplanation.textContent = "Critical propulsion instability detected. Mandatory engineering inspection.";
            highlightProgNode("HIGH RISK");
        }
    }

    function highlightProgNode(level) {
        [progNodeNormal, progNodeAttn, progNodeWarn, progNodeHigh].forEach(n => n.className = "prog-node");
        if (level === "NORMAL") progNodeNormal.classList.add("active");
        else if (level === "ATTENTION") progNodeAttn.classList.add("active-attn");
        else if (level === "WARNING") progNodeWarn.classList.add("active-warn");
        else if (level === "HIGH RISK") progNodeHigh.classList.add("active-high");
    }

    // RENDER RISK TIMELINE CHART
    function renderTimelineChart(cursorIndex = null) {
        if (!currentAnalysis || !timelineCanvas) return;
        const ctx = timelineCanvas.getContext("2d");
        const w = timelineCanvas.width;
        const h = timelineCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const timeline = currentAnalysis.risk_timeline;
        if (!timeline || timeline.length === 0) return;

        const padLeft = 45;
        const padRight = 15;
        const padTop = 15;
        const padBottom = 22;
        const chartW = w - padLeft - padRight;
        const chartH = h - padTop - padBottom;

        function yPos(prob) {
            return padTop + chartH * (1.0 - Math.max(0, Math.min(1.0, prob)));
        }

        // Zone fills
        ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
        ctx.fillRect(padLeft, yPos(1.0), chartW, yPos(0.60) - yPos(1.0));
        
        ctx.fillStyle = "rgba(249, 115, 22, 0.05)";
        ctx.fillRect(padLeft, yPos(0.60), chartW, yPos(operatingThreshold) - yPos(0.60));

        // Grid lines & labels
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#64748b";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(p => {
            const y = yPos(p);
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();
            ctx.fillText(`${(p * 100).toFixed(0)}%`, padLeft - 6, y + 3);
        });

        // Threshold Dashed Line
        const thY = yPos(operatingThreshold);
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padLeft, thY);
        ctx.lineTo(w - padRight, thY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = "#f97316";
        ctx.textAlign = "left";
        ctx.fillText(`OPERATING THRESHOLD (${(operatingThreshold * 100).toFixed(0)}%)`, padLeft + 6, thY - 4);

        // Plot Failure Probability Line
        ctx.beginPath();
        const numPts = timeline.length;
        for (let i = 0; i < numPts; i++) {
            const x = padLeft + (i / (numPts - 1)) * chartW;
            const y = yPos(timeline[i].failure_probability);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.0;
        ctx.stroke();

        // Area fill
        const lastX = padLeft + chartW;
        ctx.lineTo(lastX, padTop + chartH);
        ctx.lineTo(padLeft, padTop + chartH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
        grad.addColorStop(0, "rgba(56, 189, 248, 0.20)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // Cursor scrubber
        if (cursorIndex !== null && cursorIndex >= 0 && cursorIndex < numPts) {
            const curX = padLeft + (cursorIndex / (numPts - 1)) * chartW;
            const curY = yPos(timeline[cursorIndex].failure_probability);
            
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(curX, padTop);
            ctx.lineTo(curX, padTop + chartH);
            ctx.stroke();

            ctx.fillStyle = timeline[cursorIndex].failure_probability >= operatingThreshold ? "#ef4444" : "#10b981";
            ctx.beginPath();
            ctx.arc(curX, curY, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        const endTs = timeline[numPts - 1].timestamp;
        dispTimelineEndLabel.textContent = `T+ ${formatTime(endTs)}`;
    }

    // TIMELINE HOVER INTERACTION
    timelineContainer.addEventListener("mousemove", (e) => {
        if (!currentAnalysis) return;
        const rect = timelineContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const padLeft = 45;
        const padRight = 15;
        const chartW = rect.width - padLeft - padRight;
        
        if (mouseX >= padLeft && mouseX <= (rect.width - padRight)) {
            const ratio = (mouseX - padLeft) / chartW;
            const timeline = currentAnalysis.risk_timeline;
            const idx = Math.min(timeline.length - 1, Math.max(0, Math.floor(ratio * timeline.length)));
            const item = timeline[idx];
            
            timelineCursorLine.style.display = "block";
            timelineCursorLine.style.left = `${mouseX}px`;
            
            timelineTooltip.style.display = "block";
            timelineTooltip.style.left = `${Math.min(rect.width - 170, mouseX + 10)}px`;
            timelineTooltip.style.top = `20px`;
            timelineTooltip.innerHTML = `
                <div><strong>T+ ${formatTime(item.timestamp)}</strong> [Window #${item.window_index}]</div>
                <div>Failure Risk: <strong style="color:${getRiskColor(item.failure_probability)};">${(item.failure_probability * 100).toFixed(1)}%</strong></div>
                <div>Status: ${item.risk_level}</div>
            `;
            
            renderTimelineChart(idx);
        }
    });

    timelineContainer.addEventListener("mouseleave", () => {
        timelineCursorLine.style.display = "none";
        timelineTooltip.style.display = "none";
        if (currentAnalysis) {
            renderTimelineChart(isReplaying ? replayCurrentIndex : null);
        }
    });

    function getRiskColor(p) {
        if (p >= 0.60) return "#ef4444";
        if (p >= operatingThreshold) return "#f97316";
        if (p >= (operatingThreshold * 0.66)) return "#f59e0b";
        return "#10b981";
    }

    // TELEMETRY SENSOR TABS
    document.querySelectorAll(".telem-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".telem-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            currentTelemTab = tab.getAttribute("data-tab");
            renderTelemetryChart();
        });
    });

    // RENDER TELEMETRY MULTI-CHANNEL CHART WITH PHYSICAL UNITS
    function renderTelemetryChart() {
        if (!currentAnalysis || !telemetryCanvas) return;
        const ctx = telemetryCanvas.getContext("2d");
        const w = telemetryCanvas.width;
        const h = telemetryCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const padLeft = 45;
        const padRight = 15;
        const padTop = 20;
        const padBottom = 20;
        const chartW = w - padLeft - padRight;
        const chartH = h - padTop - padBottom;

        const kin = currentAnalysis.kinematics;
        let series = [];

        if (currentTelemTab === "accel") {
            series = [
                { name: "Ax (Surge)", data: kin.acceleration_x, unit: "m/s²", color: "#38bdf8" },
                { name: "Ay (Sway)", data: kin.acceleration_y, unit: "m/s²", color: "#a855f7" },
                { name: "Az (Heave)", data: kin.acceleration_z, unit: "m/s²", color: "#f59e0b" }
            ];
        } else if (currentTelemTab === "gyro") {
            series = [
                { name: "Roll Rate (p)", data: kin.angular_velocity_x, unit: "°/s", color: "#38bdf8" },
                { name: "Pitch Rate (q)", data: kin.angular_velocity_y, unit: "°/s", color: "#a855f7" },
                { name: "Yaw Rate (r)", data: kin.angular_velocity_z, unit: "°/s", color: "#f59e0b" }
            ];
        } else if (currentTelemTab === "vel") {
            series = [
                { name: "Forward Vx", data: kin.velocity_x, unit: "m/s", color: "#10b981" },
                { name: "Lateral Vy", data: kin.velocity_y, unit: "m/s", color: "#06b6d4" },
                { name: "Vertical Vz", data: kin.velocity_z, unit: "m/s", color: "#e2e8f0" }
            ];
        } else { // mags & jitter
            series = [
                { name: "Vibration RMS", data: kin.acceleration_std, unit: "m/s² RMS", color: "#ef4444" },
                { name: "Attitude Jitter", data: kin.angular_velocity_std, unit: "°/s RMS", color: "#f59e0b" },
                { name: "Resultant Speed", data: kin.velocity_magnitude, unit: "m/s", color: "#10b981" }
            ];
        }

        telemLegendStrip.innerHTML = series.map(s => `
            <span style="display:flex; align-items:center; gap:4px;">
                <span style="width:8px; height:8px; border-radius:1px; background:${s.color};"></span>
                <span>${s.name} [${s.unit}]</span>
            </span>
        `).join("");

        let minVal = Infinity;
        let maxVal = -Infinity;
        series.forEach(s => {
            s.data.forEach(v => {
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            });
        });
        const span = (maxVal - minVal) || 1.0;
        minVal -= span * 0.05;
        maxVal += span * 0.05;

        // Grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#64748b";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        [0, 0.25, 0.5, 0.75, 1.0].forEach(frac => {
            const y = padTop + chartH * (1.0 - frac);
            const val = minVal + span * frac;
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();
            ctx.fillText(val.toFixed(1), padLeft - 6, y + 3);
        });

        const N = series[0].data.length;
        series.forEach(s => {
            ctx.beginPath();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.5;
            for (let i = 0; i < N; i++) {
                const x = padLeft + (i / (N - 1)) * chartW;
                const norm = (s.data[i] - minVal) / (maxVal - minVal);
                const y = padTop + chartH * (1.0 - Math.max(0, Math.min(1.0, norm)));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });

        // Scrub cursor line
        if (isReplaying) {
            const curX = padLeft + (replayCurrentIndex / (N - 1)) * chartW;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(curX, padTop);
            ctx.lineTo(curX, padTop + chartH);
            ctx.stroke();
        }
    }

    // RENDER EXPLAINABILITY ATTRIBUTION WITH REAL PHYSICAL UNITS
    function renderIndicators(indicators) {
        if (!indicators || indicators.length === 0) {
            indicatorList.innerHTML = `<div class="empty-state-notice">Nominal flight envelope: All telemetry parameters within calibrated cruise margins.</div>`;
            return;
        }

        indicatorList.innerHTML = indicators.map(ind => `
            <div class="indicator-item">
                <div class="ind-top-row">
                    <span class="ind-name">${ind.display_name}</span>
                    <span class="ind-contrib">${ind.contribution_percent}% ATTRIBUTION</span>
                </div>
                <div class="ind-bar-track">
                    <div class="ind-bar-fill" style="width: ${ind.contribution_percent}%;"></div>
                </div>
                <div class="ind-physical-row">
                    <span>Baseline: <strong>${ind.baseline_val}</strong></span>
                    <span>Observed: <strong style="color:#f59e0b;">${ind.observed_val}</strong></span>
                    <span>Delta: <strong style="color:#ef4444;">${ind.physical_delta}</strong></span>
                </div>
                <div class="ind-desc">${ind.description} &bull; <strong style="color:#94a3b8;">${ind.subsystem}</strong></div>
            </div>
        `).join("");
    }

    // RENDER LOCALIZED ABNORMAL EVENTS
    function renderEvents(events) {
        if (!events || events.length === 0) {
            eventList.innerHTML = `<div class="empty-state-notice">No critical abnormal event intervals detected during flight.</div>`;
            return;
        }

        eventList.innerHTML = events.map(ev => {
            const sevColor = ev.severity === "CRITICAL" ? "#ef4444" : (ev.severity === "WARNING" ? "#f97316" : "#f59e0b");
            return `
                <div class="event-item" data-start="${ev.start_timestamp}" data-end="${ev.end_timestamp}" style="border-left-color: ${sevColor};">
                    <div class="event-top-row">
                        <span class="event-time-range">${ev.event_id} // ${ev.start_time_str} – ${ev.end_time_str} (${ev.duration_seconds}s)</span>
                        <span class="event-risk-jump" style="color: ${sevColor};">${ev.initial_risk_percent}% &rarr; ${ev.peak_risk_percent}%</span>
                    </div>
                    <div class="event-summary-text">${ev.summary}</div>
                </div>
            `;
        }).join("");

        document.querySelectorAll(".event-item").forEach(item => {
            item.addEventListener("click", () => {
                const startTs = parseFloat(item.getAttribute("data-start"));
                if (currentAnalysis) {
                    const tsList = currentAnalysis.kinematics.timestamps;
                    let targetIdx = tsList.findIndex(t => t >= startTs);
                    if (targetIdx !== -1) {
                        scrubToSample(targetIdx);
                        showToast(`Scrubbed simulation to event at ${formatTime(startTs)}`);
                    }
                }
            });
        });
    }

    // RENDER MAINTENANCE ACTIONS
    function renderMaintenance(recom) {
        if (!recom) return;
        maintCallout.textContent = recom.summary_statement || "Continue nominal flight operations.";
        airworthinessBadge.textContent = recom.airworthiness_status || "STATUS VERIFIED";
        
        if (recom.affected_subsystems && recom.affected_subsystems.length > 0) {
            subsysTagContainer.innerHTML = recom.affected_subsystems.map(s => `<span class="subsys-tag">${s}</span>`).join("");
        } else {
            subsysTagContainer.innerHTML = `<span class="subsys-tag">Nominal Propulsion</span>`;
        }
        
        if (recom.recommended_actions) {
            recommendedActionsList.innerHTML = recom.recommended_actions.map(a => `<li>${a}</li>`).join("");
        }
        
        if (recom.target_checkpoints) {
            targetCheckpointsList.innerHTML = recom.target_checkpoints.map(c => `<li>${c}</li>`).join("");
        }
    }

    // FLIGHT REPLAY ENGINE
    btnReplayPlay.addEventListener("click", toggleReplay);
    btnReplayRestart.addEventListener("click", () => scrubToSample(0));
    btnReplayStop.addEventListener("click", stopReplay);

    document.querySelectorAll(".speed-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            replaySpeed = parseInt(btn.getAttribute("data-speed"));
            if (isReplaying) {
                clearInterval(replayInterval);
                startReplayTimer();
            }
        });
    });

    replayScrubber.addEventListener("input", (e) => {
        scrubToSample(parseInt(e.target.value));
    });

    function toggleReplay() {
        if (!currentAnalysis) {
            showToast("Please load a flight log before starting simulation.");
            return;
        }
        if (isReplaying) pauseReplay();
        else startReplay();
    }

    function startReplay() {
        if (replayCurrentIndex >= totalSamples - 1) replayCurrentIndex = 0;
        isReplaying = true;
        btnReplayPlay.textContent = "[PAUSE]";
        replayStateBadge.textContent = "PLAYING";
        dispSysState.textContent = "REPLAY";
        dispSysState.className = "state-badge status-replay";
        startReplayTimer();
    }

    function startReplayTimer() {
        const intervalMs = Math.max(15, Math.floor(100 / replaySpeed));
        replayInterval = setInterval(() => {
            replayCurrentIndex++;
            if (replayCurrentIndex >= totalSamples) {
                stopReplay();
                return;
            }
            updateReplayFrame(replayCurrentIndex);
        }, intervalMs);
    }

    function pauseReplay() {
        isReplaying = false;
        clearInterval(replayInterval);
        btnReplayPlay.textContent = "[PLAY]";
        replayStateBadge.textContent = "PAUSED";
    }

    function stopReplay() {
        isReplaying = false;
        clearInterval(replayInterval);
        btnReplayPlay.textContent = "[PLAY]";
        replayStateBadge.textContent = "STANDBY";
        if (currentAnalysis) {
            dispSysState.textContent = "SYNCED";
            dispSysState.className = "state-badge status-active";
        }
    }

    function scrubToSample(idx) {
        replayCurrentIndex = Math.max(0, Math.min(totalSamples - 1, idx));
        replayScrubber.value = replayCurrentIndex;
        updateReplayFrame(replayCurrentIndex);
    }

    function updateReplayFrame(sampleIdx) {
        if (!currentAnalysis) return;
        const ts = currentAnalysis.kinematics.timestamps[sampleIdx];
        replayCurrentTime.textContent = `T+ ${formatTime(ts)}`;
        replayScrubber.value = sampleIdx;

        const timeline = currentAnalysis.risk_timeline;
        const winIdx = Math.min(timeline.length - 1, Math.max(0, sampleIdx - 19));
        const currentWindow = timeline[winIdx];

        if (currentWindow) {
            const curRisk = currentWindow.failure_probability;
            const health = Math.max(0, Math.min(100, Math.round(100 * (1 - curRisk))));
            updateHealthGauges(health, curRisk, currentWindow.risk_level);
        }

        renderTimelineChart(winIdx);
        renderTelemetryChart();
    }

    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return "00:00.0";
        const m = Math.floor(seconds / 60);
        const s = (seconds % 60).toFixed(1);
        return `${m.toString().padStart(2, '0')}:${s.padStart(4, '0')}`;
    }

    // REPORT EXPORT
    btnOpenReport.addEventListener("click", () => {
        if (currentAnalysis && currentAnalysis.flight_id) {
            window.open(`/report/${currentAnalysis.flight_id}`, "_blank");
        }
    });

    // FLIGHT HISTORY ARCHIVE MODAL
    btnOpenHistory.addEventListener("click", async () => {
        historyModal.style.display = "flex";
        try {
            const res = await fetch("/flight-history");
            const data = await res.json();
            if (data.history && data.history.length > 0) {
                historyTableBody.innerHTML = data.history.map(item => `
                    <tr>
                        <td style="font-weight:bold; color:#38bdf8;">${item.flight_id}</td>
                        <td>${item.filename}</td>
                        <td>${item.analyzed_at}</td>
                        <td style="font-weight:bold;">${item.health_score} / 100</td>
                        <td>${(item.peak_risk * 100).toFixed(1)}%</td>
                        <td>${(item.average_risk * 100).toFixed(1)}%</td>
                        <td><span class="val-badge ${item.status === 'NORMAL' ? 'success' : 'model-badge'}">${item.status}</span></td>
                        <td>
                            <button class="hud-btn hud-btn-outline" style="padding:2px 8px; font-size:10px;" onclick="loadHistoryItem('${item.flight_id}')">LOAD</button>
                        </td>
                    </tr>
                `).join("");
            } else {
                historyTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:#64748b;">No flight records found in history database.</td></tr>`;
            }
        } catch (e) {
            showToast("Failed to fetch history.");
        }
    });
    btnCloseHistoryModal.addEventListener("click", () => historyModal.style.display = "none");

    window.loadHistoryItem = async function(flightId) {
        historyModal.style.display = "none";
        setLoadingState(true, "LOADING HISTORICAL RECORD...");
        try {
            const res = await fetch(`/flight-history/${flightId}`);
            if (!res.ok) throw new Error("Could not retrieve flight record.");
            const data = await res.json();
            dispFilename.textContent = data.flight_file;
            dispSampleCount.textContent = `${data.samples_available} samples`;
            dispSchemaBadge.textContent = `SCHEMA: ${data.schema_type ? data.schema_type.toUpperCase() : 'UAV-21CH'}`;
            fileStatusStrip.style.display = "block";
            btnStartAnalysis.disabled = false;
            processAnalysisResult(data);
            showToast(`Loaded historical analysis ${flightId}`);
        } catch (e) {
            showToast(`Error: ${e.message}`);
        } finally {
            setLoadingState(false);
        }
    };

    // ADVANCED FLIGHT COMPARISON MODAL
    btnOpenCompare.addEventListener("click", async () => {
        compareModal.style.display = "flex";
        try {
            const res = await fetch("/flight-history");
            const data = await res.json();
            
            // Deduplicate options by flight_id
            const uniqueMap = new Map();
            if (data.history) {
                data.history.forEach(h => {
                    if (!uniqueMap.has(h.flight_id)) {
                        uniqueMap.set(h.flight_id, `${h.flight_id} [${h.filename}] - Health: ${h.health_score}`);
                    }
                });
            }
            
            const entries = Array.from(uniqueMap.entries());
            if (entries.length >= 2) {
                const opts = entries.map(([id, label]) => `<option value="${id}">${label}</option>`).join("");
                selectFlightA.innerHTML = opts;
                selectFlightB.innerHTML = opts;
                
                // Set default selection to Flight A vs Flight B if available
                if (uniqueMap.has("FLT-A-NOMINAL")) selectFlightA.value = "FLT-A-NOMINAL";
                if (uniqueMap.has("FLT-B-ANOMALY")) selectFlightB.value = "FLT-B-ANOMALY";
                else if (entries.length > 1) selectFlightB.selectedIndex = 1;
            } else {
                showToast("Need at least 2 flights in history to execute comparison.");
            }
        } catch (e) {
            console.error(e);
        }
    });
    btnCloseCompareModal.addEventListener("click", () => compareModal.style.display = "none");

    btnExecuteCompare.addEventListener("click", async () => {
        const idA = selectFlightA.value;
        const idB = selectFlightB.value;
        if (!idA || !idB || idA === idB) {
            showToast("Please select two distinct flights to compare.");
            return;
        }

        try {
            const res = await fetch("/compare-flights", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ flight_id_a: idA, flight_id_b: idB })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Comparison calculation failed.");
            }
            const comp = await res.json();
            
            renderComparisonMatrix(comp);
            renderComparisonChart(comp);
            comparisonResults.style.display = "block";
            showToast("Comparison analysis computed successfully.");
        } catch (e) {
            showToast(`Comparison Error: ${e.message}`);
        }
    });

    function renderComparisonMatrix(comp) {
        const a = comp.flight_a;
        const b = comp.flight_b;
        const d = comp.delta;

        const metrics = [
            {
                name: "Overall Engine Health",
                valA: `${a.health_score} / 100`,
                valB: `${b.health_score} / 100`,
                delta: `${d.health_score_diff >= 0 ? '+' : ''}${d.health_score_diff} pts`,
                eval: d.health_score_diff >= 0 ? "Nominal Preservation" : "Degradation Detected",
                color: d.health_score_diff >= 0 ? "#10b981" : "#ef4444"
            },
            {
                name: "Peak Failure Risk",
                valA: `${(a.peak_risk * 100).toFixed(1)}%`,
                valB: `${(b.peak_risk * 100).toFixed(1)}%`,
                delta: `${(d.peak_risk_diff * 100) >= 0 ? '+' : ''}${(d.peak_risk_diff * 100).toFixed(1)}%`,
                eval: d.peak_risk_diff <= 0 ? "Safe Risk Envelope" : "Elevated Risk Surge",
                color: d.peak_risk_diff <= 0 ? "#10b981" : "#ef4444"
            },
            {
                name: "Average Failure Risk",
                valA: `${(a.avg_risk * 100).toFixed(1)}%`,
                valB: `${(b.avg_risk * 100).toFixed(1)}%`,
                delta: `${(d.avg_risk_diff * 100) >= 0 ? '+' : ''}${(d.avg_risk_diff * 100).toFixed(1)}%`,
                eval: d.avg_risk_diff <= 0 ? "Stable Mean" : "Sustained Elevation",
                color: d.avg_risk_diff <= 0 ? "#10b981" : "#f97316"
            },
            {
                name: "Peak Propulsion Vibration",
                valA: a.peak_vibration,
                valB: b.peak_vibration,
                delta: `${d.vibration_delta >= 0 ? '+' : ''}${d.vibration_delta.toFixed(2)} m/s² RMS`,
                eval: d.vibration_delta <= 0.3 ? "Acceptable Bearing Wear" : "Bearing Resonance Alert",
                color: d.vibration_delta <= 0.3 ? "#10b981" : "#ef4444"
            },
            {
                name: "Attitude Rate Jitter (Yaw/Pitch)",
                valA: a.peak_attitude_jitter,
                valB: b.peak_attitude_jitter,
                delta: `${(parseFloat(b.peak_attitude_jitter) - parseFloat(a.peak_attitude_jitter)) >= 0 ? '+' : ''}${(parseFloat(b.peak_attitude_jitter) - parseFloat(a.peak_attitude_jitter)).toFixed(1)}°/s RMS`,
                eval: "Dynamic Balance Shift",
                color: "#38bdf8"
            },
            {
                name: "Mean Groundspeed / Airspeed",
                valA: a.mean_airspeed,
                valB: b.mean_airspeed,
                delta: `${d.airspeed_delta >= 0 ? '+' : ''}${d.airspeed_delta.toFixed(1)} m/s`,
                eval: d.airspeed_delta < -2.0 ? "Propulsion Thrust Loss" : "Nominal Thrust",
                color: d.airspeed_delta < -2.0 ? "#f97316" : "#10b981"
            },
            {
                name: "Operating Status Annunciator",
                valA: a.status,
                valB: b.status,
                delta: `${a.status} &rarr; ${b.status}`,
                eval: b.status === "NORMAL" ? "CLEARED" : "INSPECTION DIRECTIVE",
                color: b.status === "NORMAL" ? "#10b981" : "#ef4444"
            }
        ];

        compareMatrixTbody.innerHTML = metrics.map(m => `
            <tr>
                <td style="font-weight:600; color:#ffffff;">${m.name}</td>
                <td>${m.valA}</td>
                <td style="font-weight:bold;">${m.valB}</td>
                <td style="color:${m.color}; font-weight:bold;">${m.delta}</td>
                <td style="color:${m.color};">${m.eval}</td>
            </tr>
        `).join("");

        comparativeTextBox.textContent = comp.comparative_assessment;
    }

    function renderComparisonChart(comp) {
        if (!compareTimelineCanvas) return;
        const ctx = compareTimelineCanvas.getContext("2d");
        const parent = compareTimelineCanvas.parentElement;
        compareTimelineCanvas.width = parent.clientWidth;
        compareTimelineCanvas.height = parent.clientHeight;
        const w = compareTimelineCanvas.width;
        const h = compareTimelineCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const tA = comp.flight_a.timeline;
        const tB = comp.flight_b.timeline;
        if (!tA || !tB || tA.length === 0 || tB.length === 0) return;

        const padLeft = 45;
        const padRight = 15;
        const padTop = 15;
        const padBottom = 20;
        const chartW = w - padLeft - padRight;
        const chartH = h - padTop - padBottom;

        function yPos(p) {
            return padTop + chartH * (1.0 - Math.max(0, Math.min(1.0, p)));
        }

        // Grid
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#64748b";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        [0.0, 0.25, 0.5, 0.75, 1.0].forEach(p => {
            const y = yPos(p);
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();
            ctx.fillText(`${(p * 100).toFixed(0)}%`, padLeft - 6, y + 3);
        });

        // Threshold line
        const thY = yPos(operatingThreshold);
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padLeft, thY);
        ctx.lineTo(w - padRight, thY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Plot Flight A (Cyan)
        ctx.beginPath();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.0;
        for (let i = 0; i < tA.length; i++) {
            const x = padLeft + (i / (tA.length - 1)) * chartW;
            const y = yPos(tA[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Plot Flight B (Red / Orange)
        ctx.beginPath();
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2.0;
        for (let i = 0; i < tB.length; i++) {
            const x = padLeft + (i / (tB.length - 1)) * chartW;
            const y = yPos(tB[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Auto-load Demo Flight A on startup
    setTimeout(() => {
        loadDemoFlightPreset("flight_01_healthy_nominal.csv", "FLIGHT A: Nominal Cruise");
    }, 300);

});
