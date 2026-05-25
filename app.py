# ============================================================
# IoT-Based Real-Time Alert System for Faulty Urban Elevators
# ============================================================
# Run with: streamlit run app.py
# Requirements: pip install streamlit plotly numpy pandas scikit-learn
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import io

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ElevatorGuard AI",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for industrial-dark dashboard aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #e6edf3 !important;
}

/* Main header */
.main-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #58a6ff;
    letter-spacing: 2px;
    border-bottom: 2px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 0.85rem;
    color: #8b949e;
    letter-spacing: 1px;
    margin-top: -0.8rem;
    margin-bottom: 1.5rem;
}

/* KPI Cards */
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: border-color 0.3s;
}
.kpi-card:hover { border-color: #58a6ff; }
.kpi-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    margin-top: 0.2rem;
}
.kpi-unit { font-size: 0.9rem; color: #8b949e; }

/* Alert boxes */
.alert-fault {
    background: #2d1117;
    border: 2px solid #f85149;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    color: #f85149;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: 1px;
}
.alert-safe {
    background: #0d2119;
    border: 2px solid #3fb950;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    color: #3fb950;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: 1px;
}

/* Section titles */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
    border-left: 3px solid #58a6ff;
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem;
}

/* Streamlit metric override */
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #58a6ff !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Barlow', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }

/* Download button */
.stDownloadButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stDownloadButton > button:hover { background: #2ea043; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ML MODEL (RandomForest or Rule-Based Fallback)
# ─────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """
    Tries to load a pre-saved pickle model.
    Falls back to training a simple RandomForestClassifier
    on synthetic data if no model file is found.
    """
    try:
        import pickle, os
        if os.path.exists("elevator_model.pkl"):
            with open("elevator_model.pkl", "rb") as f:
                return pickle.load(f), "loaded"
    except Exception:
        pass

    # Train a lightweight RandomForest on synthetic labeled data
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    n = 2000
    load      = np.random.uniform(0, 1200, n)
    vib       = np.random.uniform(0, 10,   n)
    temp      = np.random.uniform(15, 100, n)
    door_open = np.random.randint(0, 2,    n)   # 1 = open, 0 = closed
    floors    = np.random.randint(0, 50,   n)
    power_fluc= np.random.randint(0, 2,    n)   # 1 = fluctuating

    # Rule-based labels for synthetic training
    labels = (
        (load > 800) |
        (vib > 7) |
        (temp > 70) |
        ((door_open == 1) & (floors > 0)) |
        (power_fluc == 1)
    ).astype(int)

    X = np.column_stack([load, vib, temp, door_open, floors, power_fluc])
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    clf.fit(X, labels)
    return clf, "trained"

model, model_source = load_or_train_model()


# ─────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────
def predict_elevator(load_kg, vibration, temperature, door_open, floor_count, power_fluctuating):
    """
    Returns prediction label, fault type, confidence %, and risk score.
    Uses ML model confidence + rule-based fault type classification.
    """
    door_bin  = 1 if door_open  == "Open"        else 0
    power_bin = 1 if power_fluctuating == "Fluctuating" else 0

    features = np.array([[load_kg, vibration, temperature, door_bin, floor_count, power_bin]])

    # ML prediction
    prob_fault = model.predict_proba(features)[0][1]  # probability of fault
    prediction = "Fault Detected" if prob_fault >= 0.5 else "Safe"
    confidence = round(prob_fault * 100 if prediction == "Fault Detected" else (1 - prob_fault) * 100, 1)

    # Rule-based fault type classification (priority order)
    fault_type = "None"
    if prediction == "Fault Detected":
        if load_kg > 800:
            fault_type = "Overload"
        elif temperature > 70:
            fault_type = "Overheating"
        elif vibration > 7:
            fault_type = "Motor Failure"
        elif door_bin == 1 and floor_count > 0:
            fault_type = "Door Malfunction"
        elif power_bin == 1:
            fault_type = "Power Instability"
        else:
            fault_type = "General Fault"

    # Risk score: weighted combination of sensor readings (1–10 scale)
    risk = (
        min(load_kg / 1200, 1) * 3 +
        (vibration / 10) * 3 +
        min((temperature - 15) / 85, 1) * 2 +
        door_bin * 1 +
        power_bin * 1
    )
    risk_score = round(min(max(risk, 1), 10), 1)

    return prediction, fault_type, confidence, risk_score


# ─────────────────────────────────────────────
# IOT SENSOR SIMULATION
# ─────────────────────────────────────────────
def simulate_sensors(fault_scenario=False):
    """
    Randomly generate IoT sensor readings.
    fault_scenario=True biases values toward fault conditions.
    """
    if fault_scenario:
        return {
            "load_kg":      round(random.uniform(750, 1200), 1),
            "vibration":    round(random.uniform(6, 10), 2),
            "temperature":  round(random.uniform(65, 95), 1),
            "door_status":  random.choice(["Open", "Open", "Closed"]),
            "floor_count":  random.randint(1, 40),
            "power_status": random.choice(["Fluctuating", "Fluctuating", "Normal"]),
        }
    else:
        return {
            "load_kg":      round(random.uniform(100, 700), 1),
            "vibration":    round(random.uniform(0, 5), 2),
            "temperature":  round(random.uniform(20, 55), 1),
            "door_status":  random.choice(["Closed", "Closed", "Open"]),
            "floor_count":  random.randint(0, 20),
            "power_status": random.choice(["Normal", "Normal", "Fluctuating"]),
        }


def generate_elevator_fleet(n=8):
    """Generate a fleet of n elevators with random sensor readings and predictions."""
    rows = []
    for i in range(1, n + 1):
        fault = random.random() < 0.35  # ~35% chance of fault
        s = simulate_sensors(fault_scenario=fault)
        pred, ftype, conf, risk = predict_elevator(
            s["load_kg"], s["vibration"], s["temperature"],
            s["door_status"], s["floor_count"], s["power_status"]
        )
        rows.append({
            "Elevator ID": f"ELV-{i:03d}",
            "Load (kg)":   s["load_kg"],
            "Vibration":   s["vibration"],
            "Temp (°C)":   s["temperature"],
            "Door":        s["door_status"],
            "Floors":      s["floor_count"],
            "Power":       s["power_status"],
            "Status":      pred,
            "Fault Type":  ftype,
            "Confidence%": conf,
            "Risk Score":  risk,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
if "sensors"       not in st.session_state: st.session_state.sensors = simulate_sensors()
if "vib_history"   not in st.session_state: st.session_state.vib_history  = []
if "temp_history"  not in st.session_state: st.session_state.temp_history  = []
if "load_history"  not in st.session_state: st.session_state.load_history  = []
if "time_history"  not in st.session_state: st.session_state.time_history  = []
if "fleet_df"      not in st.session_state: st.session_state.fleet_df = generate_elevator_fleet()


# ─────────────────────────────────────────────
# SIDEBAR – MANUAL INPUT + CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛗 ElevatorGuard AI")
    st.markdown("**IoT Real-Time Fault Detection**")
    st.markdown("---")

    st.markdown("### ⚙️ Manual Sensor Input")
    load_kg     = st.slider("Load Weight (kg)",       0,   1200, int(st.session_state.sensors["load_kg"]),    step=5)
    vibration   = st.slider("Vibration Level (0–10)", 0.0, 10.0, float(st.session_state.sensors["vibration"]), step=0.1)
    temperature = st.slider("Temperature (°C)",       15,  100,  int(st.session_state.sensors["temperature"]), step=1)
    door_status = st.selectbox("Door Status",         ["Closed", "Open"],
                               index=0 if st.session_state.sensors["door_status"] == "Closed" else 1)
    floor_count = st.slider("Floor Count Movement",   0,   50,   int(st.session_state.sensors["floor_count"]), step=1)
    power_status= st.selectbox("Power Status",        ["Normal", "Fluctuating"],
                               index=0 if st.session_state.sensors["power_status"] == "Normal" else 1)

    st.markdown("---")
    st.markdown("### 🎲 Simulation Controls")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🟢 Sim Safe"):
            st.session_state.sensors = simulate_sensors(fault_scenario=False)
            st.rerun()
    with col_s2:
        if st.button("🔴 Sim Fault"):
            st.session_state.sensors = simulate_sensors(fault_scenario=True)
            st.rerun()

    if st.button("🔄 Refresh Fleet", use_container_width=True):
        st.session_state.fleet_df = generate_elevator_fleet()
        st.rerun()

    st.markdown("---")
    st.markdown(f"<small>Model: {model_source.upper()} | RandomForest</small>", unsafe_allow_html=True)
    st.markdown(f"<small>Last updated: {datetime.now().strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RUN PREDICTION ON CURRENT INPUTS
# ─────────────────────────────────────────────
prediction, fault_type, confidence, risk_score = predict_elevator(
    load_kg, vibration, temperature, door_status, floor_count, power_status
)

# Append to time-series history (keep last 30 readings)
now = datetime.now()
st.session_state.vib_history.append(vibration)
st.session_state.temp_history.append(temperature)
st.session_state.load_history.append(load_kg)
st.session_state.time_history.append(now)
if len(st.session_state.vib_history) > 30:
    for key in ["vib_history", "temp_history", "load_history", "time_history"]:
        st.session_state[key] = st.session_state[key][-30:]


# ─────────────────────────────────────────────
# MAIN DASHBOARD HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🛗 ELEVATORGUARD AI — REAL-TIME FAULT DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IoT Sensor Monitoring · Predictive Analytics · Urban Infrastructure Safety</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1: KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Live Sensor KPIs</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_data = [
    (k1, f"{load_kg}",     "Load",        "kg"),
    (k2, f"{vibration}",   "Vibration",   "/ 10"),
    (k3, f"{temperature}", "Temperature", "°C"),
    (k4, f"{floor_count}", "Floors",      "moved"),
    (k5, f"{risk_score}",  "Risk Score",  "/ 10"),
    (k6, f"{confidence}%", "Confidence",  ""),
]
for col, val, label, unit in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-unit">{unit}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 2: ALERT BOX + SAFETY GAUGE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🚨 AI Prediction & Safety Gauge</div>', unsafe_allow_html=True)
alert_col, gauge_col = st.columns([1, 1])

with alert_col:
    if prediction == "Fault Detected":
        st.markdown(f"""
        <div class="alert-fault">
            ⚠️ FAULT DETECTED<br>
            <span style="font-size:1.3rem">Fault Type: <strong>{fault_type}</strong></span><br>
            <span style="font-size:0.85rem">Confidence: {confidence}% &nbsp;|&nbsp; Risk: {risk_score}/10</span><br>
            <span style="font-size:0.8rem; color:#ff7b72">Maintenance action required immediately.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe">
            ✅ SYSTEM SAFE<br>
            <span style="font-size:1.3rem">All sensors within normal range.</span><br>
            <span style="font-size:0.85rem">Confidence: {confidence}% &nbsp;|&nbsp; Risk: {risk_score}/10</span><br>
            <span style="font-size:0.8rem; color:#56d364">No immediate action required.</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sensor status breakdown
    status_items = [
        ("Load Weight",   load_kg,     800,  "kg",  "🔴 OVERLOAD"       if load_kg > 800  else "🟢 OK"),
        ("Vibration",     vibration,   7,    "",    "🔴 MOTOR ISSUE"    if vibration > 7  else "🟢 OK"),
        ("Temperature",   temperature, 70,   "°C",  "🔴 OVERHEATING"    if temperature>70 else "🟢 OK"),
        ("Door Status",   None,        None, "",    "🔴 DOOR OPEN (moving)" if door_status=="Open" and floor_count>0 else "🟢 OK"),
        ("Power Status",  None,        None, "",    "🔴 FLUCTUATING"    if power_status=="Fluctuating" else "🟢 NORMAL"),
    ]
    sensor_display = []
    for name, val, thresh, unit, status in status_items:
        sensor_display.append({"Sensor": name, "Status": status})
    st.dataframe(pd.DataFrame(sensor_display), use_container_width=True, hide_index=True)

with gauge_col:
    # Safety gauge using Plotly indicator
    gauge_val = round((1 - risk_score / 10) * 100, 1)  # Convert risk to safety %
    gauge_color = "#3fb950" if gauge_val >= 50 else "#f85149"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        title={"text": "Safety Score (%)", "font": {"size": 14, "color": "#8b949e", "family": "Share Tech Mono"}},
        delta={"reference": 70, "increasing": {"color": "#3fb950"}, "decreasing": {"color": "#f85149"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e", "size": 10}},
            "bar": {"color": gauge_color},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  40], "color": "#2d1117"},
                {"range": [40, 70], "color": "#1f2d1f"},
                {"range": [70,100], "color": "#0d2119"},
            ],
            "threshold": {
                "line": {"color": "#58a6ff", "width": 3},
                "thickness": 0.8,
                "value": 70
            }
        },
        number={"font": {"size": 36, "color": gauge_color, "family": "Share Tech Mono"},
                "suffix": "%"}
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#c9d1d9"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3: TREND CHARTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Sensor Trend Analysis</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

times = st.session_state.time_history
vibs  = st.session_state.vib_history
temps = st.session_state.temp_history
loads = st.session_state.load_history

# Chart 1: Vibration Trend
with chart_col1:
    fig_vib = go.Figure()
    fig_vib.add_trace(go.Scatter(
        x=times, y=vibs,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=5, color=["#f85149" if v > 7 else "#58a6ff" for v in vibs]),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        name="Vibration"
    ))
    fig_vib.add_hline(y=7, line_dash="dash", line_color="#f85149",
                      annotation_text="FAULT THRESHOLD (7)", annotation_font_color="#f85149")
    fig_vib.update_layout(
        title=dict(text="Vibration Level — Live Trend", font=dict(color="#8b949e", size=13, family="Share Tech Mono")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e"),
        xaxis=dict(gridcolor="#21262d", showgrid=True),
        yaxis=dict(gridcolor="#21262d", showgrid=True, range=[0, 10]),
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_vib, use_container_width=True)

# Chart 2: Temperature vs Load scatter
with chart_col2:
    fig_tl = go.Figure()
    colors_tl = ["#f85149" if t > 70 or l > 800 else "#3fb950" for t, l in zip(temps, loads)]
    fig_tl.add_trace(go.Scatter(
        x=loads, y=temps,
        mode="markers",
        marker=dict(size=9, color=colors_tl, opacity=0.85,
                    line=dict(width=1, color="#30363d")),
        name="Readings"
    ))
    fig_tl.add_hline(y=70,  line_dash="dash", line_color="#f85149",  annotation_text="Temp Limit (70°C)",  annotation_font_color="#f85149")
    fig_tl.add_vline(x=800, line_dash="dash", line_color="#d29922",  annotation_text="Load Limit (800kg)", annotation_font_color="#d29922")
    fig_tl.update_layout(
        title=dict(text="Temperature vs. Load — Fault Zone Map", font=dict(color="#8b949e", size=13, family="Share Tech Mono")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e"),
        xaxis=dict(title="Load (kg)", gridcolor="#21262d"),
        yaxis=dict(title="Temperature (°C)", gridcolor="#21262d"),
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_tl, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4: ELEVATOR FLEET TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🏢 Elevator Fleet Status — All Units</div>', unsafe_allow_html=True)

fleet_df = st.session_state.fleet_df

# Color-code the Status column using Styler
def style_status(val):
    if val == "Fault Detected":
        return "color: #f85149; font-weight: bold;"
    return "color: #3fb950; font-weight: bold;"

def style_risk(val):
    if val >= 7:  return "color: #f85149;"
    if val >= 4:  return "color: #d29922;"
    return "color: #3fb950;"

styled_fleet = (
    fleet_df.style
    .applymap(style_risk,   subset=["Risk Score"])
    .set_properties(**{"background-color": "#161b22", "color": "#c9d1d9", "border": "1px solid #30363d"})
    .set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#21262d"), ("color", "#8b949e"),
                  ("font-family", "Share Tech Mono"), ("font-size", "0.75rem"),
                  ("letter-spacing", "1px")]
    }])
)
st.dataframe(styled_fleet, use_container_width=True, hide_index=True)

# Fleet summary bar
total   = len(fleet_df)
faults  = (fleet_df["Status"] == "Fault Detected").sum()
safe    = total - faults

f1, f2, f3 = st.columns(3)
f1.metric("🏢 Total Elevators", total)
f2.metric("✅ Safe Units",  safe,   delta=f"{round(safe/total*100)}% operational")
f3.metric("⚠️ Fault Units", faults, delta=f"-{round(faults/total*100)}% need inspection", delta_color="inverse")


# ─────────────────────────────────────────────
# SECTION 5: FAULT TYPE DISTRIBUTION CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Fault Type Distribution — Fleet</div>', unsafe_allow_html=True)

fault_counts = fleet_df[fleet_df["Status"] == "Fault Detected"]["Fault Type"].value_counts().reset_index()
fault_counts.columns = ["Fault Type", "Count"]

if not fault_counts.empty:
    fig_bar = px.bar(
        fault_counts, x="Fault Type", y="Count",
        color="Fault Type",
        color_discrete_sequence=["#f85149", "#d29922", "#58a6ff", "#bc8cff", "#3fb950"],
        text="Count"
    )
    fig_bar.update_traces(textposition="outside", textfont_color="#c9d1d9")
    fig_bar.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="Share Tech Mono"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        showlegend=False,
        height=260,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.success("✅ No faults detected in the current fleet scan.")


# ─────────────────────────────────────────────
# SECTION 6: CSV DOWNLOAD REPORT
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📥 Export Report</div>', unsafe_allow_html=True)

# Build comprehensive report
report_df = fleet_df.copy()
report_df.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
report_df["Report Generated By"] = "ElevatorGuard AI v1.0"

csv_buffer = io.StringIO()
report_df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    st.download_button(
        label="⬇️ Download Fleet Report (.csv)",
        data=csv_data,
        file_name=f"elevator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
with dl_col2:
    st.markdown(f"<small style='color:#8b949e'>Report includes {len(report_df)} elevator records · "
                f"{faults} fault(s) detected · Generated {datetime.now().strftime('%d %b %Y, %H:%M')}</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.75rem;font-family:Share Tech Mono;'>"
    "ElevatorGuard AI &nbsp;·&nbsp; IoT Fault Detection System &nbsp;·&nbsp; "
    "Built with Streamlit + Plotly + RandomForest ML"
    "</div>",
    unsafe_allow_html=True
)