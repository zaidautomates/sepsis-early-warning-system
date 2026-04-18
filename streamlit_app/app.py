# streamlit_app/app.py
# Run: streamlit run streamlit_app/app.py

import streamlit as st
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — dark medical theme with animations ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(13,20,40,0.95) !important;
    border-right: 1px solid rgba(0,200,180,0.2);
}

section[data-testid="stSidebar"] label {
    color: #a0c4d8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,200,180,0.3) !important;
    color: #e0e8f0 !important;
    border-radius: 6px !important;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,200,180,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0,200,180,0.5);
    background: rgba(0,200,180,0.05);
}
.metric-num {
    font-size: 28px;
    font-weight: 700;
    color: #00c8b4;
    margin: 0;
    line-height: 1.2;
}
.metric-label {
    font-size: 11px;
    color: #6a8fa8;
    margin: 4px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Risk display */
.risk-display {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    margin: 12px 0;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.risk-display::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
    pointer-events: none;
}
.risk-pct {
    font-size: 64px;
    font-weight: 800;
    margin: 0;
    line-height: 1;
    letter-spacing: -2px;
}
.risk-label {
    font-size: 16px;
    font-weight: 600;
    margin: 8px 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.risk-msg {
    font-size: 13px;
    margin: 8px 0 0 0;
    opacity: 0.8;
}
.high-risk {
    background: rgba(220,38,38,0.08);
    border-color: rgba(220,38,38,0.6);
    color: #ff6b6b;
}
.mod-risk {
    background: rgba(245,158,11,0.08);
    border-color: rgba(245,158,11,0.6);
    color: #fbbf24;
}
.low-risk {
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.6);
    color: #34d399;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #00c8b4;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,200,180,0.2);
}

/* Vitals grid */
.vitals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 12px 0;
}
.vital-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 12px;
}
.vital-name {
    font-size: 10px;
    color: #6a8fa8;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.vital-val {
    font-size: 16px;
    font-weight: 600;
    color: #e0e8f0;
    margin: 2px 0 0 0;
}

/* Pulse animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }
    70% { box-shadow: 0 0 0 12px rgba(220,38,38,0); }
    100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
}
.pulse-red { animation: pulse 2s infinite; }

@keyframes pulseAmber {
    0% { box-shadow: 0 0 0 0 rgba(245,158,11,0.4); }
    70% { box-shadow: 0 0 0 12px rgba(245,158,11,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
}
.pulse-amber { animation: pulseAmber 2s infinite; }

/* Progress ring */
.progress-ring-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 8px 0;
}

/* Main title */
.main-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.main-subtitle {
    font-size: 13px;
    color: #6a8fa8;
    margin: 4px 0 0 0;
}
.title-accent {
    color: #00c8b4;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(0,200,180,0.12);
    border: 1px solid rgba(0,200,180,0.3);
    color: #00c8b4;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 6px;
}

/* Divider */
.h-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 16px 0;
}

/* Info box */
.info-box {
    background: rgba(0,200,180,0.06);
    border: 1px solid rgba(0,200,180,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    color: #a0c4d8;
    line-height: 1.6;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 11px;
    color: #3a5a78;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load('models/xgboost_model.pkl')
    scaler    = joblib.load('models/scaler.pkl')
    features  = json.load(open('models/feature_list.json'))
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

try:
    model, scaler, features, explainer = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model load error: {e}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 16px">
        <div style="font-size:32px;margin-bottom:6px">🏥</div>
        <div style="font-size:14px;font-weight:700;color:#ffffff">Patient Vitals Input</div>
        <div style="font-size:11px;color:#6a8fa8;margin-top:2px">Enter current ICU readings</div>
    </div>
    """, unsafe_allow_html=True)

    feature_config = {
        'HR':              ('Heart Rate',        'bpm',    20.0,  250.0, 85.0),
        'O2Sat':           ('O2 Saturation',     '%',      50.0,  100.0, 97.0),
        'Temp':            ('Temperature',       '°C',     30.0,  42.0,  37.0),
        'SBP':             ('Systolic BP',       'mmHg',   50.0,  250.0, 120.0),
        'MAP':             ('Mean Art. Pressure','mmHg',   30.0,  200.0, 85.0),
        'DBP':             ('Diastolic BP',      'mmHg',   20.0,  180.0, 70.0),
        'Resp':            ('Respiratory Rate',  '/min',   5.0,   60.0,  18.0),
        'WBC':             ('WBC Count',         'K/µL',   0.5,   50.0,  8.0),
        'Creatinine':      ('Creatinine',        'mg/dL',  0.1,   20.0,  1.0),
        'Bilirubin_total': ('Bilirubin',         'mg/dL',  0.1,   30.0,  0.8),
        'Lactate':         ('Lactate',           'mmol/L', 0.3,   20.0,  1.5),
        'Glucose':         ('Glucose',           'mg/dL',  40.0,  500.0, 110.0),
        'Hgb':             ('Hemoglobin',        'g/dL',   3.0,   20.0,  13.0),
        'pH':              ('Blood pH',          '',       6.8,   7.8,   7.4),
        'Age':             ('Patient Age',       'yrs',    18.0,  100.0, 55.0),
        'HospAdmTime':     ('Adm. Time Offset',  'hr',    -200.0, 0.0,  -24.0),
        'ICULOS':          ('ICU Length Stay',   'days',   0.0,   100.0, 2.0),
    }

    inputs = {}
    for feat in features:
        if feat in feature_config:
            label, unit, lo, hi, default = feature_config[feat]
            display = f"{label} ({unit})" if unit else label
            inputs[feat] = st.number_input(
                display, min_value=float(lo), max_value=float(hi),
                value=float(default), step=0.1, key=feat
            )
        else:
            inputs[feat] = st.number_input(feat, value=0.0, step=0.1, key=feat)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analyze Patient Risk", type="primary", use_container_width=True)

    st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#3a5a78;line-height:1.6'>
    <b style='color:#6a8fa8'>Risk thresholds</b><br>
    🟢 &lt;30% — Low risk<br>
    🟡 30–60% — Moderate risk<br>
    🔴 &gt;60% — High risk<br><br>
    <b style='color:#6a8fa8'>Dataset</b><br>
    PhysioNet 2019 — 40,336 patients<br>
    XGBoost AUROC: 0.8158
    </div>
    """, unsafe_allow_html=True)


# ── Main panel ────────────────────────────────────────────────────────────────
# Title row
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown("""
    <p class='main-title'>Sepsis Early <span class='title-accent'>Warning System</span></p>
    <p class='main-subtitle'>AI-powered ICU monitoring — 6-hour early prediction using XGBoost + SHAP</p>
    """, unsafe_allow_html=True)
with col_t2:
    st.markdown("""
    <div style='text-align:right;padding-top:6px'>
        <span class='badge'>XGBoost</span>
        <span class='badge'>SHAP</span><br>
        <span style='font-size:11px;color:#3a5a78;margin-top:4px;display:block'>AUROC 0.8158 | PhysioNet 2019</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><p class="metric-num">40,336</p><p class="metric-label">ICU Patients Trained</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><p class="metric-num">0.8158</p><p class="metric-label">AUROC Score</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><p class="metric-num">6 hrs</p><p class="metric-label">Early Warning Window</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><p class="metric-num">17</p><p class="metric-label">Clinical Features</p></div>', unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Prediction area ───────────────────────────────────────────────────────────
if predict_btn:
    input_array = np.array([[inputs[f] for f in features]], dtype=np.float64)

    # Animated loading
    with st.spinner("Analyzing patient vitals..."):
        time.sleep(0.6)
        risk_score = float(model.predict_proba(input_array)[0][1])  # FIX: cast to Python float
        risk_pct   = round(risk_score * 100, 1)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Risk Score Display ───────────────────────────────────────
        st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

        if risk_pct >= 60:
            css_class = "high-risk pulse-red"
            icon      = "🔴"
            level     = "HIGH RISK"
            msg       = "Sepsis predicted within 6 hours — Immediate clinical review required"
        elif risk_pct >= 30:
            css_class = "mod-risk pulse-amber"
            icon      = "🟡"
            level     = "MODERATE RISK"
            msg       = "Elevated sepsis risk — Monitor closely, reassess every 30 minutes"
        else:
            css_class = "low-risk"
            icon      = "🟢"
            level     = "LOW RISK"
            msg       = "No immediate sepsis threat — Continue standard monitoring protocol"

        st.markdown(f"""
        <div class="risk-display {css_class}">
            <p class="risk-pct">{risk_pct}%</p>
            <p class="risk-label">{icon} {level}</p>
            <p class="risk-msg">{msg}</p>
        </div>
        """, unsafe_allow_html=True)

        # Risk progress bar — FIX for float32 error
        st.progress(min(max(float(risk_score), 0.0), 1.0))

        # ── Vitals summary ────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Current Vitals Summary</p>', unsafe_allow_html=True)

        vitals_display = [
            ('HR', 'Heart Rate', 'bpm'),
            ('O2Sat', 'SpO2', '%'),
            ('Temp', 'Temp', '°C'),
            ('SBP', 'Sys BP', 'mmHg'),
            ('Lactate', 'Lactate', 'mmol/L'),
            ('WBC', 'WBC', 'K/µL'),
        ]

        cols_v = st.columns(3)
        for idx, (feat, label, unit) in enumerate(vitals_display):
            if feat in inputs:
                val = inputs[feat]
                with cols_v[idx % 3]:
                    st.markdown(f"""
                    <div class="vital-item">
                        <p class="vital-name">{label}</p>
                        <p class="vital-val">{val:.1f} <span style='font-size:10px;color:#6a8fa8'>{unit}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

    with col_right:
        # ── SHAP Explanation ─────────────────────────────────────────
        st.markdown('<p class="section-header">SHAP Explanation — Why this score?</p>', unsafe_allow_html=True)

        shap_values = explainer.shap_values(input_array)
        sv          = shap_values[0]

        feat_imp = {f: float(sv[i]) for i, f in enumerate(features)}
        top10    = dict(sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        # Map feature names to readable labels
        feat_labels_map = {
            'HR':'Heart Rate','O2Sat':'O2 Sat','Temp':'Temperature',
            'SBP':'Systolic BP','MAP':'Mean BP','DBP':'Diastolic BP',
            'Resp':'Resp Rate','WBC':'WBC Count','Creatinine':'Creatinine',
            'Bilirubin_total':'Bilirubin','Lactate':'Lactate',
            'Glucose':'Glucose','Hgb':'Hemoglobin','pH':'Blood pH',
            'Age':'Age','HospAdmTime':'Adm Time','ICULOS':'ICU Stay'
        }

        labels = [feat_labels_map.get(k, k) for k in top10]
        vals   = list(top10.values())
        colors = ['#ff6b6b' if v > 0 else '#34d399' for v in vals]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')

        bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1],
                       edgecolor='none', height=0.65)

        ax.axvline(0, color='rgba(255,255,255,0.15)', linewidth=0.8)
        ax.set_xlabel('SHAP Value', color='#6a8fa8', fontsize=10)
        ax.set_title('Feature impact on sepsis risk', color='#e0e8f0',
                     fontsize=11, fontweight='600', pad=10)

        ax.tick_params(colors='#a0c4d8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.08)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add value labels
        for bar, v in zip(bars[::-1], vals):
            ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                    f'{v:+.3f}', va='center',
                    ha='left' if v >= 0 else 'right',
                    color='#e0e8f0', fontsize=8)

        red_patch   = mpatches.Patch(color='#ff6b6b', label='Increases risk')
        green_patch = mpatches.Patch(color='#34d399', label='Decreases risk')
        ax.legend(handles=[red_patch, green_patch], loc='lower right',
                  facecolor='#0d1b2a', edgecolor='rgba(255,255,255,0.1)',
                  labelcolor='#a0c4d8', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="info-box">
            <b style='color:#00c8b4'>How to read this chart:</b><br>
            🔴 Red bars = features pushing the risk score higher (danger signals)<br>
            🟢 Green bars = features pulling the risk score lower (protective signals)<br>
            Longer bar = stronger influence on this specific prediction
        </div>
        """, unsafe_allow_html=True)

else:
    # Default state — show instructions
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div class="info-box" style='text-align:center;padding:32px 24px'>
            <div style='font-size:40px;margin-bottom:12px'>👈</div>
            <div style='font-size:15px;font-weight:600;color:#e0e8f0;margin-bottom:8px'>Enter Patient Vitals</div>
            <div style='font-size:12px;color:#6a8fa8'>Use the sidebar to input current ICU readings,<br>then click <b style='color:#00c8b4'>Analyze Patient Risk</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="info-box" style='text-align:center;padding:32px 24px'>
            <div style='font-size:40px;margin-bottom:12px'>🧠</div>
            <div style='font-size:15px;font-weight:600;color:#e0e8f0;margin-bottom:8px'>SHAP Explanation</div>
            <div style='font-size:12px;color:#6a8fa8'>After prediction, SHAP will show exactly<br>which vital signs drove the risk score</div>
        </div>
        """, unsafe_allow_html=True)

    # How it works section
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">How This System Works</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "Input Vitals", "Enter 17 clinical measurements from the patient's ICU record"),
        ("2", "XGBoost Predicts", "Model trained on 40,336 patients calculates sepsis probability"),
        ("3", "Risk Scored", "0–100% risk score with green/amber/red alert level"),
        ("4", "SHAP Explains", "Feature importance chart shows exactly why the score was given"),
    ]
    for col, (n, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card" style='padding:20px 16px'>
                <div style='font-size:24px;font-weight:800;color:#00c8b4;margin-bottom:8px'>{n}</div>
                <div style='font-size:13px;font-weight:600;color:#e0e8f0;margin-bottom:6px'>{title}</div>
                <div style='font-size:11px;color:#6a8fa8;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    Sepsis Early Warning System &nbsp;|&nbsp; Zaid Ali &nbsp;|&nbsp; Roll No. 40 &nbsp;|&nbsp;
    BS CS 4th Semester &nbsp;|&nbsp; AWKUM 2025–26 &nbsp;|&nbsp;
    Dataset: PhysioNet Computing in Cardiology Challenge 2019 &nbsp;|&nbsp;
    Model: XGBoost + SHAP Explainability
</div>
""", unsafe_allow_html=True)