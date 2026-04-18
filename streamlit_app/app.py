# ─────────────────────────────────────────────────────────────────────────────
# Sepsis Early Warning System — Dashboard
# File: streamlit_app/app.py
# Run: streamlit run streamlit_app/app.py
# All matplotlib rgba errors fixed. Uses proper (r,g,b,a) float tuples.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import time
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — dark animated medical theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #060d1a; color: #dce8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1525 !important;
    border-right: 1px solid #0d3349;
}
section[data-testid="stSidebar"] label {
    color: #7aabb8 !important; font-size: 12px !important; font-weight: 500 !important;
}
section[data-testid="stSidebar"] input[type="number"] {
    background: #0d1e30 !important; border: 1px solid #0d3349 !important;
    color: #dce8f0 !important; border-radius: 6px !important;
}

/* Metric cards */
.m-card {
    background: #0a1525; border: 1px solid #0d3349;
    border-radius: 12px; padding: 16px; text-align: center;
}
.m-num { font-size: 26px; font-weight: 800; color: #00c8b4; margin: 0; }
.m-lbl { font-size: 10px; color: #4a7a8a; margin: 3px 0 0; text-transform: uppercase; letter-spacing: 0.8px; }

/* Risk card */
.r-card {
    border-radius: 14px; padding: 26px 20px; text-align: center;
    border: 2px solid; position: relative;
}
.r-pct  { font-size: 68px; font-weight: 900; margin: 0; line-height: 1; letter-spacing: -3px; }
.r-lvl  { font-size: 14px; font-weight: 700; margin: 8px 0 4px; text-transform: uppercase; letter-spacing: 3px; }
.r-msg  { font-size: 12px; margin: 6px 0 0; opacity: 0.85; }
.r-high { background: #1a0606; border-color: #dc2626; color: #f87171; }
.r-mod  { background: #160f02; border-color: #d97706; color: #fbbf24; }
.r-low  { background: #021408; border-color: #059669; color: #34d399; }

/* Pulse animations */
@keyframes pulseRed   { 0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,.5)} 60%{box-shadow:0 0 0 14px rgba(220,38,38,0)} }
@keyframes pulseAmber { 0%,100%{box-shadow:0 0 0 0 rgba(217,119,6,.5)} 60%{box-shadow:0 0 0 14px rgba(217,119,6,0)} }
@keyframes pulseGreen { 0%,100%{box-shadow:0 0 0 0 rgba(5,150,105,.4)} 60%{box-shadow:0 0 0 10px rgba(5,150,105,0)} }
@keyframes fadeIn     { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes glow       { 0%,100%{text-shadow:0 0 8px rgba(0,200,180,.4)} 50%{text-shadow:0 0 20px rgba(0,200,180,.8)} }

.r-high { animation: pulseRed 2s infinite; }
.r-mod  { animation: pulseAmber 2.2s infinite; }
.r-low  { animation: pulseGreen 2.5s infinite; }
.fadeIn { animation: fadeIn 0.5s ease-out; }
.glow   { animation: glow 2s infinite; }

/* Vital chip */
.v-chip {
    background: #0a1525; border: 1px solid #0d3349;
    border-radius: 8px; padding: 10px 12px; text-align: center;
}
.v-name { font-size: 10px; color: #4a7a8a; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.v-val  { font-size: 17px; font-weight: 700; color: #dce8f0; margin: 3px 0 0; }
.v-unit { font-size: 10px; color: #4a7a8a; }

/* Alert strip */
.alert-strip {
    border-radius: 8px; padding: 10px 16px;
    font-size: 12px; font-weight: 600; text-align: center;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 8px 0;
}
.a-high { background: #3b0000; border: 1px solid #dc2626; color: #f87171; }
.a-mod  { background: #2d1800; border: 1px solid #d97706; color: #fbbf24; }
.a-low  { background: #001f0e; border: 1px solid #059669; color: #34d399; }

/* Section header */
.s-hdr {
    font-size: 11px; font-weight: 700; color: #00c8b4;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 0 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid #0d3349;
}

/* Info box */
.i-box {
    background: #071020; border: 1px solid #0d3349;
    border-radius: 10px; padding: 14px 16px;
    font-size: 12px; color: #7aabb8; line-height: 1.6;
}

/* Title */
.main-t  { font-size: 26px; font-weight: 800; color: #fff; margin: 0; }
.main-s  { font-size: 12px; color: #4a7a8a; margin: 3px 0 0; }
.t-teal  { color: #00c8b4; }
.t-amber { color: #f59e0b; }
.badge {
    display: inline-block; background: rgba(0,200,180,.1);
    border: 1px solid rgba(0,200,180,.3); color: #00c8b4;
    font-size: 11px; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;
    margin-right: 5px;
}

/* Progress bar track */
.prog-wrap { background: #0a1525; border-radius: 6px; height: 8px; overflow: hidden; margin: 10px 0; }
.prog-fill { height: 8px; border-radius: 6px; transition: width 0.6s ease; }

/* Footer */
.footer {
    text-align: center; font-size: 10px; color: #1e3548;
    margin-top: 20px; padding-top: 12px;
    border-top: 1px solid #0d1f30;
}

/* How it works steps */
.step-card {
    background: #0a1525; border: 1px solid #0d3349;
    border-radius: 10px; padding: 18px 14px; text-align: center;
}
.step-n    { font-size: 28px; font-weight: 900; color: #00c8b4; margin: 0; }
.step-t    { font-size: 13px; font-weight: 700; color: #dce8f0; margin: 6px 0 5px; }
.step-d    { font-size: 11px; color: #4a7a8a; line-height: 1.5; }

/* Input mode toggle */
.mode-box { background: #0a1525; border: 1px solid #0d3349; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load('models/xgboost_model.pkl')
    scaler    = joblib.load('models/scaler.pkl')
    features  = json.load(open('models/feature_list.json'))
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

try:
    model, scaler, features, explainer = load_models()
except Exception as e:
    st.error(f"❌ Model load error: {e}")
    st.info("Make sure models/xgboost_model.pkl, models/scaler.pkl, and models/feature_list.json exist.")
    st.stop()


# ── Clinical reference ranges for validation ──────────────────────────────────
# Format: (min_normal, max_normal, critical_low, critical_high, unit)
CLINICAL_RANGES = {
    'HR':              (60,   100,  40,   150,  'bpm'),
    'O2Sat':           (95,   100,  88,   100,  '%'),
    'Temp':            (36.1, 37.2, 35.0, 39.5, '°C'),
    'SBP':             (90,   130,  70,   200,  'mmHg'),
    'MAP':             (70,   105,  50,   130,  'mmHg'),
    'DBP':             (60,   85,   40,   120,  'mmHg'),
    'Resp':            (12,   20,   8,    35,   '/min'),
    'WBC':             (4.5,  11.0, 2.0,  20.0, 'K/µL'),
    'Creatinine':      (0.7,  1.2,  0.3,  4.0,  'mg/dL'),
    'Bilirubin_total': (0.2,  1.2,  0.1,  5.0,  'mg/dL'),
    'Lactate':         (0.5,  2.0,  0.3,  5.0,  'mmol/L'),
    'Glucose':         (70,   100,  50,   250,  'mg/dL'),
    'Hgb':             (12,   17.5, 7.0,  20.0, 'g/dL'),
    'pH':              (7.35, 7.45, 7.20, 7.55, ''),
    'Age':             (18,   90,   18,   100,  'yrs'),
    'HospAdmTime':     (-200, 0,   -500,  0,    'hr'),
    'ICULOS':          (0,    10,   0,    30,   'days'),
}

FEATURE_LABELS = {
    'HR': 'Heart Rate', 'O2Sat': 'O2 Saturation', 'Temp': 'Temperature',
    'SBP': 'Systolic BP', 'MAP': 'Mean Art. Pressure', 'DBP': 'Diastolic BP',
    'Resp': 'Respiratory Rate', 'WBC': 'WBC Count', 'Creatinine': 'Creatinine',
    'Bilirubin_total': 'Bilirubin', 'Lactate': 'Lactate', 'Glucose': 'Glucose',
    'Hgb': 'Hemoglobin', 'pH': 'Blood pH', 'Age': 'Patient Age',
    'HospAdmTime': 'Adm Time Offset', 'ICULOS': 'ICU Length of Stay'
}

DEFAULTS = {
    'HR': 85.0, 'O2Sat': 97.0, 'Temp': 37.0, 'SBP': 120.0, 'MAP': 85.0,
    'DBP': 70.0, 'Resp': 18.0, 'WBC': 8.0, 'Creatinine': 1.0,
    'Bilirubin_total': 0.8, 'Lactate': 1.5, 'Glucose': 110.0,
    'Hgb': 13.0, 'pH': 7.4, 'Age': 55.0, 'HospAdmTime': -24.0, 'ICULOS': 2.0
}

# Preset scenarios for demo
PRESETS = {
    "Normal patient":     {'HR':78,'O2Sat':98,'Temp':36.8,'SBP':118,'MAP':82,'DBP':68,'Resp':15,'WBC':7.2,'Creatinine':0.9,'Bilirubin_total':0.7,'Lactate':1.1,'Glucose':95,'Hgb':13.5,'pH':7.41,'Age':45,'HospAdmTime':-24,'ICULOS':1},
    "Moderate risk":      {'HR':102,'O2Sat':93,'Temp':38.2,'SBP':95,'MAP':68,'DBP':58,'Resp':24,'WBC':14.5,'Creatinine':1.8,'Bilirubin_total':1.5,'Lactate':2.8,'Glucose':145,'Hgb':10.2,'pH':7.33,'Age':62,'HospAdmTime':-12,'ICULOS':3},
    "High risk / Sepsis": {'HR':128,'O2Sat':88,'Temp':39.4,'SBP':78,'MAP':52,'DBP':42,'Resp':32,'WBC':22.3,'Creatinine':3.2,'Bilirubin_total':3.8,'Lactate':5.1,'Glucose':210,'Hgb':8.5,'pH':7.22,'Age':71,'HospAdmTime':-6,'ICULOS':2},
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 14px'>
        <div style='font-size:36px'>🏥</div>
        <div style='font-size:15px;font-weight:800;color:#ffffff;margin-top:6px'>Patient Vitals</div>
        <div style='font-size:11px;color:#4a7a8a;margin-top:2px'>ICU Clinical Input</div>
    </div>""", unsafe_allow_html=True)

    # Input mode
    input_mode = st.radio(
        "Input method",
        ["Manual entry", "Load preset scenario", "Upload .txt file"],
        horizontal=False
    )

    inputs = {}

    if input_mode == "Load preset scenario":
        preset = st.selectbox("Select patient scenario", list(PRESETS.keys()))
        inputs = dict(PRESETS[preset])
        st.success(f"Loaded: {preset}")
        # Show values
        for feat in features:
            if feat in inputs:
                lo = CLINICAL_RANGES.get(feat, (0,200,0,300,''))[2]
                hi = CLINICAL_RANGES.get(feat, (0,200,0,300,''))[3]
                inputs[feat] = float(inputs[feat])

    elif input_mode == "Upload .txt file":
        st.markdown("""
        <div class='i-box' style='margin-bottom:10px'>
        Upload a .txt file with one value per line, in this order:<br>
        HR, O2Sat, Temp, SBP, MAP, DBP, Resp, WBC, Creatinine,<br>
        Bilirubin_total, Lactate, Glucose, Hgb, pH, Age, HospAdmTime, ICULOS
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload patient values (.txt)", type=["txt"])
        if uploaded:
            content = uploaded.read().decode().strip().split('\n')
            vals = [float(v.strip()) for v in content if v.strip()]
            for i, feat in enumerate(features):
                inputs[feat] = vals[i] if i < len(vals) else DEFAULTS.get(feat, 0.0)
            st.success(f"Loaded {len(vals)} values from file")
        else:
            for feat in features:
                inputs[feat] = DEFAULTS.get(feat, 0.0)

    else:  # Manual entry
        for feat in features:
            lbl  = FEATURE_LABELS.get(feat, feat)
            cfg  = CLINICAL_RANGES.get(feat, (0, 200, 0, 300, ''))
            unit = cfg[4]
            lo   = float(cfg[2])
            hi   = float(cfg[3])
            dflt = float(DEFAULTS.get(feat, (lo+hi)/2))
            display = f"{lbl} ({unit})" if unit else lbl
            inputs[feat] = st.number_input(
                display, min_value=lo, max_value=hi,
                value=dflt, step=0.1, key=f"inp_{feat}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Analyze Sepsis Risk", type="primary", use_container_width=True)

    st.markdown("<hr style='border:1px solid #0d1f30;margin:14px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#1e3548;line-height:1.7'>
    <b style='color:#4a7a8a'>Risk thresholds</b><br>
    🟢 &lt;30% — Low risk<br>
    🟡 30–60% — Moderate risk<br>
    🔴 &gt;60% — High risk<br><br>
    <b style='color:#4a7a8a'>Model</b><br>
    XGBoost | AUROC 0.8158<br>
    PhysioNet 2019 | 40,336 patients
    </div>""", unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <p class='main-t'>Sepsis Early <span class='t-teal'>Warning</span> System</p>
    <p class='main-s'>AI-powered ICU monitoring — 6-hour early prediction using XGBoost + SHAP Explainability</p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style='text-align:right;padding-top:4px'>
        <span class='badge'>XGBoost</span><span class='badge'>SHAP</span>
        <div style='font-size:10px;color:#1e3548;margin-top:6px'>AUROC 0.8158 | PhysioNet 2019</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #0d1f30;margin:8px 0 12px'>", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
sm1, sm2, sm3, sm4 = st.columns(4)
for col, num, lbl in zip(
    [sm1, sm2, sm3, sm4],
    ["40,336", "0.8158", "6 hrs", "17"],
    ["ICU Patients Trained", "AUROC Score", "Early Warning Window", "Clinical Features"]
):
    with col:
        st.markdown(f'<div class="m-card"><p class="m-num">{num}</p><p class="m-lbl">{lbl}</p></div>', unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ── Prediction ────────────────────────────────────────────────────────────────
def get_vital_status(feat, val):
    """Returns 'normal', 'abnormal', or 'critical' based on clinical ranges."""
    if feat not in CLINICAL_RANGES:
        return 'normal'
    mn, mx, clo, chi, _ = CLINICAL_RANGES[feat]
    if val < clo or val > chi:
        return 'critical'
    if val < mn or val > mx:
        return 'abnormal'
    return 'normal'


if predict_btn:
    input_array = np.array([[float(inputs.get(f, DEFAULTS.get(f, 0))) for f in features]])

    with st.spinner("Analyzing patient vitals..."):
        time.sleep(0.4)
        risk_score = float(model.predict_proba(input_array)[0][1])
        risk_pct   = round(risk_score * 100, 1)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<p class="s-hdr">Risk Assessment</p>', unsafe_allow_html=True)

        if risk_pct >= 60:
            css, icon, level = "r-high", "🔴", "HIGH RISK"
            msg     = "Sepsis predicted within 6 hours — Immediate clinical review required"
            bar_clr = "#dc2626"
            alert_c = "a-high"
        elif risk_pct >= 30:
            css, icon, level = "r-mod", "🟡", "MODERATE RISK"
            msg     = "Elevated sepsis risk — Monitor closely, reassess every 30 minutes"
            bar_clr = "#d97706"
            alert_c = "a-mod"
        else:
            css, icon, level = "r-low", "🟢", "LOW RISK"
            msg     = "No immediate sepsis threat — Continue standard monitoring"
            bar_clr = "#059669"
            alert_c = "a-low"

        st.markdown(f"""
        <div class="r-card {css} fadeIn">
            <p class="r-pct">{risk_pct}%</p>
            <p class="r-lvl">{icon} {level}</p>
            <p class="r-msg">{msg}</p>
        </div>""", unsafe_allow_html=True)

        # Progress bar (HTML — avoids Streamlit type issues entirely)
        pct_w = min(int(risk_pct), 100)
        st.markdown(f"""
        <div class="prog-wrap">
            <div class="prog-fill" style="width:{pct_w}%;background:{bar_clr}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#1e3548">
            <span>0%</span><span>Low</span><span>Moderate</span><span>High</span><span>100%</span>
        </div>""", unsafe_allow_html=True)

        # ── Vitals grid with status colors ────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="s-hdr">Current Vitals Status</p>', unsafe_allow_html=True)

        vitals_show = ['HR','O2Sat','Temp','SBP','Lactate','WBC','Resp','Creatinine','pH']
        vitals_show = [v for v in vitals_show if v in inputs]

        rows = [vitals_show[i:i+3] for i in range(0, len(vitals_show), 3)]
        for row_feats in rows:
            cols = st.columns(len(row_feats))
            for col, feat in zip(cols, row_feats):
                val    = inputs[feat]
                status = get_vital_status(feat, val)
                unit   = CLINICAL_RANGES.get(feat, ('','','','',''))[4]
                lbl    = FEATURE_LABELS.get(feat, feat)
                clr    = {"normal":"#34d399", "abnormal":"#fbbf24", "critical":"#f87171"}[status]
                bdr    = {"normal":"#0d3349", "abnormal":"#854d0e", "critical":"#7f1d1d"}[status]
                with col:
                    st.markdown(f"""
                    <div class="v-chip" style="border-color:{bdr}">
                        <p class="v-name">{lbl}</p>
                        <p class="v-val" style="color:{clr}">{val:.1f}<span class="v-unit"> {unit}</span></p>
                    </div>""", unsafe_allow_html=True)

        # ── Clinical interpretation ────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        critical_vitals = [FEATURE_LABELS.get(f,f) for f in vitals_show if get_vital_status(f, inputs[f]) == 'critical']
        abnormal_vitals = [FEATURE_LABELS.get(f,f) for f in vitals_show if get_vital_status(f, inputs[f]) == 'abnormal']

        if critical_vitals:
            st.markdown(f'<div class="alert-strip a-high">⚠️ Critical values: {", ".join(critical_vitals)}</div>', unsafe_allow_html=True)
        if abnormal_vitals:
            st.markdown(f'<div class="alert-strip a-mod">⚡ Abnormal values: {", ".join(abnormal_vitals)}</div>', unsafe_allow_html=True)
        if not critical_vitals and not abnormal_vitals:
            st.markdown('<div class="alert-strip a-low">✅ All displayed vitals within normal reference range</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="s-hdr">SHAP Explanation — Why this score?</p>', unsafe_allow_html=True)

        # Compute SHAP
        shap_values = explainer.shap_values(input_array)
        sv          = shap_values[0]
        feat_imp    = {f: float(sv[i]) for i, f in enumerate(features)}
        top10       = dict(sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        labels = [FEATURE_LABELS.get(k, k) for k in top10]
        vals   = list(top10.values())
        colors_bar = [(0.95, 0.42, 0.42, 0.9) if v > 0 else (0.20, 0.83, 0.60, 0.9) for v in vals]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#060d1a')
        ax.set_facecolor('#0a1525')

        bars = ax.barh(
            labels[::-1], vals[::-1],
            color=colors_bar[::-1],
            edgecolor='none', height=0.65
        )

        # FIX: use proper matplotlib color tuples — NOT rgba() strings
        ax.axvline(0, color=(1.0, 1.0, 1.0, 0.15), linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.08))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('SHAP Value (impact on prediction)', color='#4a7a8a', fontsize=9)
        ax.set_title('Feature impact on sepsis risk score', color='#dce8f0', fontsize=10, fontweight='600', pad=10)
        ax.tick_params(colors='#7aabb8', labelsize=8.5)
        ax.xaxis.label.set_color('#4a7a8a')

        # Value labels on bars
        for bar, v in zip(bars[::-1], vals):
            offset = 0.003 if v >= 0 else -0.003
            ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                    f'{v:+.3f}', va='center',
                    ha='left' if v >= 0 else 'right',
                    color='#dce8f0', fontsize=7.5)

        red_p   = mpatches.Patch(color=(0.95, 0.42, 0.42), label='↑ Increases risk')
        green_p = mpatches.Patch(color=(0.20, 0.83, 0.60), label='↓ Decreases risk')
        ax.legend(handles=[red_p, green_p], loc='lower right',
                  facecolor='#0a1525', edgecolor=(1,1,1,0.1),
                  labelcolor='#7aabb8', fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top driver explanation
        top_feat     = list(top10.keys())[0]
        top_val      = list(top10.values())[0]
        top_lbl      = FEATURE_LABELS.get(top_feat, top_feat)
        direction    = "increasing" if top_val > 0 else "decreasing"
        top_inp_val  = inputs.get(top_feat, 0)
        unit_top     = CLINICAL_RANGES.get(top_feat, ('','','','',''))[4]

        st.markdown(f"""
        <div class="i-box">
            <b style='color:#00c8b4'>Top driver:</b>
            <b style='color:#fbbf24'>{top_lbl}</b> = {top_inp_val:.1f} {unit_top}
            (SHAP: {top_val:+.3f}) is the strongest factor
            <b style='color:#{"f87171" if top_val>0 else "34d399"}'>{direction}</b> sepsis risk for this patient.<br><br>
            🔴 Red = pushes risk higher &nbsp;|&nbsp; 🟢 Green = pulls risk lower<br>
            Longer bar = stronger influence on <em>this specific</em> prediction.
        </div>""", unsafe_allow_html=True)

        # ── Model confidence ─────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="s-hdr">Model Confidence Breakdown</p>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(7, 1.8))
        fig2.patch.set_facecolor('#060d1a')
        ax2.set_facecolor('#060d1a')

        no_risk = 1 - risk_score
        ax2.barh(['No Sepsis Risk', 'Sepsis Risk'],
                 [no_risk, risk_score],
                 color=[(0.20, 0.83, 0.60, 0.8), (0.95, 0.42, 0.42, 0.8)],
                 edgecolor='none', height=0.5)

        ax2.axvline(0.5, color=(1,1,1,0.2), linewidth=0.8, linestyle='--')
        ax2.set_xlim(0, 1)
        ax2.tick_params(colors='#7aabb8', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax2.text(no_risk - 0.02, 0, f'{no_risk*100:.1f}%', va='center', ha='right', color='#34d399', fontsize=9, fontweight='bold')
        ax2.text(risk_score - 0.02, 1, f'{risk_score*100:.1f}%', va='center', ha='right', color='#f87171', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()


else:
    # ── Default state ─────────────────────────────────────────────────────────
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        <div class="i-box" style="text-align:center;padding:32px 24px">
            <div style="font-size:42px;margin-bottom:10px">👈</div>
            <div style="font-size:15px;font-weight:700;color:#dce8f0;margin-bottom:8px">Enter Patient Vitals</div>
            <div style="font-size:12px;color:#4a7a8a">Use the sidebar to input ICU readings,<br>then click <b style='color:#00c8b4'>Analyze Sepsis Risk</b></div>
        </div>""", unsafe_allow_html=True)
    with col_d2:
        st.markdown("""
        <div class="i-box" style="text-align:center;padding:32px 24px">
            <div style="font-size:42px;margin-bottom:10px">💡</div>
            <div style="font-size:15px;font-weight:700;color:#dce8f0;margin-bottom:8px">Try a Preset</div>
            <div style="font-size:12px;color:#4a7a8a">Switch to <b style='color:#00c8b4'>Load preset scenario</b><br>to see High Risk, Moderate, or Normal examples</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="s-hdr">How This System Works</p>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (n, t, d) in zip(
        [sc1, sc2, sc3, sc4],
        [("1","Input Vitals","Enter 17 clinical measurements from the patient's ICU record"),
         ("2","XGBoost Predicts","Model trained on 40,336 patients calculates sepsis probability"),
         ("3","Risk Scored","0–100% risk with green / amber / red alert + progress bar"),
         ("4","SHAP Explains","Bar chart shows exactly which vital sign drove the prediction")]
    ):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <p class="step-n">{n}</p>
                <p class="step-t">{t}</p>
                <p class="step-d">{d}</p>
            </div>""", unsafe_allow_html=True)

    # Reference ranges table
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="s-hdr">Clinical Reference Ranges Used</p>', unsafe_allow_html=True)
    ref_data = {
        'Feature':      ['Heart Rate','O2 Saturation','Temperature','Lactate','WBC Count','Creatinine','Blood pH','Resp Rate'],
        'Normal Range': ['60–100 bpm','95–100%','36.1–37.2°C','0.5–2.0 mmol/L','4.5–11.0 K/µL','0.7–1.2 mg/dL','7.35–7.45','12–20 /min'],
        'Sepsis Signal':['> 90 or < 60','< 94%','> 38.3°C or < 36°C','> 2.0 mmol/L','> 12 or < 4 K/µL','> 1.5 mg/dL','< 7.35','> 22 /min'],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
Sepsis Early Warning System &nbsp;|&nbsp; Zaid Ali &nbsp;|&nbsp; Roll No. 40 &nbsp;|&nbsp;
BS CS 6th Semester &nbsp;|&nbsp; AWKUM 2025–26 &nbsp;|&nbsp;
Model: XGBoost AUROC 0.8158 &nbsp;|&nbsp; Dataset: PhysioNet Computing in Cardiology Challenge 2019
</div>""", unsafe_allow_html=True)