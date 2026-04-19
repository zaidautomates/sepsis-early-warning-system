# ─────────────────────────────────────────────────────────────────────────────
# Sepsis Early Warning System — ICU Dashboard v3.0
# Author : Zaid Ali | Roll No. 40 | BS CS 6th Semester | AWKUM 2025–26
# Dataset: PhysioNet Computing in Cardiology Challenge 2019 (40,336 patients)
# Model  : XGBoost | AUROC 0.8158 | 6-Hour Early Prediction
# Run    : streamlit run app.py
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
import time
import pandas as pd
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sepsis EWS — ICU Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — FIXED: readable text, proper contrast, no broken animations
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Background */
.stApp {
    background: #07111f;
    color: #e0eaf2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #04090f !important;
    border-right: 1px solid #0e2a3a !important;
}
section[data-testid="stSidebar"] label {
    color: #7ab8c8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
section[data-testid="stSidebar"] input[type="number"] {
    background: #0a1c2a !important;
    border: 1px solid #1a3a4a !important;
    color: #e0eaf2 !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #a0c8d8 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #7ab8c8 !important;
}

/* Animations — clean, no broken rgba strings */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseRedGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
    50%      { box-shadow: 0 0 0 16px rgba(220,38,38,0); }
}
@keyframes pulseAmberGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(217,119,6,0.4); }
    50%      { box-shadow: 0 0 0 14px rgba(217,119,6,0); }
}
@keyframes pulseGreenGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(5,150,105,0.35); }
    50%      { box-shadow: 0 0 0 12px rgba(5,150,105,0); }
}
@keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(200%); }
}
@keyframes glowTeal {
    0%,100% { text-shadow: 0 0 6px rgba(0,200,180,0.25); }
    50%      { text-shadow: 0 0 18px rgba(0,200,180,0.7); }
}
@keyframes blink {
    0%,80%,100% { opacity: 1; }
    40%          { opacity: 0.3; }
}

/* ── Header card ────────────────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #0a1e30 0%, #071524 100%);
    border: 1px solid #1a3a4a;
    border-top: 2px solid #00c8b4;
    border-radius: 14px;
    padding: 20px 24px 18px;
    margin-bottom: 6px;
}
.header-title {
    font-size: 27px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
}
.header-sub {
    font-size: 12px;
    color: #5a8fa0;
    margin: 5px 0 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}
.teal  { color: #00c8b4; }
.red   { color: #f87171; }
.amber { color: #fbbf24; }
.green { color: #34d399; }

/* ── Stat cards ─────────────────────────────────────────── */
.stat-card {
    background: #0a1e30;
    border: 1px solid #1a3a4a;
    border-bottom: 2px solid #00c8b4;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-num {
    font-size: 22px;
    font-weight: 800;
    color: #00c8b4;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
    animation: glowTeal 3s infinite;
}
.stat-lbl {
    font-size: 10px;
    color: #4a7080;
    margin: 4px 0 0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Risk cards ─────────────────────────────────────────── */
.risk-card {
    border-radius: 16px;
    padding: 26px 20px;
    text-align: center;
    border: 2px solid;
    animation: fadeIn 0.45s ease-out;
}
.risk-pct {
    font-size: 70px;
    font-weight: 900;
    margin: 0;
    line-height: 1;
    letter-spacing: -3px;
    font-family: 'JetBrains Mono', monospace;
}
.risk-level {
    font-size: 13px;
    font-weight: 700;
    margin: 10px 0 6px;
    text-transform: uppercase;
    letter-spacing: 3px;
}
.risk-msg {
    font-size: 12px;
    margin: 6px 0 0;
    line-height: 1.55;
}

/* Risk card colour variants */
.risk-high {
    background: linear-gradient(160deg, #1e0505 0%, #120202 100%);
    border-color: #dc2626;
    color: #fca5a5;
    animation: pulseRedGlow 2s infinite, fadeIn 0.45s ease-out;
}
.risk-mod {
    background: linear-gradient(160deg, #1e1005 0%, #120a02 100%);
    border-color: #d97706;
    color: #fde68a;
    animation: pulseAmberGlow 2.2s infinite, fadeIn 0.45s ease-out;
}
.risk-low {
    background: linear-gradient(160deg, #03180c 0%, #020e08 100%);
    border-color: #059669;
    color: #6ee7b7;
    animation: pulseGreenGlow 2.5s infinite, fadeIn 0.45s ease-out;
}

/* ── Progress bar ───────────────────────────────────────── */
.prog-track {
    background: #0a1e30;
    border: 1px solid #1a3a4a;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin: 12px 0 4px;
}
.prog-fill {
    height: 10px;
    border-radius: 6px;
    position: relative;
    overflow: hidden;
}
.prog-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    animation: shimmer 2s linear infinite;
}
.prog-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9.5px;
    color: #2a4a5a;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section headers ────────────────────────────────────── */
.s-hdr {
    font-size: 10px;
    font-weight: 700;
    color: #00c8b4;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0 0 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid #0e2a3a;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Score boxes ─────────────────────────────────────────── */
.score-box {
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
    border: 1px solid;
}
.score-num {
    font-size: 24px;
    font-weight: 800;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}
.score-lbl {
    font-size: 9.5px;
    margin: 4px 0 0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.75;
}

/* ── Vital chips — FIXED CONTRAST ──────────────────────── */
.vital-chip {
    border-radius: 9px;
    padding: 10px 8px;
    text-align: center;
    border: 1px solid;
    animation: fadeIn 0.4s ease-out both;
    transition: transform 0.15s ease;
}
.vital-chip:hover { transform: translateY(-2px); }

/* Normal: dark green tint, white text */
.vc-normal {
    background: #051a0e;
    border-color: #1a5a3a;
}
/* Abnormal: dark amber tint */
.vc-warn {
    background: #1a1000;
    border-color: #7a4800;
}
/* Critical: dark red tint */
.vc-critical {
    background: #1e0404;
    border-color: #8a1a1a;
}

.vital-name {
    font-size: 9px;
    color: #6a9aaa;          /* FIXED: was too dark */
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-family: 'JetBrains Mono', monospace;
}
.vital-val {
    font-size: 17px;
    font-weight: 700;
    margin: 4px 0 1px;
    font-family: 'JetBrains Mono', monospace;
}
.vital-unit {
    font-size: 9px;
    color: #4a7080;
}
.vital-range {
    font-size: 8px;
    color: #3a5a6a;          /* FIXED: visible range text */
    margin: 2px 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── SIRS criteria ──────────────────────────────────────── */
.sirs-wrap {
    background: #07111f;
    border: 1px solid #0e2a3a;
    border-radius: 10px;
    padding: 12px 14px;
}
.sirs-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid #0a1e2a;
    font-size: 12px;
}
.sirs-row:last-child { border-bottom: none; }
.sirs-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Alert strips ───────────────────────────────────────── */
.alert-strip {
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 11.5px;
    font-weight: 700;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 5px 0;
    font-family: 'JetBrains Mono', monospace;
}
.a-high { background: #2a0404; border: 1px solid #7a1a1a; color: #fca5a5; }
.a-mod  { background: #2a1500; border: 1px solid #7a4800; color: #fde68a; }
.a-low  { background: #01180a; border: 1px solid #1a6a3a; color: #6ee7b7; }
.a-info { background: #001a28; border: 1px solid #1a5a7a; color: #7dd3fc; }

/* ── Recommendation cards ───────────────────────────────── */
.rec-card {
    background: #07111f;
    border-left: 3px solid;
    border-radius: 0 9px 9px 0;
    padding: 10px 13px;
    margin: 5px 0;
    font-size: 12px;
    line-height: 1.6;
    animation: fadeIn 0.4s ease-out both;
}
.rec-title {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.rc-high { border-color: #dc2626; color: #fca5a5; }
.rc-mod  { border-color: #d97706; color: #fde68a; }
.rc-low  { border-color: #059669; color: #6ee7b7; }
.rc-info { border-color: #0891b2; color: #7dd3fc; }

/* ── Quick scores box ───────────────────────────────────── */
.scores-box {
    background: #07111f;
    border: 1px solid #0e2a3a;
    border-radius: 10px;
    padding: 13px 15px;
}
.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #0a1e2a;
    font-size: 12px;
}
.score-row:last-child { border-bottom: none; }
.score-label { color: #5a8090; }
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 13px;
}

/* ── Info box ───────────────────────────────────────────── */
.i-box {
    background: #07111f;
    border: 1px solid #0e2a3a;
    border-radius: 9px;
    padding: 13px 15px;
    font-size: 12px;
    color: #7ab8c8;
    line-height: 1.65;
}

/* ── Live badge ─────────────────────────────────────────── */
.badge {
    display: inline-block;
    background: rgba(0,200,180,0.1);
    border: 1px solid rgba(0,200,180,0.3);
    color: #00c8b4;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    margin-right: 4px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.badge-red {
    background: rgba(220,38,38,0.12);
    border-color: rgba(220,38,38,0.35);
    color: #f87171;
    animation: blink 2s infinite;
}
.time-badge {
    background: rgba(0,200,180,0.07);
    border: 1px solid rgba(0,200,180,0.18);
    border-radius: 7px;
    padding: 5px 11px;
    font-size: 11px;
    color: #00c8b4;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

/* ── Step cards (default view) ──────────────────────────── */
.step-card {
    background: #0a1e30;
    border: 1px solid #1a3a4a;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.25s;
}
.step-card:hover { border-color: #2a6a7a; }
.step-n { font-size: 30px; font-weight: 900; color: #00c8b4; margin: 0; font-family: 'JetBrains Mono', monospace; }
.step-t { font-size: 13px; font-weight: 700; color: #d0e8f0; margin: 6px 0 5px; }
.step-d { font-size: 11px; color: #4a7080; line-height: 1.55; }

/* ── Summary alert bar ──────────────────────────────────── */
.summary-bar {
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 10px;
    animation: fadeIn 0.5s ease-out;
}

/* ── Footer ─────────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 10px;
    color: #1a3a4a;
    margin-top: 24px;
    padding-top: 12px;
    border-top: 1px solid #0a1e2a;
    font-family: 'JetBrains Mono', monospace;
}

/* Streamlit dataframe dark theming */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
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
    st.error(f"❌ Model load failed: {e}")
    st.info("Ensure models/xgboost_model.pkl, models/scaler.pkl, and models/feature_list.json exist.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL DATA
# ══════════════════════════════════════════════════════════════════════════════
# (min_normal, max_normal, hard_min, hard_max, unit)
CLINICAL_RANGES = {
    'HR':              (60,   100,  20,   220,  'bpm'),
    'O2Sat':           (95,   100,  50,   100,  '%'),
    'Temp':            (36.1, 37.2, 32.0, 42.0, '°C'),
    'SBP':             (90,   130,  40,   250,  'mmHg'),
    'MAP':             (70,   105,  20,   180,  'mmHg'),
    'DBP':             (60,   85,   20,   150,  'mmHg'),
    'Resp':            (12,   20,   4,    60,   '/min'),
    'WBC':             (4.5,  11.0, 0.5,  100.0,'K/µL'),
    'Creatinine':      (0.7,  1.2,  0.1,  20.0, 'mg/dL'),
    'Bilirubin_total': (0.2,  1.2,  0.1,  30.0, 'mg/dL'),
    'Lactate':         (0.5,  2.0,  0.1,  25.0, 'mmol/L'),
    'Glucose':         (70,   100,  10,   800,  'mg/dL'),
    'Hgb':             (12,   17.5, 3.0,  25.0, 'g/dL'),
    'pH':              (7.35, 7.45, 6.80, 7.80, ''),
    'Age':             (18,   90,   18,   110,  'yrs'),
    'HospAdmTime':     (-200, 0,   -600,  0,    'hr'),
    'ICULOS':          (0,    10,   0,    90,   'days'),
}

FEATURE_LABELS = {
    'HR':              'Heart Rate',
    'O2Sat':           'O₂ Saturation',
    'Temp':            'Temperature',
    'SBP':             'Systolic BP',
    'MAP':             'Mean Art. Pr.',
    'DBP':             'Diastolic BP',
    'Resp':            'Resp. Rate',
    'WBC':             'WBC Count',
    'Creatinine':      'Creatinine',
    'Bilirubin_total': 'Bilirubin',
    'Lactate':         'Lactate',
    'Glucose':         'Glucose',
    'Hgb':             'Hemoglobin',
    'pH':              'Blood pH',
    'Age':             'Patient Age',
    'HospAdmTime':     'Adm. Offset',
    'ICULOS':          'ICU LOS',
}

DEFAULTS = {
    'HR': 85.0, 'O2Sat': 97.0, 'Temp': 37.0, 'SBP': 120.0, 'MAP': 85.0,
    'DBP': 70.0, 'Resp': 18.0, 'WBC': 8.0, 'Creatinine': 1.0,
    'Bilirubin_total': 0.8, 'Lactate': 1.5, 'Glucose': 110.0,
    'Hgb': 13.0, 'pH': 7.4, 'Age': 55.0, 'HospAdmTime': -24.0, 'ICULOS': 2.0
}

PRESETS = {
    "🟢 Normal Patient": {
        'HR':78,'O2Sat':98,'Temp':36.8,'SBP':118,'MAP':82,'DBP':68,'Resp':15,
        'WBC':7.2,'Creatinine':0.9,'Bilirubin_total':0.7,'Lactate':1.1,
        'Glucose':95,'Hgb':13.5,'pH':7.41,'Age':45,'HospAdmTime':-24,'ICULOS':1
    },
    "🟡 Moderate Risk": {
        'HR':105,'O2Sat':92,'Temp':38.4,'SBP':94,'MAP':66,'DBP':56,'Resp':26,
        'WBC':15.8,'Creatinine':2.1,'Bilirubin_total':1.8,'Lactate':3.2,
        'Glucose':158,'Hgb':9.8,'pH':7.31,'Age':64,'HospAdmTime':-10,'ICULOS':4
    },
    "🔴 High Risk / Sepsis": {
        'HR':132,'O2Sat':86,'Temp':39.6,'SBP':74,'MAP':48,'DBP':38,'Resp':34,
        'WBC':24.1,'Creatinine':3.8,'Bilirubin_total':4.2,'Lactate':6.5,
        'Glucose':225,'Hgb':7.8,'pH':7.19,'Age':73,'HospAdmTime':-5,'ICULOS':2
    },
}

GROUPS = [
    ("📊 Vital Signs",      ['HR','O2Sat','Temp','Resp']),
    ("🩸 Blood Pressure",   ['SBP','DBP','MAP']),
    ("🔬 Laboratory",       ['WBC','Creatinine','Bilirubin_total','Lactate','Glucose','Hgb','pH']),
    ("👤 Patient Info",     ['Age','HospAdmTime','ICULOS']),
]


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def vital_status(feat, val):
    """Return 'normal', 'abnormal', or 'critical'."""
    if feat not in CLINICAL_RANGES:
        return 'normal'
    mn, mx, _, _, _ = CLINICAL_RANGES[feat]
    if feat == 'O2Sat':
        if val < 88: return 'critical'
        if val < 95: return 'abnormal'
        return 'normal'
    if feat == 'pH':
        if val < 7.20 or val > 7.55: return 'critical'
        if val < 7.35 or val > 7.45: return 'abnormal'
        return 'normal'
    if feat == 'Lactate':
        if val > 4.0: return 'critical'
        if val > 2.0: return 'abnormal'
        return 'normal'
    if feat == 'MAP':
        if val < 50: return 'critical'
        if val < 70: return 'abnormal'
        return 'normal'
    # General rule
    span = mx - mn
    if val < mn - span * 0.5 or val > mx + span * 0.5:
        return 'critical'
    if val < mn or val > mx:
        return 'abnormal'
    return 'normal'


def vital_color(status):
    return {'normal': '#4ade80', 'abnormal': '#fbbf24', 'critical': '#f87171'}[status]


def vital_chip_class(status):
    return {'normal': 'vc-normal', 'abnormal': 'vc-warn', 'critical': 'vc-critical'}[status]


def compute_sirs(inp):
    return {
        'Temp >38.3°C or <36°C': inp.get('Temp',37) > 38.3 or inp.get('Temp',37) < 36.0,
        'HR > 90 bpm':           inp.get('HR',80) > 90,
        'Resp > 20 /min':        inp.get('Resp',16) > 20,
        'WBC >12 or <4 K/µL':    inp.get('WBC',8) > 12 or inp.get('WBC',8) < 4,
    }


def compute_sofa(inp):
    """Simplified SOFA approximation (0–12)."""
    s = 0
    o2 = inp.get('O2Sat', 97)
    if o2 < 90: s += 3
    elif o2 < 94: s += 2
    elif o2 < 96: s += 1

    bili = inp.get('Bilirubin_total', 0.8)
    if bili >= 6: s += 3
    elif bili >= 2: s += 2
    elif bili >= 1.2: s += 1

    mp = inp.get('MAP', 85)
    if mp < 50: s += 3
    elif mp < 65: s += 2
    elif mp < 70: s += 1

    cr = inp.get('Creatinine', 1.0)
    if cr >= 3.5: s += 3
    elif cr >= 2.0: s += 2
    elif cr >= 1.2: s += 1
    return s


def shock_index(inp):
    hr  = inp.get('HR', 80)
    sbp = inp.get('SBP', 120)
    return round(hr / sbp, 2) if sbp > 0 else 0.0


def clinical_recs(risk_pct, inp, sirs_met, sofa):
    recs = []
    if risk_pct >= 60:
        recs.append(("high","⚠️ IMMEDIATE — SEP-1 PROTOCOL",
                     "Activate Sepsis Alert. Notify attending physician NOW. Start 30 mL/kg IV crystalloid within 3 hours."))
        recs.append(("high","🧫 CULTURES + LABS",
                     "Blood cultures × 2 BEFORE antibiotics. STAT: CBC, BMP, LFTs, coagulation, procalcitonin."))
        recs.append(("high","💊 ANTIBIOTICS",
                     f"Broad-spectrum IV antibiotics within 1 hour. MAP currently {inp.get('MAP',0):.0f} mmHg — target ≥65."))
        recs.append(("info","📊 MONITORING",
                     "Continuous vitals, foley catheter for urine output. Target UO ≥0.5 mL/kg/hr. Reassess every 15 min."))
    elif risk_pct >= 30:
        recs.append(("mod","⚡ INCREASED VIGILANCE",
                     f"SIRS criteria met: {sirs_met}/4. Reassess every 30 minutes. Watch for clinical deterioration."))
        recs.append(("mod","🔬 DIAGNOSTIC WORKUP",
                     "Order CBC, CMP, Lactate, Blood Cultures if febrile. Consider empiric antibiotics if source identified."))
        recs.append(("info","💧 FLUID ASSESSMENT",
                     "Assess volume responsiveness. IV fluid challenge 500 mL NS if hypotensive or tachycardic."))
    else:
        recs.append(("low","✅ ROUTINE MONITORING",
                     "Continue standard ICU monitoring. Reassess every 4 hours."))
        if sofa >= 2:
            recs.append(("info","📋 SOFA NOTE",
                         f"SOFA score {sofa} — document organ function baseline. Trend daily LFTs and creatinine."))
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
BG  = '#040d18'
AX  = '#07111f'
GRID_C = (1, 1, 1, 0.05)
TICK_C = '#4a7080'
TEXT_C = '#c8dde8'

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX)
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.07))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TICK_C, labelsize=8.5)
    ax.xaxis.label.set_color('#3a6070')
    ax.yaxis.label.set_color('#3a6070')


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 16px'>
        <div style='font-size:38px'>🏥</div>
        <div style='font-size:14px;font-weight:800;color:#ffffff;margin-top:7px;letter-spacing:-0.3px'>Patient Vitals Input</div>
        <div style='font-size:10px;color:#2a5a6a;margin-top:3px;font-family:JetBrains Mono'>ICU Clinical Parameters</div>
    </div>""", unsafe_allow_html=True)

    # ── Input mode ────────────────────────────────────────────
    input_mode = st.radio(
        "Input method",
        ["Manual entry", "Load preset scenario", "Upload .txt file"],
        horizontal=False
    )

    inputs = {}

    # ── PRESET MODE ───────────────────────────────────────────
    if input_mode == "Load preset scenario":
        preset = st.selectbox("Select scenario", list(PRESETS.keys()))
        inputs = {k: float(v) for k, v in PRESETS[preset].items()}
        # Ensure all features present
        for f in features:
            if f not in inputs:
                inputs[f] = float(DEFAULTS.get(f, 0.0))
        st.success(f"✅ Scenario loaded — {len(inputs)} values")

    # ── FILE UPLOAD MODE ──────────────────────────────────────
    elif input_mode == "Upload .txt file":
        st.markdown("""<div class='i-box' style='margin-bottom:10px;font-size:11px'>
        One numeric value per line in this order:<br><br>
        HR, O2Sat, Temp, SBP, MAP, DBP, Resp,<br>
        WBC, Creatinine, Bilirubin_total, Lactate,<br>
        Glucose, Hgb, pH, Age, HospAdmTime, ICULOS
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload .txt", type=["txt"])
        if uploaded:
            lines = uploaded.read().decode().strip().split('\n')
            vals  = [float(v.strip()) for v in lines if v.strip()]
            for i, feat in enumerate(features):
                inputs[feat] = vals[i] if i < len(vals) else float(DEFAULTS.get(feat, 0.0))
            st.success(f"✅ Loaded {len(vals)} values")
        else:
            for f in features:
                inputs[f] = float(DEFAULTS.get(f, 0.0))

    # ── MANUAL ENTRY MODE ─────────────────────────────────────
    else:
        for grp_name, feats in GROUPS:
            with st.expander(grp_name, expanded=(grp_name == "📊 Vital Signs")):
                for feat in feats:
                    lbl  = FEATURE_LABELS.get(feat, feat)
                    cfg  = CLINICAL_RANGES.get(feat, (0, 200, 0, 300, ''))
                    unit = cfg[4]
                    lo   = float(cfg[2])
                    hi   = float(cfg[3])
                    dflt = float(DEFAULTS.get(feat, (lo + hi) / 2))
                    disp = f"{lbl}  [{unit}]" if unit else lbl
                    val  = st.number_input(
                        disp,
                        min_value=lo,
                        max_value=hi,
                        value=dflt,
                        step=0.1,
                        key=f"inp_{feat}"
                    )
                    inputs[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "⚡  Analyze Sepsis Risk",
        type="primary",
        use_container_width=True
    )

    st.markdown("<hr style='border:1px solid #0a1e2a;margin:14px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10.5px;color:#1e3548;line-height:1.9;font-family:JetBrains Mono'>
        <span style='color:#3a6a7a;font-weight:700'>Risk thresholds</span><br>
        🟢 &lt;30% — Low risk<br>
        🟡 30–60% — Moderate<br>
        🔴 &gt;60% — High risk<br><br>
        <span style='color:#3a6a7a;font-weight:700'>Model info</span><br>
        XGBoost · AUROC 0.8158<br>
        Sensitivity 0.74<br>
        Specificity 0.81<br>
        PhysioNet 2019 · 40,336 pts
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
hc1, hc2 = st.columns([3.2, 1])
with hc1:
    st.markdown(f"""
    <div class='header-card'>
        <p class='header-title'>Sepsis Early <span class='teal'>Warning</span> System</p>
        <p class='header-sub'>
            AI-powered ICU monitoring · 6-hour early prediction · XGBoost + SHAP Explainability<br>
            <span style='opacity:0.45'>PhysioNet CinC 2019 · 40,336 patients · AUROC 0.8158</span>
        </p>
    </div>""", unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:8px'>
        <span class='badge'>XGBoost</span>
        <span class='badge'>SHAP</span>
        <span class='badge badge-red'>LIVE</span><br>
        <div class='time-badge' style='margin-top:10px'>{now_str}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Stats bar
s1, s2, s3, s4, s5 = st.columns(5)
for col, (num, lbl) in zip([s1, s2, s3, s4, s5], [
    ("40,336", "Training Patients"),
    ("0.8158", "AUROC Score"),
    ("6 hrs",  "Early Warning"),
    ("17",     "Clinical Features"),
    ("0.74/0.81", "Sens / Spec"),
]):
    with col:
        st.markdown(
            f'<div class="stat-card"><p class="stat-num">{num}</p><p class="stat-lbl">{lbl}</p></div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION BLOCK
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    # Build input array in the exact feature order the model expects
    input_array = np.array([[float(inputs.get(f, DEFAULTS.get(f, 0.0))) for f in features]])

    with st.spinner("🔬 Analyzing patient vitals..."):
        time.sleep(0.4)
        proba      = model.predict_proba(input_array)[0]
        risk_score = float(proba[1])
        risk_pct   = round(risk_score * 100, 1)

    # Clinical scores
    sirs_dict = compute_sirs(inputs)
    sirs_met  = sum(sirs_dict.values())
    sofa      = compute_sofa(inputs)
    si        = shock_index(inputs)
    recs      = clinical_recs(risk_pct, inputs, sirs_met, sofa)

    # Risk tier
    if risk_pct >= 60:
        css, icon, level = "risk-high", "🔴", "HIGH RISK"
        msg      = "Sepsis predicted within 6 hours — Immediate clinical review required"
        bar_clr  = "#dc2626"
        bar_grad = "linear-gradient(90deg, #7f1d1d, #dc2626)"
        alert_c  = "a-high"
        sum_bg   = "#2a0404"; sum_brd = "#8a1a1a"; sum_clr = "#fca5a5"
    elif risk_pct >= 30:
        css, icon, level = "risk-mod", "🟡", "MODERATE RISK"
        msg      = "Elevated sepsis risk — Monitor closely, reassess every 30 minutes"
        bar_clr  = "#d97706"
        bar_grad = "linear-gradient(90deg, #78350f, #d97706)"
        alert_c  = "a-mod"
        sum_bg   = "#2a1500"; sum_brd = "#7a4800"; sum_clr = "#fde68a"
    else:
        css, icon, level = "risk-low", "🟢", "LOW RISK"
        msg      = "No immediate sepsis threat — Continue standard monitoring"
        bar_clr  = "#059669"
        bar_grad = "linear-gradient(90deg, #064e3b, #059669)"
        alert_c  = "a-low"
        sum_bg   = "#01180a"; sum_brd = "#1a6a3a"; sum_clr = "#6ee7b7"

    pct_w = min(int(risk_pct), 100)

    # ── ROW 1: Risk score (left) + SHAP bar (right) ───────────────────────────
    col_l, col_r = st.columns([1, 1.08], gap="large")

    with col_l:
        # Risk card — NO circle/ring
        st.markdown('<p class="s-hdr">Risk Assessment</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card {css}">
            <p class="risk-pct">{risk_pct}%</p>
            <p class="risk-level">{icon} {level}</p>
            <p class="risk-msg">{msg}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prog-track">
            <div class="prog-fill" style="width:{pct_w}%;background:{bar_grad}"></div>
        </div>
        <div class="prog-labels">
            <span>0%</span><span>◂30%</span><span>◂60%</span><span>100%</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── Clinical severity scores ───────────────────────────────────────
        st.markdown('<p class="s-hdr">Clinical Severity Scores</p>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            c = "#f87171" if sirs_met >= 2 else "#fbbf24" if sirs_met == 1 else "#4ade80"
            b = "#7a1a1a" if sirs_met >= 2 else "#7a4800" if sirs_met == 1 else "#1a5a3a"
            st.markdown(f"""<div class="score-box" style="border-color:{b};background:#07111f">
                <p class="score-num" style="color:{c}">{sirs_met}/4</p>
                <p class="score-lbl" style="color:{c}88">SIRS Criteria</p></div>""",
                unsafe_allow_html=True)
        with sc2:
            c = "#f87171" if sofa >= 6 else "#fbbf24" if sofa >= 3 else "#4ade80"
            b = "#7a1a1a" if sofa >= 6 else "#7a4800" if sofa >= 3 else "#1a5a3a"
            st.markdown(f"""<div class="score-box" style="border-color:{b};background:#07111f">
                <p class="score-num" style="color:{c}">{sofa}</p>
                <p class="score-lbl" style="color:{c}88">SOFA Score</p></div>""",
                unsafe_allow_html=True)
        with sc3:
            c = "#f87171" if si > 1.0 else "#fbbf24" if si > 0.7 else "#4ade80"
            b = "#7a1a1a" if si > 1.0 else "#7a4800" if si > 0.7 else "#1a5a3a"
            st.markdown(f"""<div class="score-box" style="border-color:{b};background:#07111f">
                <p class="score-num" style="color:{c}">{si}</p>
                <p class="score-lbl" style="color:{c}88">Shock Index</p></div>""",
                unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── SIRS breakdown ─────────────────────────────────────────────────
        st.markdown('<p class="s-hdr">SIRS Criteria Breakdown</p>', unsafe_allow_html=True)
        st.markdown("<div class='sirs-wrap'>", unsafe_allow_html=True)
        for criterion, met in sirs_dict.items():
            dot_clr = '#dc2626' if met else '#059669'
            txt_clr = '#fca5a5' if met else '#6ee7b7'
            icon_s  = '✗' if met else '✓'
            st.markdown(f"""
            <div class='sirs-row'>
                <div class='sirs-dot' style='background:{dot_clr}'></div>
                <span style='color:{txt_clr}'>{icon_s} {criterion}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Recommendations ────────────────────────────────────────────────
        st.markdown('<p class="s-hdr">Clinical Recommendations</p>', unsafe_allow_html=True)
        rc_map = {"high":"rc-high","mod":"rc-mod","low":"rc-low","info":"rc-info"}
        for i, (typ, title, text) in enumerate(recs):
            st.markdown(f"""
            <div class="rec-card {rc_map.get(typ,'rc-info')}" style="animation-delay:{i*0.1}s">
                <div class="rec-title">{title}</div>
                {text}
            </div>""", unsafe_allow_html=True)

    with col_r:
        # ── SHAP top-12 horizontal bar ─────────────────────────────────────
        st.markdown('<p class="s-hdr">SHAP Explanation — Feature Impact</p>', unsafe_allow_html=True)

        shap_values = explainer.shap_values(input_array)
        sv          = shap_values[0]
        feat_imp    = {f: float(sv[i]) for i, f in enumerate(features)}
        top12       = dict(sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:12])

        labels_s = [FEATURE_LABELS.get(k, k) for k in top12]
        vals_s   = list(top12.values())
        clrs_s   = [(0.96, 0.37, 0.37, 0.85) if v > 0 else (0.22, 0.83, 0.56, 0.85) for v in vals_s]

        fig1, ax1 = plt.subplots(figsize=(7.0, 5.2))
        style_ax(ax1, fig1)

        bars1 = ax1.barh(labels_s[::-1], vals_s[::-1], color=clrs_s[::-1],
                         edgecolor='none', height=0.62)
        ax1.axvline(0, color=(1, 1, 1, 0.14), linewidth=0.8)

        # Subtle row stripes
        for i, bar in enumerate(bars1):
            if i % 2 == 0:
                ax1.barh(bar.get_y() - 0.01, 9e3, left=-9e3,
                         height=bar.get_height() + 0.02,
                         color=(1, 1, 1, 0.015), zorder=0)

        # Value labels on bars — readable white text
        for bar, v in zip(bars1, vals_s[::-1]):
            off = 0.002 if v >= 0 else -0.002
            ax1.text(v + off, bar.get_y() + bar.get_height() / 2,
                     f'{v:+.3f}', va='center',
                     ha='left' if v >= 0 else 'right',
                     color='#d0e8f0', fontsize=7.5,
                     fontfamily='monospace', fontweight='500')

        ax1.set_xlabel('SHAP value  (impact on prediction)', fontsize=8.5)
        ax1.set_title('Feature contributions — current patient', color=TEXT_C,
                      fontsize=9.5, fontweight='600', pad=10)
        ax1.grid(axis='x', color=GRID_C, linewidth=0.5)

        red_p   = mpatches.Patch(color=(0.96, 0.37, 0.37), label='↑ Increases risk')
        green_p = mpatches.Patch(color=(0.22, 0.83, 0.56), label='↓ Decreases risk')
        ax1.legend(handles=[red_p, green_p], loc='lower right',
                   facecolor=BG, edgecolor=(1, 1, 1, 0.08),
                   labelcolor=TICK_C, fontsize=8.5)

        plt.tight_layout(pad=1.0)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        # Top driver explainer box
        top_feat = list(top12.keys())[0]
        top_val  = list(top12.values())[0]
        top_lbl  = FEATURE_LABELS.get(top_feat, top_feat)
        top_inp  = inputs.get(top_feat, 0)
        unit_t   = CLINICAL_RANGES.get(top_feat, ('', '', '', '', ''))[4]
        direction = "↑ increasing" if top_val > 0 else "↓ decreasing"
        dir_clr   = "#f87171" if top_val > 0 else "#4ade80"

        st.markdown(f"""
        <div class="i-box" style="margin-top:8px">
            <b style='color:#00c8b4'>Top driver:</b>
            <b style='color:#fbbf24'> {top_lbl}</b>
            <span style='font-family:JetBrains Mono;color:#d0e8f0'> = {top_inp:.1f} {unit_t}</span>
            <span style='color:#6a9aaa'> (SHAP </span>
            <span style='font-family:JetBrains Mono;color:#d0e8f0'>{top_val:+.3f}</span>
            <span style='color:#6a9aaa'>)</span> —
            strongest factor <b style='color:{dir_clr}'>{direction}</b> risk.<br><br>
            <span style='color:#3a6070'>🔴 Red bar = pushes risk higher &nbsp;·&nbsp;
            🟢 Green bar = pulls risk lower</span>
        </div>""", unsafe_allow_html=True)

        # ── Model probability bar ──────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="s-hdr">Model Probability Distribution</p>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(7.0, 1.55))
        style_ax(ax2, fig2)

        no_risk = 1 - risk_score
        ax2.barh(['No Sepsis', 'Sepsis'],
                 [no_risk, risk_score],
                 color=[(0.22, 0.83, 0.56, 0.78), (0.96, 0.37, 0.37, 0.78)],
                 edgecolor='none', height=0.45)
        ax2.axvline(0.5, color=(1, 1, 1, 0.18), linewidth=0.8, linestyle='--')
        ax2.set_xlim(0, 1)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
        ax2.tick_params(colors=TICK_C, labelsize=8)
        for s in ax2.spines.values():
            s.set_visible(False)

        ax2.text(no_risk - 0.01, 0, f'{no_risk*100:.1f}%', va='center', ha='right',
                 color='#4ade80', fontsize=10, fontweight='bold')
        ax2.text(risk_score - 0.01, 1, f'{risk_score*100:.1f}%', va='center', ha='right',
                 color='#f87171', fontsize=10, fontweight='bold')

        plt.tight_layout(pad=0.6)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── ROW 2: Vitals grid (left) + Radar + Quick Scores (right) ─────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    v1, v2 = st.columns([1.7, 1], gap="large")

    with v1:
        st.markdown('<p class="s-hdr">Current Vitals — Status Overview</p>', unsafe_allow_html=True)

        vitals_show = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP',
                       'Resp', 'WBC', 'Lactate', 'Creatinine', 'pH', 'Glucose', 'Hgb']
        vitals_show = [v for v in vitals_show if v in inputs]

        rows_v = [vitals_show[i:i+4] for i in range(0, len(vitals_show), 4)]
        for row_feats in rows_v:
            cols_v = st.columns(len(row_feats))
            for col_v, feat in zip(cols_v, row_feats):
                val    = inputs[feat]
                status = vital_status(feat, val)
                unit   = CLINICAL_RANGES.get(feat, ('', '', '', '', ''))[4]
                lbl    = FEATURE_LABELS.get(feat, feat)
                clr    = vital_color(status)
                chip   = vital_chip_class(status)
                mn     = CLINICAL_RANGES.get(feat, (0, 0, 0, 0, ''))[0]
                mx     = CLINICAL_RANGES.get(feat, (0, 0, 0, 0, ''))[1]
                with col_v:
                    st.markdown(f"""
                    <div class="vital-chip {chip}">
                        <p class="vital-name">{lbl}</p>
                        <p class="vital-val" style="color:{clr}">{val:.1f}
                            <span class="vital-unit">{unit}</span>
                        </p>
                        <p class="vital-range">{mn}–{mx}</p>
                    </div>""", unsafe_allow_html=True)

        # Alert strips
        st.markdown("<div style='height:7px'></div>", unsafe_allow_html=True)
        crit_v = [FEATURE_LABELS.get(f, f) for f in vitals_show if vital_status(f, inputs[f]) == 'critical']
        abno_v = [FEATURE_LABELS.get(f, f) for f in vitals_show if vital_status(f, inputs[f]) == 'abnormal']
        norm_v = [FEATURE_LABELS.get(f, f) for f in vitals_show if vital_status(f, inputs[f]) == 'normal']

        if crit_v:
            st.markdown(f'<div class="alert-strip a-high">⚠️ CRITICAL — {", ".join(crit_v)}</div>',
                        unsafe_allow_html=True)
        if abno_v:
            st.markdown(f'<div class="alert-strip a-mod">⚡ ABNORMAL — {", ".join(abno_v)}</div>',
                        unsafe_allow_html=True)
        if not crit_v and not abno_v:
            st.markdown('<div class="alert-strip a-low">✅ All displayed vitals within normal range</div>',
                        unsafe_allow_html=True)

        lact_val = inputs.get('Lactate', 1.5)
        if lact_val > 4:
            lact_msg = "Severe hyperlactatemia — tissue shock likely"
            lact_cls = "a-high"
        elif lact_val > 2:
            lact_msg = "Elevated — possible tissue hypoperfusion"
            lact_cls = "a-mod"
        else:
            lact_msg = "Within acceptable range"
            lact_cls = "a-low"
        st.markdown(f'<div class="alert-strip {lact_cls}">🧬 LACTATE {lact_val:.1f} mmol/L — {lact_msg}</div>',
                    unsafe_allow_html=True)

    with v2:
        # ── Radar chart ────────────────────────────────────────────────────
        st.markdown('<p class="s-hdr">Vital Signs Radar</p>', unsafe_allow_html=True)

        radar_feats = ['HR', 'O2Sat', 'Temp', 'SBP', 'Resp', 'WBC', 'Lactate', 'pH']
        radar_feats = [f for f in radar_feats if f in inputs]
        N = len(radar_feats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

        def norm_vital(feat, val):
            mn, mx, clo, chi, _ = CLINICAL_RANGES.get(feat, (0, 1, 0, 1, ''))
            r = chi - clo
            return max(0.0, min(1.0, (val - clo) / r)) if r > 0 else 0.5

        patient_norm = [norm_vital(f, inputs[f]) for f in radar_feats] + [norm_vital(radar_feats[0], inputs[radar_feats[0]])]
        default_norm = [norm_vital(f, DEFAULTS[f]) for f in radar_feats] + [norm_vital(radar_feats[0], DEFAULTS[radar_feats[0]])]

        fig3, ax3 = plt.subplots(figsize=(4.0, 4.0), subplot_kw=dict(polar=True))
        fig3.patch.set_facecolor(BG)
        ax3.set_facecolor(BG)

        for r in [0.25, 0.5, 0.75, 1.0]:
            ax3.plot(angles, [r] * len(angles), color=(1, 1, 1, 0.07), linewidth=0.5)

        # Normal baseline
        ax3.plot(angles, default_norm, color=(0.22, 0.83, 0.56, 0.45),
                 linewidth=1.2, linestyle='--')
        ax3.fill(angles, default_norm, color=(0.22, 0.83, 0.56, 0.05))

        # Patient
        if risk_pct >= 60:
            p_clr = (0.96, 0.37, 0.37)
        elif risk_pct >= 30:
            p_clr = (0.95, 0.77, 0.22)
        else:
            p_clr = (0.22, 0.83, 0.56)

        ax3.plot(angles, patient_norm, color=(*p_clr, 0.9), linewidth=2.0)
        ax3.fill(angles, patient_norm, color=(*p_clr, 0.12))

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(
            [FEATURE_LABELS.get(f, f) for f in radar_feats],
            color='#5a8090', fontsize=7.5
        )
        ax3.set_yticks([])
        ax3.spines['polar'].set_color((1, 1, 1, 0.08))
        ax3.grid(color=(1, 1, 1, 0.06), linewidth=0.5)
        ax3.set_title('Patient vs. Normal Baseline', color='#7ab8c8',
                      fontsize=9, pad=12)

        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # Quick scores table
        st.markdown(f"""
        <div class="scores-box" style="margin-top:10px">
            <div style="font-size:9.5px;color:#2a5a6a;text-transform:uppercase;
                        letter-spacing:1.2px;margin-bottom:8px;
                        font-family:JetBrains Mono;font-weight:700">Quick Scores</div>
            <div class='score-row'>
                <span class='score-label'>Shock Index (HR/SBP)</span>
                <span class='score-value' style='color:{("#f87171" if si>1.0 else "#fbbf24" if si>0.7 else "#4ade80")}'>{si}</span>
            </div>
            <div class='score-row'>
                <span class='score-label'>SOFA Score (approx)</span>
                <span class='score-value' style='color:{("#f87171" if sofa>=6 else "#fbbf24" if sofa>=3 else "#4ade80")}'>{sofa}/12</span>
            </div>
            <div class='score-row'>
                <span class='score-label'>SIRS Criteria</span>
                <span class='score-value' style='color:{("#f87171" if sirs_met>=2 else "#fbbf24" if sirs_met==1 else "#4ade80")}'>{sirs_met}/4</span>
            </div>
            <div class='score-row'>
                <span class='score-label'>Risk Probability</span>
                <span class='score-value' style='color:{bar_clr}'>{risk_pct}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── All-features SHAP bar ─────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="s-hdr">All Feature Contributions — Extended View (All 17 Features)</p>',
                unsafe_allow_html=True)

    all_sorted  = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    all_labels  = [FEATURE_LABELS.get(k, k) for k, _ in all_sorted]
    all_vals    = [v for _, v in all_sorted]
    all_colors  = [(0.96, 0.37, 0.37, 0.82) if v > 0 else (0.22, 0.83, 0.56, 0.82) for v in all_vals]

    fig4, ax4 = plt.subplots(figsize=(12, 3.8))
    style_ax(ax4, fig4)

    x_pos = range(len(all_labels))
    bars4 = ax4.bar(list(x_pos), all_vals, color=all_colors, edgecolor='none', width=0.65)
    ax4.axhline(0, color=(1, 1, 1, 0.12), linewidth=0.8)

    # Alternating faint column bands
    for i in range(0, len(all_labels), 2):
        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.03, color='white', zorder=0)

    ax4.set_xticks(list(x_pos))
    ax4.set_xticklabels(all_labels, rotation=32, ha='right', color='#6a9aaa', fontsize=8.5)
    ax4.set_ylabel('SHAP Value', fontsize=8.5)
    ax4.set_title('SHAP Values — All 17 Features (current patient)', color=TEXT_C,
                  fontsize=10, fontweight='600', pad=8)
    ax4.grid(axis='y', color=GRID_C, linewidth=0.5)

    for bar, v in zip(bars4, all_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 v + (0.002 if v >= 0 else -0.002),
                 f'{v:+.3f}', ha='center',
                 va='bottom' if v >= 0 else 'top',
                 color='#c8dde8', fontsize=6.5, fontfamily='monospace')

    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()

    # ── Final summary bar ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="summary-bar" style="background:{sum_bg};border:1px solid {sum_brd};color:{sum_clr}">
        {icon} {level} — {risk_pct}% probability of sepsis within 6 hours &nbsp;|&nbsp;
        SIRS {sirs_met}/4 &nbsp;·&nbsp; SOFA {sofa} &nbsp;·&nbsp; Shock Index {si}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT STATE (before analysis button)
# ══════════════════════════════════════════════════════════════════════════════
else:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="i-box" style="text-align:center;padding:34px 22px">
            <div style="font-size:42px;margin-bottom:10px">👈</div>
            <div style="font-size:15px;font-weight:700;color:#d0e8f0;margin-bottom:8px">Enter Patient Vitals</div>
            <div style="font-size:12px;color:#3a6070">Use the sidebar to input ICU readings,<br>
            then click <b style='color:#00c8b4'>Analyze Sepsis Risk</b></div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="i-box" style="text-align:center;padding:34px 22px">
            <div style="font-size:42px;margin-bottom:10px">💡</div>
            <div style="font-size:15px;font-weight:700;color:#d0e8f0;margin-bottom:8px">Quick Demo</div>
            <div style="font-size:12px;color:#3a6070">Switch to
            <b style='color:#00c8b4'>Load preset scenario</b><br>
            to instantly see High / Moderate / Normal examples</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="s-hdr">How This System Works</p>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (n, t, d) in zip([sc1, sc2, sc3, sc4], [
        ("1", "Input Vitals",      "Enter 17 ICU measurements — vitals, labs, and patient history"),
        ("2", "XGBoost Predicts",  "Model trained on 40,336 ICU patients calculates 6-hour sepsis probability"),
        ("3", "Risk Scored",       "Green / Amber / Red alert with SIRS, SOFA, and Shock Index computed live"),
        ("4", "SHAP Explains",     "Bar chart shows which vital drove the prediction — transparent AI"),
    ]):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <p class="step-n">{n}</p>
                <p class="step-t">{t}</p>
                <p class="step-d">{d}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="s-hdr">Clinical Reference Ranges</p>', unsafe_allow_html=True)
    ref_df = pd.DataFrame({
        'Feature':        ['Heart Rate', 'O₂ Sat', 'Temperature', 'Lactate',
                           'WBC', 'Creatinine', 'Blood pH', 'Resp Rate', 'MAP'],
        'Normal Range':   ['60–100 bpm', '95–100%', '36.1–37.2 °C', '0.5–2.0 mmol/L',
                           '4.5–11.0 K/µL', '0.7–1.2 mg/dL', '7.35–7.45', '12–20 /min', '70–105 mmHg'],
        'Sepsis Signal':  ['>90 or <60', '<94%', '>38.3°C or <36°C', '>2.0 mmol/L',
                           '>12 or <4 K/µL', '>1.5 mg/dL', '<7.35', '>22 /min', '<65 mmHg'],
        'Severity':       ['Moderate', 'High', 'Moderate', 'High',
                           'Moderate', 'Moderate', 'High', 'Moderate', 'Critical'],
    })
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    Sepsis Early Warning System &nbsp;·&nbsp; Zaid Ali &nbsp;·&nbsp; AWKUM 2025–26 &nbsp;·&nbsp;
    XGBoost AUROC 0.8158 &nbsp;·&nbsp; PhysioNet CinC 2019
</div>""", unsafe_allow_html=True)
