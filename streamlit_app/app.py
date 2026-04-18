# streamlit_app/app.py
# Run with: streamlit run streamlit_app/app.py

import streamlit as st
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide"
)

# ── Load models ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model    = joblib.load('models/xgboost_model.pkl')
    scaler   = joblib.load('models/scaler.pkl')
    features = json.load(open('models/feature_list.json'))
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

model, scaler, features, explainer = load_models()

# ── Title ─────────────────────────────────────────────────────────
st.title("🏥 Sepsis Early Warning System")
st.markdown("**AI-powered sepsis prediction — enter patient vitals to get risk score**")
st.markdown("---")

# ── Sidebar: patient vitals input ────────────────────────────────
st.sidebar.header("Patient Vitals Input")

feature_defaults = {
    'HR': 85.0, 'O2Sat': 97.0, 'Temp': 37.0, 'SBP': 120.0,
    'MAP': 85.0, 'DBP': 70.0, 'Resp': 18.0, 'WBC': 8.0,
    'Creatinine': 1.0, 'Bilirubin_total': 0.8, 'Lactate': 1.5,
    'Glucose': 110.0, 'Hgb': 13.0, 'pH': 7.4,
    'Age': 55.0, 'HospAdmTime': -24.0, 'ICULOS': 2.0
}

feature_ranges = {
    'HR': (20, 250), 'O2Sat': (50, 100), 'Temp': (30.0, 42.0),
    'SBP': (50, 250), 'MAP': (30, 200), 'DBP': (20, 180),
    'Resp': (5, 60), 'WBC': (0.5, 50.0), 'Creatinine': (0.1, 20.0),
    'Bilirubin_total': (0.1, 30.0), 'Lactate': (0.3, 20.0),
    'Glucose': (40, 500), 'Hgb': (3.0, 20.0), 'pH': (6.8, 7.8),
    'Age': (18, 100), 'HospAdmTime': (-200, 0), 'ICULOS': (0, 100)
}

feature_labels = {
    'HR': 'Heart Rate (bpm)', 'O2Sat': 'O2 Saturation (%)',
    'Temp': 'Temperature (°C)', 'SBP': 'Systolic BP (mmHg)',
    'MAP': 'Mean Art. Pressure', 'DBP': 'Diastolic BP (mmHg)',
    'Resp': 'Respiratory Rate', 'WBC': 'WBC Count (K/µL)',
    'Creatinine': 'Creatinine (mg/dL)', 'Bilirubin_total': 'Bilirubin (mg/dL)',
    'Lactate': 'Lactate (mmol/L)', 'Glucose': 'Glucose (mg/dL)',
    'Hgb': 'Hemoglobin (g/dL)', 'pH': 'Blood pH',
    'Age': 'Patient Age', 'HospAdmTime': 'Time since Admission (hr)',
    'ICULOS': 'ICU Length of Stay (days)'
}

inputs = {}
for feat in features:
    label = feature_labels.get(feat, feat)
    lo, hi = feature_ranges.get(feat, (0.0, 200.0))
    default = feature_defaults.get(feat, (lo+hi)/2)
    inputs[feat] = st.sidebar.number_input(
        label, min_value=float(lo), max_value=float(hi),
        value=float(default), step=0.1
    )

predict_btn = st.sidebar.button("🔍 Predict Sepsis Risk", type="primary")

# ── Main panel ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

if predict_btn:
    input_array = np.array([[inputs[f] for f in features]])
    risk_score  = model.predict_proba(input_array)[0][1]
    risk_pct    = round(risk_score * 100, 1)

    with col1:
        st.subheader("Risk Assessment")

        if risk_pct >= 60:
            st.error(f"🔴 HIGH RISK — {risk_pct}%")
            st.error("⚠️ Sepsis predicted within 6 hours. Immediate clinical review recommended.")
        elif risk_pct >= 30:
            st.warning(f"🟡 MODERATE RISK — {risk_pct}%")
            st.warning("Monitor closely. Reassess vitals every 30 minutes.")
        else:
            st.success(f"🟢 LOW RISK — {risk_pct}%")
            st.success("No immediate sepsis risk. Continue standard monitoring.")

        # Risk bar
        st.progress(risk_score)

        st.markdown("---")
        st.caption("Risk thresholds: <30% = Low | 30–60% = Moderate | >60% = High")

    with col2:
        st.subheader("SHAP Explanation — Why this score?")

        shap_values = explainer.shap_values(input_array)

        fig, ax = plt.subplots(figsize=(7, 4))
        feat_imp = dict(zip(features, shap_values[0]))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        colors = ['#E24B4A' if v > 0 else '#1D9E75' for v in feat_imp.values()]
        bars = ax.barh(list(feat_imp.keys())[::-1],
                       list(feat_imp.values())[::-1],
                       color=colors[::-1], edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on risk score)')
        ax.set_title('Top 10 Features Driving This Prediction', fontweight='bold')
        st.pyplot(fig)
        plt.close()

        st.caption("🔴 Red = increases sepsis risk | 🟢 Green = decreases sepsis risk")

else:
    with col1:
        st.info("👈 Enter patient vitals in the sidebar and click **Predict Sepsis Risk**")
    with col2:
        st.info("SHAP explanation will appear here after prediction")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Sepsis Early Warning System | Zaid Ali | BS CS 4th Semester | AWKUM 2025–26 | Built with XGBoost + SHAP")