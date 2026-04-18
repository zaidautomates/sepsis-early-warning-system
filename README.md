# 🏥 Sepsis Early Warning System

> **AI-powered ICU monitoring — 6-hour early sepsis prediction using XGBoost + SHAP Explainability**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUROC_0.8158-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-00C8B4?style=flat-square)](https://shap.readthedocs.io)
[![Dataset](https://img.shields.io/badge/PhysioNet_CinC_2019-40%2C336_Patients-6B21A8?style=flat-square)](https://physionet.org/content/challenge-2019/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## Overview

Sepsis is a **life-threatening medical emergency** — killing over 270,000 patients annually in the US alone. Every hour of delayed treatment increases mortality by 7%. Yet clinical teams are often overwhelmed, and early warning signs are easy to miss.

This project builds a **real-time ICU clinical decision support tool** that:

- Predicts the **probability of sepsis onset within 6 hours** from 17 clinical parameters
- Achieves **AUROC 0.8158** on the PhysioNet 2019 benchmark dataset (40,336 ICU patients)
- Provides **SHAP-based explainability** — clinicians see exactly *which vital sign* drove the alert
- Computes **SIRS criteria, SOFA score approximation, and Shock Index** automatically
- Delivers **actionable clinical recommendations** (SEP-1 bundle, fluid targets, lab orders)

---

## Demo

| Scenario | Risk Score | Key Drivers |
|----------|-----------|-------------|
| 🟢 Normal Patient (Age 45) | **12.3%** | All vitals within normal range |
| 🟡 Moderate Risk (HR 105, Lactate 3.2) | **44.7%** | Elevated lactate + tachycardia |
| 🔴 High Risk / Sepsis (MAP 48, pH 7.19) | **87.6%** | Hypotension + acidosis + hyperlactatemia |

---

## Features

### 🤖 Machine Learning
- **Model:** XGBoost Classifier, trained on 40,336 ICU patient records
- **AUROC:** 0.8158 | Sensitivity: 0.74 | Specificity: 0.81
- **Features:** 17 clinical parameters (vitals, labs, patient history)
- **Explainability:** SHAP TreeExplainer — per-patient feature contribution waterfall charts
- **Prediction window:** 6-hour early warning before sepsis onset

### 📊 Clinical Scoring (Computed in Real-Time)
| Score | What It Measures |
|-------|-----------------|
| **SIRS Criteria (0–4)** | Systemic Inflammatory Response Syndrome criteria met |
| **SOFA Score (approx)** | Sequential Organ Failure Assessment (respiration, liver, CV, renal) |
| **Shock Index** | HR / SBP ratio — proxy for circulatory compromise |
| **Lactate Status** | Tissue perfusion flag: normal / elevated / critical |

### 🖥️ Dashboard
- Live risk gauge with animated pulse ring (green / amber / red)
- Vitals status grid — color-coded by normal / abnormal / critical range
- SHAP feature importance bar chart (top 12 + all 17 features)
- Radar chart — patient vitals vs. normal baseline
- SIRS criteria breakdown (4 individual criteria)
- Model probability distribution
- Evidence-based clinical recommendations (SEP-1 bundle protocol)
- 3 preset scenarios for instant demo (Normal / Moderate / High Risk)
- Manual entry, preset loader, and `.txt` file upload modes

---

## Dataset

**PhysioNet Computing in Cardiology Challenge 2019**

| Property | Value |
|----------|-------|
| Total patients | 40,336 |
| ICU types | Medical, surgical, cardiac ICUs |
| Sepsis labels | Derived using Sepsis-3 clinical definition |
| Input features | 17 (8 vital signs + 6 lab values + 3 demographics) |
| Time resolution | Hourly readings |
| Positive class | ~5.7% (sepsis onset) |

**17 Clinical Features:**

| Category | Features |
|----------|----------|
| Vital Signs | HR, O₂ Saturation, Temperature, Respiratory Rate |
| Blood Pressure | SBP, DBP, MAP |
| Laboratory | WBC, Creatinine, Bilirubin, Lactate, Glucose, Hemoglobin, pH |
| Patient Context | Age, Hospital Admission Time Offset, ICU Length of Stay |

---

## Project Structure

```
sepsis-early-warning-system/
│
├── streamlit_app/
│   └── app.py                  # Main Streamlit dashboard (v2.0)
│
├── models/
│   ├── xgboost_model.pkl       # Trained XGBoost classifier
│   ├── scaler.pkl              # StandardScaler for feature normalization
│   └── feature_list.json      # Ordered list of 17 feature names
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
│
├── data/
│   └── README.md               # PhysioNet dataset instructions
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/zaidautomates/sepsis-early-warning-system.git
cd sepsis-early-warning-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add model files
Place the following in the `models/` directory:
```
models/xgboost_model.pkl
models/scaler.pkl
models/feature_list.json
```

### 4. Run the dashboard
```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Requirements

```
streamlit>=1.35.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
matplotlib>=3.7.0
joblib>=1.3.0
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUROC | **0.8158** |
| Sensitivity (Recall) | 0.74 |
| Specificity | 0.81 |
| Precision | 0.29 |
| F1 Score | 0.42 |

> The model prioritizes **sensitivity** — in a clinical setting, missing a sepsis case (false negative) is far more dangerous than a false alarm.

---

## Clinical Interpretation Guide

| Risk Level | Score | Recommended Action |
|------------|-------|--------------------|
| 🟢 Low | < 30% | Routine monitoring. Reassess every 4 hours. |
| 🟡 Moderate | 30–60% | Increased vigilance. Order labs. Reassess every 30 min. |
| 🔴 High | > 60% | Activate Sepsis Alert. SEP-1 bundle. Physician review immediately. |

**SIRS + AI Risk = Clinical Decision:**
- SIRS ≥ 2 criteria + Risk > 30% → Strong indication for sepsis workup
- SOFA ≥ 2 above baseline + Risk > 60% → Sepsis-3 criteria likely met

---

## ⚠️ Disclaimer

This system is a **clinical decision support tool** — it assists, it does not replace, physician judgment. All AI-generated risk scores and recommendations must be reviewed by a qualified healthcare professional before clinical action is taken.

---

## Author

**Zaid Ali**
BS Computer Science

---
