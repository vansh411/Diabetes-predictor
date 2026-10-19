"""
app/streamlit_app.py
Patient-friendly Streamlit web app for the Diabetes Risk Predictor.
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb

from utils.explain import get_top_risk_factors

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Check",
    page_icon="🩺",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS — friendly, clean, no jargon feel
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    .risk-box {
        text-align: center;
        padding: 35px 20px;
        border-radius: 18px;
        margin: 10px 0 20px 0;
    }
    .risk-number { font-size: 72px; font-weight: 800; margin: 0; }
    .risk-label  { font-size: 26px; font-weight: 700; margin: 6px 0 0 0; }
    .factor-card {
        padding: 14px 18px;
        border-radius: 12px;
        margin: 8px 0;
        font-size: 15px;
        line-height: 1.5;
    }
    .action-card {
        padding: 18px 22px;
        border-radius: 12px;
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.6;
    }
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 14px;
        color: #78350f;
        margin-top: 20px;
    }
    .reliability-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 18px 22px;
        font-size: 15px;
        line-height: 1.7;
    }
    .tip-box {
        background: #eff6ff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_diabetes.json")
    scaler    = joblib.load("models/scaler.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, scaler, explainer

try:
    model, scaler, explainer = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "Glucose_BMI", "Age_Group"
]

# Human-readable names for each feature
FEATURE_LABELS = {
    "Glucose":                  "Blood Sugar (Glucose) Level",
    "BMI":                      "Body Mass Index (BMI)",
    "Age":                      "Age",
    "BloodPressure":            "Blood Pressure",
    "Insulin":                  "Insulin Level",
    "SkinThickness":            "Skin Thickness",
    "Pregnancies":              "Number of Pregnancies",
    "DiabetesPedigreeFunction": "Family History of Diabetes",
    "Glucose_BMI":              "Blood Sugar + Weight Combination",
    "Age_Group":                "Age Group",
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#1e3a5f; font-size:36px; margin-bottom:4px;'>
        🩺 Diabetes Risk Check
    </h1>
    <p style='text-align:center; color:#64748b; font-size:16px; margin-bottom:24px;'>
        Enter your health details to get a personalised diabetes risk assessment
    </p>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model not ready yet. Please run `python train.py` first.\n\nDetails: {load_error}")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — User-friendly input labels
# ─────────────────────────────────────────────
st.sidebar.markdown("## 📋 Your Health Details")
st.sidebar.markdown("Fill in your most recent measurements:")

st.sidebar.markdown("---")

age            = st.sidebar.slider("🎂 Your Age",                        18, 85, 35)
glucose        = st.sidebar.slider("🩸 Blood Sugar Level (mg/dL)",       50, 200, 120,
                                    help="Measured after a 2-hour oral glucose tolerance test")
bmi            = st.sidebar.slider("⚖️ BMI (Body Mass Index)",           15.0, 67.0, 25.0, step=0.1,
                                    help="Weight(kg) ÷ Height(m)²  |  Normal: 18.5–24.9")
blood_pressure = st.sidebar.slider("💉 Blood Pressure (mm Hg)",          40, 130, 70,
                                    help="Diastolic blood pressure (the lower number)")
insulin        = st.sidebar.slider("💊 Insulin Level (µU/mL)",           10, 850, 80,
                                    help="2-Hour serum insulin level")
skin_thickness = st.sidebar.slider("📏 Skin Fold Thickness (mm)",         5, 100, 20,
                                    help="Triceps skin fold thickness")
pregnancies    = st.sidebar.slider("🤰 Number of Pregnancies",            0, 17, 1)
dpf            = st.sidebar.slider("👨‍👩‍👧 Family Diabetes History Score",  0.05, 2.5, 0.5, step=0.01,
                                    help="Higher = stronger family history of diabetes")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Check My Risk", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Tabs — plain English labels
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 My Risk Result", "💡 Why This Result?", "ℹ️ About This Tool"])

# ════════════════════════════════════════════
# TAB 1 — Result
# ════════════════════════════════════════════
with tab1:
    if predict_btn:
        # Feature engineering
        glucose_bmi = glucose * bmi
        age_group   = int(pd.cut([age], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])[0])
        raw         = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age, glucose_bmi, age_group]], dtype=float)
        scaled      = scaler.transform(raw)

        prob      = model.predict_proba(scaled)[0][1]
        risk_pct  = round(prob * 100, 1)

        # Risk tier
        if prob < 0.3:
            tier    = "Low Risk"
            color   = "#16a34a"
            bg      = "#f0fdf4"
            border  = "#86efac"
            emoji   = "🟢"
            summary = "Your results look healthy. Keep up your good habits!"
        elif prob < 0.6:
            tier    = "Moderate Risk"
            color   = "#d97706"
            bg      = "#fffbeb"
            border  = "#fcd34d"
            emoji   = "🟡"
            summary = "Some factors need attention. Small lifestyle changes can make a big difference."
        else:
            tier    = "High Risk"
            color   = "#dc2626"
            bg      = "#fef2f2"
            border  = "#fca5a5"
            emoji   = "🔴"
            summary = "Several risk factors were detected. We strongly recommend seeing a doctor soon."

        # ── Risk Score Box ──
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
                <div class='risk-box' style='background:{bg}; border: 2px solid {border};'>
                    <div style='font-size:56px;'>{emoji}</div>
                    <p class='risk-number' style='color:{color};'>{risk_pct}%</p>
                    <p class='risk-label'  style='color:{color};'>{tier}</p>
                    <p style='color:#555; font-size:15px; margin-top:10px;'>{summary}</p>
                </div>
            """, unsafe_allow_html=True)

        # ── Risk Factors in plain English ──
        st.markdown("### 📊 What's affecting your result?")
        factors = get_top_risk_factors(explainer, scaled, FEATURE_NAMES, top_n=6)

        raising  = [(f, v) for f, v in factors if v > 0]
        lowering = [(f, v) for f, v in factors if v < 0]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**⚠️ Factors raising your risk:**")
            if raising:
                for feat, val in raising[:3]:
                    label     = FEATURE_LABELS.get(feat, feat)
                    intensity = "strongly" if abs(val) > 0.3 else "slightly"
                    st.markdown(f"""
                        <div class='factor-card' style='background:#fef2f2; border-left:4px solid #dc2626;'>
                            🔴 <b>{label}</b><br>
                            <span style='color:#666;'>This is {intensity} increasing your risk</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='factor-card' style='background:#f0fdf4;'>✅ No strong risk-raising factors found</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("**✅ Factors in your favour:**")
            if lowering:
                for feat, val in lowering[:3]:
                    label     = FEATURE_LABELS.get(feat, feat)
                    intensity = "strongly" if abs(val) > 0.3 else "slightly"
                    st.markdown(f"""
                        <div class='factor-card' style='background:#f0fdf4; border-left:4px solid #16a34a;'>
                            🟢 <b>{label}</b><br>
                            <span style='color:#666;'>This is {intensity} reducing your risk</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='factor-card' style='background:#fffbeb;'>⚠️ No strong protective factors found</div>", unsafe_allow_html=True)

        # ── What should I do? ──
        st.markdown("---")
        st.markdown("### 🗺️ What should I do next?")

        if prob < 0.3:
            actions = [
                ("✅ Keep it up!", "Continue your healthy diet and regular exercise routine."),
                ("📅 Annual check-up", "Get a routine blood sugar test once a year as a precaution."),
                ("💧 Stay hydrated", "Drink enough water daily and maintain a balanced diet."),
            ]
        elif prob < 0.6:
            actions = [
                ("🥗 Review your diet", "Reduce sugary drinks, white bread, and processed foods."),
                ("🚶 Move more", "Aim for at least 30 minutes of walking 5 days a week."),
                ("🩺 See your doctor", "Book a blood sugar test with your GP in the next few weeks."),
                ("⚖️ Check your weight", "Even a small reduction in weight can significantly lower risk."),
            ]
        else:
            actions = [
                ("🚨 See a doctor soon", "Please book an appointment with your GP as soon as possible."),
                ("🩸 Get a blood test", "Ask for a fasting blood glucose and HbA1c test."),
                ("🚫 Avoid sugary foods", "Cut out sugary drinks, sweets, and refined carbs immediately."),
                ("🏃 Start moving", "Even light daily walks can help lower blood sugar levels."),
                ("👨‍👩‍👧 Tell your family", "Diabetes has a genetic component — let close family know to get checked too."),
            ]

        for title, desc in actions:
            st.markdown(f"""
                <div class='action-card' style='background:#eff6ff; border-left:4px solid #3b82f6;'>
                    <b>{title}</b><br>
                    <span style='color:#374151;'>{desc}</span>
                </div>
            """, unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown(f"""
            <div class='disclaimer'>
                ⚠️ <b>Important:</b> This tool is for <b>screening purposes only</b> and does <b>not</b>
                replace professional medical advice. Always consult a qualified doctor or healthcare
                provider for a proper diagnosis and treatment plan.
            </div>
        """, unsafe_allow_html=True)

        # Store for tab 2
        st.session_state["scaled"]  = scaled
        st.session_state["factors"] = factors
        st.session_state["prob"]    = prob

    else:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px; color:#94a3b8;'>
                <div style='font-size:64px;'>🩺</div>
                <h3 style='color:#94a3b8;'>Enter your details on the left and click <b>Check My Risk</b></h3>
                <p>It only takes 30 seconds</p>
            </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — Why this result (plain English SHAP)
# ════════════════════════════════════════════
with tab2:
    if "factors" in st.session_state:
        factors = st.session_state["factors"]
        prob    = st.session_state["prob"]

        st.markdown("### 💡 Why did you get this result?")
        st.markdown(
            "Our tool looked at **10 different health measurements** from your input. "
            "Here's a simple explanation of what mattered most:"
        )

        for i, (feat, val) in enumerate(factors):
            label     = FEATURE_LABELS.get(feat, feat)
            impact    = abs(val)
            direction = "raising ⬆️" if val > 0 else "lowering ⬇️"
            color     = "#dc2626" if val > 0 else "#16a34a"
            bg        = "#fef2f2" if val > 0 else "#f0fdf4"
            border    = "#fca5a5" if val > 0 else "#86efac"

            # Bar width proportional to impact
            bar_pct = min(int(impact / 0.6 * 100), 100)

            st.markdown(f"""
                <div style='background:{bg}; border:1px solid {border}; border-radius:12px;
                            padding:14px 18px; margin:8px 0;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <b style='font-size:15px;'>{i+1}. {label}</b>
                        <span style='color:{color}; font-weight:600;'>{direction} risk</span>
                    </div>
                    <div style='background:#e5e7eb; border-radius:999px; height:8px; margin-top:10px;'>
                        <div style='background:{color}; width:{bar_pct}%; height:8px; border-radius:999px;'></div>
                    </div>
                    <p style='color:#6b7280; font-size:13px; margin:6px 0 0 0;'>
                        Impact strength: {"High" if impact > 0.3 else "Medium" if impact > 0.1 else "Low"}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
            <div class='tip-box'>
                💬 <b>How to read this:</b> The longer the coloured bar, the more that factor
                influenced your result. Red bars are things raising your risk.
                Green bars are things protecting you.
            </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Run your risk check first (My Risk Result tab) to see the explanation here.")


# ════════════════════════════════════════════
# TAB 3 — About this tool (no jargon)
# ════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ About This Tool")

    st.markdown("""
        <div class='reliability-box'>
            <b>🤖 How reliable is this tool?</b><br><br>
            This tool was trained on health data from <b>768 real patients</b> and tested
            on patients it had never seen before.<br><br>
            Out of every <b>100 people</b> it checks:
            <ul>
                <li>✅ It gives the <b>correct result for about 78–83 people</b></li>
                <li>⚠️ About <b>17–22 people</b> may get a result that needs a follow-up check</li>
            </ul>
            This is <b>comparable to many standard screening tools</b> used in clinics —
            but it is <b>not a replacement for a real medical test</b>.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 What do the health measurements mean?")

    explanations = [
        ("🩸 Blood Sugar (Glucose)", "The amount of sugar in your blood after a glucose test. High levels can signal diabetes."),
        ("⚖️ BMI", "A measure of body fat based on your height and weight. Normal range is 18.5–24.9."),
        ("💉 Blood Pressure", "The pressure of blood in your arteries. Normal is around 60–80 mm Hg (diastolic)."),
        ("💊 Insulin Level", "A hormone that controls blood sugar. High levels can mean your body is struggling."),
        ("👨‍👩‍👧 Family History Score", "How strongly diabetes runs in your family. Higher means more relatives affected."),
        ("📏 Skin Fold Thickness", "A way to estimate body fat by measuring a pinch of skin at the tricep."),
    ]

    for title, desc in explanations:
        st.markdown(f"""
            <div class='tip-box'>
                <b>{title}</b><br>
                <span style='color:#374151;'>{desc}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div class='disclaimer'>
            ⚠️ <b>Medical Disclaimer:</b> This application is intended for <b>educational and
            screening purposes only</b>. It does not provide medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or other qualified health provider with any
            questions you may have regarding a medical condition.
        </div>
    """, unsafe_allow_html=True)