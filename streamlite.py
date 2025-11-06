# streamlit_app.py
# Clean, robust Streamlit app for Mood Reboot — Digital Detox Impact Analyzer

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mood Reboot — Digital Detox", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------------
# Paths (adjust only if your repo structure differs)
# ---------------------------
MODELS_DIR = "models"
DATA_DIR = "data"

# fallback if running from repo root directly
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(os.getcwd(), "models")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(os.getcwd(), "data")

# ---------------------------
# Load models & artifacts (safe load with friendly errors)
# ---------------------------
lr = None
scaler_reg = None
try:
    lr = joblib.load(os.path.join(MODELS_DIR, "linear_mood_change_model.joblib"))
    scaler_reg = joblib.load(os.path.join(MODELS_DIR, "scaler_reg.joblib"))
except Exception as e:
    st.error(f"Could not load regression model/scaler from '{MODELS_DIR}'.\nError: {e}")
    st.stop()

# Classification artifacts (optional)
clf = None
scaler_clf = None
le = None
if os.path.exists(os.path.join(MODELS_DIR, "logistic_difficulty_model.joblib")):
    try:
        clf = joblib.load(os.path.join(MODELS_DIR, "logistic_difficulty_model.joblib"))
        scaler_clf = joblib.load(os.path.join(MODELS_DIR, "scaler_clf.joblib"))
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder_difficulty.joblib"))
    except Exception as e:
        st.warning("Classification artifacts found but couldn't be loaded. Classification will be disabled.")
        st.write(e)
        clf = scaler_clf = le = None

# ---------------------------
# Page header
# ---------------------------
st.title("💆 Mood Reboot — Digital Detox Impact Analyzer")
st.markdown(
    "Interactive demo: adjust sliders on the left to predict **Mood Improvement** and "
    "**Detox Difficulty**. Models are loaded from `models/` and dataset (optional) from `data/`."
)

# ---------------------------
# Sidebar inputs (define these FIRST)
# ---------------------------
with st.sidebar:
    st.header("Your Inputs")
    duration = st.selectbox("Detox Duration (hrs)", options=[24, 48], index=0)
    baseline_mood = st.slider("Baseline Mood (0–10)", 0, 10, 5)
    baseline_stress = st.slider("Baseline Stress (%)", 0, 100, 60)
    baseline_sleep = st.slider("Baseline Sleep Quality (0–10)", 0, 10, 6)
    baseline_focus = st.slider("Baseline Focus (0–10)", 0, 10, 5)
    screen_time = st.slider("Screen Time (hrs/day)", 0, 24, 6)
    sleep_hours = st.slider("Average Sleep Hours (per night)", 0, 12, 7)

    st.markdown("---")
    st.markdown("**Notes:** These sliders match your dataset columns. If your trained model "
                "expects different features, the app will show the expected order below.")

# ---------------------------
# Build slider -> value map
# ---------------------------
slider_map = {
    "Detox Duration": int(duration),
    "Baseline Mood": int(baseline_mood),
    "Baseline Stress": int(baseline_stress),
    "Baseline Sleep": int(baseline_sleep),
    "Baseline Focus": int(baseline_focus),
    "Screen Time": int(screen_time),
    "Sleep Hours": int(sleep_hours)
}

# ---------------------------
# Determine expected feature order (from scaler/model if possible)
# ---------------------------
expected_features = None
# Prefer scaler.feature_names_in_ (scaler was fit with column names)
if scaler_reg is not None and hasattr(scaler_reg, "feature_names_in_"):
    expected_features = list(scaler_reg.feature_names_in_)
# Fallback to linear model feature names
elif lr is not None and hasattr(lr, "feature_names_in_"):
    expected_features = list(lr.feature_names_in_)

# If still None, choose a reasonable default based on your CSV (keeps same names)
if expected_features is None:
    expected_features = [
        "Detox Duration",
        "Baseline Mood",
        "Baseline Stress",
        "Baseline Sleep",
        "Baseline Focus",
        "Screen Time",
        "Sleep Hours"
    ]

# Show expected feature order for debugging / clarity
st.info(f"Model expects features (in order): {expected_features}")

# ---------------------------
# Build input row in exact expected order; fill missing with 0
# ---------------------------
input_values = []
missing = []
for feat in expected_features:
    if feat in slider_map:
        input_values.append(slider_map[feat])
    else:
        # Not provided by slider -> set safe default 0
        input_values.append(0)
        missing.append(feat)

if missing:
    st.warning(f"The following expected features were not provided by sliders and were set to 0: {missing}")

X_reg = pd.DataFrame([input_values], columns=expected_features)

# ---------------------------
# Make predictions (regression + optional classification)
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    try:
        X_reg_scaled = scaler_reg.transform(X_reg)
        pred_mood_change = float(lr.predict(X_reg_scaled)[0])
        st.metric(label="Predicted Mood Improvement (0–10)", value=f"{pred_mood_change:.2f}")
        st.write("Predicted Post Mood (Baseline + Improvement):", round(min(10, baseline_mood + pred_mood_change), 2))
    except Exception as e:
        st.error("Prediction failed — feature names/order mismatch. See logs. Error: " + str(e))

with col2:
    if clf is not None and scaler_clf is not None and le is not None:
        try:
            # Try to use classifier's expected features if available
            clf_expected = None
            if hasattr(scaler_clf, "feature_names_in_"):
                clf_expected = list(scaler_clf.feature_names_in_)
            elif hasattr(clf, "feature_names_in_"):
                clf_expected = list(clf.feature_names_in_)
            else:
                clf_expected = expected_features

            # Build classifier input row in its expected order
            X_clf = pd.DataFrame([{f: slider_map.get(f, 0) for f in clf_expected}])
            X_clf_scaled = scaler_clf.transform(X_clf)
            pred_class = clf.predict(X_clf_scaled)[0]
            pred_label = le.inverse_transform([pred_class])[0]
            st.metric(label="Predicted Detox Difficulty", value=pred_label)
        except Exception as e:
            st.warning("Classification failed (feature mismatch). Error: " + str(e))
    else:
        st.info("Classification model not available or artifacts missing.")

# ---------------------------
# Bottom: Data & Visuals
# ---------------------------
st.markdown("---")
st.subheader("Dataset Summary & Visuals")

# Try to load dataset if exists (using names you used earlier)
csv_candidates = [
    os.path.join(DATA_DIR, "Digital_Detoxx_Final_RealBalanced.csv"),
    os.path.join(DATA_DIR, "Digital_Detoxx_Final_200.csv"),
    os.path.join(DATA_DIR, "Digital_Detoxx_Realistic_Accuracy.csv"),
    os.path.join(DATA_DIR, "Digital_Detoxx_Final.csv"),
    os.path.join(DATA_DIR, "Digital detox 3.csv"),  # your CSV exact name included as fallback
]

df = None
for p in csv_candidates:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            csv_path = p
            break
        except Exception as e:
            st.warning(f"Found file at {p} but failed to read: {e}")

if df is None:
    st.warning("Dataset not found in data/ folder. Place your CSV (final dataset) inside 'data/' for dataset visuals.")
else:
    st.write(f"Loaded dataset: `{os.path.basename(csv_path)}` — shape: {df.shape}")
    st.markdown("**Head of dataset:**")
    st.dataframe(df.head())

    with st.expander("Show summary statistics"):
        st.dataframe(df.describe(include="all").T)

    # Make charts safe: only use columns that exist
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if "Difficulty Level" in df.columns and "Stress Reduction" in df.columns:
        sns.boxplot(x=df["Difficulty Level"], y=df["Stress Reduction"], ax=ax[0])
        ax[0].set_title("Stress Reduction by Difficulty Level")
    else:
        ax[0].text(0.5, 0.5, "Required columns for this plot not found", ha="center")
        ax[0].set_axis_off()

    if "Mood Change" in df.columns:
        sns.histplot(df["Mood Change"], bins=12, ax=ax[1])
        ax[1].set_title("Distribution of Mood Change")
    else:
        ax[1].text(0.5, 0.5, "Mood Change column not found", ha="center")
        ax[1].set_axis_off()

    st.pyplot(fig)

    # Correlation heatmap (numeric only)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 3:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2)
        st.pyplot(fig2)

# ---------------------------
# Footer: How to use / Troubleshooting
# ---------------------------
st.markdown("---")
st.markdown(
    """
**How to use this app in your repo**

1. Put your models in the repo root `models/` folder:
   - `linear_mood_change_model.joblib`
   - `scaler_reg.joblib`
   - `logistic_difficulty_model.joblib` (optional)
   - `scaler_clf.joblib` (optional)
   - `label_encoder_difficulty.joblib` (optional)

2. Put your final CSV in `data/` (recommended filename: `Digital_Detoxx_Final_RealBalanced.csv` or `Digital detox 3.csv`).

3. From repo root run:
   `streamlit run dashboard/streamlit_app.py`

If the app cannot find models or data, check file paths and ensure the files are in the correct folders.
"""
)
