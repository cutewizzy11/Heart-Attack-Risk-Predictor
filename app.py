from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import ensure_model_artifact


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"


def encode_raw_batch(raw_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0, index=raw_df.index, columns=feature_names)

    out["Age"] = pd.to_numeric(raw_df["Age"], errors="raise")
    out["RestingBP"] = pd.to_numeric(raw_df["RestingBP"], errors="raise")
    out["Cholesterol"] = pd.to_numeric(raw_df["Cholesterol"], errors="raise")
    out["FastingBS"] = pd.to_numeric(raw_df["FastingBS"], errors="raise").astype(int)
    out["MaxHR"] = pd.to_numeric(raw_df["MaxHR"], errors="raise")
    out["Oldpeak"] = pd.to_numeric(raw_df["Oldpeak"], errors="raise")

    out["Sex_M"] = (raw_df["Sex"].astype(str).str.lower() == "male").astype(int)

    cp = raw_df["ChestPainType"].astype(str).str.upper()
    if "ChestPainType_ATA" in out.columns:
        out["ChestPainType_ATA"] = (cp == "ATA").astype(int)
    if "ChestPainType_NAP" in out.columns:
        out["ChestPainType_NAP"] = (cp == "NAP").astype(int)
    if "ChestPainType_TA" in out.columns:
        out["ChestPainType_TA"] = (cp == "TA").astype(int)

    ecg = raw_df["RestingECG"].astype(str)
    if "RestingECG_Normal" in out.columns:
        out["RestingECG_Normal"] = (ecg.str.lower() == "normal").astype(int)
    if "RestingECG_ST" in out.columns:
        out["RestingECG_ST"] = (ecg.str.upper() == "ST").astype(int)

    ang = raw_df["ExerciseAngina"].astype(str).str.lower()
    out["ExerciseAngina_Y"] = (ang.isin(["y", "yes", "true", "1"])).astype(int)

    slope = raw_df["ST_Slope"].astype(str).str.lower()
    if "ST_Slope_Flat" in out.columns:
        out["ST_Slope_Flat"] = (slope == "flat").astype(int)
    if "ST_Slope_Up" in out.columns:
        out["ST_Slope_Up"] = (slope == "up").astype(int)

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="raise")

    return out


def build_feature_row(feature_names: list[str], raw: dict) -> pd.DataFrame:
    row = {k: 0 for k in feature_names}

    row["Age"] = raw["Age"]
    row["RestingBP"] = raw["RestingBP"]
    row["Cholesterol"] = raw["Cholesterol"]
    row["FastingBS"] = int(raw["FastingBS"])
    row["MaxHR"] = raw["MaxHR"]
    row["Oldpeak"] = raw["Oldpeak"]

    row["Sex_M"] = 1 if raw["Sex"] == "Male" else 0

    # ChestPainType baseline is ASY (all zeros)
    if raw["ChestPainType"] == "ATA":
        row["ChestPainType_ATA"] = 1
    elif raw["ChestPainType"] == "NAP":
        row["ChestPainType_NAP"] = 1
    elif raw["ChestPainType"] == "TA":
        row["ChestPainType_TA"] = 1

    # RestingECG baseline is LVH (all zeros)
    if raw["RestingECG"] == "Normal":
        row["RestingECG_Normal"] = 1
    elif raw["RestingECG"] == "ST":
        row["RestingECG_ST"] = 1

    row["ExerciseAngina_Y"] = 1 if raw["ExerciseAngina"] == "Yes" else 0

    # ST_Slope baseline is Down (all zeros)
    if raw["ST_Slope"] == "Flat":
        row["ST_Slope_Flat"] = 1
    elif raw["ST_Slope"] == "Up":
        row["ST_Slope_Up"] = 1

    df = pd.DataFrame([row], columns=feature_names)

    # Ensure numeric dtypes
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    return df


st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

st.title("Heart Attack Risk Predictor")
st.caption("Hackathon demo app – risk prediction from routine clinical + ECG-derived features")


@st.cache_resource(show_spinner=True)
def _get_artifact_and_results():
    return ensure_model_artifact(MODEL_PATH)


artifact, training_results = _get_artifact_and_results()
model_name = str(artifact.get("model_name", "(unknown)"))
feature_names = list(artifact["feature_names"])
model = artifact["model"]

with st.sidebar:
    st.subheader("Model")
    st.write(model_name)
    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    if training_results is not None:
        st.write("Model trained on first run")
        st.dataframe(training_results, use_container_width=True)

st.subheader("About")
st.write(
    "This tool estimates the probability of heart disease based on a tabular model trained on an encoded dataset. "
    "It is intended for hackathon/demo purposes."
)

with st.expander("Disclaimer"):
    st.write(
        "This is not a medical device and must not be used for diagnosis or clinical decision-making. "
        "Predictions can be wrong; consult a qualified clinician."
    )

st.subheader("Patient inputs")

presets = {
    "Custom": None,
    "Low risk example": {
        "Age": 40,
        "RestingBP": 120,
        "Cholesterol": 180,
        "FastingBS": 0,
        "MaxHR": 170,
        "Oldpeak": 0.0,
        "Sex": "Female",
        "ChestPainType": "ATA",
        "RestingECG": "Normal",
        "ExerciseAngina": "No",
        "ST_Slope": "Up",
    },
    "High risk example": {
        "Age": 62,
        "RestingBP": 160,
        "Cholesterol": 310,
        "FastingBS": 1,
        "MaxHR": 110,
        "Oldpeak": 2.5,
        "Sex": "Male",
        "ChestPainType": "ASY",
        "RestingECG": "ST",
        "ExerciseAngina": "Yes",
        "ST_Slope": "Flat",
    },
}

preset_name = st.selectbox("Demo preset", options=list(presets.keys()), index=0)
apply_preset = st.button("Apply preset")
if apply_preset and presets[preset_name] is not None:
    p = presets[preset_name]
    st.session_state["age"] = p["Age"]
    st.session_state["resting_bp"] = p["RestingBP"]
    st.session_state["cholesterol"] = p["Cholesterol"]
    st.session_state["fasting_bs"] = p["FastingBS"]
    st.session_state["max_hr"] = p["MaxHR"]
    st.session_state["oldpeak"] = p["Oldpeak"]
    st.session_state["sex"] = p["Sex"]
    st.session_state["chest_pain"] = p["ChestPainType"]
    st.session_state["resting_ecg"] = p["RestingECG"]
    st.session_state["exercise_angina"] = p["ExerciseAngina"]
    st.session_state["st_slope"] = p["ST_Slope"]

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1, key="age")
    resting_bp = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=130, step=1, key="resting_bp")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=800, value=200, step=1, key="cholesterol")
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL?", options=[0, 1], index=0, key="fasting_bs")

with col2:
    max_hr = st.number_input("Max HR", min_value=40, max_value=250, value=150, step=1, key="max_hr")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="oldpeak")
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0, key="sex")

chest_pain = st.selectbox(
    "Chest Pain Type",
    options=["ASY", "ATA", "NAP", "TA"],
    index=0,
    key="chest_pain",
)

resting_ecg = st.selectbox(
    "Resting ECG",
    options=["LVH", "Normal", "ST"],
    index=0,
    key="resting_ecg",
)

exercise_angina = st.selectbox("Exercise Angina", options=["No", "Yes"], index=0, key="exercise_angina")

st_slope = st.selectbox(
    "ST Slope",
    options=["Down", "Flat", "Up"],
    index=1,
    key="st_slope",
)

raw = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingECG": resting_ecg,
    "ExerciseAngina": exercise_angina,
    "ST_Slope": st_slope,
}

x_row = build_feature_row(feature_names, raw)

st.divider()

predict_clicked = st.button("Predict risk", type="primary")

if predict_clicked:
    proba = float(model.predict_proba(x_row)[:, 1][0])
    pred = 1 if proba >= threshold else 0

    st.subheader("Result")
    st.metric("Predicted risk (probability)", f"{proba:.3f}")

    if pred == 1:
        st.error(f"High risk (>= {threshold:.2f})")
    else:
        st.success(f"Low risk (< {threshold:.2f})")

    with st.expander("Show model input row"):
        st.dataframe(x_row)

st.divider()

st.subheader("Batch scoring (optional)")
st.write(
    "Upload either: (1) an encoded CSV with the same columns as heart_processed.csv (except HeartDisease), "
    "or (2) a raw CSV with columns: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope."
)

uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    if "Sex" in batch.columns and "ChestPainType" in batch.columns and "ST_Slope" in batch.columns:
        batch_enc = encode_raw_batch(batch, feature_names)
    else:
        missing = [c for c in feature_names if c not in batch.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()

        batch_enc = batch[feature_names].copy()
        for c in batch_enc.columns:
            if batch_enc[c].dtype == "bool":
                batch_enc[c] = batch_enc[c].astype(int)

    probas = model.predict_proba(batch_enc)[:, 1]
    out = batch_enc.copy()
    out["risk_proba"] = probas
    out["risk_label"] = (out["risk_proba"] >= threshold).astype(int)

    st.write("Scored rows")
    st.dataframe(out.head(50))
    st.download_button(
        "Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scored_heart_risk.csv",
        mime="text/csv",
    )

