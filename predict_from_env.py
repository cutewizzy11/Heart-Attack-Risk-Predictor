#!/usr/bin/env python3

import os
import joblib
import pandas as pd

def load_artifact(model_path="model.joblib"):
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["feature_names"]

def build_feature_row(feature_names, raw):
    row = {k: 0 for k in feature_names}

    row["Age"] = float(os.getenv("AGE", 50))
    row["RestingBP"] = float(os.getenv("RESTING_BP", 130))
    row["Cholesterol"] = float(os.getenv("CHOLESTEROL", 200))
    row["FastingBS"] = int(os.getenv("FASTING_BS", 0))
    row["MaxHR"] = float(os.getenv("MAX_HR", 150))
    row["Oldpeak"] = float(os.getenv("OLDPEAK", 1.0))

    row["Sex_M"] = 1 if os.getenv("SEX", "Female").lower() == "male" else 0

    cp = os.getenv("CHEST_PAIN_TYPE", "ASY").upper()
    if "ChestPainType_ATA" in row:
        row["ChestPainType_ATA"] = 1 if cp == "ATA" else 0
    if "ChestPainType_NAP" in row:
        row["ChestPainType_NAP"] = 1 if cp == "NAP" else 0
    if "ChestPainType_TA" in row:
        row["ChestPainType_TA"] = 1 if cp == "TA" else 0

    ecg = os.getenv("RESTING_ECG", "Normal")
    if "RestingECG_Normal" in row:
        row["RestingECG_Normal"] = 1 if ecg.lower() == "normal" else 0
    if "RestingECG_ST" in row:
        row["RestingECG_ST"] = 1 if ecg.upper() == "ST" else 0

    ang = os.getenv("EXERCISE_ANGINA", "No").lower()
    row["ExerciseAngina_Y"] = 1 if ang in ["y", "yes", "true", "1"] else 0

    slope = os.getenv("ST_SLOPE", "Flat").lower()
    if "ST_Slope_Flat" in row:
        row["ST_Slope_Flat"] = 1 if slope == "flat" else 0
    if "ST_Slope_Up" in row:
        row["ST_Slope_Up"] = 1 if slope == "up" else 0

    df = pd.DataFrame([row], columns=feature_names)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df

def main():
    model, feature_names = load_artifact()
    raw = {}  # Not needed since we use env vars
    x_row = build_feature_row(feature_names, raw)
    proba = float(model.predict_proba(x_row)[:, 1][0])
    threshold = float(os.getenv("THRESHOLD", 0.5))
    pred = 1 if proba >= threshold else 0

    print("Heart Attack Risk Prediction")
    print("=" * 30)
    print(f"Predicted risk probability: {proba:.3f}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Classification: {'High risk' if pred == 1 else 'Low risk'}")
    print("\nInput features:")
    for k, v in x_row.iloc[0].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
