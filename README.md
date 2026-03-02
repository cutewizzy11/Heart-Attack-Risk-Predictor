# Heart Attack Risk Predictor (HARPA)

Predicts the probability of heart disease from routine clinical + ECG-derived features using a trained ML model.

## Live demo

https://harpredictor.streamlit.app/

## Local run

```bash
pip install -r requirements.txt
python train_model.py          # optional – app auto-trains if model.joblib is missing
streamlit run app.py
```

## Batch scoring

The app accepts CSV uploads in two formats:

- **Encoded CSV**: same columns as `Datasets/Heart Attack/heart_processed.csv` (excluding `HeartDisease`)
- **Raw CSV**: columns `Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope`

## Coder Workspace Template

This repo also includes a Terraform template (`main.tf`) that integrates **Coder's product suite**.

### What it does

- Uses `coder_parameter` to render an **interactive workspace creation form** in the Coder dashboard where users enter patient clinical values (age, blood pressure, chest pain type, etc.).
- On workspace provision the agent **clones this repo, trains the model, runs a one-shot prediction** from the form values, and **launches the full Streamlit app** inside the workspace.
- A `coder_app` resource exposes the Streamlit UI directly in the Coder dashboard.

### How to use

1. Import `main.tf` as a template in your Coder instance.
2. Create a workspace — fill in the patient data form.
3. The workspace provisions a Docker container, trains the model, prints the prediction result, and starts Streamlit on port 8501.

### Key files

| File | Purpose |
|---|---|
| `main.tf` | Coder workspace template (Terraform) |
| `predict_from_env.py` | Reads patient values from env vars, runs prediction |
| `app.py` | Streamlit web app (single + batch scoring) |
| `model_utils.py` | Model training, saving, loading utilities |
| `train_model.py` | CLI script to train and export the best model |

## Notes

- `model.joblib` is gitignored; the app and workspace both auto-train on first run.
- Dataset: `Datasets/Heart Attack/heart_processed.csv`
