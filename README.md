# Heart Attack Risk Predictor

Streamlit web app that predicts the probability of heart disease using a model trained on `heart_processed.csv`.

## Local run

From the `b2b/` folder:

```bash
python -m pip install -r requirements.txt
python train_model.py  # optional (app can auto-train if model.joblib is missing)
streamlit run app.py
```

## Batch scoring

The app supports uploading:

- **Encoded CSV**: same columns as `Datasets/Heart Attack/heart_processed.csv` (excluding `HeartDisease`)
- **Raw CSV**: columns
  `Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope`


### Notes

- `model.joblib` is intentionally gitignored. On Streamlit Cloud the app will auto-train on first run (cached).
- If you prefer deterministic deploys, commit `model.joblib` and remove it from `.gitignore`.
