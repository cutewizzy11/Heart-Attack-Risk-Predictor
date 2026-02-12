from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "xgboost is required. Install dependencies from requirements.txt"
    ) from e


RANDOM_STATE = 42


def find_data_path() -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd / "b2b" / "Datasets" / "Heart Attack" / "heart_processed.csv",
        cwd / "Datasets" / "Heart Attack" / "heart_processed.csv",
    ]
    for p in [cwd] + list(cwd.parents):
        candidates.append(p / "b2b" / "Datasets" / "Heart Attack" / "heart_processed.csv")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not find heart_processed.csv. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def build_models(random_state: int = RANDOM_STATE):
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    return {
        "LogReg (scaled)": logreg,
        "RandomForest": rf,
        "XGBoost": xgb,
    }


def main() -> None:
    data_path = find_data_path()
    df = pd.read_csv(data_path)

    if "HeartDisease" not in df.columns:
        raise ValueError("Expected target column 'HeartDisease'")

    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
        "accuracy": "accuracy",
        "f1": "f1",
    }

    rows = []
    for name, m in models.items():
        scores = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "cv_pr_auc_mean": float(np.mean(scores["test_avg_precision"])),
                "cv_f1_mean": float(np.mean(scores["test_f1"])),
                "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    best_name = str(results.iloc[0]["model"])
    best_model = models[best_name]

    best_model.fit(X_train, y_train)

    model_path = Path(__file__).resolve().parent / "model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "feature_names": X.columns.tolist(),
            "model": best_model,
        },
        model_path,
    )

    print("Saved:", model_path)
    print("Best model:", best_name)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
