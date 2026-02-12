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
    raise RuntimeError("xgboost is required. Install dependencies from requirements.txt") from e


RANDOM_STATE = 42


def find_data_path() -> Path:
    app_dir = Path(__file__).resolve().parent

    candidates = [
        app_dir / "Datasets" / "Heart Attack" / "heart_processed.csv",
        app_dir / "b2b" / "Datasets" / "Heart Attack" / "heart_processed.csv",
    ]

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "b2b" / "Datasets" / "Heart Attack" / "heart_processed.csv",
            cwd / "Datasets" / "Heart Attack" / "heart_processed.csv",
        ]
    )

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


def train_best_model(df: pd.DataFrame):
    if "HeartDisease" not in df.columns:
        raise ValueError("Expected target column 'HeartDisease'")

    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)

    X_train, _, y_train, _ = train_test_split(
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

    return best_name, best_model, X.columns.tolist(), results


def save_artifact(model_path: Path, model_name: str, model, feature_names: list[str]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name,
            "feature_names": feature_names,
            "model": model,
        },
        model_path,
    )


def load_artifact(model_path: Path):
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Invalid model artifact format")
    if "model" not in artifact or "feature_names" not in artifact:
        raise ValueError("Model artifact is missing required keys")
    return artifact


def ensure_model_artifact(model_path: Path):
    if model_path.exists():
        return load_artifact(model_path), None

    data_path = find_data_path()
    df = pd.read_csv(data_path)
    model_name, model, feature_names, results = train_best_model(df)
    save_artifact(model_path, model_name, model, feature_names)

    return load_artifact(model_path), results
