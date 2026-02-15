"""
Random Forest for Obesity classification (binary target: Not Obese vs Obese).

Reusable module:
- build_model()
- train_and_save()

Cloud-safe note:
- Use n_jobs=1 to avoid over-using Streamlit Cloud CPU.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

from model.preprocess import prepare_X_y
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "ObesityDataSet.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_model(
    random_state: int = RANDOM_STATE,
    n_estimators: int = 200,   # reduced from 300 (lighter, still strong)
    max_depth: int = 10,
    min_samples_leaf: int = 15,
    n_jobs: int = 1,           # IMPORTANT for Streamlit Cloud
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_and_save(
    df: pd.DataFrame,
    out_path: str = "models/random_forest.pkl",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_leaf: int = 15,
    verbose: bool = True,
) -> str:
    """
    Train RandomForest on df and save it as a .pkl.
    Returns the output path.
    """
    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # quick sanity metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(model, out_path)

    if verbose:
        print(f"Saved Random Forest model to: {out_path}")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")
        print(f"Params: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, n_jobs=1")

    return out_path


def main():
    df = pd.read_csv(CSV_FILE)

    out_path = os.path.join(BASE_DIR, "models", "model_name.pkl")
    train_and_save(df, out_path=out_path)


if __name__ == "__main__":
    main()
