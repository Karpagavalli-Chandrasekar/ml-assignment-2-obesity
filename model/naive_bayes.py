"""
Gaussian Naive Bayes for Obesity classification (binary target: Not Obese vs Obese).

Reusable module:
- build_model()
- train_and_save()
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

from preprocess import prepare_X_y

CSV_FILE = "ObesityDataSet.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_model() -> GaussianNB:
    """Return a GaussianNB model."""
    return GaussianNB()


def train_and_save(
    df: pd.DataFrame,
    out_path: str = "models/naive_bayes.pkl",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> str:
    """
    Train GaussianNB on df and save it as a .pkl.
    Returns the output path.
    """
    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model()
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
        print(f"Saved Naive Bayes model to: {out_path}")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

    return out_path


def main() -> None:
    """Optional: run as standalone script (local training only)."""
    np.random.seed(RANDOM_STATE)
    df = pd.read_csv(CSV_FILE)
    train_and_save(df, out_path="models/naive_bayes.pkl", verbose=True)


if __name__ == "__main__":
    main()
