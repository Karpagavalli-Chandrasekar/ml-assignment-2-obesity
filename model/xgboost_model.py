"""
XGBoost classifier for Obesity classification (binary target: Not Obese vs Obese).

Reusable module:
- build_model()
- train_and_save()

Cloud-safe notes:
- n_jobs=1 (avoid CPU spikes)
- fewer estimators (lighter)
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

from model.preprocess import prepare_X_y

# Optional import: keep project working even if xgboost is not installed
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    XGBClassifier = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "ObesityDataSet.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_model(
    random_state: int = RANDOM_STATE,
    n_estimators: int = 150,   # reduced from 300 (lighter)
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    n_jobs: int = 1,           # IMPORTANT for Streamlit Cloud
):
    """Return an XGBClassifier, or None if xgboost not available."""
    if not XGB_AVAILABLE:
        return None

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_and_save(
    df: pd.DataFrame,
    out_path: str = "models/xgboost.pkl",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    verbose: bool = True,
) -> str:
    """
    Train XGBoost on df and save it as a .pkl.
    Returns the output path.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Add xgboost to requirements.txt and reinstall.")

    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=1,
    )
    assert model is not None

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
        print(f"Saved XGBoost model to: {out_path}")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")
        print(f"Params: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, n_jobs=1")

    return out_path


def main() -> None:
    """Optional: run as standalone script (local training only)."""
    np.random.seed(RANDOM_STATE)
    if not XGB_AVAILABLE:
        print("XGBoost not available. Install with: pip install xgboost")
        return

    df = pd.read_csv(CSV_FILE)
    out_path = os.path.join(BASE_DIR, "models", "xgboost.pkl")
    train_and_save(df, out_path=out_path, verbose=True)

if __name__ == "__main__":
    main()
