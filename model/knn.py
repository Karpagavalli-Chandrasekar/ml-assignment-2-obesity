"""
KNN classifier for Obesity classification (binary target: Not Obese vs Obese).

KNN is distance-based, so feature scaling is essential.
This script trains a scaled KNN model and prints a detailed evaluation summary.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

CSV_FILE = "ObesityDataSet.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50
N_NEIGHBORS = 7

TARGET_MAP = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 1,
    "Obesity_Type_I": 1,
    "Obesity_Type_II": 1,
    "Obesity_Type_III": 1,
}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) after binary mapping + encoding."""
    df = df.copy()

    df["Obese"] = df["NObeyesdad"].map(TARGET_MAP).fillna(0).astype(int)
    y = df["Obese"]

    X = df.drop(columns=["NObeyesdad", "Obese"])

    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    for col in binary_cols:
        if col in X.columns:
            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
                .fillna(0)
                .astype(int)
            )

    multi_cols = ["Gender", "CAEC", "CALC", "MTRANS"]
    multi_cols = [c for c in multi_cols if c in X.columns]
    X = pd.get_dummies(X, columns=multi_cols, drop_first=True)

    return X.astype(float), y


def evaluate_binary(y_true: pd.Series, y_prob: np.ndarray, threshold: float = THRESHOLD) -> None:
    """Print a compact evaluation summary using a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===================== KNN RESULTS =====================")
    print(f"n_neighbors : {N_NEIGHBORS}")
    print(f"Threshold   : {threshold:.2f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"AUC         : {auc:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print("------------------------------------------------------")
    print("Confusion Matrix:")
    print(cm)
    print("------------------------------------------------------")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("======================================================\n")


def main() -> None:
    np.random.seed(RANDOM_STATE)

    df = pd.read_csv(CSV_FILE)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
    ])

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluate_binary(y_test, y_prob, threshold=THRESHOLD)


if __name__ == "__main__":
    main()
