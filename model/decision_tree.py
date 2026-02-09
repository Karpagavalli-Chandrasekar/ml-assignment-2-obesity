import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

np.random.seed(42)

CSV_FILE = "ObesityDataSet.csv"

def print_model_performance(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n======== MODEL PERFORMANCE ========")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"AUC         : {auc:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("===================================")

def main():
    data = pd.read_csv(CSV_FILE).dropna()

    target_map = {
        "Insufficient_Weight": 0,
        "Normal_Weight": 0,
        "Overweight_Level_I": 1,
        "Overweight_Level_II": 1,
        "Obesity_Type_I": 1,
        "Obesity_Type_II": 1,
        "Obesity_Type_III": 1
    }

    data["Obese"] = data["NObeyesdad"].map(target_map)
    y = data["Obese"].astype(int)

    X = data.drop(columns=["NObeyesdad", "Obese"]).copy()

    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    for col in binary_cols:
        X[col] = X[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    multi_cols = ["Gender", "CAEC", "CALC", "MTRANS"]
    X = pd.get_dummies(X, columns=multi_cols, drop_first=True)
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nDECISION TREE METRICS")
    print_model_performance(y_test, y_prob)

if __name__ == "__main__":
    main()
