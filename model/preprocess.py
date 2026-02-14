from __future__ import annotations
import pandas as pd

TARGET_MAP = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 1,
    "Obesity_Type_I": 1,
    "Obesity_Type_II": 1,
    "Obesity_Type_III": 1,
}

TARGET_COL = "NObeyesdad"

def prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Shared preprocessing used by:
    - training scripts
    - Streamlit app
    - all model modules

    Output:
    - X: numeric dataframe with one-hot encoding applied
    - y: binary series (0 = Not Obese, 1 = Obese)
    """
    df2 = df.copy()

    df2["Obese"] = df2[TARGET_COL].map(TARGET_MAP).fillna(0).astype(int)
    y = df2["Obese"]

    X = df2.drop(columns=[TARGET_COL, "Obese"])

    # yes/no -> 1/0
    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    for col in binary_cols:
        if col in X.columns:
            X[col] = (
                X[col].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
                .fillna(0).astype(int)
            )

    # One-hot for categoricals
    multi_cols = ["Gender", "CAEC", "CALC", "MTRANS"]
    multi_cols = [c for c in multi_cols if c in X.columns]
    X = pd.get_dummies(X, columns=multi_cols, drop_first=True)

    return X.astype(float), y
