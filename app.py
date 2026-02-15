import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
)

# Use the shared preprocessing (same as training)
from model.preprocess import prepare_X_y


# ---------------- Page Config ----------------
st.set_page_config(page_title="Obesity ML Assignment", layout="wide")


# ---------------- Theme / Styling ----------------
def apply_theme():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #40826D 0%, #2F6F5F 100%); }

        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp label, .stMarkdown, .stText, .stCaption { color: #ffffff !important; }

        section[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.65) !important; }
        section[data-testid="stSidebar"] * { color: #ffffff !important; }

        header[data-testid="stHeader"] { background: transparent !important; }
        header[data-testid="stHeader"] > div { background: transparent !important; }
        div[data-testid="stToolbar"] { background: transparent !important; }

        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] button,
        div[data-testid="stToolbar"] svg,
        div[data-testid="stToolbar"] button {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .app-title { color: #ffffff !important; font-weight: 800; margin-bottom: 0.6rem; }

        .card {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 15px;
            padding: 15px;
            margin: 12px 0;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .muted { color: rgba(255,255,255,0.85) !important; font-size: 0.92rem; margin: 0; }

        .card-blue   { border-left: 6px solid rgba(37, 99, 235, 0.85); }
        .card-green  { border-left: 6px solid rgba(16, 185, 129, 0.85); }
        .card-purple { border-left: 6px solid rgba(124, 58, 237, 0.85); }
        .card-amber  { border-left: 6px solid rgba(245, 158, 11, 0.85); }
        .card-rose   { border-left: 6px solid rgba(244, 63, 94, 0.85); }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.60);
            border-radius: 12px;
            padding: 10px;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: rgba(15, 23, 42, 0.70) !important;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
        }

        div[role="listbox"] {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #ffffff !important;
        }
        div[role="option"] { color: #ffffff !important; }

        .stButton button {
            background-color: rgba(255,255,255,0.15) !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            color: #ffffff !important;
        }

        div[data-testid="stDownloadButton"] button {
            background-color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-radius: 10px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 700 !important;
            color: #000000 !important;
        }
        div[data-testid="stDownloadButton"] button * {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stDownloadButton"] button svg { fill: #000000 !important; }

        .stTable, .stDataFrame { background: rgba(15, 23, 42, 0.55) !important; }

/* ===== File Uploader Styling ===== */

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Browse files button */
div[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
}

/* Text inside button */
div[data-testid="stFileUploader"] button * {
    color: #000000 !important;
}

        </style>
        """,
        unsafe_allow_html=True
    )


def card(title, subtitle, color_class="card-blue"):
    st.markdown(
        f"<div class='card {color_class}'><h3>{title}</h3><p class='muted'>{subtitle}</p></div>",
        unsafe_allow_html=True
    )


apply_theme()


# ---------------- Title ----------------
st.markdown(
    "<h1 class='app-title'>Obesity Dataset – (Interactive Model Comparison)</h1>",
    unsafe_allow_html=True
)


# ---------------- Sidebar ----------------
st.sidebar.markdown("## Controls")
st.sidebar.caption("Pick a saved model, set split/threshold, and run evaluation.")

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes (GaussianNB)",
        "Random Forest",
        "XGBoost",
    ]
)

st.sidebar.header("Train/Test")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.25, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("Prediction")
threshold = st.sidebar.slider("Probability threshold", 0.10, 0.90, 0.50, 0.05)

st.sidebar.header("Display")

# Light version: comparison uses SAVED models (no training)
show_compare = st.sidebar.checkbox("Generate comparison table (fast - uses saved models)", value=True)

run_btn = st.sidebar.button("🚀 Run / Re-run")

st.sidebar.header("Test CSV Upload")
uploaded_test_file = st.sidebar.file_uploader(
    "⬆️ Upload Test CSV (features only – no NObeyesdad)",
    type=["csv"]
)


# ---------------- Paths ----------------
# Obesity_Streamlit_App/model/
# models are saved inside: Obesity_Streamlit_App/model/models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "model", "ObesityDataSet.csv")

MODEL_PATHS = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "models", "logistic_regression.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "model", "models", "decision_tree.pkl"),
    "KNN": os.path.join(BASE_DIR, "model", "models", "knn.pkl"),
    "Naive Bayes (GaussianNB)": os.path.join(BASE_DIR, "model", "models", "naive_bayes.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "model", "models", "random_forest.pkl"),
    "XGBoost": os.path.join(BASE_DIR, "model", "models", "xgboost.pkl"),
}


# ---------------- Load data ----------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


df = load_data(CSV_PATH)
X, y = prepare_X_y(df)

# ---------------- Test CSV Download (Without Target) ----------------
# Create test split based on sidebar settings
_, X_test_tmp, _, y_test_tmp = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y
)

# Recover original raw rows using index
test_raw = df.loc[X_test_tmp.index].copy()

# Remove target column
test_features_only = test_raw.drop(columns=["NObeyesdad"], errors="ignore")

# Convert to CSV bytes
test_csv_bytes = test_features_only.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download Test CSV (features only)",
    data=test_csv_bytes,
    file_name="Obesity_Test_features_only.csv",
    mime="text/csv"
)


# ---------------- Dataset Preview (Always Visible) ----------------
card("Dataset Preview", "First few rows of the dataset.", "card-blue")

st.dataframe(df.head(10), width="stretch")

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Full Dataset (CSV)",
    data=csv_data,
    file_name="ObesityDataSet.csv",
    mime="text/csv"
)



# ---------------- Target Distributions (Always Visible) ----------------
card("Target Distributions", "Class balance overview.", "card-amber")

with st.expander("Target distributions", expanded=True):
    st.write(df["NObeyesdad"].value_counts())
    st.write(y.value_counts())


if not run_btn:
    card("Ready to Run", "Choose options on the left and click Run.", "card-green")
    st.stop()

# ---------------- Test data for evaluation: Upload overrides internal split ----------------
if uploaded_test_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_test_file)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

    if "NObeyesdad" in uploaded_df.columns:
        st.error("Uploaded test CSV should NOT contain target column.")
        st.stop()

    # Prepare features only
    from model.preprocess import prepare_X
    X_test_df = prepare_X(uploaded_df)

    card("Using Uploaded Test CSV", "Prediction mode (no target column).", "card-green")
    y_test = None
    prediction_only = True
else:
    _, X_test_df, _, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )
    prediction_only = False



# ---------------- Load Selected Model (NO TRAINING) ----------------
model_path = MODEL_PATHS.get(model_name)

if not model_path or not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.info("Run `python model/train_all_models.py` locally to generate .pkl files.")
    st.stop()

# Safe model loading
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load selected model ({model_name}): {e}")
    st.stop()

# Safe prediction
try:
    y_prob = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_df)

        # Binary case
        if proba.shape[1] == 2:
            y_prob = proba[:, 1]
            y_pred = (y_prob >= float(threshold)).astype(int)

        # Multi-class case
        else:
            y_pred = np.argmax(proba, axis=1)
    else:
        y_pred = model.predict(X_test_df)

except Exception as e:
    st.error(f"Failed to run prediction for selected model ({model_name}): {e}")
    st.stop()

# ---------------- Prediction Only Mode ----------------
if prediction_only:
    output_df = uploaded_df.copy()
    output_df["Predicted_Class"] = y_pred

    if y_prob is not None:
        output_df["Predicted_Prob_Obese"] = y_prob

    st.subheader("Predictions")
    st.dataframe(output_df.head(30), width="stretch")

    st.download_button(
        "Download Predictions",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.stop()   # Much needed



# ---------------- Metrics ----------------
card("Model Performance", "Evaluation metrics (model is loaded from .pkl, no training here).", "card-purple")

acc = accuracy_score(y_test, y_pred)
st.metric("Overall Accuracy", round(acc, 4))

col1, col2 = st.columns(2)

if y_prob is not None:
    auc = roc_auc_score(y_test, y_prob)
    col1.metric("AUC Score", round(auc, 4))

mcc = matthews_corrcoef(y_test, y_pred)
col2.metric("MCC", round(mcc, 4))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T.round(4)
# Remove duplicate accuracy row (if already shown)
if "accuracy" in report_df.index:
    report_df = report_df.drop(index="accuracy")
report_df = report_df.rename(index={
    "0": "Class 0 – Not Obese",
    "1": "Class 1 – Obese"
})
st.table(report_df)


# ---------------- Confusion Matrix ----------------
card("Confusion Matrix", "Prediction counts.", "card-rose")
st.table(
    pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    )
)


# ---------------- Comparison Table (uses saved models) ----------------
def safe_predict(model_loaded, X_test_df, threshold):
    """Return (y_pred, y_prob or None) safely."""
    if hasattr(model_loaded, "predict_proba"):
        prob = model_loaded.predict_proba(X_test_df)[:, 1]
        pred = (prob >= threshold).astype(int)
        return pred, prob
    pred = model_loaded.predict(X_test_df)
    return pred, None

def compute_saved_models_comparison(X_test_df, y_test, threshold: float):
    rows = []

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            continue

        try:
            m = load_model(path)
            # fixed 0.5 for fair comparison across models
            y_pred_i, y_prob_i = safe_predict(m, X_test_df, threshold=0.5)

            acc_i = accuracy_score(y_test, y_pred_i)
            mcc_i = matthews_corrcoef(y_test, y_pred_i)
            auc_i = roc_auc_score(y_test, y_prob_i) if y_prob_i is not None else np.nan

            rep = classification_report(y_test, y_pred_i, output_dict=True, zero_division=0)

            rows.append([
                name,
                acc_i,
                auc_i,
                mcc_i,
                rep["weighted avg"]["precision"],
                rep["weighted avg"]["recall"],
                rep["weighted avg"]["f1-score"],
            ])

        except Exception as e:
            st.warning(f"Skipped {name}: {e}")
            continue

    df_results = pd.DataFrame(
        rows,
        columns=["Model", "Accuracy", "AUC", "MCC", "Precision", "Recall", "F1"]
    ).sort_values(by=["Accuracy", "AUC", "MCC"], ascending=False).reset_index(drop=True)

    df_results.index = df_results.index + 1
    return df_results


st.subheader("Models performance Comparison")

if show_compare:
    results_df = compute_saved_models_comparison(X_test_df, y_test, float(threshold))

    st.dataframe(
        results_df.round(4),
        width="stretch",
        hide_index=True
    )
else:
    st.info("Enable comparison to display results.")

