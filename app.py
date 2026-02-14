import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ---------------- Page Config ----------------
st.set_page_config(page_title="Obesity ML Assignment", layout="wide")


# ---------------- Theme / Styling ----------------
def apply_theme():
    st.markdown(
        """
        <style>
        /* --- App background --- */
        .stApp {
            background: linear-gradient(180deg, #40826D 0%, #2F6F5F 100%);
        }

        /* --- Safer text coloring (DO NOT override all elements) --- */
        .stApp p,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp label,
        .stMarkdown,
        .stText,
        .stCaption {
            color: #ffffff !important;
        }

        /* --- Sidebar readable --- */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.65) !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* --- Fix the top Streamlit header / toolbar (prevents white bar / white square) --- */
/* Remove default white background */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Fix internal header container that causes white square */
header[data-testid="stHeader"] > div {
    background: transparent !important;
}

/* Fix deploy/menu area */
div[data-testid="stToolbar"] {
    background: transparent !important;
}

/* Ensure icons and text are white */
header[data-testid="stHeader"] svg,
header[data-testid="stHeader"] button,
div[data-testid="stToolbar"] svg,
div[data-testid="stToolbar"] button {
    color: #ffffff !important;
    fill: #ffffff !important;
}


        /* --- Titles --- */
        .app-title {
            color: #ffffff !important;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }

        /* --- Card UI --- */
        .card {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 15px;
            padding: 15px;
            margin: 12px 0;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .muted {
            color: rgba(255,255,255,0.85) !important;
            font-size: 0.92rem;
            margin: 0;
        }

        /* Optional card accents */
        .card-blue   { border-left: 6px solid rgba(37, 99, 235, 0.85); }
        .card-green  { border-left: 6px solid rgba(16, 185, 129, 0.85); }
        .card-purple { border-left: 6px solid rgba(124, 58, 237, 0.85); }
        .card-amber  { border-left: 6px solid rgba(245, 158, 11, 0.85); }
        .card-rose   { border-left: 6px solid rgba(244, 63, 94, 0.85); }

        /* --- Metrics block --- */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.60);
            border-radius: 12px;
            padding: 10px;
        }

        /* --- Inputs (selectbox/slider/text) backgrounds --- */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: rgba(15, 23, 42, 0.70) !important;
        }

        /* --- Dropdown / popover menu background + text --- */
        div[role="listbox"] {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #ffffff !important;
        }
        div[role="option"] {
            color: #ffffff !important;
        }

        /* --- Buttons --- */
        .stButton button {
            background-color: rgba(255,255,255,0.15) !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            color: #ffffff !important;
        }

        /* --- Download button: white background + BLACK text --- */
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
        div[data-testid="stDownloadButton"] button svg {
            fill: #000000 !important;
        }

        /* --- Tables --- */
        .stTable, .stDataFrame {
            background: rgba(15, 23, 42, 0.55) !important;
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
st.sidebar.caption("Pick a model, set split/threshold, and run the evaluation.")

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes (GaussianNB)",
        "Random Forest",
        "XGBoost (optional)"
    ]
)

st.sidebar.header("Train/Test")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.25, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("Prediction")
threshold = st.sidebar.slider("Probability threshold (if available)", 0.10, 0.90, 0.50, 0.05)

st.sidebar.header("Display")
show_data = st.sidebar.checkbox("Show dataset preview", value=True)
show_distributions = st.sidebar.checkbox("Show target distributions", value=True)

run_btn = st.sidebar.button("🚀 Run / Re-run")


# ---------------- Load data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("ObesityDataSet.csv")


df = load_data()
target_col = "NObeyesdad"


# ---------------- Preview ----------------
if show_data:
    card("Dataset Preview", "First few rows of the dataset.", "card-blue")
    st.dataframe(df.head(10), width="stretch")

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Dataset (CSV)",
        data=csv_data,
        file_name="ObesityDataSet.csv",
        mime="text/csv"
    )


# ---------------- Target mapping ----------------
target_map = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 1,
    "Obesity_Type_I": 1,
    "Obesity_Type_II": 1,
    "Obesity_Type_III": 1,
}

y = df[target_col].map(target_map).fillna(0).astype(int)


# ---------------- Distributions ----------------
if show_distributions:
    card("Target Distributions", "Class balance overview.", "card-amber")
    with st.expander("Target distributions"):
        st.write(df[target_col].value_counts())
        st.write(y.value_counts())


# ---------------- Features ----------------
binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)

X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)


if not run_btn:
    card("Ready to Run", "Choose options on the left and click Run.", "card-green")
    st.stop()


def generate_readme_table(X_train_df, X_test_df, y_train, y_test, random_state):
    """
    Uses the SAME train/test split as the selected model section.
    Applies scaling ONLY for LR and KNN inside this function.
    Uses threshold=0.50 for fair comparison across models.
    """
    rows = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_df)
            X_test = scaler.transform(X_test_df)
        else:
            X_train = X_train_df
            X_test = X_test_df

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.50).astype(int)
            auc = roc_auc_score(y_test, y_prob)
        else:
            y_pred = model.predict(X_test)
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        rows.append([name, acc, auc, mcc, precision, recall, f1])

    df_results = pd.DataFrame(
        rows,
        columns=["Model", "Accuracy", "AUC", "MCC", "Precision", "Recall", "F1"]
    ).sort_values(by=["Accuracy", "AUC", "MCC"], ascending=False).reset_index(drop=True)

    # Start index from 1 instead of 0
    df_results.index = df_results.index + 1

    return df_results


# ---------------- Train/Test split (keep DF copies) ----------------
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Selected model training data (may be scaled)
X_train, X_test = X_train_df, X_test_df

needs_scaling = model_name in ["Logistic Regression", "KNN"]
if needs_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)


# ---------------- Model ----------------
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=2000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=random_state)
elif model_name == "KNN":
    model = KNeighborsClassifier()
elif model_name == "Naive Bayes (GaussianNB)":
    model = GaussianNB()
elif model_name == "Random Forest":
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
elif model_name == "XGBoost (optional)":
    if not XGB_AVAILABLE:
        st.error("XGBoost is not available in this environment. Please check requirements.txt and redeploy.")
        st.stop()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
else:
    st.error("Unknown model selection")
    st.stop()

model.fit(X_train, y_train)


# ---------------- Predict ----------------
y_prob = None
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
else:
    y_pred = model.predict(X_test)


# ---------------- Metrics ----------------
card("Model Performance", "Evaluation metrics.", "card-purple")

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


# ---------------- Models performance comparison table ----------------
st.subheader("Models performance Comparison")

results_df = generate_readme_table(X_train_df, X_test_df, y_train, y_test, random_state)
st.table(results_df.round(4))
