import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef

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
        /* ================= MAIN APP BACKGROUND ================= */
        .stApp {
            background: linear-gradient(180deg, #40826D 0%, #2F6F5F 100%) !important;
        }

        /* Right-side content spacing */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
        }

        /* ================= TOP HEADER / TOOLBAR  ================= */
        header[data-testid="stHeader"] {
            background: linear-gradient(180deg, #1f3f36 0%, #16332c 100%) !important;
        }
        header[data-testid="stHeader"] * {
            color: #FFFFFF !important;
        }
        header[data-testid="stHeader"]::after {
            background: none !important;
        }

        /* ================= SIDEBAR (VIRIDIAN + BLACK TEXT) ================= */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #40826D 0%, #2F6F5F 100%) !important;
            border-right: 1px solid rgba(0,0,0,0.25) !important;
        }

        /* FORCE ALL SIDEBAR TEXT TO BLACK */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            color: #000000 !important;
        }

        /* Dropdown arrow */
        section[data-testid="stSidebar"] svg {
            fill: #000000 !important;
        }

        /* Sidebar Run button */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.4);
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            color: #ffffff !important;   /* button text stays white */
            font-weight: 700;
            padding: 0.7rem 1rem;
        }

        /* ================= TITLE ================= */
        .app-title {
            color: #ffffff;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        /* ================= CARDS (IMPROVED FOR VIRIDIAN BACKGROUND) ================= */
        .card {
            background: rgba(15, 23, 42, 0.75);      /* deep slate overlay */
            border-radius: 18px;
            padding: 18px;
            margin: 14px 0;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        }

        .card h3 {
            color: #f9fafb;
            margin-bottom: 8px;
        }

        .muted {
            color: rgba(255,255,255,0.85);
            font-size: 0.92rem;
        }

        /* Keep your accent variants (optional; they sit on top of the base card bg if used elsewhere) */
        .card-blue { background: rgba(37, 99, 235, 0.16); }
        .card-green { background: rgba(16, 185, 129, 0.16); }
        .card-purple { background: rgba(124, 58, 237, 0.16); }
        .card-amber { background: rgba(245, 158, 11, 0.16); }
        .card-rose { background: rgba(244, 63, 94, 0.14); }

        /* ================= METRICS ================= */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 12px;
            border-radius: 14px;
            text-align: left !important;
        }

        div[data-testid="stMetric"] * {
            color: #ffffff !important;
            font-weight: 600;
            text-align: left !important;
            justify-content: flex-start !important;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
            font-weight: 800;
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* ================= TABLES & DATAFRAMES ================= */
        [data-testid="stTable"],
        div[data-testid="stDataFrame"] {
            background: rgba(15, 23, 42, 0.55);
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
        }

        [data-testid="stTable"] th,
        [data-testid="stTable"] td,
        div[data-testid="stDataFrame"] * {
            color: #ffffff !important;
        }

        /* ================= GENERAL TEXT ================= */
        div[data-testid="stMarkdownContainer"] *,
        div[data-testid="stExpander"] * {
            color: #ffffff !important;
        }

        pre {
            color: #ffffff !important;
            background-color: rgba(15, 23, 42, 0.55) !important;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.12);
        }

        /* ================= FORCE TABLE TEXT LEFT ALIGN ================= */

        /* st.table() */
        div[data-testid="stTable"] div {
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* st.dataframe() cells */
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* Headers */
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* Prevent numeric auto-alignment */
        div[data-testid="stDataFrame"] * {
            justify-content: flex-start !important;
        }

/* ================= DOWNLOAD BUTTON FIX (FORCE BLACK + BOLD) ================= */
div.stDownloadButton > button,
div.stDownloadButton > button * {
    color: #000000 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

/* Optional: make button background visible and nicer */
div.stDownloadButton > button {
    background: #ffffff !important;
    border: 2px solid rgba(0,0,0,0.35) !important;
    border-radius: 12px !important;
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
    st.dataframe(df.head(10), use_container_width=True)  # show 10 rows (you can change)

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

y = df[target_col].map(target_map)
y = y.fillna(0)

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
        df[col] = df[col].astype(str).str.lower().map({"yes":1, "no":0})
        df[col] = df[col].fillna(0)

X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)


if not run_btn:
    card("Ready to Run", "Choose options on the left and click Run.", "card-green")
    st.stop()


# ---------------- Train/Test ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

needs_scaling = model_name in ["Logistic Regression", "KNN"]

if needs_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
        n_estimators=100,      # reduced from 300
        max_depth=4,           # reduced depth
        learning_rate=0.1,     # slightly higher learning rate
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

report = classification_report(y_test, y_pred, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).T.round(4)

# Rename class labels for clarity
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
