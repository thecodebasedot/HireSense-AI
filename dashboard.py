"""
HireSense AI — live analytics dashboard (Streamlit).

A multi-tab operational dashboard on top of the trained models:

  * Overview    — model metadata + headline metrics, with an SVM/NN selector
  * Dataset     — class balance and feature distributions
  * Batch score — upload a CSV, validate, score, rank, and download results
  * Fairness    — per-group selection rates + disparate-impact check
  * Drift       — compare an uploaded batch against the training distribution

Run:
    streamlit run dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import pandas as pd
import streamlit as st

from config import (
    DATASET_PATH, GROUP_COLUMN, MODEL_DIR, NUMERIC_FEATURES, TARGET,
)
from predict import screen, rank
from explain import global_importance
from fairness import audit, summarize
from drift import drift_report
from validation import validate

st.set_page_config(page_title="HireSense AI — Dashboard", page_icon="📊", layout="wide")

MODEL_FILES = {
    "SVM (primary)": MODEL_DIR / "hiresense_svm.joblib",
    "Neural Network": MODEL_DIR / "hiresense_mlp.joblib",
}


@st.cache_resource
def load_bundle(path_str: str):
    return joblib.load(path_str)


@st.cache_data
def load_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return None


st.title("📊 HireSense AI — Analytics Dashboard")

available = {name: p for name, p in MODEL_FILES.items() if p.exists()}
if not available:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

choice = st.sidebar.selectbox("Model", list(available.keys()))
bundle = load_bundle(str(available[choice]))
model = bundle["model"]
df = load_dataset()

tab_overview, tab_data, tab_batch, tab_fair, tab_drift = st.tabs(
    ["Overview", "Dataset", "Batch score", "Fairness", "Drift"]
)

# --- Overview -------------------------------------------------------------- #
with tab_overview:
    st.subheader("Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", bundle.get("algorithm", "?"))
    c2.metric("Version", bundle.get("version", "?"))
    c3.metric("Accuracy", f"{bundle.get('test_accuracy', float('nan')):.1%}")
    c4.metric("ROC-AUC", f"{bundle.get('test_roc_auc', float('nan')):.3f}")
    st.caption(f"Trained at {bundle.get('trained_at', 'unknown')} "
               f"· tuning: {bundle.get('tuning', 'n/a')} "
               f"· params: {bundle.get('best_params', {})}")

    if df is not None:
        st.subheader("Feature importance (permutation)")
        imp = global_importance(model, df[NUMERIC_FEATURES], df[TARGET])
        st.bar_chart(imp.set_index("feature")["importance"])

# --- Dataset --------------------------------------------------------------- #
with tab_data:
    if df is None:
        st.info("Dataset not found. Run `python src/generate_data.py`.")
    else:
        st.subheader("Class balance")
        counts = df[TARGET].value_counts().rename({0: "Rejected", 1: "Shortlisted"})
        st.bar_chart(counts)
        st.subheader("Feature distributions")
        feat = st.selectbox("Feature", NUMERIC_FEATURES)
        st.bar_chart(df[feat].value_counts(bins=20).sort_index())
        st.subheader("Sample rows")
        st.dataframe(df.head(20), use_container_width=True)

# --- Batch score ----------------------------------------------------------- #
with tab_batch:
    st.subheader("Upload applicants CSV")
    st.caption(f"Required columns: {', '.join(NUMERIC_FEATURES)}")
    up = st.file_uploader("CSV file", type="csv")
    top_n = st.number_input("Show top N (0 = all)", min_value=0, value=0, step=1)
    if up is not None:
        batch = pd.read_csv(up)
        issues = validate(batch)
        if issues:
            st.error("Validation issues:\n- " + "\n- ".join(issues))
        else:
            result = (rank(model, batch, top=top_n or None) if top_n
                      else screen(model, batch))
            st.success(f"Scored {len(batch)} applicants.")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download scored CSV",
                               result.to_csv(index=False).encode(),
                               "scored.csv", "text/csv")

# --- Fairness -------------------------------------------------------------- #
with tab_fair:
    if df is None or GROUP_COLUMN not in df.columns:
        st.info("Fairness audit needs the dataset with a `group` column.")
    else:
        metrics = audit(model, df)
        summary = summarize(metrics)
        st.subheader("Per-group metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Demographic parity diff", f"{summary['demographic_parity_difference']:.3f}")
        di = summary["disparate_impact_ratio"]
        c2.metric("Disparate impact ratio", f"{di:.3f}",
                  delta="pass" if di >= 0.8 else "review")
        c3.metric("Equal-opportunity diff", f"{summary['equal_opportunity_difference']:.3f}")

# --- Drift ----------------------------------------------------------------- #
with tab_drift:
    st.subheader("Data drift vs training distribution")
    if df is None:
        st.info("Training dataset not found.")
    else:
        up2 = st.file_uploader("Upload a new applicant batch (CSV)", type="csv", key="drift")
        if up2 is not None:
            new = pd.read_csv(up2)
            issues = validate(new)
            if issues:
                st.error("Validation issues:\n- " + "\n- ".join(issues))
            else:
                report = drift_report(df, new)
                st.dataframe(report, use_container_width=True, hide_index=True)
                major = (report["severity"] == "MAJOR").sum()
                if major:
                    st.warning(f"{major} feature(s) show MAJOR drift — consider retraining.")
                else:
                    st.success("No major drift detected.")
