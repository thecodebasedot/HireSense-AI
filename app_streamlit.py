"""
HireSense AI — interactive web UI (Streamlit).

Run:
    streamlit run app_streamlit.py

Enter an applicant's details and get an instant Shortlist / Reject decision,
a confidence score, and a per-feature explanation of why.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from config import EDUCATION_LABELS, NUMERIC_FEATURES  # noqa: E402
from explain import explain_instance  # noqa: E402
from predict import load_model, screen  # noqa: E402

st.set_page_config(page_title="HireSense AI", page_icon="🧠", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


st.title("🧠 HireSense AI")
st.caption("SVM-based job applicant screening")

try:
    bundle = get_model()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train.py` first, then reload.")
    st.stop()

model = bundle["model"]
acc = bundle.get("test_accuracy")
auc = bundle.get("test_roc_auc")
if acc is not None:
    c1, c2 = st.columns(2)
    c1.metric("Model accuracy", f"{acc:.1%}")
    c2.metric("Model ROC-AUC", f"{auc:.3f}")

st.subheader("Applicant details")
col1, col2 = st.columns(2)
with col1:
    years_experience = st.slider("Years of experience", 0.0, 25.0, 5.0, 0.5)
    education_level = st.selectbox(
        "Education level",
        options=list(EDUCATION_LABELS.keys()),
        format_func=lambda k: EDUCATION_LABELS[k],
        index=1,
    )
    skill_match_score = st.slider("Skill match score", 0, 100, 60)
    interview_score = st.slider("Interview score", 0, 100, 62)
with col2:
    communication_score = st.slider("Communication score", 0, 100, 65)
    num_certifications = st.slider("Certifications", 0, 10, 1)
    num_projects = st.slider("Projects completed", 0, 20, 3)
    gpa = st.slider("GPA", 2.0, 4.0, 3.1, 0.05)

applicant = pd.DataFrame([{
    "years_experience": years_experience,
    "education_level": education_level,
    "skill_match_score": skill_match_score,
    "interview_score": interview_score,
    "communication_score": communication_score,
    "num_certifications": num_certifications,
    "num_projects": num_projects,
    "gpa": gpa,
}])[NUMERIC_FEATURES]

if st.button("Screen applicant", type="primary"):
    result = screen(model, applicant).iloc[0]
    prob = float(result["shortlist_probability"])
    decision = result["decision"]

    if decision == "Shortlist":
        st.success(f"✅ **{decision}**  —  shortlist probability **{prob:.1%}**")
    else:
        st.error(f"❌ **{decision}**  —  shortlist probability **{prob:.1%}**")
    st.progress(prob)

    st.subheader("Why? (per-feature contribution)")
    contrib = explain_instance(model, applicant)
    contrib = contrib.rename(columns={
        "feature": "Feature", "value": "Value", "contribution": "Impact on shortlist prob.",
    })
    st.dataframe(
        contrib.style.format({"Impact on shortlist prob.": "{:+.3f}"})
        .background_gradient(cmap="RdYlGn", subset=["Impact on shortlist prob."]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Positive impact means the applicant's value raised their chances "
        "versus an average candidate; negative lowered them."
    )
