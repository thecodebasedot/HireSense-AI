"""
HireSense AI — smoke tests for data generation, training, and prediction.
Run with:  python -m pytest tests/  (or  python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from config import NUMERIC_FEATURES, TARGET  # noqa: E402
from config import GROUP_COLUMN  # noqa: E402
from generate_data import generate  # noqa: E402
from train import build_search, calibrate  # noqa: E402
from predict import rank, screen  # noqa: E402
from explain import explain_instance, global_importance  # noqa: E402
from job_profiles import list_profiles, load_profile, meets_requirements  # noqa: E402
from fairness import audit, summarize  # noqa: E402
from resume_parser import parse_resume  # noqa: E402
from validation import validate, is_valid  # noqa: E402
from conformal import ConformalClassifier, LABELS  # noqa: E402
from threshold import youden_threshold, cost_threshold  # noqa: E402
from drift import psi, drift_report, _simulate_drift  # noqa: E402
from bias_mitigation import mitigate  # noqa: E402
from deep_model import build_model as build_deep_model  # noqa: E402
from load_dataset import ingest  # noqa: E402


def _fit_calibrated(df):
    """Helper: tune + calibrate a model on a dataframe."""
    search = build_search()
    search.fit(df[NUMERIC_FEATURES], df[TARGET])
    return calibrate(search.best_estimator_, df[NUMERIC_FEATURES], df[TARGET])


def test_generate_shape_and_columns():
    df = generate(n_samples=500, seed=1)
    assert len(df) == 500
    assert list(df.columns) == NUMERIC_FEATURES + [TARGET, GROUP_COLUMN]
    assert set(df[TARGET].unique()) <= {0, 1}


def test_generate_has_both_classes():
    df = generate(n_samples=500, seed=1)
    counts = df[TARGET].value_counts()
    assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0


def test_train_and_predict_end_to_end():
    df = generate(n_samples=800, seed=2)
    X, y = df[NUMERIC_FEATURES], df[TARGET]

    search = build_search()
    search.fit(X, y)
    model = calibrate(search.best_estimator_, X, y)

    # Model should beat random on its own training data.
    assert model.score(X, y) > 0.7

    scored = screen(model, df.head(5).copy())
    assert "decision" in scored.columns
    assert "shortlist_probability" in scored.columns
    assert scored["shortlist_probability"].between(0, 1).all()
    assert set(scored["decision"].unique()) <= {"Shortlist", "Reject"}


def test_screen_rejects_missing_columns():
    # The column check in screen() runs before the model is touched, so a
    # tuned (uncalibrated) pipeline is enough here.
    df = generate(n_samples=100, seed=3)
    search = build_search()
    search.fit(df[NUMERIC_FEATURES], df[TARGET])
    model = search.best_estimator_

    bad = pd.DataFrame({"years_experience": [1, 2]})
    try:
        screen(model, bad)
        assert False, "Expected ValueError for missing columns"
    except ValueError:
        pass


def test_global_importance_ranks_all_features():
    df = generate(n_samples=600, seed=4)
    model = _fit_calibrated(df)

    imp = global_importance(model, df[NUMERIC_FEATURES], df[TARGET])
    assert set(imp["feature"]) == set(NUMERIC_FEATURES)
    # Importance is sorted descending.
    assert imp["importance"].is_monotonic_decreasing


def test_explain_instance_contributions():
    df = generate(n_samples=600, seed=5)
    model = _fit_calibrated(df)
    baseline = df[NUMERIC_FEATURES].median()

    row = df[NUMERIC_FEATURES].head(1).copy()
    contrib = explain_instance(model, row, baseline=baseline)

    assert set(contrib["feature"]) == set(NUMERIC_FEATURES)
    assert {"feature", "value", "contribution"} <= set(contrib.columns)
    # Contributions are ordered by absolute impact.
    assert contrib["contribution"].abs().is_monotonic_decreasing


def test_threshold_changes_decisions():
    df = generate(n_samples=600, seed=6)
    model = _fit_calibrated(df)
    sample = df[NUMERIC_FEATURES].head(50).copy()

    lenient = screen(model, sample, threshold=0.2)
    strict = screen(model, sample, threshold=0.8)
    n_lenient = (lenient["decision"] == "Shortlist").sum()
    n_strict = (strict["decision"] == "Shortlist").sum()
    assert n_lenient >= n_strict


def test_rank_orders_by_probability():
    df = generate(n_samples=200, seed=7)
    model = _fit_calibrated(df)
    ranked = rank(model, df[NUMERIC_FEATURES].copy(), top=5)
    assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
    assert ranked["shortlist_probability"].is_monotonic_decreasing


def test_job_profiles_load_and_requirements():
    assert "senior_engineer" in list_profiles()
    profile = load_profile("senior_engineer")
    assert "threshold" in profile and "requirements" in profile

    strong = {"years_experience": 8, "education_level": 2, "skill_match_score": 90}
    weak = {"years_experience": 1, "education_level": 2, "skill_match_score": 90}
    assert meets_requirements(strong, profile) is True
    assert meets_requirements(weak, profile) is False


def test_fairness_audit_runs():
    df = generate(n_samples=800, seed=8)
    assert GROUP_COLUMN in df.columns
    model = _fit_calibrated(df)
    metrics = audit(model, df)
    summary = summarize(metrics)
    assert 0.0 <= summary["disparate_impact_ratio"] <= 1.0
    assert set(metrics["group"]) == {"A", "B"}


def test_resume_parser_extracts_features():
    text = (
        "Senior Engineer with 7 years of experience. "
        "Master of Science in Computer Science. CGPA: 3.8 / 4.0. "
        "Skills: Python, SQL. AWS Certified. Delivered a project."
    )
    features = parse_resume(text, job_skills=["python", "sql"])
    assert features.loc[0, "years_experience"] == 7
    assert features.loc[0, "education_level"] == 2   # Master
    assert features.loc[0, "gpa"] == 3.8
    assert features.loc[0, "skill_match_score"] == 100.0


def test_validation_detects_bad_data():
    df = generate(n_samples=50, seed=9)
    assert is_valid(df[NUMERIC_FEATURES])
    bad = df[NUMERIC_FEATURES].copy()
    bad.loc[0, "gpa"] = 9.0  # out of [0, 4]
    issues = validate(bad)
    assert any("gpa" in i for i in issues)


def test_conformal_coverage_meets_target():
    df = generate(n_samples=1200, seed=10)
    model = _fit_calibrated(df)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    Xc, Xt = X.iloc[:800], X.iloc[800:]
    yc, yt = y.iloc[:800], y.iloc[800:]

    cp = ConformalClassifier(model, alpha=0.1).calibrate(Xc, yc)
    sets = cp.predict_sets(Xt)
    covered = sum(LABELS[t] in s for s, t in zip(sets, yt))
    # Allow a small finite-sample slack below the 0.90 target.
    assert covered / len(yt) >= 0.85


def test_threshold_finders_return_valid_values():
    df = generate(n_samples=600, seed=11)
    model = _fit_calibrated(df)
    proba = model.predict_proba(df[NUMERIC_FEATURES])[:, 1]
    jt = youden_threshold(df[TARGET], proba)
    ct, cost = cost_threshold(df[TARGET], proba)
    assert 0.0 <= jt <= 1.0
    assert 0.0 <= ct <= 1.0 and cost >= 0


def test_drift_detects_shift():
    df = generate(n_samples=500, seed=12)
    # Identical distribution -> near-zero PSI.
    assert psi(df["gpa"].values, df["gpa"].values) < 0.01
    # Simulated drift -> at least one MAJOR feature.
    report = drift_report(df, _simulate_drift(df))
    assert (report["severity"] == "MAJOR").any()


def test_bias_mitigation_improves_parity():
    df = generate(n_samples=1000, seed=13)
    model = _fit_calibrated(df)
    _, before, after = mitigate(model, df)

    def di(m):
        r = m["selection_rate"]
        return r.min() / r.max()

    # Group-specific thresholds should not worsen demographic parity.
    assert di(after) >= di(before) - 1e-6


def test_deep_model_trains_and_predicts():
    df = generate(n_samples=400, seed=14)
    model = build_deep_model()
    model.fit(df[NUMERIC_FEATURES], df[TARGET])

    proba = model.predict_proba(df[NUMERIC_FEATURES])
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    preds = model.predict(df[NUMERIC_FEATURES])
    assert set(pd.unique(preds)) <= {0, 1}
    # Beats random on its own training data.
    assert model.score(df[NUMERIC_FEATURES], df[TARGET]) > 0.7


def test_load_dataset_ingest_maps_columns():
    import io
    raw = io.StringIO(
        "exp_years,degree,skills_pct,interview,comm,certs,projects,cgpa,hired\n"
        "7,Master,88,82,80,4,10,3.8,1\n"
        "0,High School,35,40,45,0,1,2.5,0\n"
    )
    mapping = {
        "exp_years": "years_experience", "degree": "education_level",
        "skills_pct": "skill_match_score", "interview": "interview_score",
        "comm": "communication_score", "certs": "num_certifications",
        "projects": "num_projects", "cgpa": "gpa", "hired": "shortlisted",
    }
    df = ingest(raw, mapping)
    assert set(NUMERIC_FEATURES + [TARGET]) <= set(df.columns)
    # Textual education mapped to the ordinal scale.
    assert df.loc[0, "education_level"] == 2   # Master
    assert df.loc[1, "education_level"] == 0   # High School


if __name__ == "__main__":
    test_generate_shape_and_columns()
    test_generate_has_both_classes()
    test_train_and_predict_end_to_end()
    test_screen_rejects_missing_columns()
    test_global_importance_ranks_all_features()
    test_explain_instance_contributions()
    test_threshold_changes_decisions()
    test_rank_orders_by_probability()
    test_job_profiles_load_and_requirements()
    test_fairness_audit_runs()
    test_resume_parser_extracts_features()
    test_validation_detects_bad_data()
    test_conformal_coverage_meets_target()
    test_threshold_finders_return_valid_values()
    test_drift_detects_shift()
    test_bias_mitigation_improves_parity()
    test_deep_model_trains_and_predicts()
    test_load_dataset_ingest_maps_columns()
    print("✓ All tests passed")
