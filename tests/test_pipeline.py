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
    print("✓ All tests passed")
