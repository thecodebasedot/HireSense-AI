"""
HireSense AI — smoke tests for data generation, training, and prediction.
Run with:  python -m pytest tests/  (or  python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from config import NUMERIC_FEATURES, TARGET  # noqa: E402
from generate_data import generate  # noqa: E402
from train import build_search, calibrate  # noqa: E402
from predict import screen  # noqa: E402


def test_generate_shape_and_columns():
    df = generate(n_samples=500, seed=1)
    assert len(df) == 500
    assert list(df.columns) == NUMERIC_FEATURES + [TARGET]
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


if __name__ == "__main__":
    test_generate_shape_and_columns()
    test_generate_has_both_classes()
    test_train_and_predict_end_to_end()
    test_screen_rejects_missing_columns()
    print("✓ All tests passed")
