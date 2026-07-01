"""
HireSense AI — bias mitigation.

The fairness audit only *detects* disparity. This module *reduces* it using a
post-processing technique: group-specific thresholds. Instead of one global
cutoff, each protected group gets a threshold chosen so that every group's
selection rate matches a common target (demographic parity).

Post-processing is chosen because it needs no retraining and leaves the model
untouched — only the decision rule adapts.
"""
import numpy as np
import pandas as pd

from config import DATASET_PATH, GROUP_COLUMN, NUMERIC_FEATURES
from predict import load_model


def group_thresholds(proba: np.ndarray, groups: pd.Series, target_rate: float) -> dict:
    """Per-group threshold so each group's selection rate ~= target_rate."""
    thresholds = {}
    for g in groups.unique():
        mask = (groups == g).values
        g_proba = proba[mask]
        # The threshold is the (1 - target_rate) quantile of that group's scores.
        thresholds[g] = float(np.quantile(g_proba, 1 - target_rate))
    return thresholds


def _metrics_from_decisions(df, decisions, group_col):
    scored = df.copy()
    scored["_shortlisted"] = decisions.astype(int)
    rows = []
    for g, sub in scored.groupby(group_col):
        rows.append({"group": g, "selection_rate": round(sub["_shortlisted"].mean(), 4),
                     "true_positive_rate": float("nan")})
    return pd.DataFrame(rows)


def mitigate(model, df: pd.DataFrame, group_col: str = GROUP_COLUMN):
    """Return (thresholds, before_summary, after_summary)."""
    proba = model.predict_proba(df[NUMERIC_FEATURES])[:, 1]
    target = float((proba >= 0.5).mean())  # overall baseline selection rate

    before = _metrics_from_decisions(df, proba >= 0.5, group_col)

    thr = group_thresholds(proba, df[group_col], target)
    decisions = np.array([p >= thr[g] for p, g in zip(proba, df[group_col])])
    after = _metrics_from_decisions(df, decisions, group_col)

    return thr, before, after


def main() -> None:
    model = load_model()["model"]
    df = pd.read_csv(DATASET_PATH)

    if GROUP_COLUMN not in df.columns:
        print(f"No '{GROUP_COLUMN}' column. Regenerate with src/generate_data.py.")
        return

    thr, before, after = mitigate(model, df)

    def di(metrics):
        r = metrics["selection_rate"]
        return r.min() / r.max() if r.max() else float("nan")

    print("Bias mitigation via group-specific thresholds\n")
    print("Per-group thresholds:")
    for g, t in thr.items():
        print(f"  group {g}: {t:.3f}")

    print("\nSelection rates BEFORE (global 0.5 threshold):")
    print(before.to_string(index=False))
    print(f"  disparate impact ratio: {di(before):.4f}")

    print("\nSelection rates AFTER (group-specific thresholds):")
    print(after.to_string(index=False))
    print(f"  disparate impact ratio: {di(after):.4f}")

    print("\n→ Ratio closer to 1.0 means fairer selection across groups.")


if __name__ == "__main__":
    main()
