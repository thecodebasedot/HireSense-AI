"""
HireSense AI — model explainability.

Two kinds of explanation:

* Global — which features matter most across the whole population, computed
  with permutation importance (model-agnostic, works for any kernel).
* Local — for one applicant, how much each feature pushes the shortlist
  probability up or down, computed by ablating each feature to a neutral
  baseline (the training median) and measuring the change in probability.
"""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET


def _baseline_row() -> pd.Series:
    """Neutral applicant: the median of every feature in the dataset."""
    df = pd.read_csv(DATASET_PATH)
    return df[NUMERIC_FEATURES].median()


def global_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features by permutation importance (drop in ROC-AUC when shuffled)."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": NUMERIC_FEATURES,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_instance(model, row: pd.DataFrame, baseline: pd.Series | None = None) -> pd.DataFrame:
    """Explain one applicant: per-feature contribution to shortlist probability.

    For each feature we replace its value with the neutral baseline and see how
    much the shortlist probability changes. A positive contribution means the
    applicant's actual value raised their chances relative to an average
    candidate; negative means it lowered them.
    """
    if baseline is None:
        baseline = _baseline_row()

    row = row[NUMERIC_FEATURES].reset_index(drop=True)
    base_prob = float(model.predict_proba(row)[:, 1][0])

    contributions = []
    for feat in NUMERIC_FEATURES:
        perturbed = row.copy()
        perturbed.loc[0, feat] = baseline[feat]
        ablated_prob = float(model.predict_proba(perturbed)[:, 1][0])
        # How much this feature's real value added over the baseline value.
        contributions.append({
            "feature": feat,
            "value": row.loc[0, feat],
            "contribution": base_prob - ablated_prob,
        })

    out = pd.DataFrame(contributions)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def main() -> None:
    """CLI: print global feature importance for the trained model."""
    from predict import load_model  # local import to avoid a cycle

    bundle = load_model()
    model = bundle["model"]

    df = pd.read_csv(DATASET_PATH)
    imp = global_importance(model, df[NUMERIC_FEATURES], df[TARGET])

    print("Global feature importance (permutation, drop in ROC-AUC):\n")
    for _, r in imp.iterrows():
        bar = "█" * max(1, int(r["importance"] * 200))
        print(f"  {r['feature']:22s} {r['importance']:.4f}  {bar}")


if __name__ == "__main__":
    main()
