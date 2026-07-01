"""
HireSense AI — fairness / bias audit.

Checks whether the model's shortlist decisions are distributed fairly across a
protected group (the `group` column in the dataset, which is NOT a model
feature). Reports:

  * Selection rate per group        — fraction shortlisted
  * Demographic parity difference   — max gap in selection rate
  * Disparate impact ratio          — min/max selection rate (the "80% rule":
                                       below 0.8 is a common red flag)
  * Equal-opportunity difference     — gap in true-positive rate across groups

These are diagnostics, not guarantees — always review hiring models with a
human and domain expertise.
"""
import pandas as pd

from config import DATASET_PATH, GROUP_COLUMN, NUMERIC_FEATURES, TARGET
from predict import load_model, screen


def audit(model, df: pd.DataFrame, group_col: str = GROUP_COLUMN) -> pd.DataFrame:
    """Per-group fairness metrics."""
    scored = screen(model, df)
    scored[group_col] = df[group_col].values
    scored["_shortlisted"] = (scored["decision"] == "Shortlist").astype(int)
    scored["_true"] = df[TARGET].values

    rows = []
    for g, sub in scored.groupby(group_col):
        qualified = sub[sub["_true"] == 1]
        tpr = qualified["_shortlisted"].mean() if len(qualified) else float("nan")
        rows.append({
            "group": g,
            "n": len(sub),
            "selection_rate": round(sub["_shortlisted"].mean(), 4),
            "true_positive_rate": round(tpr, 4),
        })
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict:
    """Aggregate fairness numbers across groups."""
    rates = metrics["selection_rate"]
    tprs = metrics["true_positive_rate"]
    return {
        "demographic_parity_difference": round(rates.max() - rates.min(), 4),
        "disparate_impact_ratio": round(rates.min() / rates.max(), 4) if rates.max() else float("nan"),
        "equal_opportunity_difference": round(tprs.max() - tprs.min(), 4),
    }


def main() -> None:
    bundle = load_model()
    model = bundle["model"]
    df = pd.read_csv(DATASET_PATH)

    if GROUP_COLUMN not in df.columns:
        print(f"No '{GROUP_COLUMN}' column in the dataset. Regenerate with "
              f"`python src/generate_data.py`.")
        return

    metrics = audit(model, df)
    summary = summarize(metrics)

    print("Per-group metrics:\n")
    print(metrics.to_string(index=False))
    print("\nFairness summary:")
    print(f"  Demographic parity difference : {summary['demographic_parity_difference']:.4f} "
          f"(0 = perfectly equal selection rates)")
    print(f"  Disparate impact ratio        : {summary['disparate_impact_ratio']:.4f} "
          f"(>= 0.80 passes the 80% rule)")
    print(f"  Equal-opportunity difference  : {summary['equal_opportunity_difference']:.4f} "
          f"(0 = equal true-positive rates)")

    di = summary["disparate_impact_ratio"]
    verdict = "PASS ✅" if di >= 0.8 else "REVIEW ⚠️"
    print(f"\n  80% rule: {verdict}")


if __name__ == "__main__":
    main()
