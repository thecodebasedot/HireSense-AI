"""
HireSense AI — data drift detection.

Compares an incoming batch of applicants against a reference distribution (the
training data) to detect distribution shift that could silently degrade the
model. For each feature it reports:

  * PSI (Population Stability Index)
        < 0.10  no significant shift
        0.10–0.25  moderate shift
        > 0.25  major shift — investigate / retrain
  * KS test p-value (two-sample Kolmogorov–Smirnov); small p => distributions differ.

    python src/drift.py data/new_batch.csv
    python src/drift.py --demo          # simulate drift for demonstration
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import DATASET_PATH, NUMERIC_FEATURES


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    # Avoid division by zero / log(0).
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _severity(psi_value: float) -> str:
    if psi_value < 0.10:
        return "none"
    if psi_value < 0.25:
        return "moderate"
    return "MAJOR"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in NUMERIC_FEATURES:
        p = psi(reference[feat].values, current[feat].values)
        ks_p = ks_2samp(reference[feat].values, current[feat].values).pvalue
        rows.append({
            "feature": feat,
            "psi": round(p, 4),
            "severity": _severity(p),
            "ks_pvalue": round(float(ks_p), 4),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _simulate_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Shift a few features to mimic a changed applicant population."""
    rng = np.random.default_rng(seed)
    shifted = df.copy()
    shifted["years_experience"] = (shifted["years_experience"] + 4).clip(0, 50)
    shifted["skill_match_score"] = (shifted["skill_match_score"] - 15).clip(0, 100)
    shifted["gpa"] = (shifted["gpa"] + rng.normal(0.3, 0.1, len(shifted))).clip(0, 4)
    return shifted


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect data drift vs the training set.")
    parser.add_argument("csv", nargs="?", help="New applicant batch CSV.")
    parser.add_argument("--demo", action="store_true", help="Simulate a drifted batch.")
    args = parser.parse_args()

    reference = pd.read_csv(DATASET_PATH)
    if args.demo or not args.csv:
        current = _simulate_drift(reference)
        print("(demo) Comparing training data against a simulated drifted batch.\n")
    else:
        current = pd.read_csv(args.csv)
        print(f"Comparing {args.csv} against the training distribution.\n")

    report = drift_report(reference, current)
    print(report.to_string(index=False))

    major = report[report["severity"] == "MAJOR"]
    if len(major):
        print(f"\n⚠️  {len(major)} feature(s) show MAJOR drift — consider retraining.")
    else:
        print("\n✓ No major drift detected.")


if __name__ == "__main__":
    main()
