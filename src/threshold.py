"""
HireSense AI — optimal decision threshold.

The default 0.5 cutoff is rarely optimal. This finds a better threshold on a
held-out set using two strategies:

  * Youden's J   — maximises (TPR - FPR); balances sensitivity and specificity.
  * Cost-sensitive — minimises expected cost given the business cost of a false
                     negative (missing a good candidate) vs a false positive
                     (interviewing a weak one).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET
from predict import load_model


def youden_threshold(y_true, y_proba) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def cost_threshold(y_true, y_proba, cost_fn: float = 5.0, cost_fp: float = 1.0) -> tuple[float, float]:
    """Threshold minimising expected cost. Returns (threshold, min_cost)."""
    y_true = np.asarray(y_true)
    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, float("inf")
    for t in grid:
        pred = (y_proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Find an optimal decision threshold.")
    parser.add_argument("--cost-fn", type=float, default=5.0,
                        help="Cost of a false negative (missing a good candidate).")
    parser.add_argument("--cost-fp", type=float, default=1.0,
                        help="Cost of a false positive (interviewing a weak one).")
    args = parser.parse_args()

    model = load_model()["model"]
    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    proba = model.predict_proba(X_test)[:, 1]

    j_t = youden_threshold(y_test, proba)
    c_t, c_cost = cost_threshold(y_test, proba, args.cost_fn, args.cost_fp)

    print("Optimal decision thresholds (held-out set):\n")
    print(f"  Default            : 0.50")
    print(f"  Youden's J optimal : {j_t:.3f}")
    print(f"  Cost-sensitive     : {c_t:.3f}  "
          f"(FN cost={args.cost_fn}, FP cost={args.cost_fp}, total cost={c_cost:.0f})")
    print("\nUse with:  python src/predict.py --csv <file> --threshold "
          f"{c_t:.2f}")


if __name__ == "__main__":
    main()
