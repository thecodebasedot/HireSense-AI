"""
HireSense AI — conformal prediction (uncertainty quantification).

Split-conformal classification wraps a fitted probabilistic model and, for a
chosen error rate alpha, returns a *prediction set* per applicant with a
guaranteed marginal coverage of >= 1 - alpha.

Prediction sets are interpreted as:
  {Shortlist}          -> confident shortlist
  {Reject}             -> confident reject
  {Reject, Shortlist}  -> uncertain — send to a human reviewer
  {}                   -> out of distribution (rare)

Calibration: on a held-out set, the nonconformity score for a labelled example
is 1 - p(true class). qhat is the (1-alpha) empirical quantile of those scores.
A class is included in an applicant's set when 1 - p(class) <= qhat.
"""
import numpy as np
import pandas as pd

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGET

LABELS = {0: "Reject", 1: "Shortlist"}


class ConformalClassifier:
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.qhat_ = None

    def calibrate(self, X_cal: pd.DataFrame, y_cal) -> "ConformalClassifier":
        proba = self.model.predict_proba(X_cal[NUMERIC_FEATURES])
        y_cal = np.asarray(y_cal)
        true_proba = proba[np.arange(len(y_cal)), y_cal]
        scores = 1 - true_proba
        n = len(scores)
        # Finite-sample corrected quantile level.
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, X: pd.DataFrame) -> list[list[str]]:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate() before predict_sets().")
        proba = self.model.predict_proba(X[NUMERIC_FEATURES])
        sets = []
        for row in proba:
            included = [LABELS[c] for c in range(len(row)) if (1 - row[c]) <= self.qhat_]
            sets.append(included)
        return sets


def _label_set(s: list[str]) -> str:
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return "Uncertain"
    return "OOD"


def main() -> None:
    import argparse
    from sklearn.model_selection import train_test_split
    from predict import load_model

    parser = argparse.ArgumentParser(description="Conformal prediction demo.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Target error rate (coverage = 1 - alpha).")
    args = parser.parse_args()

    model = load_model()["model"]
    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]

    X_cal, X_test, y_cal, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    cp = ConformalClassifier(model, alpha=args.alpha).calibrate(X_cal, y_cal)
    sets = cp.predict_sets(X_test)

    # Empirical coverage: fraction of test points whose true label is in the set.
    covered = sum(LABELS[t] in s for s, t in zip(sets, y_test))
    coverage = covered / len(y_test)

    kinds = pd.Series([_label_set(s) for s in sets]).value_counts()

    print(f"Conformal prediction (alpha={args.alpha}, target coverage "
          f">= {1 - args.alpha:.0%})\n")
    print(f"  qhat            : {cp.qhat_:.4f}")
    print(f"  Empirical cover : {coverage:.3f}")
    print("\n  Prediction-set breakdown:")
    for kind, count in kinds.items():
        print(f"    {kind:10s} {count:5d}  ({count / len(sets):.1%})")
    uncertain = kinds.get("Uncertain", 0)
    print(f"\n  → {uncertain} applicant(s) flagged Uncertain for human review.")


if __name__ == "__main__":
    main()
