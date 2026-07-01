"""
HireSense AI — model diagnostics: calibration curve (reliability diagram).

A calibrated model's predicted probabilities should match observed
frequencies: among applicants given P≈0.7, about 70% should truly be
shortlisted. This plots predicted vs actual and reports the Brier score
(lower is better) and Expected Calibration Error (ECE).
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, REPORTS_DIR, TARGET
from predict import load_model


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Weighted average gap between confidence and accuracy across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = y_proba[mask].mean()
        acc = np.asarray(y_true)[mask].mean()
        ece += (mask.sum() / len(y_proba)) * abs(conf - acc)
    return float(ece)


def main() -> None:
    model = load_model()["model"]
    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    proba = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    brier = brier_score_loss(y_test, proba)
    ece = expected_calibration_error(np.asarray(y_test), proba)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#3d5a80", label="HireSense SVM")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve\nBrier={brier:.3f}, ECE={ece:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=120)
    plt.close(fig)

    print("Model calibration (held-out set):")
    print(f"  Brier score : {brier:.4f}  (lower is better)")
    print(f"  ECE         : {ece:.4f}  (lower is better)")
    print(f"\n✓ Plot saved to {REPORTS_DIR / 'calibration_curve.png'}")


if __name__ == "__main__":
    main()
