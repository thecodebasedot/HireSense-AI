"""
HireSense AI — generate evaluation plots as PNG files.

Produces, into the reports/ directory:
  * class_distribution.png  — shortlisted vs rejected counts
  * confusion_matrix.png     — on a held-out test split
  * roc_curve.png            — ROC curve with AUC
  * feature_importance.png   — permutation importance ranking
"""
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, ROOT_DIR, TARGET
from explain import global_importance
from predict import load_model

REPORTS_DIR = ROOT_DIR / "reports"


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Rejected", "Shortlisted"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts.values, color=["#e07a5f", "#81b29a"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Applicant Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_confusion_and_roc(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Rejected", "Shortlisted"], cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(model, X, y) -> None:
    imp = global_importance(model, X, y).sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp["feature"], imp["importance"], color="#3d5a80")
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean drop in ROC-AUC")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    bundle = load_model()
    model = bundle["model"]

    plot_class_distribution(df)
    plot_confusion_and_roc(model, X_test, y_test)
    plot_feature_importance(model, X, y)

    print(f"Saved plots to {REPORTS_DIR}/")
    for p in sorted(REPORTS_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
