"""
HireSense AI — compare SVM against other classifiers.

Trains several models on the same data with identical preprocessing and
reports cross-validated ROC-AUC plus held-out test metrics. Confirms whether
SVM is a justified choice for this problem. Saves a bar chart to reports/.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, REPORTS_DIR, TARGET


def candidate_models() -> dict:
    """The classifiers to compare, each behind a StandardScaler pipeline."""
    return {
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "SVM (linear)": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    for name, clf in candidate_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "cv_roc_auc": round(cv_auc, 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        })

    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def plot(results: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#3d5a80" if "SVM" in m else "#98c1d9" for m in results["model"]]
    ax.barh(results["model"], results["cv_roc_auc"], color=colors)
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated ROC-AUC")
    ax.set_title("Model Comparison (SVM highlighted)")
    for i, v in enumerate(results["cv_roc_auc"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    results = evaluate()
    print("Model comparison (5-fold CV ROC-AUC + held-out test):\n")
    print(results.to_string(index=False))
    plot(results)
    print(f"\n✓ Chart saved to {REPORTS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
