"""
HireSense AI — train a Support Vector Machine to screen job applicants.

Pipeline: StandardScaler -> SVC. SVMs are distance-based, so scaling the
features is essential. GridSearchCV tunes the kernel and regularisation.
"""
import argparse

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    DATASET_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def load_data(path=DATASET_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/generate_data.py` first."
        )
    df = pd.read_csv(path)
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]
    return X, y


def build_search() -> GridSearchCV:
    """Build the StandardScaler + SVC pipeline and hyperparameter grid.

    Tuning uses SVC's decision_function via the roc_auc scorer, so we do not
    need probability estimates here. Calibrated probabilities are added to
    the winning model afterwards (see `calibrate`).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=RANDOM_STATE)),
    ])

    param_grid = [
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.01, 0.1],
        },
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10],
        },
    ]

    return GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )


def calibrate(best_pipeline: Pipeline, X_train, y_train) -> CalibratedClassifierCV:
    """Calibrate the winning pipeline so it supports predict_proba.

    SVC(probability=True) is deprecated in scikit-learn 1.9; the recommended
    replacement is wrapping the estimator in CalibratedClassifierCV.
    """
    calibrated = CalibratedClassifierCV(best_pipeline, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HireSense SVM model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    print("Tuning SVM hyperparameters via 5-fold cross-validation...")
    search = build_search()
    search.fit(X_train, y_train)

    print(f"\nBest params : {search.best_params_}")
    print(f"CV ROC-AUC  : {search.best_score_:.4f}")

    # Wrap the tuned pipeline so it exposes calibrated predict_proba.
    print("Calibrating probabilities...")
    model = calibrate(search.best_estimator_, X_train, y_train)

    # --- Evaluate on the held-out test set ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Test set performance ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Shortlisted"]))

    # --- Persist the fitted pipeline plus metadata ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": NUMERIC_FEATURES,
            "best_params": search.best_params_,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        MODEL_PATH,
    )
    print(f"\n✓ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
