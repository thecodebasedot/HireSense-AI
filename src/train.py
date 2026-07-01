"""
HireSense AI — train a Support Vector Machine to screen job applicants.

Pipeline: StandardScaler -> SVC. SVMs are distance-based, so scaling the
features is essential. GridSearchCV tunes the kernel and regularisation.

Optional extras:
  --optuna         tune with Optuna (Bayesian) instead of GridSearchCV
  --no-registry    skip registering the model in the registry
MLflow tracking is used automatically when the `mlflow` package is installed.
"""
import argparse
import logging
from datetime import datetime, timezone

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
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from validation import validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiresense.train")

# MLflow is optional — training works with or without it.
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MLFLOW = False


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
    parser.add_argument("--optuna", action="store_true",
                        help="Tune with Optuna (Bayesian) instead of GridSearchCV.")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials.")
    parser.add_argument("--no-registry", action="store_true",
                        help="Do not register the model in the registry.")
    args = parser.parse_args()

    X, y = load_data()

    # Validate the training data up front.
    issues = validate(pd.concat([X], axis=1))
    if issues:
        logger.warning("Data validation issues: %s", issues)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    if args.optuna:
        print(f"Tuning SVM hyperparameters with Optuna ({args.trials} trials)...")
        from tune_optuna import tune
        best_pipe, best_params, cv_auc = tune(X_train, y_train, n_trials=args.trials)
        tuning = "optuna"
    else:
        print("Tuning SVM hyperparameters via GridSearchCV (5-fold CV)...")
        search = build_search()
        search.fit(X_train, y_train)
        best_pipe, best_params, cv_auc = (
            search.best_estimator_, search.best_params_, search.best_score_
        )
        tuning = "gridsearch"

    print(f"\nBest params : {best_params}")
    print(f"CV ROC-AUC  : {cv_auc:.4f}")

    # Wrap the tuned pipeline so it exposes calibrated predict_proba.
    print("Calibrating probabilities...")
    model = calibrate(best_pipe, X_train, y_train)

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

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_roc_auc = float(roc_auc_score(y_test, y_proba))

    # --- Persist the fitted pipeline plus metadata (versioned) ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle = {
        "model": model,
        "version": MODEL_VERSION,
        "trained_at": trained_at,
        "algorithm": "SVM",
        "tuning": tuning,
        "features": NUMERIC_FEATURES,
        "best_params": best_params,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info("Model v%s trained at %s saved to %s", MODEL_VERSION, trained_at, MODEL_PATH)
    print(f"\n✓ Model v{MODEL_VERSION} saved to {MODEL_PATH}")

    # --- Experiment tracking (MLflow, if available) ---
    if _HAS_MLFLOW:
        try:
            mlflow.set_experiment("hiresense")
            with mlflow.start_run():
                mlflow.log_params({f"param_{k}": v for k, v in best_params.items()})
                mlflow.log_param("tuning", tuning)
                mlflow.log_param("version", MODEL_VERSION)
                mlflow.log_metrics({
                    "cv_roc_auc": float(cv_auc),
                    "test_accuracy": test_accuracy,
                    "test_roc_auc": test_roc_auc,
                })
            logger.info("Logged run to MLflow (./mlruns)")
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow logging skipped: %s", exc)

    # --- Register in the model registry ---
    if not args.no_registry:
        from registry import register
        entry = register(bundle, make_current=True)
        print(f"✓ Registered as {entry['file']}")


if __name__ == "__main__":
    main()
