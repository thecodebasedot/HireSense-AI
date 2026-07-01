"""
HireSense AI — deep learning model (neural network).

An alternative to the core SVM: a multi-layer perceptron (feed-forward neural
network) trained with backpropagation. SVM remains the project's primary
algorithm; this module lets you train, evaluate, and compare a deep model on
the same data and schema.

If PyTorch is available a torch MLP (with dropout + early stopping) is used;
otherwise scikit-learn's MLPClassifier (also a backprop-trained neural net) is
the fallback. Either way the saved bundle matches the SVM bundle format, so
predict / explain / conformal / fairness all work unchanged when it is made
the current model.

    python src/deep_model.py                 # train + evaluate
    python src/deep_model.py --make-current  # also set as the active model
"""
import argparse
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    DATASET_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiresense.deep")

DEEP_MODEL_PATH = MODEL_DIR / "hiresense_mlp.joblib"
HIDDEN_LAYERS = (128, 64, 32)

try:
    import torch  # noqa: F401
    from torch_mlp import TorchMLP
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def build_model() -> Pipeline:
    """StandardScaler + a deep neural network (torch if available)."""
    if _HAS_TORCH:
        clf = TorchMLP()
        backend = f"PyTorch {torch.__version__}"
    else:  # pragma: no cover
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
            alpha=1e-4, batch_size=64, max_iter=300, early_stopping=True,
            n_iter_no_change=15, random_state=RANDOM_STATE,
        )
        backend = "scikit-learn MLP"
    logger.info("Deep model backend: %s", backend)
    return Pipeline([("scaler", StandardScaler()), ("mlp", clf)])


def train_and_evaluate(make_current: bool = False) -> dict:
    df = pd.read_csv(DATASET_PATH)
    X, y = df[NUMERIC_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    print(f"Training deep MLP {HIDDEN_LAYERS} ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, y_pred))
    test_auc = float(roc_auc_score(y_test, y_proba))

    print("\n=== Test set performance (deep MLP) ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"ROC-AUC  : {test_auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Shortlisted"]))

    backend = f"PyTorch {torch.__version__}" if _HAS_TORCH else "scikit-learn MLP"
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle = {
        "model": model,
        "version": "1.0.0-mlp",
        "trained_at": trained_at,
        "algorithm": f"Neural Network ({backend})",
        "tuning": "fixed",
        "features": NUMERIC_FEATURES,
        "best_params": {"hidden_layer_sizes": HIDDEN_LAYERS},
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
    }

    DEEP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, DEEP_MODEL_PATH)
    print(f"\n✓ Deep model saved to {DEEP_MODEL_PATH}")

    if make_current:
        joblib.dump(bundle, MODEL_PATH)
        print(f"✓ Set as the active model at {MODEL_PATH}")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deep learning (MLP) model.")
    parser.add_argument("--make-current", action="store_true",
                        help="Also set the deep model as the active model.")
    args = parser.parse_args()
    train_and_evaluate(make_current=args.make_current)


if __name__ == "__main__":
    main()
