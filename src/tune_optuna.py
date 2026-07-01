"""
HireSense AI — Optuna hyperparameter tuning.

A smarter alternative to GridSearchCV: Optuna uses Bayesian optimization
(TPE) to search the SVM hyperparameter space more efficiently, exploring
continuous ranges for C and gamma instead of a fixed grid.

Returns a fitted pipeline (best params) ready to be calibrated in train.py.
"""
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_pipe(params: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=RANDOM_STATE, **params)),
    ])


def tune(X, y, n_trials: int = 40) -> tuple[Pipeline, dict, float]:
    """Search SVM hyperparameters with Optuna; return (fitted pipe, params, cv_auc)."""

    def objective(trial: optuna.Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
        pipe = _build_pipe(params)
        return cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_pipe = _build_pipe(best_params).fit(X, y)
    return best_pipe, best_params, study.best_value


def main() -> None:
    import pandas as pd
    from config import DATASET_PATH, NUMERIC_FEATURES, TARGET

    df = pd.read_csv(DATASET_PATH)
    _, params, auc = tune(df[NUMERIC_FEATURES], df[TARGET], n_trials=40)
    print(f"Best params : {params}")
    print(f"CV ROC-AUC  : {auc:.4f}")


if __name__ == "__main__":
    main()
