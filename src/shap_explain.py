"""
HireSense AI — SHAP explainability.

SHAP (SHapley Additive exPlanations) gives game-theoretic feature attributions
that are more principled than permutation importance: each feature's
contribution to a prediction is its average marginal effect across all
feature orderings.

We use a model-agnostic KernelExplainer over the calibrated pipeline's
predict_proba, with a small background sample for speed.

    python src/shap_explain.py            # global summary -> reports/shap_summary.png
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import DATASET_PATH, NUMERIC_FEATURES, RANDOM_STATE, REPORTS_DIR
from predict import load_model


def _shortlist_proba(model):
    """Return a function giving P(shortlist) for a feature matrix."""
    return lambda data: model.predict_proba(pd.DataFrame(data, columns=NUMERIC_FEATURES))[:, 1]


def build_explainer(model, background: pd.DataFrame):
    """KernelExplainer over the model's shortlist probability."""
    bg = shap.sample(background[NUMERIC_FEATURES], min(50, len(background)),
                     random_state=RANDOM_STATE)
    return shap.KernelExplainer(_shortlist_proba(model), bg)


def global_summary(model, X: pd.DataFrame, n_explain: int = 60) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (global importance)."""
    explainer = build_explainer(model, X)
    sample = shap.sample(X[NUMERIC_FEATURES], min(n_explain, len(X)),
                         random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample, silent=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": NUMERIC_FEATURES, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot = summary.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(plot["feature"], plot["mean_abs_shap"], color="#5f0f40")
    ax.set_title("SHAP Feature Importance (mean |SHAP|)")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_summary.png", dpi=120)
    plt.close(fig)

    return summary


def explain_instance(model, row: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """SHAP contribution of each feature for a single applicant."""
    explainer = build_explainer(model, background)
    values = explainer.shap_values(row[NUMERIC_FEATURES], silent=True)[0]
    out = pd.DataFrame({
        "feature": NUMERIC_FEATURES,
        "value": row[NUMERIC_FEATURES].iloc[0].values,
        "shap_value": values,
    })
    out["abs"] = out["shap_value"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def main() -> None:
    model = load_model()["model"]
    df = pd.read_csv(DATASET_PATH)
    summary = global_summary(model, df)
    print("SHAP global feature importance:\n")
    for _, r in summary.iterrows():
        bar = "█" * max(1, int(r["mean_abs_shap"] * 200))
        print(f"  {r['feature']:22s} {r['mean_abs_shap']:.4f}  {bar}")
    print(f"\n✓ Plot saved to {REPORTS_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
