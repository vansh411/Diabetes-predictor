"""
utils/explain.py
SHAP-based explainability helpers for the XGBoost diabetes model.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import os


def get_explainer(model, X_train: np.ndarray):
    """Create and return a SHAP TreeExplainer for the XGBoost model."""
    explainer = shap.TreeExplainer(model)
    return explainer


def plot_summary(explainer, X_test: np.ndarray, feature_names: list,
                 save_path: str = "models/shap_summary.png"):
    """
    Global SHAP summary bar + beeswarm plot.
    Shows which features matter most across the entire test set.
    """
    shap_values = explainer.shap_values(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    axes[0].set_title("Feature Importance (mean |SHAP|)", fontsize=13)

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False)
    axes[1].set_title("SHAP Value Distribution", fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP summary plot saved → {save_path}")
    return shap_values


def plot_single_patient(explainer, patient_array: np.ndarray,
                        feature_names: list, save_path: str = None):
    """
    Waterfall chart explaining a single patient's prediction.
    patient_array: shape (1, n_features) — already scaled
    """
    shap_values = explainer.shap_values(patient_array)
    expected_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=patient_array[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.title("Why this prediction? (Patient SHAP Breakdown)", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        return fig


def get_top_risk_factors(explainer, patient_array: np.ndarray,
                         feature_names: list, top_n: int = 3) -> list:
    """
    Return top N features driving risk UP for a single patient.
    Returns list of (feature_name, shap_value) tuples sorted by impact.
    """
    shap_values = explainer.shap_values(patient_array)[0]
    pairs = list(zip(feature_names, shap_values))
    # Sort by absolute impact descending
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    return pairs_sorted[:top_n]
