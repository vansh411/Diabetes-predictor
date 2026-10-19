"""
train.py
Full XGBoost training pipeline with Optuna hyperparameter tuning,
evaluation metrics, and SHAP explainability generation.

Usage:
    python train.py
"""

import os
import json
import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocess import prepare_pipeline
from utils.explain import get_explainer, plot_summary


# ─────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  🏥 Diabetes Risk Predictor — Training")
print("="*50 + "\n")

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_pipeline()

# Class imbalance weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"⚖️  Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}\n")


# ─────────────────────────────────────────────
# 2. Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "gamma":              trial.suggest_float("gamma", 0, 5),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3),
        "scale_pos_weight":   scale_pos_weight,
        "use_label_encoder":  False,
        "eval_metric":        "auc",
        "random_state":       42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)


print("🔍 Running Optuna hyperparameter search (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
best_params["scale_pos_weight"] = scale_pos_weight
best_params["use_label_encoder"] = False
best_params["eval_metric"] = "auc"
best_params["random_state"] = 42

print(f"✅ Best ROC-AUC: {study.best_value:.4f}")
print(f"   Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()}, indent=4)}\n")


# ─────────────────────────────────────────────
# 3. Train Final Model
# ─────────────────────────────────────────────
print("🏋️  Training final model with best hyperparameters...")
model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Save model
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_diabetes.json")
joblib.dump(best_params, "models/best_params.pkl")
print("✅ Model saved → models/xgb_diabetes.json\n")


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc   = roc_auc_score(y_test, y_prob)
f1    = f1_score(y_test, y_pred)
acc   = accuracy_score(y_test, y_pred)

print("📈 Evaluation Results:")
print(f"   ROC-AUC  : {auc:.4f}")
print(f"   F1 Score : {f1:.4f}")
print(f"   Accuracy : {acc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Save metrics
metrics = {"roc_auc": round(auc, 4), "f1": round(f1, 4), "accuracy": round(acc, 4)}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# ─────────────────────────────────────────────
# 5. Plots — Confusion Matrix & ROC Curve
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
axes[0].set_title("Confusion Matrix", fontsize=13)
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve", fontsize=13)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("models/evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Evaluation plots saved → models/evaluation_plots.png")


# ─────────────────────────────────────────────
# 6. SHAP Explainability
# ─────────────────────────────────────────────
print("\n🔍 Generating SHAP explainability plots...")
explainer = get_explainer(model, X_train)
joblib.dump(explainer, "models/shap_explainer.pkl")

plot_summary(explainer, X_test, feature_names, save_path="models/shap_summary.png")

print("\n" + "="*50)
print("  ✅ Training Complete!")
print("="*50)
print("\nNext step → launch the web app:")
print("  streamlit run app/streamlit_app.py\n")
