"""
notebooks/eda.py
Exploratory Data Analysis for the Pima Diabetes Dataset.
Run with: python notebooks/eda.py
(Or open as a Jupyter notebook by converting with: jupytext --to notebook notebooks/eda.py)
"""

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import load_data, clean_data, engineer_features, FEATURE_COLS, TARGET_COL

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("notebooks/figures", exist_ok=True)

# ── 1. Load & clean ──────────────────────────────────────
df_raw  = load_data()
df      = clean_data(df_raw)
df      = engineer_features(df)
print(df.describe().round(2).to_string())

# ── 2. Class distribution ────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
df[TARGET_COL].value_counts().plot(kind="bar", ax=ax, color=["#2563EB", "#DC2626"], edgecolor="white")
ax.set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/figures/class_dist.png", dpi=150)
plt.close()
print("✅ Saved class_dist.png")

# ── 3. Feature distributions by class ───────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    for label, color in [(0, "#2563EB"), (1, "#DC2626")]:
        subset = df[df[TARGET_COL] == label][col]
        axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=str(label), edgecolor="none")
    axes[i].set_title(col, fontsize=11)
    axes[i].legend(["No Diabetes", "Diabetes"], fontsize=8)
plt.suptitle("Feature Distributions by Class", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("notebooks/figures/feature_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved feature_dist.png")

# ── 4. Correlation heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("notebooks/figures/correlation.png", dpi=150)
plt.close()
print("✅ Saved correlation.png")

# ── 5. Glucose vs BMI scatter ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for label, color, name in [(0, "#2563EB", "No Diabetes"), (1, "#DC2626", "Diabetes")]:
    subset = df[df[TARGET_COL] == label]
    ax.scatter(subset["Glucose"], subset["BMI"], alpha=0.4, color=color,
               label=name, s=30, edgecolors="none")
ax.set_xlabel("Glucose (mg/dL)")
ax.set_ylabel("BMI")
ax.set_title("Glucose vs BMI by Class")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/glucose_bmi.png", dpi=150)
plt.close()
print("✅ Saved glucose_bmi.png")

print("\n📊 EDA complete. Figures saved in notebooks/figures/")
