"""
utils/preprocess.py
Data loading, cleaning, and feature engineering for the Pima Diabetes dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Features where 0 is biologically impossible → treat as missing
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"


def load_data() -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset.
    Tries ucimlrepo first, falls back to a bundled CSV if unavailable.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=34)  # Pima Indians Diabetes
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df.columns = FEATURE_COLS + [TARGET_COL]
        print("✅ Dataset loaded from UCI ML Repository")
    except Exception:
        # Fallback: well-known Kaggle mirror URL embedded as inline data
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = FEATURE_COLS + [TARGET_COL]
        df = pd.read_csv(url, header=None, names=cols)
        print("✅ Dataset loaded from fallback URL")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zeros with NaN, then impute with median."""
    df = df.copy()

    for col in ZERO_AS_NAN_COLS:
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0:
            print(f"  → {col}: replacing {n_zeros} zeros with median")
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few useful derived features."""
    df = df.copy()
    # Insulin resistance proxy
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    # Age group buckets
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100],
                              labels=[0, 1, 2, 3]).astype(int)
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train/test (stratified) and fit a StandardScaler on train.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_scaler(scaler, path: str = "models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"✅ Scaler saved → {path}")


def load_scaler(path: str = "models/scaler.pkl"):
    return joblib.load(path)


def prepare_pipeline(save_artifacts: bool = True):
    """Full preprocessing pipeline. Returns train/test splits + metadata."""
    df = load_data()
    print(f"📊 Raw data shape: {df.shape} | Positive rate: {df[TARGET_COL].mean():.2%}")

    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    if save_artifacts:
        save_scaler(scaler)

    print(f"✅ Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, feature_names
