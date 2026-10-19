# 🏥 Early Diabetes Risk Predictor

An end-to-end ML project using **XGBoost** to predict Type 2 diabetes risk, with **SHAP explainability** and a **Streamlit web app**.

## Project Structure

```
diabetes_predictor/
├── data/                   # Raw and processed datasets
├── models/                 # Saved model files
├── notebooks/              # EDA and experimentation
├── app/                    # Streamlit web app
│   └── streamlit_app.py
├── utils/                  # Helper modules
│   ├── preprocess.py
│   └── explain.py
├── train.py                # Full training pipeline
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Launch the web app
```bash
streamlit run app/streamlit_app.py
```

## Dataset
Uses the **Pima Indians Diabetes Dataset** — automatically downloaded via `sklearn` / `ucimlrepo` or from Kaggle.

Features:
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

Target: `Outcome` (0 = No Diabetes, 1 = Diabetes)

## Model
- **Algorithm**: XGBoost Classifier
- **Tuning**: Optuna hyperparameter optimization
- **Imbalance**: Handled via `scale_pos_weight`
- **Explainability**: SHAP values (global + per-patient)

## Results (expected)
| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.83 |
| F1 Score | ~0.72 |
| Accuracy | ~0.78 |
