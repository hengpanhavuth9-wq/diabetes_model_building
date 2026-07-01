# Diabetes Health Indicators — Model Building

## Overview
This project builds and evaluates machine learning models to predict diabetes status 
(no diabetes / prediabetes / diabetes) using the BRFSS2015 (Behavioral Risk Factor 
Surveillance System) dataset. Three models are trained, tuned, and compared: 
Logistic Regression, XGBoost, and LightGBM.

## Dataset
- **Source:** [Diabetes Health Indicators Dataset (BRFSS2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) 
  — originally collected by the CDC's Behavioral Risk Factor Surveillance System, made 
  available on Kaggle.
- **File:** `Data/diabetes_012_health_indicators_BRFSS2015.csv`
- **Shape:** 253,680 rows × 22 columns (validated on load)
- **Target:** `Diabetes_012` — 3-class label (0 = no diabetes, 1 = prediabetes, 2 = diabetes)
- **License:** Open-source / publicly available for research and educational use.

## Pipeline

1. **Data validation** — checks expected shape and confirms no missing values.
2. **Feature engineering** — creates a `HealthStatusScore` from `GenHlth`, `PhysHlth`, 
   and `MentHlth`, min-max scaled and weighted (0.4 / 0.3 / 0.3).
3. **Train/Val/Test split** — 70% / 15% / 15%, stratified on the target, seed = 42.
4. **Class imbalance handling** — balanced sample weights (for XGBoost) and 
   `class_weight='balanced'` (for Logistic Regression and LightGBM).
5. **Preprocessing** — `StandardScaler` applied for Logistic Regression only; 
   tree-based models use raw features.
6. **Hyperparameter tuning** — Optuna, 5-fold stratified CV, optimizing macro F1:
   - Logistic Regression: 20 trials (C, solver, penalty)
   - XGBoost: 30 trials (depth, learning rate, regularization, subsampling)
   - LightGBM: 30 trials (num_leaves, depth, learning rate, regularization, subsampling)
7. **Final training** — best params refit on the full train+val set (train only for 
   tree models, with early stopping against the validation set).
8. **Evaluation** — classification report, confusion matrices (raw + normalized), 
   one-vs-rest ROC AUC, macro F1/precision/recall, accuracy, per-class F1 
   (with particular attention to Class 1 / prediabetes, the hardest class).
9. **Feature importance** — LR coefficients, XGBoost gain importance, LightGBM gain 
   importance, saved as CSV/plots.
10. **Model comparison** — all three models' metrics summarized and the best model 
    (by macro F1) is reported.

## Models & Evaluation Metric
- **Models:** Logistic Regression, XGBoost, LightGBM
- **Primary metric:** Macro F1-score (chosen to fairly weight the minority prediabetes class)
- **Secondary metrics:** Accuracy, macro precision/recall, one-vs-rest ROC AUC, per-class F1

## Project Structure
```
Project/
├── Data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── train_evaluate.py       # Full pipeline: EDA validation, feature engineering,
│                            # tuning, training, and evaluation
├── Output/                  # Trained models (.joblib), confusion matrices,
│                            # feature importance plots, comparison CSV (excluded from repo)
└── README.md
```
## Output Artifacts (in `Output/`, gitignored)
- `model_lr.joblib`, `model_xgb.joblib`, `model_lgb.joblib`
- `cm_<model>.png` — confusion matrices per model
- `feat_imp_xgb.png`, `feat_imp_lgb.png` — feature importance plots
- `lr_coefficients.csv` — Logistic Regression coefficients per class
- `model_comparison.csv` — final metric comparison across all models

## How to Run
```bash
python train_evaluate.py
```

**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, optuna, xgboost, lightgbm, joblib

## Notes
This project was developed as part of coursework at CAMTECH.
