# Diabetes Health Indicators — Model Building

## Overview
This project builds and evaluates machine learning models to predict diabetes risk 
using the BRFSS2015 (Behavioral Risk Factor Surveillance System) dataset.

## Dataset
- **Source:** [Diabetes Health Indicators Dataset (BRFSS2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) 
  — originally collected by the CDC's Behavioral Risk Factor Surveillance System, made 
  available on Kaggle.
- **File:** `Data/diabetes_012_health_indicators_BRFSS2015.csv`
- **License:** Open-source / publicly available for research and educational use.

## Approach
- **Data split:** 70% train / 15% validation / 15% test, stratified on the target class.
- **Models compared:**
  - Logistic Regression (baseline)
  - LightGBM
  - XGBoost
- **Primary evaluation metric:** Macro F1-score (to account for class imbalance).

## Project Structure
```
Project/
├── Data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── train_evaluate.py       # Main training and evaluation script
├── Output/                 # Model outputs, figures, results (excluded from repo)
└── README.md
```
## How to Run
```bash
python train_evaluate.py
```

## Notes
This project was developed as part of coursework at CAMTECH.
