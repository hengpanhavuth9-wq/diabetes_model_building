import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import optuna
import xgboost as xgb
import lightgbm as lgb
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    print("Step 1: Load and Validate Data")
    data_path = 'Data/diabetes_012_health_indicators_BRFSS2015.csv'
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    assert df.shape == (253680, 22), f"Unexpected shape {df.shape}"
    assert df.isnull().sum().sum() == 0, "Missing values found!"
    
    print("\nStep 2: Feature Engineering (HealthStatusScore)")
    
    print("\nStep 3: Train/Validation/Test Split")
    # First split off 15% as the test set
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    # 85% remaining, we want 70% of total for train, 15% of total for val
    val_size_fraction = 15.0 / 85.0
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_fraction, stratify=y_temp, random_state=42
    )
    
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")
    
    # Feature Engineering logic (fitting scaler on train)
    scaler_hs = MinMaxScaler()
    hs_cols = ['GenHlth', 'PhysHlth', 'MentHlth']
    
    def add_health_status(X_df, fit=False):
        X_out = X_df.copy()
        if fit:
            scaler_hs.fit(X_out[hs_cols])
        scaled_hs = scaler_hs.transform(X_out[hs_cols])
        # Weights: 0.4, 0.3, 0.3
        X_out['HealthStatusScore'] = (scaled_hs[:, 0] * 0.4) + (scaled_hs[:, 1] * 0.3) + (scaled_hs[:, 2] * 0.3)
        return X_out
        
    X_train = add_health_status(X_train, fit=True)
    X_val = add_health_status(X_val, fit=False)
    X_test = add_health_status(X_test, fit=False)
    
    # Also create the combined Train+Val set for Optuna CV (Step 3)
    X_temp = pd.concat([X_train, X_val], axis=0)
    y_temp = pd.concat([y_train, y_val], axis=0)
    
    print("\nStep 4: Handle Class Imbalance")
    # Computed dynamically during model training
    sample_weight_train = compute_sample_weight('balanced', y_train)
    sample_weight_val = compute_sample_weight('balanced', y_val)
    sample_weight_temp = compute_sample_weight('balanced', y_temp)

    print("\nStep 5: Preprocessing per Model")
    # For LR, we need a standard scaler. Fit on train only.
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_val_lr = scaler_lr.transform(X_val)
    X_test_lr = scaler_lr.transform(X_test)
    X_temp_lr = scaler_lr.transform(X_temp)
    
    output_dir = 'Output'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Logistic Regression Tuning ---
    print("\n--- Tuning Logistic Regression ---")
    def objective_lr(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        
        if solver == 'lbfgs':
            penalty = 'l2'
        else:
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        
        model = LogisticRegression(
            solver=solver, penalty=penalty, C=C, 
            class_weight='balanced', max_iter=200, random_state=42, n_jobs=-1
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_temp_lr, y_temp):
            X_cv_train, X_cv_val = X_temp_lr[train_idx], X_temp_lr[val_idx]
            y_cv_train, y_cv_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
            
            model.fit(X_cv_train, y_cv_train)
            preds = model.predict(X_cv_val)
            cv_scores.append(f1_score(y_cv_val, preds, average='macro'))
            
        return np.mean(cv_scores)

    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(objective_lr, n_trials=20) 
    print("Best LR params:", study_lr.best_params)
    
    # --- XGBoost Tuning ---
    print("\n--- Tuning XGBoost ---")
    def objective_xgb(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42,
            'tree_method': 'hist',
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_temp, y_temp):
            X_cv_train, X_cv_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_cv_train, y_cv_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
            sw_cv_train = compute_sample_weight('balanced', y_cv_train)
            
            dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train, weight=sw_cv_train)
            dval = xgb.DMatrix(X_cv_val, label=y_cv_val)
            
            model = xgb.train(
                params, dtrain, num_boost_round=300, evals=[(dval, 'val')],
                early_stopping_rounds=15, verbose_eval=False
            )
            
            preds_prob = model.predict(dval)
            preds = np.argmax(preds_prob, axis=1)
            cv_scores.append(f1_score(y_cv_val, preds, average='macro'))
            
        return np.mean(cv_scores)
        
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=30)
    print("Best XGB params:", study_xgb.best_params)

    # --- LightGBM Tuning ---
    print("\n--- Tuning LightGBM ---")
    def objective_lgb(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced'
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_temp, y_temp):
            X_cv_train, X_cv_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_cv_train, y_cv_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params, n_estimators=300)
            model.fit(
                X_cv_train, y_cv_train, 
                eval_set=[(X_cv_val, y_cv_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
            
            preds = model.predict(X_cv_val)
            cv_scores.append(f1_score(y_cv_val, preds, average='macro'))
            
        return np.mean(cv_scores)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=30)
    print("Best LGB params:", study_lgb.best_params)
    
    print("\nStep 6 & 7: Final Training and Evaluation")
    
    # 1. Logistic Regression Refit
    best_lr_params = study_lr.best_params
    solver = best_lr_params['solver']
    penalty = 'l2' if solver == 'lbfgs' else best_lr_params.get('penalty', 'l2')
    
    final_lr = LogisticRegression(
        solver=solver, penalty=penalty, C=best_lr_params['C'],
        class_weight='balanced', max_iter=300, random_state=42, n_jobs=-1
    )
    final_lr.fit(X_temp_lr, y_temp)
    joblib.dump(final_lr, f"{output_dir}/model_lr.joblib")
    
    # 2. XGBoost Refit
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss', 'random_state': 42, 'tree_method': 'hist'})
    dtrain_full = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
    dval_full = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    final_xgb = xgb.train(
        best_xgb_params, dtrain_full, num_boost_round=500, evals=[(dval_full, 'val')],
        early_stopping_rounds=30, verbose_eval=50
    )
    joblib.dump(final_xgb, f"{output_dir}/model_xgb.joblib")
    
    # 3. LightGBM Refit
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'random_state': 42, 'verbose': -1, 'class_weight': 'balanced'})
    final_lgb = lgb.LGBMClassifier(**best_lgb_params, n_estimators=500, importance_type='gain')
    final_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
    )
    joblib.dump(final_lgb, f"{output_dir}/model_lgb.joblib")
    
    # Evaluation
    def eval_model(name, y_true, preds, preds_prob):
        print(f"\n--- {name} Evaluation ---")
        print(classification_report(y_true, preds))
        
        cm = confusion_matrix(y_true, preds)
        cm_norm = confusion_matrix(y_true, preds, normalize='true')
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Raw Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues')
        plt.title(f"{name} - Normalized")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.savefig(f"{output_dir}/cm_{name}.png")
        plt.close()
        
        roc_auc = roc_auc_score(y_true, preds_prob, multi_class='ovr')
        print(f"One-vs-Rest ROC AUC: {roc_auc:.4f}")
        
        f1_macro = f1_score(y_true, preds, average='macro')
        precision_macro = precision_score(y_true, preds, average='macro')
        recall_macro = recall_score(y_true, preds, average='macro')
        acc = accuracy_score(y_true, preds)
        
        f1_class = f1_score(y_true, preds, average=None)
        
        return {
            'Model': name,
            'Macro F1': f1_macro,
            'Macro Precision': precision_macro,
            'Macro Recall': recall_macro,
            'Accuracy': acc,
            'AUC-ROC': roc_auc,
            'F1 Class 0': f1_class[0],
            'F1 Class 1 (Prediabetes)': f1_class[1],
            'F1 Class 2': f1_class[2]
        }
        
    preds_lr = final_lr.predict(X_test_lr)
    preds_prob_lr = final_lr.predict_proba(X_test_lr)
    res_lr = eval_model("LogisticRegression", y_test, preds_lr, preds_prob_lr)
    
    preds_prob_xgb = final_xgb.predict(dtest)
    preds_xgb = np.argmax(preds_prob_xgb, axis=1)
    res_xgb = eval_model("XGBoost", y_test, preds_xgb, preds_prob_xgb)
    
    preds_prob_lgb = final_lgb.predict_proba(X_test)
    preds_lgb = final_lgb.predict(X_test)
    res_lgb = eval_model("LightGBM", y_test, preds_lgb, preds_prob_lgb)
    
    # Feature Importance Plots
    # LR Coefficients
    coef_df = pd.DataFrame(final_lr.coef_, columns=X_train.columns, index=['Class 0', 'Class 1', 'Class 2'])
    coef_df.to_csv(f"{output_dir}/lr_coefficients.csv")
    
    # XGB Feature Importance (gain)
    xgb_imp = final_xgb.get_score(importance_type='gain')
    xgb_imp_df = pd.DataFrame(list(xgb_imp.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Gain', y='Feature', data=xgb_imp_df.head(20))
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feat_imp_xgb.png")
    plt.close()
    
    # LGBM Feature Importance (gain)
    lgb_imp = final_lgb.feature_importances_
    lgb_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Gain': lgb_imp}).sort_values('Gain', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Gain', y='Feature', data=lgb_imp_df.head(20))
    plt.title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feat_imp_lgb.png")
    plt.close()
    
    # Step 8: Final Comparison
    summary_df = pd.DataFrame([res_lr, res_xgb, res_lgb])
    print("\n--- Final Comparison ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    best_model_idx = summary_df['Macro F1'].idxmax()
    best_model = summary_df.loc[best_model_idx, 'Model']
    print(f"\nBest Model based on Macro F1 is: {best_model}")
    print(f"Class 1 (Prediabetes) Performance: LR={res_lr['F1 Class 1 (Prediabetes)']:.4f}, XGB={res_xgb['F1 Class 1 (Prediabetes)']:.4f}, LGB={res_lgb['F1 Class 1 (Prediabetes)']:.4f}")

if __name__ == "__main__":
    main()
