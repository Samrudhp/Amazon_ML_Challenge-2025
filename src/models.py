"""
Model training: LightGBM and Ridge ensemble
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error

def smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def train_lightgbm_cv(
    X_train, y_train,
    params, num_rounds=2000, early_stopping_rounds=100,
    n_folds=5, random_state=42
):
    """Train LightGBM with stratified K-fold cross-validation"""
    print(f"Training LightGBM with {n_folds}-fold CV...")
    
    # Create stratified folds based on log(price) quantiles
    y_log = np.log1p(y_train)
    bins = pd.qcut(y_log, q=n_folds, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    oof_predictions = np.zeros(len(X_train))
    models = []
    feature_importance = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, bins)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Predict
        oof_predictions[val_idx] = model.predict(X_fold_val)
        
        # Calculate fold score
        fold_smape = smape(y_fold_val, oof_predictions[val_idx])
        fold_scores.append(fold_smape)
        print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}")
        
        models.append(model)
        feature_importance.append(model.feature_importance(importance_type='gain'))
    
    # Overall OOF score
    oof_smape = smape(y_train, oof_predictions)
    print(f"\n=== Overall OOF SMAPE: {oof_smape:.4f} ===")
    print(f"Fold scores: {fold_scores}")
    print(f"Std: {np.std(fold_scores):.4f}")
    
    # Average feature importance
    avg_importance = np.mean(feature_importance, axis=0)
    
    return models, oof_predictions, oof_smape, avg_importance

def predict_lightgbm_ensemble(models, X_test):
    """Predict using ensemble of LightGBM models"""
    predictions = np.zeros(len(X_test))
    
    for model in models:
        predictions += model.predict(X_test)
    
    predictions /= len(models)
    
    return predictions

def train_ridge_meta(oof_predictions, additional_features, y_train, alpha=1.0):
    """Train Ridge meta-model on OOF predictions"""
    print(f"\nTraining Ridge meta-model (alpha={alpha})...")
    
    # Combine OOF predictions with additional features
    X_meta = np.column_stack([oof_predictions.reshape(-1, 1), additional_features])
    
    # Train Ridge
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_meta, y_train)
    
    # Calculate meta-model score
    meta_predictions = ridge.predict(X_meta)
    meta_smape = smape(y_train, meta_predictions)
    print(f"Ridge meta-model SMAPE: {meta_smape:.4f}")
    
    return ridge

def predict_ridge_meta(ridge, lgb_predictions, additional_features):
    """Predict using Ridge meta-model"""
    X_meta = np.column_stack([lgb_predictions.reshape(-1, 1), additional_features])
    return ridge.predict(X_meta)

def train_full_pipeline(
    X_train, y_train,
    lgb_params, num_rounds, early_stopping_rounds,
    n_folds, ridge_alpha, random_state=42
):
    """Train full pipeline: LightGBM + Ridge"""
    
    # Convert y to log scale
    y_train_log = np.log1p(y_train)
    
    # Train LightGBM
    lgb_models, oof_preds_log, oof_smape, feature_importance = train_lightgbm_cv(
        X_train, y_train_log,
        lgb_params, num_rounds, early_stopping_rounds,
        n_folds, random_state
    )
    
    # Select features for meta-model (RAG features work well)
    # We'll use the last 9 features which should be RAG features
    meta_features = X_train[:, -9:]
    
    # Train Ridge meta-model
    ridge_model = train_ridge_meta(
        oof_preds_log, meta_features, y_train_log, ridge_alpha
    )
    
    return lgb_models, ridge_model, feature_importance

def predict_full_pipeline(lgb_models, ridge_model, X_test):
    """Predict using full pipeline"""
    
    # LightGBM predictions
    lgb_preds_log = predict_lightgbm_ensemble(lgb_models, X_test)
    
    # Ridge meta predictions
    meta_features = X_test[:, -9:]
    ridge_preds_log = predict_ridge_meta(ridge_model, lgb_preds_log, meta_features)
    
    # Convert back from log scale
    predictions = np.expm1(ridge_preds_log)
    
    return predictions
