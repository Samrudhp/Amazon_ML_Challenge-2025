"""
Model training: LightGBM and Ridge ensemble, plus Deep MLP Fusion Network
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import os

class PriceDataset(Dataset):
    """PyTorch Dataset for price prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DeepMLPFusion(nn.Module):
    """Deep MLP for multimodal price prediction"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3):
        super(DeepMLPFusion, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)

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
    import lightgbm as lgb
    
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
    n_folds, ridge_alpha, random_state=42,
    use_deep_mlp=False, mlp_hidden_dims=[512, 256, 128, 64], mlp_dropout_rate=0.3,
    mlp_learning_rate=1e-3, mlp_batch_size=1024, mlp_num_epochs=100, mlp_patience=10
):
    """Train full pipeline: LightGBM + Ridge or Deep MLP"""
    
    if use_deep_mlp:
        print("Using Deep MLP pipeline...")
        # Train Deep MLP directly (no meta-model needed)
        mlp_models, oof_preds_log, oof_smape, scaler = train_deep_mlp_cv(
            X_train, y_train,
            hidden_dims=mlp_hidden_dims, dropout_rate=mlp_dropout_rate,
            learning_rate=mlp_learning_rate, batch_size=mlp_batch_size, 
            num_epochs=mlp_num_epochs, early_stopping_patience=mlp_patience,
            n_folds=n_folds, random_state=random_state
        )
        
        return mlp_models, None, None, scaler  # No ridge model or feature importance for MLP
    
    else:
        print("Using LightGBM + Ridge pipeline...")
        # Convert y to log scale
        y_train_log = np.log1p(y_train)
        
        # Train LightGBM
        lgb_models, oof_preds_log, oof_smape, feature_importance = train_lightgbm_cv(
            X_train, y_train_log,
            lgb_params, num_rounds, early_stopping_rounds,
            n_folds, random_state
        )
        
        # Select features for meta-model (RAG features work well)
        # We'll use the last 22 features which should be advanced RAG features
        meta_features = X_train[:, -22:]
        
        # Train Ridge meta-model
        ridge_model = train_ridge_meta(
            oof_preds_log, meta_features, y_train_log, ridge_alpha
        )
        
        return lgb_models, ridge_model, feature_importance, None  # No scaler for LightGBM

def predict_full_pipeline(lgb_models, ridge_model, X_test, use_deep_mlp=False, mlp_batch_size=1024, scaler=None):
    """Predict using full pipeline"""
    
    if use_deep_mlp:
        # Scale test features using the same scaler from training
        X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
        # Use Deep MLP prediction
        predictions = predict_deep_mlp_ensemble(lgb_models, X_test_scaled, batch_size=mlp_batch_size)
        return predictions
    else:
        # LightGBM predictions
        lgb_preds_log = predict_lightgbm_ensemble(lgb_models, X_test)
        
        # Ridge meta predictions
        meta_features = X_test[:, -22:]  # Use advanced RAG features
        ridge_preds_log = predict_ridge_meta(ridge_model, lgb_preds_log, meta_features)
        
        # Convert back from log scale
        ridge_preds_log_clipped = np.clip(ridge_preds_log, -5, 10)
        predictions = np.expm1(ridge_preds_log_clipped)
        
        return predictions

def train_deep_mlp_cv(
    X_train, y_train,
    hidden_dims=[512, 256, 128, 64], dropout_rate=0.3,
    learning_rate=1e-3, batch_size=1024, num_epochs=100,
    early_stopping_patience=10, n_folds=5, random_state=42
):
    """Train Deep MLP with stratified K-fold cross-validation"""
    print(f"Training Deep MLP with {n_folds}-fold CV...")
    print(f"Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, hidden_dims))} -> 1")
    
    # Scale features to prevent numerical instability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"Features scaled - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
    
    # Create stratified folds based on log(price) quantiles
    y_log = np.log1p(y_train)
    bins = pd.qcut(y_log, q=n_folds, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    oof_predictions = np.zeros(len(X_train))
    models = []
    fold_scores = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, bins)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_log[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_log[val_idx]
        
        # Create datasets
        train_dataset = PriceDataset(X_fold_train, y_fold_train)
        val_dataset = PriceDataset(X_fold_val, y_fold_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = DeepMLPFusion(X_train.shape[1], hidden_dims, dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = F.mse_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(X_batch)
            
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = F.mse_loss(outputs, y_batch)
                    val_loss += loss.item() * len(X_batch)
                    
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(y_batch.cpu().numpy().flatten())
            
            val_loss /= len(val_dataset)
            scheduler.step(val_loss)
            
            # Calculate SMAPE on original scale
            val_predictions_clipped = np.clip(val_predictions, -5, 10)  # Prevent exp overflow
            val_preds_orig = np.expm1(val_predictions_clipped)
            val_targets_orig = np.expm1(val_targets)
            val_smape = smape(val_targets_orig, val_preds_orig)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val SMAPE: {val_smape:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and predict
        model.load_state_dict(best_model_state)
        model.eval()
        
        val_dataset_full = PriceDataset(X_fold_val, y_fold_val)
        val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size, shuffle=False)
        
        fold_predictions = []
        with torch.no_grad():
            for X_batch, _ in val_loader_full:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                fold_predictions.extend(outputs.cpu().numpy().flatten())
        
        # Store predictions (still in log scale)
        oof_predictions[val_idx] = fold_predictions
        
        # Calculate fold score
        fold_predictions_clipped = np.clip(fold_predictions, -5, 10)
        fold_preds_orig = np.expm1(fold_predictions_clipped)
        fold_targets_orig = np.expm1(y_fold_val)
        fold_smape = smape(fold_targets_orig, fold_preds_orig)
        fold_scores.append(fold_smape)
        print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}")
        
        models.append(model.cpu())  # Move to CPU for storage
    
    # Overall OOF score
    oof_predictions_clipped = np.clip(oof_predictions, -5, 10)
    oof_predictions_orig = np.expm1(oof_predictions_clipped)
    y_train_orig = np.expm1(y_log)
    oof_smape = smape(y_train_orig, oof_predictions_orig)
    print(f"\n=== Overall OOF SMAPE: {oof_smape:.4f} ===")
    print(f"Fold scores: {fold_scores}")
    print(f"Std: {np.std(fold_scores):.4f}")
    
    return models, oof_predictions, oof_smape, scaler

def predict_deep_mlp_ensemble(models, X_test, batch_size=1024):
    """Predict using ensemble of Deep MLP models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = PriceDataset(X_test, np.zeros(len(X_test)))  # Dummy y values
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = np.zeros(len(X_test))
    
    for model in models:
        model.to(device)
        model.eval()
        
        fold_predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                fold_predictions.extend(outputs.cpu().numpy().flatten())
        
        predictions += np.array(fold_predictions)
        model.cpu()  # Move back to CPU
    
    predictions /= len(models)
    
    # Convert from log scale with clipping to prevent overflow
    predictions = np.clip(predictions, -5, 10)
    predictions = np.expm1(predictions)
    
    return predictions

def load_models(model_path=None):
    """Load trained models from pickle file"""
    if model_path is None:
        from config import CACHE_DIR
        model_path = os.path.join(CACHE_DIR, 'trained_models.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        models_dict = pickle.load(f)
    
    return (models_dict['lgb_models'], models_dict['ridge_model'], 
            models_dict.get('feature_importance', None), models_dict.get('use_deep_mlp', False),
            models_dict.get('scaler', None))
