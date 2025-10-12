"""
Quick test of Deep MLP pipeline integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import MLP_HIDDEN_DIMS, MLP_DROPOUT_RATE, MLP_LEARNING_RATE, MLP_BATCH_SIZE, MLP_NUM_EPOCHS, MLP_EARLY_STOPPING_PATIENCE
from models import train_deep_mlp_cv, predict_deep_mlp_ensemble

def test_deep_mlp_integration():
    """Test Deep MLP pipeline with synthetic data"""
    print("Testing Deep MLP pipeline integration...")

    # Create synthetic multimodal features (547D total: 256 text + 256 image + 13 numeric + 22 RAG)
    np.random.seed(42)
    n_samples = 1000
    n_features = 547

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.exponential(100, n_samples).astype(np.float32)  # Price-like distribution

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Test Deep MLP training
    print("\nTraining Deep MLP...")
    models, oof_preds, oof_smape = train_deep_mlp_cv(
        X_train, y_train,
        hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rate=MLP_DROPOUT_RATE,
        learning_rate=MLP_LEARNING_RATE,
        batch_size=min(MLP_BATCH_SIZE, len(X_train)),  # Smaller batch for test
        num_epochs=MLP_NUM_EPOCHS,
        early_stopping_patience=MLP_EARLY_STOPPING_PATIENCE,
        n_folds=3, random_state=42
    )

    # Test prediction
    print("\nTesting prediction...")
    predictions = predict_deep_mlp_ensemble(models, X_test)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Predictions mean: {predictions.mean():.2f}")

    # Basic sanity check
    assert len(predictions) == len(X_test), "Prediction length mismatch"
    assert np.all(predictions > 0), "All predictions should be positive"
    assert not np.any(np.isnan(predictions)), "No NaN predictions"

    print("✅ Deep MLP integration test passed!")

if __name__ == "__main__":
    test_deep_mlp_integration()