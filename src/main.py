"""
Main pipeline for Smart Product Pricing ML Challenge
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import *
from preprocessing import load_and_clean_data, get_numeric_feature_names
from embeddings import generate_and_cache_embeddings
from rag import build_fused_features, compute_rag_features, get_rag_feature_names
from models import (train_full_pipeline, predict_full_pipeline)
from postprocessing import post_process_predictions

# Import utils for image downloading
sys.path.append(os.path.dirname(__file__))
from utils import download_images

def main():
    """Main pipeline execution"""
    
    print("=" * 80)
    print("SMART PRODUCT PRICING ML PIPELINE")
    print("=" * 80)
    
    # ========== STEP 1: Load and Clean Data ==========
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading & Cleaning")
    print("=" * 80)
    
    train_df, test_df = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # ========== STEP 2: Extract Numeric Features ==========
    print("\n" + "=" * 80)
    print("STEP 2: Numeric Features Extracted")
    print("=" * 80)
    
    numeric_feature_names = get_numeric_feature_names()
    X_train_numeric = train_df[numeric_feature_names].values
    X_test_numeric = test_df[numeric_feature_names].values
    
    print(f"Numeric features shape - Train: {X_train_numeric.shape}, Test: {X_test_numeric.shape}")
    
    # Target variable
    y_train = train_df['price'].values
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Target mean: {y_train.mean():.2f}, median: {np.median(y_train):.2f}")
    
    # ========== STEP 3 & 4: Generate Embeddings ==========
    print("\n" + "=" * 80)
    print("STEP 3 & 4: Text (MiniLM) and Image (CLIP) Embeddings")
    print("=" * 80)
    
    text_emb_train, text_emb_test, image_emb_train, image_emb_test = \
        generate_and_cache_embeddings(
            train_df, test_df,
            TEXT_MODEL, IMAGE_MODEL,
            TRAIN_IMAGE_DIR, TEST_IMAGE_DIR,
            CACHE_DIR,
            text_batch_size=TEXT_BATCH_SIZE,
            image_batch_size=IMAGE_BATCH_SIZE,
            text_dim_reduced=TEXT_DIM_REDUCED,
            image_dim_reduced=IMAGE_DIM_REDUCED,
            download_func=download_images
        )
    
    # ========== STEP 5: RAG Features ==========
    print("\n" + "=" * 80)
    print("STEP 5: RAG Features (FAISS Nearest Neighbors)")
    print("=" * 80)
    
    # Build fused features for RAG
    train_fused = build_fused_features(text_emb_train, image_emb_train, X_train_numeric)
    test_fused = build_fused_features(text_emb_test, image_emb_test, X_test_numeric)
    
    print(f"Fused features shape - Train: {train_fused.shape}, Test: {test_fused.shape}")
    
    # Compute RAG features
    X_test_rag = compute_rag_features(
        train_fused, test_fused, y_train,
        k=RAG_K, use_gpu=FAISS_USE_GPU,
        use_advanced_features=RAG_USE_ADVANCED, n_categories=RAG_N_CATEGORIES
    )
    
    # For training, we need to compute RAG features using leave-one-out or pseudo-labeling
    # For simplicity, we'll compute RAG on full training set (slightly optimistic)
    print("\nComputing training RAG features...")
    X_train_rag = compute_rag_features(
        train_fused, train_fused, y_train,
        k=RAG_K + 1, use_gpu=FAISS_USE_GPU,
        use_advanced_features=RAG_USE_ADVANCED, n_categories=RAG_N_CATEGORIES
    )
    
    # ========== STEP 6: Feature Fusion ==========
    print("\n" + "=" * 80)
    print("STEP 6: Feature Fusion")
    print("=" * 80)
    
    X_train_full = np.hstack([
        text_emb_train,
        image_emb_train,
        X_train_numeric,
        X_train_rag
    ])
    
    X_test_full = np.hstack([
        text_emb_test,
        image_emb_test,
        X_test_numeric,
        X_test_rag
    ])
    
    print(f"Final feature shape - Train: {X_train_full.shape}, Test: {X_test_full.shape}")
    
    # ========== STEP 6.5: Feature Selection ==========
    print("\n" + "=" * 80)
    print("STEP 6.5: Feature Selection")
    print("=" * 80)
    
    # Try to load pre-computed selected feature indices
    feature_selection_enabled = True
    selected_indices = None
    
    if feature_selection_enabled:
        try:
            # First try to load pre-selected indices
            selected_indices_path = os.path.join(CACHE_DIR, 'selected_feature_indices.npy')
            if os.path.exists(selected_indices_path):
                selected_indices = np.load(selected_indices_path)
                print(f"Loaded pre-selected {len(selected_indices)} feature indices")
            else:
                # Fallback: compute from previous model
                with open(os.path.join(CACHE_DIR, 'trained_models.pkl'), 'rb') as f:
                    prev_models = pickle.load(f)
                    prev_importance = prev_models['feature_importance']
                    
                    # Keep top N most important features (configurable)
                    N_FEATURES_TO_KEEP = 250  # Keep top 250 features
                    if len(prev_importance) > N_FEATURES_TO_KEEP:
                        # Get indices of top N features
                        top_indices = np.argsort(prev_importance)[-N_FEATURES_TO_KEEP:]
                        selected_indices = np.sort(top_indices)
                        print(f"Computed and selected {N_FEATURES_TO_KEEP} most important features")
                    else:
                        print("Not enough features to select from, keeping all features")
            
            if selected_indices is not None:
                # Filter features
                X_train_full = X_train_full[:, selected_indices]
                X_test_full = X_test_full[:, selected_indices]
                
                print(f"Applied feature selection - reduced to {len(selected_indices)} features")
                print(f"Reduced feature shape - Train: {X_train_full.shape}, Test: {X_test_full.shape}")
                
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"No feature selection data found ({e}), keeping all features")
    
    # ========== STEP 7 & 8: Train LightGBM + Ridge Ensemble ==========
    print("\n" + "=" * 80)
    print("STEP 7 & 8: Training LightGBM + Ridge Ensemble")
    print("=" * 80)

    # Train ensemble with multiple seeds
    all_lgb_models = []
    all_ridge_models = []
    all_feature_importances = []
    all_test_predictions = []

    for seed_idx in range(N_SEEDS):
        current_seed = RANDOM_SEED + seed_idx
        print(f"\n--- Training seed {seed_idx + 1}/{N_SEEDS} (seed={current_seed}) ---")

        # Train LightGBM + Ridge ensemble
        lgb_models, ridge_model, feature_importance, scaler = train_full_pipeline(
            X_train_full, y_train,
            LGB_PARAMS, LGB_NUM_ROUNDS, LGB_EARLY_STOPPING,
            N_FOLDS, RIDGE_ALPHA, current_seed,
            use_deep_mlp=USE_DEEP_MLP, mlp_hidden_dims=MLP_HIDDEN_DIMS,
            mlp_dropout_rate=MLP_DROPOUT_RATE, mlp_learning_rate=MLP_LEARNING_RATE,
            mlp_batch_size=MLP_BATCH_SIZE, mlp_num_epochs=MLP_NUM_EPOCHS,
            mlp_patience=MLP_EARLY_STOPPING_PATIENCE
        )

        # Store models
        all_lgb_models.append(lgb_models)
        all_ridge_models.append(ridge_model)
        all_feature_importances.append(feature_importance)

        # Get predictions for this seed
        seed_predictions = predict_full_pipeline(lgb_models, ridge_model, X_test_full,
                                                use_deep_mlp=USE_DEEP_MLP, mlp_batch_size=MLP_BATCH_SIZE)
        all_test_predictions.append(seed_predictions)

    # Average predictions across seeds
    predictions = np.mean(all_test_predictions, axis=0)
    feature_importance = np.mean(all_feature_importances, axis=0)

    # Use the last trained models for saving (they're equivalent)
    lgb_models, ridge_model = all_lgb_models[-1], all_ridge_models[-1]

    print(f"Ensemble predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Ensemble predictions mean: {predictions.mean():.2f}, median: {np.median(predictions):.2f}")
    
    # ========== STEP 10: Post-Processing & Calibration ==========
    print("\n" + "=" * 80)
    print("STEP 10: Post-Processing & Calibration")
    print("=" * 80)
    
    if USE_DEEP_MLP:
        # Skip post-processing for Deep MLP as it's designed for LightGBM
        print("Skipping post-processing for Deep MLP predictions...")
        final_predictions = predictions
    else:
        final_predictions = post_process_predictions(
            predictions,
            train_fused, y_train,
            test_fused,
            n_clusters=N_CLUSTERS,
            use_quantile_mapping=True,
            use_advanced_calibration=True,
            use_ensemble_calibration=True
        )
    
    print(f"Final predictions range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
    print(f"Final predictions mean: {final_predictions.mean():.2f}, median: {np.median(final_predictions):.2f}")
    
    # ========== Save Models ==========
    print("\n" + "=" * 80)
    print("Saving Models")
    print("=" * 80)
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, 'trained_models.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'lgb_models': lgb_models,
            'ridge_model': ridge_model,
            'feature_importance': feature_importance,
            'scaler': scaler,
            'use_deep_mlp': USE_DEEP_MLP,
            'train_data': {
                'fused': train_fused,
                'prices': y_train
            }
        }, f)
    
    print(f"Models saved to: {model_path}")
    
    # ========== Save Submission ==========
    print("\n" + "=" * 80)
    print("Saving Submission")
    print("=" * 80)
    
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved to: {OUTPUT_PATH}")
    print(f"Submission shape: {submission.shape}")
    print("\nFirst few predictions:")
    print(submission.head(10))
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Total features used: {X_train_full.shape[1]}")
    print(f"  - Text embeddings: {TEXT_DIM_REDUCED}")
    print(f"  - Image embeddings: {IMAGE_DIM_REDUCED}")
    print(f"  - Numeric features: {len(numeric_feature_names)}")
    print(f"  - RAG features: {len(get_rag_feature_names())}")
    print(f"\nOutput file: {OUTPUT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()
