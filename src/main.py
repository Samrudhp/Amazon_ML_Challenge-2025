"""
Main pipeline for Smart Product Pricing ML Challenge
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import *
from preprocessing import load_and_clean_data, get_numeric_feature_names
from embeddings import generate_and_cache_embeddings
from rag import build_fused_features, compute_rag_features, get_rag_feature_names
from models import train_full_pipeline, predict_full_pipeline
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
        k=RAG_K, use_gpu=FAISS_USE_GPU
    )
    
    # For training, we need to compute RAG features using leave-one-out or pseudo-labeling
    # For simplicity, we'll compute RAG on full training set (slightly optimistic)
    print("\nComputing training RAG features...")
    X_train_rag = compute_rag_features(
        train_fused, train_fused, y_train,
        k=RAG_K + 1, use_gpu=FAISS_USE_GPU
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
    
    # ========== STEP 7 & 8: Train LightGBM + Ridge ==========
    print("\n" + "=" * 80)
    print("STEP 7 & 8: Training LightGBM + Ridge Meta-Model")
    print("=" * 80)
    
    lgb_models, ridge_model, feature_importance = train_full_pipeline(
        X_train_full, y_train,
        LGB_PARAMS, LGB_NUM_ROUNDS, LGB_EARLY_STOPPING,
        N_FOLDS, RIDGE_ALPHA, RANDOM_SEED
    )
    
    # ========== STEP 9: Predictions ==========
    print("\n" + "=" * 80)
    print("STEP 9: Making Predictions")
    print("=" * 80)
    
    predictions = predict_full_pipeline(lgb_models, ridge_model, X_test_full)
    print(f"Raw predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Raw predictions mean: {predictions.mean():.2f}, median: {np.median(predictions):.2f}")
    
    # ========== STEP 10: Post-Processing ==========
    print("\n" + "=" * 80)
    print("STEP 10: Post-Processing & Calibration")
    print("=" * 80)
    
    final_predictions = post_process_predictions(
        predictions,
        train_fused, y_train,
        test_fused,
        n_clusters=N_CLUSTERS,
        use_quantile_mapping=True
    )
    
    print(f"Final predictions range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
    print(f"Final predictions mean: {final_predictions.mean():.2f}, median: {np.median(final_predictions):.2f}")
    
    # ========== Save Models ==========
    print("\n" + "=" * 80)
    print("Saving Models")
    print("=" * 80)
    
    import pickle
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, 'trained_models.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'lgb_models': lgb_models,
            'ridge_model': ridge_model,
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
