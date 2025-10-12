"""
Fast inference script - uses cached embeddings if available
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

from config import *
from preprocessing import load_and_clean_data, get_numeric_feature_names
from embeddings import generate_and_cache_embeddings
from rag import build_fused_features, compute_rag_features
from models import predict_full_pipeline
from postprocessing import post_process_predictions
from utils import download_images

def load_models():
    """Load trained models"""
    import pickle
    
    model_path = os.path.join(CACHE_DIR, 'trained_models.pkl')
    
    if os.path.exists(model_path):
        print("Loading trained models...")
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        return models['lgb_models'], models['ridge_model'], models['train_data']
    else:
        raise FileNotFoundError(
            "No trained models found. Please run train.py first."
        )

def inference():
    """Run inference on test data"""
    
    print("=" * 80)
    print("FAST INFERENCE MODE")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    
    # Extract numeric features
    numeric_feature_names = get_numeric_feature_names()
    X_train_numeric = train_df[numeric_feature_names].values
    X_test_numeric = test_df[numeric_feature_names].values
    y_train = train_df['price'].values
    
    # Generate/load embeddings (uses cache)
    print("\nLoading embeddings...")
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
    
    # Build fused features
    print("\nBuilding fused features...")
    train_fused = build_fused_features(text_emb_train, image_emb_train, X_train_numeric)
    test_fused = build_fused_features(text_emb_test, image_emb_test, X_test_numeric)
    
    # Compute RAG features
    print("\nComputing RAG features...")
    X_test_rag = compute_rag_features(train_fused, test_fused, y_train, k=RAG_K,
                                     use_advanced_features=RAG_USE_ADVANCED, n_categories=RAG_N_CATEGORIES)
    X_train_rag = compute_rag_features(train_fused, train_fused, y_train, k=RAG_K + 1,
                                       use_advanced_features=RAG_USE_ADVANCED, n_categories=RAG_N_CATEGORIES)
    
    # Fuse all features
    X_test_full = np.hstack([text_emb_test, image_emb_test, X_test_numeric, X_test_rag])
    
    # Load models
    try:
        lgb_models, ridge_model, _, use_deep_mlp, scaler = load_models()
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Running full training pipeline instead...")
        from main import main
        main()
        return
    
    # Predict
    print("\nMaking predictions...")
    predictions = predict_full_pipeline(lgb_models, ridge_model, X_test_full, use_deep_mlp=use_deep_mlp, scaler=scaler)
    
    # Post-process
    print("\nPost-processing...")
    final_predictions = post_process_predictions(
        predictions, train_fused, y_train, test_fused,
        n_clusters=N_CLUSTERS, use_quantile_mapping=True
    )
    
    # Save
    print("\nSaving submission...")
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    submission.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Predictions saved to: {OUTPUT_PATH}")
    print(f"Prediction statistics:")
    print(f"  Mean: {final_predictions.mean():.2f}")
    print(f"  Median: {np.median(final_predictions):.2f}")
    print(f"  Range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")

if __name__ == "__main__":
    inference()
