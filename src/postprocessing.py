"""
Post-processing and calibration
"""

import numpy as np
from sklearn.cluster import KMeans

def cluster_median_calibration(predictions, train_features, train_prices, n_clusters=50):
    """Apply cluster-based median calibration"""
    print(f"Applying cluster median calibration with {n_clusters} clusters...")
    
    # Cluster training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(train_features)
    
    # Calculate median price per cluster
    cluster_medians = {}
    for cluster_id in range(n_clusters):
        cluster_mask = train_clusters == cluster_id
        if cluster_mask.sum() > 0:
            cluster_medians[cluster_id] = np.median(train_prices[cluster_mask])
        else:
            cluster_medians[cluster_id] = np.median(train_prices)
    
    return kmeans, cluster_medians

def apply_cluster_calibration(predictions, features, kmeans, cluster_medians, strength=0.1):
    """Apply cluster calibration to predictions"""
    
    # Assign test samples to clusters
    test_clusters = kmeans.predict(features)
    
    # Adjust predictions towards cluster median
    calibrated = []
    for i, pred in enumerate(predictions):
        cluster_id = test_clusters[i]
        cluster_median = cluster_medians[cluster_id]
        
        # Blend prediction with cluster median
        adjusted = pred * (1 - strength) + cluster_median * strength
        calibrated.append(adjusted)
    
    return np.array(calibrated)

def quantile_mapping(predictions, train_prices, n_quantiles=100):
    """Map predictions to match training distribution quantiles"""
    print("Applying quantile mapping...")
    
    # Get quantiles from training data
    train_quantiles = np.percentile(train_prices, np.linspace(0, 100, n_quantiles))
    
    # Get quantiles from predictions
    pred_quantiles = np.percentile(predictions, np.linspace(0, 100, n_quantiles))
    
    # Map predictions
    mapped = np.interp(predictions, pred_quantiles, train_quantiles)
    
    return mapped

def post_process_predictions(
    predictions, train_features, train_prices,
    test_features, n_clusters=50, use_quantile_mapping=True
):
    """Apply all post-processing steps"""
    print("\n=== Post-processing predictions ===")
    
    # Ensure positive predictions
    predictions = np.maximum(predictions, 0.01)
    
    # Cluster calibration
    kmeans, cluster_medians = cluster_median_calibration(
        predictions, train_features, train_prices, n_clusters
    )
    
    calibrated = apply_cluster_calibration(
        predictions, test_features, kmeans, cluster_medians, strength=0.05
    )
    
    # Quantile mapping
    if use_quantile_mapping:
        calibrated = quantile_mapping(calibrated, train_prices, n_quantiles=100)
    
    # Final clipping
    calibrated = np.maximum(calibrated, 0.01)
    
    print("Post-processing complete")
    
    return calibrated
