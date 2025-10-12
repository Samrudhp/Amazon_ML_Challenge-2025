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

def advanced_cluster_calibration(predictions, features, kmeans, cluster_medians, 
                               train_prices, strength=0.15, adaptive=True):
    """Advanced cluster calibration with adaptive strength and error correction"""
    
    # Assign test samples to clusters
    test_clusters = kmeans.predict(features)
    
    calibrated = []
    for i, pred in enumerate(predictions):
        cluster_id = test_clusters[i]
        cluster_median = cluster_medians[cluster_id]
        
        if adaptive:
            # Adaptive strength based on cluster size and variance
            cluster_mask = kmeans.labels_ == cluster_id
            cluster_size = cluster_mask.sum()
            cluster_std = np.std(train_prices[cluster_mask]) if cluster_size > 1 else np.std(train_prices)
            
            # Higher strength for smaller, more consistent clusters
            adaptive_strength = min(strength * (50 / max(cluster_size, 10)) * (1 / max(cluster_std/train_prices.mean(), 0.1)), 0.3)
        else:
            adaptive_strength = strength
        
        # Blend prediction with cluster median
        adjusted = pred * (1 - adaptive_strength) + cluster_median * adaptive_strength
        calibrated.append(adjusted)
    
    return np.array(calibrated)

def advanced_quantile_mapping(predictions, train_prices, n_quantiles=200, smoothing=True):
    """Advanced quantile mapping with smoothing and outlier handling"""
    print("Applying advanced quantile mapping...")
    
    # Remove outliers from training data for more robust mapping
    q1, q3 = np.percentile(train_prices, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_train_prices = train_prices[(train_prices >= lower_bound) & (train_prices <= upper_bound)]
    
    # Get quantiles from clean training data
    train_quantiles = np.percentile(clean_train_prices, np.linspace(0, 100, n_quantiles))
    
    # Get quantiles from predictions
    pred_quantiles = np.percentile(predictions, np.linspace(0, 100, n_quantiles))
    
    if smoothing:
        # Apply slight smoothing to prevent overfitting to quantile boundaries
        from scipy.ndimage import gaussian_filter1d
        train_quantiles = gaussian_filter1d(train_quantiles, sigma=2)
        pred_quantiles = gaussian_filter1d(pred_quantiles, sigma=2)
    
    # Map predictions with linear interpolation
    mapped = np.interp(predictions, pred_quantiles, train_quantiles)
    
    return mapped

def ensemble_calibration(predictions, train_features, train_prices, n_models=5):
    """Ensemble calibration using multiple clustering approaches"""
    print("Applying ensemble calibration...")
    
    calibrated_predictions = []
    
    for i in range(n_models):
        # Different random seeds for ensemble diversity
        n_clusters = np.random.choice([30, 40, 50, 60, 70])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42+i, n_init=10)
        train_clusters = kmeans.fit_predict(train_features)
        
        cluster_medians = {}
        for cluster_id in range(n_clusters):
            cluster_mask = train_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_medians[cluster_id] = np.median(train_prices[cluster_mask])
            else:
                cluster_medians[cluster_id] = np.median(train_prices)
        
        # Apply calibration with different strengths
        strength = np.random.uniform(0.1, 0.2)
        cal_pred = apply_cluster_calibration(predictions, train_features, kmeans, cluster_medians, strength)
        calibrated_predictions.append(cal_pred)
    
    # Ensemble the calibrated predictions
    ensemble_pred = np.mean(calibrated_predictions, axis=0)
    
    return ensemble_pred

def residual_calibration(predictions, train_predictions, train_residuals, k=50):
    """Calibrate based on prediction residuals"""
    print("Applying residual calibration...")
    
    # Use KNN to find similar predictions and adjust based on their residual patterns
    from sklearn.neighbors import KNeighborsRegressor
    
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(train_predictions.reshape(-1, 1), train_residuals)
    
    # Predict residual adjustments
    adjustments = knn.predict(predictions.reshape(-1, 1))
    
    # Apply adjustments (dampened)
    calibrated = predictions + 0.3 * adjustments
    
    return calibrated

def post_process_predictions(
    predictions, train_features, train_prices,
    test_features, n_clusters=50, use_quantile_mapping=True,
    use_advanced_calibration=True, use_ensemble_calibration=True
):
    """Apply all post-processing steps with advanced techniques"""
    print("\n=== Post-processing predictions ===")
    
    # Ensure positive predictions
    predictions = np.maximum(predictions, 0.01)
    
    original_predictions = predictions.copy()
    
    if use_advanced_calibration:
        print("Applying advanced cluster calibration...")
        # Basic cluster setup
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_clusters = kmeans.fit_predict(train_features)
        
        cluster_medians = {}
        for cluster_id in range(n_clusters):
            cluster_mask = train_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_medians[cluster_id] = np.median(train_prices[cluster_mask])
            else:
                cluster_medians[cluster_id] = np.median(train_prices)
        
        # Advanced cluster calibration
        predictions = advanced_cluster_calibration(
            predictions, test_features, kmeans, cluster_medians, train_prices, 
            strength=0.15, adaptive=True
        )
    
    # Ensemble calibration
    if use_ensemble_calibration:
        predictions = ensemble_calibration(predictions, test_features, train_prices, n_models=3)
    
    # Advanced quantile mapping
    if use_quantile_mapping:
        predictions = advanced_quantile_mapping(predictions, train_prices, n_quantiles=200, smoothing=True)
    
    # Residual calibration (using original predictions as reference)
    predictions = residual_calibration(predictions, original_predictions, 
                                     train_prices - original_predictions, k=30)
    
    # Final clipping and bounds
    predictions = np.maximum(predictions, 0.01)
    predictions = np.minimum(predictions, train_prices.max() * 1.5)  # Reasonable upper bound
    
    print("Post-processing complete")
    
    return predictions
