"""
RAG (Retrieval-Augmented Generation) features with advanced techniques
"""

import numpy as np
import faiss
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist, squareform

def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms

def create_similarity_weights(distances, method='inverse_distance', sigma=1.0):
    """Create different types of similarity weights from distances"""
    # Ensure distances is 2D
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    
    if method == 'inverse_distance':
        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-8)
    elif method == 'gaussian':
        # Gaussian kernel weighting
        weights = np.exp(-distances**2 / (2 * sigma**2))
    elif method == 'rank':
        # Rank-based weighting (higher rank = lower weight)
        ranks = np.argsort(np.argsort(distances, axis=1), axis=1) + 1
        weights = 1.0 / ranks
    elif method == 'uniform':
        # Uniform weights
        weights = np.ones_like(distances)
    else:
        weights = np.ones_like(distances)

    # Normalize weights to sum to 1
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    return weights.flatten()  # Return 1D for single sample

def create_pseudo_categories(embeddings, n_categories=20, random_state=42):
    """Create pseudo-categories using clustering on embeddings"""
    print(f"Creating {n_categories} pseudo-categories using K-means clustering...")

    # Use a subset for clustering if too large
    if len(embeddings) > 10000:
        indices = np.random.choice(len(embeddings), 10000, replace=False)
        subset_embeddings = embeddings[indices]
    else:
        subset_embeddings = embeddings

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_categories, random_state=random_state, n_init=10)
    categories = kmeans.fit_predict(embeddings)

    print(f"Created {n_categories} categories with sizes: {np.bincount(categories)}")
    return categories, kmeans

def compute_multiple_similarities(query_emb, candidate_embs, metrics=['cosine', 'euclidean']):
    """Compute multiple similarity/distance metrics"""
    similarities = {}

    if 'cosine' in metrics:
        similarities['cosine'] = cosine_similarity(query_emb.reshape(1, -1), candidate_embs).flatten()

    if 'euclidean' in metrics:
        eucl_dist = euclidean_distances(query_emb.reshape(1, -1), candidate_embs).flatten()
        # Convert distance to similarity (higher = more similar)
        similarities['euclidean'] = 1.0 / (1.0 + eucl_dist)

    if 'manhattan' in metrics:
        manh_dist = manhattan_distances(query_emb.reshape(1, -1), candidate_embs).flatten()
        similarities['manhattan'] = 1.0 / (1.0 + manh_dist)

    return similarities

def adaptive_k_selection(distances, base_k=10, density_threshold=0.8):
    """Adaptively select k based on local density"""
    # Compute local density (inverse of average distance to k nearest neighbors)
    avg_distances = distances.mean(axis=1)

    # Higher density (lower avg distance) = use more neighbors
    density_percentile = np.percentile(avg_distances, density_threshold * 100)

    adaptive_k = np.where(avg_distances <= density_percentile,
                         base_k * 2,  # High density areas
                         base_k // 2)  # Low density areas

    return np.clip(adaptive_k, 3, base_k * 3)  # Reasonable bounds

def create_faiss_index(embeddings, use_gpu=False):
    """Create FAISS index for nearest neighbor search"""
    print("Creating FAISS index...")
    
    # Normalize for cosine similarity
    embeddings_norm = normalize_embeddings(embeddings)
    
    # Create index
    d = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
    
    # Add embeddings
    index.add(embeddings_norm.astype('float32'))
    
    print(f"FAISS index created with {index.ntotal} vectors")
    
    return index, embeddings_norm

def build_fused_features(text_emb, image_emb, numeric_features):
    """Fuse multimodal features"""
    # Normalize each modality
    text_norm = StandardScaler().fit_transform(text_emb)
    image_norm = StandardScaler().fit_transform(image_emb)
    numeric_norm = StandardScaler().fit_transform(numeric_features)
    
    # Concatenate
    fused = np.hstack([text_norm, image_norm, numeric_norm])
    
    return fused

def compute_rag_features(
    train_fused, test_fused, train_prices,
    k=10, use_gpu=False, batch_size=1000,
    use_advanced_features=True, n_categories=20
):
    """Compute advanced RAG features with multiple enhancements"""
    print(f"Computing advanced RAG features with k={k}...")

    # Reduce dimensionality of fused features for memory efficiency
    from sklearn.decomposition import PCA
    print("Reducing fused features dimensionality for RAG...")
    
    # Adaptive PCA components based on data size
    n_samples = len(train_fused)
    n_features = train_fused.shape[1]
    n_components = min(128, n_samples - 1, n_features)
    
    pca = PCA(n_components=n_components, random_state=42)

    # Fit on training data and transform both
    train_fused_reduced = pca.fit_transform(train_fused)
    test_fused_reduced = pca.transform(test_fused)

    print(f"Reduced fused features - Train: {train_fused_reduced.shape}, Test: {test_fused_reduced.shape}")

    if use_advanced_features:
        # Create pseudo-categories for category-specific retrieval
        train_categories, kmeans = create_pseudo_categories(train_fused_reduced, n_categories)
        test_categories = kmeans.predict(test_fused_reduced)

    # Use scikit-learn NearestNeighbors instead of FAISS for stability
    from sklearn.neighbors import NearestNeighbors
    print("Using scikit-learn NearestNeighbors...")

    # Normalize for cosine similarity
    train_norm = normalize_embeddings(train_fused_reduced)
    test_norm = normalize_embeddings(test_fused_reduced)

    # Create and fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k*2, metric='cosine', algorithm='brute')  # Get more neighbors for advanced features
    nn.fit(train_norm)

    # Search for nearest neighbors in batches
    print("Searching for nearest neighbors...")
    all_distances = []
    all_indices = []

    n_test = len(test_norm)
    for start_idx in tqdm(range(0, n_test, batch_size), desc="NN search"):
        end_idx = min(start_idx + batch_size, n_test)
        batch_queries = test_norm[start_idx:end_idx]

        distances, indices = nn.kneighbors(batch_queries)
        all_distances.append(distances)
        all_indices.append(indices)

    # Concatenate results
    distances = np.vstack(all_distances)
    indices = np.vstack(all_indices)

    # Compute advanced RAG features
    rag_features = []

    print("Computing advanced statistical features from neighbors...")
    for i in tqdm(range(len(test_fused)), desc="Advanced RAG features"):
        neighbor_indices = indices[i][:k]  # Use top k
        neighbor_distances = distances[i][:k]

        if use_advanced_features:
            # Category-specific filtering
            query_category = test_categories[i]
            neighbor_categories = train_categories[neighbor_indices]

            # Boost similarity for same-category neighbors
            category_boost = np.where(neighbor_categories == query_category, 1.5, 1.0)
            adjusted_distances = neighbor_distances / category_boost

            # Re-sort by adjusted distances
            sorted_idx = np.argsort(adjusted_distances)
            neighbor_indices = neighbor_indices[sorted_idx][:k]
            neighbor_distances = adjusted_distances[sorted_idx][:k]

        neighbor_prices = train_prices[neighbor_indices]

        # Multiple weighting schemes
        weights_uniform = create_similarity_weights(neighbor_distances, method='uniform')
        weights_inverse = create_similarity_weights(neighbor_distances, method='inverse_distance')
        weights_gaussian = create_similarity_weights(neighbor_distances, method='gaussian', sigma=0.5)
        weights_rank = create_similarity_weights(neighbor_distances, method='rank')

        # Basic statistics
        mean_price = np.mean(neighbor_prices)
        std_price = np.std(neighbor_prices)
        min_price = np.min(neighbor_prices)
        max_price = np.max(neighbor_prices)
        median_price = np.median(neighbor_prices)

        # Weighted statistics with different schemes
        weighted_mean_uniform = np.sum(weights_uniform * neighbor_prices)
        weighted_mean_inverse = np.sum(weights_inverse * neighbor_prices)
        weighted_mean_gaussian = np.sum(weights_gaussian * neighbor_prices)
        weighted_mean_rank = np.sum(weights_rank * neighbor_prices)

        # Price range and distribution features
        price_range = max_price - min_price
        price_iqr = np.subtract(*np.percentile(neighbor_prices, [75, 25]))

        # Higher-order moments
        price_skewness = skew(neighbor_prices) if len(neighbor_prices) > 2 else 0
        price_kurtosis = kurtosis(neighbor_prices) if len(neighbor_prices) > 2 else 0

        # Quantiles
        q10 = np.percentile(neighbor_prices, 10)
        q25 = np.percentile(neighbor_prices, 25)
        q75 = np.percentile(neighbor_prices, 75)
        q90 = np.percentile(neighbor_prices, 90)

        # Confidence interval (assuming normal distribution)
        confidence_interval = 1.96 * std_price / np.sqrt(len(neighbor_prices))

        # Distance-based features
        mean_distance = np.mean(neighbor_distances)
        std_distance = np.std(neighbor_distances)
        max_similarity = 1.0 / (neighbor_distances.min() + 1e-8)  # Convert min distance to max similarity

        # Category diversity (if using categories)
        if use_advanced_features:
            unique_categories = len(np.unique(neighbor_categories[:k]))
            category_diversity = unique_categories / k
        else:
            category_diversity = 0.0

        features = [
            # Basic stats
            mean_price, std_price, min_price, max_price, median_price,
            # Weighted means with different schemes
            weighted_mean_uniform, weighted_mean_inverse, weighted_mean_gaussian, weighted_mean_rank,
            # Distribution features
            price_range, price_iqr, price_skewness, price_kurtosis,
            # Quantiles
            q10, q25, q75, q90,
            # Confidence and uncertainty
            confidence_interval,
            # Distance/similarity features
            mean_distance, std_distance, max_similarity,
            # Category features
            category_diversity
        ]

        rag_features.append(features)

    rag_features = np.array(rag_features)
    print(f"Advanced RAG features shape: {rag_features.shape}")

    return rag_features

def get_rag_feature_names():
    """Return names of advanced RAG features"""
    return [
        # Basic statistics
        'rag_mean_price', 'rag_std_price', 'rag_min_price', 'rag_max_price', 'rag_median_price',
        # Weighted means with different schemes
        'rag_weighted_mean_uniform', 'rag_weighted_mean_inverse', 'rag_weighted_mean_gaussian', 'rag_weighted_mean_rank',
        # Distribution features
        'rag_price_range', 'rag_price_iqr', 'rag_price_skewness', 'rag_price_kurtosis',
        # Quantiles
        'rag_q10', 'rag_q25', 'rag_q75', 'rag_q90',
        # Confidence and uncertainty
        'rag_confidence_interval',
        # Distance/similarity features
        'rag_mean_distance', 'rag_std_distance', 'rag_max_similarity',
        # Category features
        'rag_category_diversity'
    ]
