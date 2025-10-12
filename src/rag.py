"""
RAG (Retrieval-Augmented Generation) features using FAISS
"""

import numpy as np
import faiss
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms

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
    k=10, use_gpu=False
):
    """Compute RAG features using FAISS nearest neighbors"""
    print(f"Computing RAG features with k={k}...")
    
    # Create FAISS index from training data
    index, train_fused_norm = create_faiss_index(train_fused, use_gpu)
    
    # Normalize test embeddings
    test_fused_norm = normalize_embeddings(test_fused)
    
    # Search for nearest neighbors
    print("Searching for nearest neighbors...")
    distances, indices = index.search(test_fused_norm.astype('float32'), k)
    
    # Compute RAG features
    rag_features = []
    
    print("Computing statistical features from neighbors...")
    for i in tqdm(range(len(test_fused))):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        neighbor_prices = train_prices[neighbor_indices]
        
        # Basic statistics
        mean_price = np.mean(neighbor_prices)
        std_price = np.std(neighbor_prices)
        min_price = np.min(neighbor_prices)
        max_price = np.max(neighbor_prices)
        median_price = np.median(neighbor_prices)
        
        # Weighted statistics (weight by similarity)
        weights = neighbor_distances / (neighbor_distances.sum() + 1e-8)
        weighted_mean = np.sum(weights * neighbor_prices)
        
        # Price range
        price_range = max_price - min_price
        
        # Quantiles
        q25 = np.percentile(neighbor_prices, 25)
        q75 = np.percentile(neighbor_prices, 75)
        
        features = [
            mean_price, std_price, min_price, max_price, median_price,
            weighted_mean, price_range, q25, q75
        ]
        
        rag_features.append(features)
    
    rag_features = np.array(rag_features)
    print(f"RAG features shape: {rag_features.shape}")
    
    return rag_features

def get_rag_feature_names():
    """Return names of RAG features"""
    return [
        'rag_mean_price', 'rag_std_price', 'rag_min_price', 'rag_max_price',
        'rag_median_price', 'rag_weighted_mean', 'rag_price_range',
        'rag_q25', 'rag_q75'
    ]
