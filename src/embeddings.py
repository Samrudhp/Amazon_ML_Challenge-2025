"""
Embedding generation using MiniLM and CLIP
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.decomposition import IncrementalPCA

def generate_text_embeddings(texts, model_name, batch_size=64, device='cpu'):
    """Generate text embeddings using SentenceTransformer"""
    print(f"Generating text embeddings using {model_name}...")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Text embeddings shape: {embeddings.shape}")
    return embeddings

def reduce_dimensions_pca(embeddings, n_components, batch_size=10000):
    """Reduce embedding dimensions using Incremental PCA"""
    print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}...")
    
    pca = IncrementalPCA(n_components=n_components)
    
    # Fit in batches for memory efficiency
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        pca.partial_fit(batch)
    
    # Transform
    reduced = pca.transform(embeddings)
    
    print(f"Reduced shape: {reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return reduced, pca

def load_image_safe(image_path, default_size=(224, 224)):
    """Safely load image with fallback"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        # Return blank image if loading fails
        return Image.new('RGB', default_size, color=(128, 128, 128))

def generate_image_embeddings(image_paths, model_name, batch_size=16, device='cpu'):
    """Generate image embeddings using CLIP"""
    print(f"Generating image embeddings using {model_name}...")
    
    # Load CLIP model
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images
            images = [load_image_safe(path) for path in batch_paths]
            
            # Process images
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model.get_image_features(**inputs)
            batch_emb = outputs.cpu().numpy()
            
            embeddings.append(batch_emb)
    
    embeddings = np.vstack(embeddings)
    print(f"Image embeddings shape: {embeddings.shape}")
    
    return embeddings

def get_image_paths(df, image_dir, download_images_func=None):
    """Get image paths from dataframe, optionally download first"""
    from pathlib import Path
    
    image_paths = []
    
    # Download images if function provided
    if download_images_func is not None and not os.path.exists(image_dir):
        print(f"Downloading images to {image_dir}...")
        download_images_func(df['image_link'].tolist(), image_dir)
    
    # Build image paths
    for link in df['image_link']:
        if pd.notna(link):
            filename = Path(link).name
            image_path = os.path.join(image_dir, filename)
            image_paths.append(image_path)
        else:
            image_paths.append("")
    
    return image_paths

def generate_and_cache_embeddings(
    train_df, test_df,
    text_model_name, image_model_name,
    train_image_dir, test_image_dir,
    cache_dir,
    text_batch_size=64, image_batch_size=16,
    text_dim_reduced=256, image_dim_reduced=256,
    download_func=None
):
    """Generate and cache all embeddings"""
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Text embeddings
    text_train_cache = os.path.join(cache_dir, "text_emb_train.npy")
    text_test_cache = os.path.join(cache_dir, "text_emb_test.npy")
    
    if os.path.exists(text_train_cache) and os.path.exists(text_test_cache):
        print("Loading cached text embeddings...")
        text_emb_train = np.load(text_train_cache)
        text_emb_test = np.load(text_test_cache)
    else:
        # Generate text embeddings
        text_emb_train_raw = generate_text_embeddings(
            train_df['catalog_content'].tolist(),
            text_model_name,
            batch_size=text_batch_size
        )
        text_emb_test_raw = generate_text_embeddings(
            test_df['catalog_content'].tolist(),
            text_model_name,
            batch_size=text_batch_size
        )
        
        # Reduce dimensions
        text_emb_combined = np.vstack([text_emb_train_raw, text_emb_test_raw])
        text_emb_reduced, _ = reduce_dimensions_pca(text_emb_combined, text_dim_reduced)
        
        text_emb_train = text_emb_reduced[:len(train_df)]
        text_emb_test = text_emb_reduced[len(train_df):]
        
        # Cache
        np.save(text_train_cache, text_emb_train)
        np.save(text_test_cache, text_emb_test)
    
    print(f"Text embeddings - Train: {text_emb_train.shape}, Test: {text_emb_test.shape}")
    
    # Image embeddings
    image_train_cache = os.path.join(cache_dir, "image_emb_train.npy")
    image_test_cache = os.path.join(cache_dir, "image_emb_test.npy")
    
    if os.path.exists(image_train_cache) and os.path.exists(image_test_cache):
        print("Loading cached image embeddings...")
        image_emb_train = np.load(image_train_cache)
        image_emb_test = np.load(image_test_cache)
    else:
        # Get image paths
        train_image_paths = get_image_paths(train_df, train_image_dir, download_func)
        test_image_paths = get_image_paths(test_df, test_image_dir, download_func)
        
        # Generate image embeddings
        image_emb_train_raw = generate_image_embeddings(
            train_image_paths,
            image_model_name,
            batch_size=image_batch_size
        )
        image_emb_test_raw = generate_image_embeddings(
            test_image_paths,
            image_model_name,
            batch_size=image_batch_size
        )
        
        # Reduce dimensions
        image_emb_combined = np.vstack([image_emb_train_raw, image_emb_test_raw])
        image_emb_reduced, _ = reduce_dimensions_pca(image_emb_combined, image_dim_reduced)
        
        image_emb_train = image_emb_reduced[:len(train_df)]
        image_emb_test = image_emb_reduced[len(train_df):]
        
        # Cache
        np.save(image_train_cache, image_emb_train)
        np.save(image_test_cache, image_emb_test)
    
    print(f"Image embeddings - Train: {image_emb_train.shape}, Test: {image_emb_test.shape}")
    
    return text_emb_train, text_emb_test, image_emb_train, image_emb_test

import pandas as pd
