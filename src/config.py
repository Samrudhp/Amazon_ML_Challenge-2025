"""
Configuration file for Smart Product Pricing ML Pipeline
"""

import os

# Paths
DATA_DIR = "dataset"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_OUT_PATH = os.path.join(DATA_DIR, "sample_test_out.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "test_out.csv")

# Image paths
IMAGE_DIR = "images"
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test")

# Cache paths
CACHE_DIR = "cache"
TEXT_EMB_TRAIN = os.path.join(CACHE_DIR, "text_emb_train.npy")
TEXT_EMB_TEST = os.path.join(CACHE_DIR, "text_emb_test.npy")
IMAGE_EMB_TRAIN = os.path.join(CACHE_DIR, "image_emb_train.npy")
IMAGE_EMB_TEST = os.path.join(CACHE_DIR, "image_emb_test.npy")

# Model parameters
RANDOM_SEED = 42
N_FOLDS = 5
N_SEEDS = 3  # Increased from 1 for ensemble averaging

# Text embedding
TEXT_MODEL = "all-MiniLM-L6-v2"
TEXT_BATCH_SIZE = 64
TEXT_DIM_ORIGINAL = 384
TEXT_DIM_REDUCED = 256

# Image embedding
IMAGE_MODEL = "openai/clip-vit-base-patch32"
IMAGE_BATCH_SIZE = 16
IMAGE_DIM_ORIGINAL = 512
IMAGE_DIM_REDUCED = 256

# RAG parameters
RAG_K = 10  # Number of neighbors
FAISS_USE_GPU = False
RAG_USE_ADVANCED = True  # Enable advanced RAG features
RAG_N_CATEGORIES = 20  # Number of pseudo-categories for category-specific retrieval

# LightGBM parameters - Optimized for better performance
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 100,  # Increased from 31 for more complex trees
    'max_depth': 12,    # Limited depth to prevent overfitting
    'learning_rate': 0.01,  # Lower learning rate for better convergence
    'feature_fraction': 0.9,  # Use more features per tree
    'bagging_fraction': 0.9,  # Use more data per tree
    'bagging_freq': 5,
    'lambda_l1': 0.5,  # Increased L1 regularization
    'lambda_l2': 0.5,  # Increased L2 regularization
    'min_data_in_leaf': 20,  # Minimum samples per leaf
    'min_sum_hessian_in_leaf': 1e-3,  # Minimum hessian per leaf
    'num_threads': 4,
    'verbosity': -1,
    'seed': RANDOM_SEED
}

LGB_NUM_ROUNDS = 5000  # Increased from 2000 for lower learning rate
LGB_EARLY_STOPPING = 200  # Increased patience for convergence

# Ridge parameters
RIDGE_ALPHA = 1.0

# Post-processing
N_CLUSTERS = 50  # For cluster-based calibration

# Deep MLP parameters
USE_DEEP_MLP = False
MLP_HIDDEN_DIMS = [512, 256, 128, 64]  # Hidden layer dimensions
MLP_DROPOUT_RATE = 0.3
MLP_LEARNING_RATE = 1e-3
MLP_BATCH_SIZE = 1024
MLP_NUM_EPOCHS = 100
MLP_EARLY_STOPPING_PATIENCE = 10
