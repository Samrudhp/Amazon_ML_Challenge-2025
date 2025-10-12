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
N_SEEDS = 1  # Can increase to 3-5 for ensemble

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

# LightGBM parameters
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_threads': 4,
    'verbosity': -1,
    'seed': RANDOM_SEED
}

LGB_NUM_ROUNDS = 2000
LGB_EARLY_STOPPING = 100

# Ridge parameters
RIDGE_ALPHA = 1.0

# Post-processing
N_CLUSTERS = 50  # For cluster-based calibration
