# Smart Product Pricing ML Pipeline

## Overview

Complete end-to-end ML pipeline for product price prediction using multimodal features (text, images, numeric) with RAG-enhanced predictions. 

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python train.py

# Output: dataset/test_out.csv
```

## Problem Statement

## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 75k products with complete product details and prices
- **Test Set:** 75k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.

Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

## Pipeline Architecture

### Feature Engineering
- **Text Processing**: Cleaning, keyword extraction, IPQ parsing
- **Numeric Features**: 13 engineered features (length, word count, ratios)
- **Text Embeddings**: MiniLM-L6-v2 → 384D → PCA to 256D
- **Image Embeddings**: CLIP ViT-B/32 → 512D → PCA to 256D
- **RAG Features**: FAISS k-NN (k=10) → 9 statistical price features

### Models
- **LightGBM**: 5-fold stratified CV on log(price)
- **Ridge Meta-Model**: Ensemble predictions + RAG features
- **Post-Processing**: Cluster calibration + quantile mapping

## Project Structure

```
src/
├── config.py          # Configuration and hyperparameters
├── preprocessing.py   # Data cleaning and feature engineering
├── embeddings.py      # MiniLM and CLIP embedding generation
├── rag.py            # FAISS-based RAG feature computation
├── models.py         # LightGBM and Ridge training
├── postprocessing.py # Calibration and post-processing
├── main.py           # Main pipeline orchestrator
├── inference.py      # Fast inference with cached models
└── utils.py          # Image download utilities

dataset/
├── train.csv         # Training data (75k samples)
├── test.csv          # Test data (75k samples)
└── test_out.csv      # Output predictions (generated)
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Full Training Pipeline
```bash
python train.py
```

This will:
1. Download images using `src/utils.py`
2. Generate and cache embeddings
3. Train models with 5-fold CV
4. Generate predictions with post-processing
5. Save to `dataset/test_out.csv`

### Fast Inference (with cached embeddings)
```bash
python src/inference.py
```
## Key Features

✅ **Multimodal Fusion**: Text + Image + Numeric features  
✅ **RAG Enhancement**: FAISS k-NN for local price priors  
✅ **Dimension Reduction**: PCA for memory efficiency  
✅ **Ensemble Learning**: LightGBM + Ridge stacking  
✅ **Smart Calibration**: Cluster-based + quantile mapping  
✅ **CPU Optimized**: No GPU required  
✅ **Reproducible**: Fixed random seeds throughout
