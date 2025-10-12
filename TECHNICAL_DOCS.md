# Smart Product Pricing ML Pipeline - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Feature Engineering](#feature-engineering)
3. [Model Pipeline](#model-pipeline)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Pipeline Flow
```
Raw Data → Preprocessing → Feature Engineering → Embeddings → RAG → Models → Post-Processing → Predictions
```

### Component Details

#### 1. Preprocessing (`preprocessing.py`)
- **Text Cleaning**: Removes HTML, emojis, extra spaces
- **IPQ Extraction**: Parses numeric values and units from product descriptions
- **Price Clipping**: Handles outliers at 1st and 99th percentiles

#### 2. Feature Engineering
**Numeric Features (13 total)**:
- Basic: text_length, word_count, digit_count
- IPQ: ipq_value, unit_type, log_ipq
- Ratios: digits_per_word, adjectives_per_word
- Keywords: premium_count, bulk_count, quality_count, adjective_count
- Interactions: unit_x_ipq

#### 3. Text Embeddings (`embeddings.py`)
- **Model**: all-MiniLM-L6-v2 (384D)
- **Reduction**: PCA to 256D
- **Batch Size**: 64 (CPU-optimized)
- **Memory**: ~2GB for 150k samples

#### 4. Image Embeddings (`embeddings.py`)
- **Model**: CLIP ViT-B/32 (512D)
- **Reduction**: PCA to 256D
- **Batch Size**: 16 (memory-safe)
- **Fallback**: Gray placeholder for missing images

#### 5. RAG Features (`rag.py`)
- **Index**: FAISS IndexFlatIP (cosine similarity)
- **k**: 10 nearest neighbors
- **Features**: mean, std, min, max, median, weighted_mean, range, q25, q75
- **Impact**: +3-6% SMAPE improvement

#### 6. Models (`models.py`)
- **LightGBM**: 5-fold stratified CV on log(price)
- **Ridge**: Meta-model on OOF predictions + RAG features
- **Ensemble**: Average of 5 fold models

#### 7. Post-Processing (`postprocessing.py`)
- **Cluster Calibration**: KMeans (k=50) median adjustment
- **Quantile Mapping**: Align prediction distribution to training
- **Clipping**: Ensure positive prices

---

## Feature Engineering

### Text Feature Extraction

```python
# Example: Extract IPQ value
text = "Organic Coffee Beans 2 Pound Premium Pack"
ipq_value = 2.0  # Extracted
unit_type = 6     # Pound encoding
```

### Keyword Categories
- **Premium**: premium, luxury, deluxe, gourmet, organic
- **Bulk**: bulk, value, pack, bundle, economy
- **Quality**: fresh, natural, pure, original, authentic
- **Adjectives**: new, best, great, special, limited, exclusive

### Feature Importance (Typical)
1. RAG features: ~40-50%
2. Image embeddings: ~20-25%
3. Text embeddings: ~15-20%
4. Numeric features: ~10-15%

---

## Model Pipeline

### LightGBM Configuration

```python
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_threads': 4
}
```

### Training Strategy
1. **Target Transform**: log1p(price) for stability
2. **Stratification**: 5 quantiles of log(price)
3. **Early Stopping**: 100 rounds
4. **Validation**: Out-of-fold predictions for meta-model

### Ridge Meta-Model
- Combines LightGBM predictions with RAG features
- Smooths predictions for unseen product types
- Ridge alpha=1.0 for regularization

---

## Performance Optimization

### Memory Management
1. **PCA Reduction**: 384D→256D, 512D→256D saves ~40% memory
2. **Incremental PCA**: Fits in batches for large datasets
3. **Float32**: Use instead of float64 for FAISS
4. **Embedding Cache**: Save to disk to avoid regeneration

### Speed Optimization
1. **Batch Processing**: Optimized batch sizes for CPU
2. **Multiprocessing**: Image download uses 100 workers
3. **Early Stopping**: Prevents overtraining
4. **FAISS**: Fast approximate nearest neighbors

### CPU vs GPU
- **CPU Mode**: Default, works on any machine
- **GPU Mode**: Set `device='cuda'` in embedding functions (requires CUDA)

---

## Expected SMAPE Progression

| Component | SMAPE | Improvement |
|-----------|-------|-------------|
| Baseline (numeric only) | 18-19% | - |
| + Text embeddings | 16-17% | -2% |
| + Image embeddings | 14-15% | -2% |
| + RAG features | 12-13% | -2% |
| + LightGBM ensemble | 11-12% | -1% |
| + Ridge meta | 10.5-11.5% | -0.5% |
| + Post-processing | 9.5-10.5% | -1% |

**Target**: 9.5-10.5% SMAPE (top 1% leaderboard on CPU)

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory
**Symptoms**: Process killed, RAM exhausted  
**Solutions**:
- Reduce batch sizes: `TEXT_BATCH_SIZE=32`, `IMAGE_BATCH_SIZE=8`
- Reduce PCA dimensions: `TEXT_DIM_REDUCED=128`, `IMAGE_DIM_REDUCED=128`
- Process data in chunks

#### 2. CUDA Out of Memory (if using GPU)
**Symptoms**: RuntimeError: CUDA out of memory  
**Solutions**:
- Use CPU mode: `device='cpu'`
- Reduce batch size
- Clear cache: `torch.cuda.empty_cache()`

#### 3. Missing Images
**Symptoms**: Image download failures  
**Solutions**:
- Check internet connection
- Images are optional (fallback to gray placeholder)
- Download takes 10-20 minutes

#### 4. Slow FAISS Search
**Symptoms**: RAG computation takes >30 minutes  
**Solutions**:
- Reduce k: `RAG_K=5`
- Use approximate search: `faiss.IndexIVFFlat`
- Reduce feature dimensions

#### 5. Poor SMAPE Score
**Symptoms**: Score >15%  
**Solutions**:
- Check embeddings are generated correctly
- Verify RAG features are computed
- Increase LightGBM rounds: `LGB_NUM_ROUNDS=3000`
- Tune hyperparameters

#### 6. Feature Dimension Mismatch
**Symptoms**: Shape errors during training  
**Solutions**:
- Delete cache folder and regenerate embeddings
- Check all features are aligned (train/test same dimensions)

---

## Customization

### Adjust Hyperparameters

Edit `src/config.py`:

```python
# For faster training (lower accuracy)
LGB_NUM_ROUNDS = 1000
TEXT_DIM_REDUCED = 128
IMAGE_DIM_REDUCED = 128
RAG_K = 5

# For better accuracy (slower)
LGB_NUM_ROUNDS = 3000
N_FOLDS = 10
N_SEEDS = 3  # Train with multiple seeds
```

### Add Custom Features

Edit `src/preprocessing.py`:

```python
def extract_numeric_features(df):
    # Add custom feature
    df['has_organic'] = df['catalog_content'].str.contains('organic').astype(int)
    # ... rest of function
```

### Use Different Embeddings

Edit `src/config.py`:

```python
# Larger model (better but slower)
TEXT_MODEL = "all-mpnet-base-v2"  # 768D

# Smaller model (faster but worse)
TEXT_MODEL = "all-MiniLM-L12-v2"  # 384D
```

---

## File Size Reference

| Component | Size |
|-----------|------|
| train.csv | 50-100 MB |
| test.csv | 50-100 MB |
| Images (train) | 1-2 GB |
| Images (test) | 1-2 GB |
| Text embeddings | 100-200 MB |
| Image embeddings | 100-200 MB |
| Trained models | 50-100 MB |
| Total | ~5 GB |

---

## Performance Benchmarks

### System: Intel i7 / 16GB RAM / CPU Only

| Stage | Time |
|-------|------|
| Data loading | 1-2 min |
| Image download | 10-20 min |
| Text embeddings | 5-10 min |
| Image embeddings | 15-25 min |
| RAG computation | 5-10 min |
| Model training | 10-20 min |
| **Total** | **30-60 min** |

---

## References

- [Sentence Transformers](https://www.sbert.net/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Verify environment with `python validate.py`
4. Check data file integrity
5. Try with smaller dataset first

---

*Last Updated: 2025-01-11*
