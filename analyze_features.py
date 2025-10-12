import pickle
import numpy as np
import pandas as pd

# Load the saved models
with open('cache/trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

feature_importance = models['feature_importance']

# Construct feature names
feature_names = []

# Text embeddings (256)
for i in range(256):
    feature_names.append(f'text_emb_{i}')

# Image embeddings (256)
for i in range(256):
    feature_names.append(f'image_emb_{i}')

# Numeric features (13)
numeric_features = ['num_chars_title', 'num_chars_description', 'num_words_title',
                   'num_words_description', 'title_word_density', 'desc_word_density',
                   'title_desc_ratio', 'avg_word_length_title', 'avg_word_length_desc',
                   'has_price_in_text', 'price_position_ratio', 'text_similarity_score', 'combined_text_score']
feature_names.extend(numeric_features)

# RAG features (22)
rag_features = [
    'rag_mean_price', 'rag_std_price', 'rag_min_price', 'rag_max_price', 'rag_median_price',
    'rag_weighted_mean_uniform', 'rag_weighted_mean_inverse', 'rag_weighted_mean_gaussian', 'rag_weighted_mean_rank',
    'rag_price_range', 'rag_price_iqr', 'rag_price_skewness', 'rag_price_kurtosis',
    'rag_q10', 'rag_q25', 'rag_q75', 'rag_q90',
    'rag_confidence_interval',
    'rag_mean_distance', 'rag_std_distance', 'rag_max_similarity',
    'rag_category_diversity'
]
feature_names.extend(rag_features)

print(f'Total features: {len(feature_names)}, Importance values: {len(feature_importance)}')

# Create DataFrame for feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

print('Top 20 most important features:')
for i, row in importance_df.head(20).iterrows():
    print(f'{i+1:2d}. {row["feature"]:<30} {row["importance"]:.4f}')

print('\n' + '='*60)
print('RAG feature importance analysis:')
rag_features_df = importance_df[importance_df['feature'].str.startswith('rag_')]
print(f'Number of RAG features: {len(rag_features_df)}')
print(f'Total RAG importance: {rag_features_df["importance"].sum():.4f}')
print(f'Average RAG importance: {rag_features_df["importance"].mean():.4f}')
print(f'Max RAG importance: {rag_features_df["importance"].max():.4f}')

print('\nTop 10 RAG features:')
for i, row in rag_features_df.head(10).iterrows():
    print(f'{i+1:2d}. {row["feature"]:<30} {row["importance"]:.4f}')

print('\n' + '='*60)
print('Feature type breakdown (top 10 avg importance):')
text_importance = importance_df[importance_df['feature'].str.startswith('text_emb')]['importance'].head(10).mean()
image_importance = importance_df[importance_df['feature'].str.startswith('image_emb')]['importance'].head(10).mean()
num_importance = importance_df[importance_df['feature'].str.startswith('num_')]['importance'].head(10).mean()
rag_top10_importance = rag_features_df['importance'].head(10).mean()

print(f'Text embeddings (top 10 avg): {text_importance:.4f}')
print(f'Image embeddings (top 10 avg): {image_importance:.4f}')
print(f'Numeric features (top 10 avg): {num_importance:.4f}')
print(f'RAG features (top 10 avg): {rag_top10_importance:.4f}')

print('\n' + '='*60)
print('RAG feature ranking:')
print(f'RAG features in top 50: {len(importance_df.head(50).merge(rag_features_df, on="feature"))}')
print(f'RAG features in top 100: {len(importance_df.head(100).merge(rag_features_df, on="feature"))}')
print(f'Best RAG rank: {importance_df[importance_df["feature"].isin(rag_features_df["feature"])].index[0] + 1}')

print('\n' + '='*60)
print('Top RAG features details:')
for i, row in rag_features_df.head(5).iterrows():
    rank = importance_df.index.get_loc(i) + 1
    print(f'Rank {rank:3d}: {row["feature"]:<30} Importance: {row["importance"]:.4f}')