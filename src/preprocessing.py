"""
Data preprocessing and feature engineering functions
"""

import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_ipq_value(text):
    """Extract IPQ value from text"""
    if pd.isna(text):
        return 0.0
    
    text = str(text).lower()
    
    # Try to find numeric value followed by unit
    patterns = [
        r'(\d+\.?\d*)\s*(?:ounce|oz|fl oz|count|gram|g|kg|lb|pound|ml|liter|l)',
        r'(\d+\.?\d*)\s*(?:pack|count|ct)',
        r'(\d+\.?\d*)x',
        r'(\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    
    return 0.0

def extract_unit(text):
    """Extract unit type from text"""
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    
    # Unit encoding
    if 'fl oz' in text or 'fluid ounce' in text:
        return 2
    elif 'ounce' in text or 'oz' in text:
        return 1
    elif 'count' in text or 'pack' in text or 'ct' in text:
        return 3
    elif 'gram' in text or ' g ' in text:
        return 4
    elif 'kg' in text or 'kilogram' in text:
        return 5
    elif 'lb' in text or 'pound' in text:
        return 6
    elif 'ml' in text or 'milliliter' in text:
        return 7
    elif 'liter' in text or ' l ' in text:
        return 8
    
    return 0

def count_keywords(text, keywords):
    """Count occurrences of keywords in text"""
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    count = 0
    for keyword in keywords:
        count += text.count(keyword)
    
    return count

def extract_numeric_features(df):
    """Extract all numeric features from catalog content"""
    print("Extracting numeric features...")
    
    # Basic text features
    df['text_length'] = df['catalog_content'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['word_count'] = df['catalog_content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['digit_count'] = df['catalog_content'].apply(lambda x: sum(c.isdigit() for c in str(x)) if pd.notna(x) else 0)
    
    # IPQ and Unit extraction
    df['ipq_value'] = df['catalog_content'].apply(extract_ipq_value)
    df['unit_type'] = df['catalog_content'].apply(extract_unit)
    
    # Log transform
    df['log_ipq'] = np.log1p(df['ipq_value'])
    
    # Ratios
    df['digits_per_word'] = df['digit_count'] / (df['word_count'] + 1)
    
    # Keyword counts
    premium_keywords = ['premium', 'luxury', 'deluxe', 'gourmet', 'organic']
    bulk_keywords = ['bulk', 'value', 'pack', 'bundle', 'economy']
    quality_keywords = ['fresh', 'natural', 'pure', 'original', 'authentic']
    
    df['premium_count'] = df['catalog_content'].apply(lambda x: count_keywords(x, premium_keywords))
    df['bulk_count'] = df['catalog_content'].apply(lambda x: count_keywords(x, bulk_keywords))
    df['quality_count'] = df['catalog_content'].apply(lambda x: count_keywords(x, quality_keywords))
    
    # Adjective-like count
    adjective_keywords = ['new', 'best', 'great', 'special', 'limited', 'exclusive']
    df['adjective_count'] = df['catalog_content'].apply(lambda x: count_keywords(x, adjective_keywords))
    df['adjectives_per_word'] = df['adjective_count'] / (df['word_count'] + 1)
    
    # Cross features
    df['unit_x_ipq'] = df['unit_type'] * df['log_ipq']
    
    return df

def load_and_clean_data(train_path, test_path):
    """Load and clean training and test data"""
    print("Loading data...")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Clean catalog content
    print("Cleaning text...")
    train_df['catalog_content'] = train_df['catalog_content'].apply(clean_text)
    test_df['catalog_content'] = test_df['catalog_content'].apply(clean_text)
    
    # Clip extreme prices for stability
    if 'price' in train_df.columns:
        price_99 = train_df['price'].quantile(0.99)
        price_1 = train_df['price'].quantile(0.01)
        train_df['price'] = train_df['price'].clip(price_1, price_99)
    
    # Extract numeric features
    train_df = extract_numeric_features(train_df)
    test_df = extract_numeric_features(test_df)
    
    return train_df, test_df

def get_numeric_feature_names():
    """Return list of numeric feature names"""
    return [
        'text_length', 'word_count', 'digit_count',
        'ipq_value', 'unit_type', 'log_ipq',
        'digits_per_word', 'premium_count', 'bulk_count',
        'quality_count', 'adjective_count', 'adjectives_per_word',
        'unit_x_ipq'
    ]
