import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re
import pandas as pd
from config import (
    EMBEDDING_MODEL_NAME, 
    TEXT_EMBEDDING_DIM, 
    TEMPORAL_FEATURES_DIM,
    FEATURE_DIM,
    MAX_SEQUENCE_LENGTH
)

class TemporalFeatureExtractor:
    """Extract temporal features from user posts"""
    
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        print(f"🤖 Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.feature_dim = FEATURE_DIM
        
    def extract_post_features(self, posts_df):
        """
        Extract features for all posts of a user
        
        Args:
            posts_df: DataFrame with columns [text, timestamp]
        
        Returns:
            features: numpy array of shape (n_posts, 771)
        """
        features = []
        previous_time = None
        
        # Get text embeddings for all posts at once (faster)
        texts = posts_df['text'].tolist()
        text_embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Get max time gap for normalization
        timestamps = posts_df['timestamp'].values
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
            time_diffs.append(diff)
        
        max_time_gap = max(time_diffs) if time_diffs else 1.0
        
        # Extract features for each post
        for idx, row in posts_df.iterrows():
            text_embedding = text_embeddings[idx]  # Shape: (768,)
            current_time = row['timestamp']
            
            # === Temporal Features ===
            
            # 1. Time gap since previous post (normalized to [0, 1])
            if previous_time is None:
                time_gap = 0.0
            else:
                gap_hours = (current_time - previous_time).total_seconds() / 3600
                time_gap = min(gap_hours / max_time_gap, 1.0) if max_time_gap > 0 else 0.0
            
            # 2. Hour of day (normalized to [0, 1])
            hour_of_day = current_time.hour / 24.0
            
            # 3. Day of week (normalized to [0, 1])
            day_of_week = current_time.weekday() / 7.0
            
            # Concatenate all features
            feature_vector = np.concatenate([
                text_embedding,      # (768,)
                [time_gap],          # (1,)
                [hour_of_day],       # (1,)
                [day_of_week]        # (1,)
            ])  # Total: (771,)
            
            features.append(feature_vector)
            previous_time = current_time
        
        return np.array(features)  # Shape: (n_posts, 771)
    
    def build_time_series(self, post_features, max_len=MAX_SEQUENCE_LENGTH):
        """
        Build fixed-length time series with padding/truncation
        
        Args:
            post_features: numpy array of shape (n_posts, 771)
            max_len: maximum sequence length
        
        Returns:
            time_series: numpy array of shape (max_len, 771)
        """
        n_posts = len(post_features)
        
        # Truncate if too long (keep most recent posts)
        if n_posts > max_len:
            sequence = post_features[-max_len:]
        else:
            sequence = post_features
        
        # Pad if too short (pad at the end with zeros)
        if n_posts < max_len:
            padding_size = max_len - n_posts
            zero_padding = np.zeros((padding_size, self.feature_dim))
            sequence = np.vstack([sequence, zero_padding])
        
        return sequence  # Shape: (max_len, 771)
    
    def extract_user_time_series(self, user_data):
        """
        Complete feature extraction pipeline for one user
        
        Args:
            user_data: Dictionary with 'posts' (DataFrame) and 'label'
        
        Returns:
            time_series: numpy array of shape (max_len, 771)
            label: int
            user_id: str
        """
        posts_df = user_data['posts']
        
        # Step 1: Extract features for each post
        post_features = self.extract_post_features(posts_df)
        
        # Step 2: Build fixed-length time series
        time_series = self.build_time_series(post_features)
        
        return {
            'user_id': user_data['user_id'],
            'label': user_data['label'],
            'time_series': time_series,
            'n_posts': len(posts_df)
        }

def clean_tweet_text(text, remove_emojis=False, remove_mentions=True, remove_urls=True):
    """
    Clean tweet text with robust handling of special characters
    
    Args:
        text: Raw tweet text
        remove_emojis: Whether to remove emoji characters
        remove_mentions: Whether to remove @mentions
        remove_urls: Whether to remove URLs
    
    Returns:
        cleaned_text: Cleaned text
    """
    if pd.isna(text) or text.strip() == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove URLs and URL placeholders
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+|HTTPURL', '', text, flags=re.IGNORECASE)
    
    # Remove @mentions
    if remove_mentions:
        text = re.sub(r'@\w+|@USER', '', text, flags=re.IGNORECASE)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove 'RT' at the beginning (retweet marker)
    text = re.sub(r'^RT\s+', '', text, flags=re.IGNORECASE)
    
    # Remove &amp; and other HTML entities (just in case they weren't decoded)
    text = re.sub(r'&amp;?', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    
    # Remove emojis (optional)
    if remove_emojis:
        # Remove emoji characters
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace and quotes
    text = text.strip().strip('"').strip("'").strip()
    
    return text

if __name__ == "__main__":
    # Test feature extractor
    from data_loader import DataLoader
    
    loader = DataLoader()
    users = loader.load_all_users()
    users = loader.filter_by_post_count(users, min_posts=10)
    
    if len(users) > 0:
        extractor = TemporalFeatureExtractor()
        
        print("\n🧪 Testing feature extraction on first user...")
        result = extractor.extract_user_time_series(users[0])
        
        print(f"   User ID: {result['user_id']}")
        print(f"   Label: {result['label']}")
        print(f"   Number of posts: {result['n_posts']}")
        print(f"   Time series shape: {result['time_series'].shape}")
        print(f"   Expected shape: ({MAX_SEQUENCE_LENGTH}, {FEATURE_DIM})")
        
        # Check first feature vector
        print(f"\n   First feature vector (first 10 values):")
        print(f"   {result['time_series'][0][:10]}")