import numpy as np
import os
from pathlib import Path
from utils.data_loader import DataLoader
from utils.feature_extractor import TemporalFeatureExtractor
from config import PROCESSED_DIR, MAX_SEQUENCE_LENGTH, FEATURE_DIM

def main():
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    # Create output directories
    emb_dir = PROCESSED_DIR / "embeddings"
    ts_dir = PROCESSED_DIR / "time_series"
    emb_dir.mkdir(exist_ok=True)
    ts_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    loader = DataLoader()
    users = loader.load_all_users()
    
    # Filter by post count
    users = loader.filter_by_post_count(users, min_posts=100, max_posts=None)
    
    if len(users) == 0:
        print("❌ No users found! Check your dataset directory.")
        return
    
    # Extract features
    print("\n🔧 Extracting temporal features...")
    extractor = TemporalFeatureExtractor()
    
    all_time_series = []
    all_labels = []
    all_user_ids = []
    
    for i, user in enumerate(users):
        if (i + 1) % 10 == 0:
            print(f"   Processing user {i+1}/{len(users)}...")
        
        try:
            result = extractor.extract_user_time_series(user)
            
            all_time_series.append(result['time_series'])
            all_labels.append(result['label'])
            all_user_ids.append(result['user_id'])
            
            # Save individual time series
            np.save(ts_dir / f"{result['user_id']}.npy", result['time_series'])
            
        except Exception as e:
            print(f"   ⚠️ Error processing {user['user_id']}: {e}")
            continue
    
    # Convert to arrays
    X = np.array(all_time_series)  # Shape: (n_users, max_len, 771)
    y = np.array(all_labels)        # Shape: (n_users,)
    user_ids = np.array(all_user_ids)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Expected X shape: (num_users, {MAX_SEQUENCE_LENGTH}, {FEATURE_DIM})")
    
    # Save complete dataset
    np.savez(
        PROCESSED_DIR / "dataset.npz",
        X=X,
        y=y,
        user_ids=user_ids
    )
    
    print(f"\n💾 Saved complete dataset to: {PROCESSED_DIR}/dataset.npz")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Final class distribution:")
    for label, count in zip(unique, counts):
        print(f"   Class {label}: {count} users")

if __name__ == "__main__":
    main()