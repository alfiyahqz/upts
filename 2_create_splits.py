import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
from config import (
    PROCESSED_DIR, 
    TRAIN_RATIO, 
    VAL_RATIO, 
    TEST_RATIO,
    RANDOM_SEED,
    CLASS_MAPPING
)

def create_splits_from_presplit(partition_csv_path):
    """
    Create train/val/test splits using existing partition file
    
    Args:
        partition_csv_path: Path to partition CSV file
    """
    print("=" * 60)
    print("CREATE SPLITS FROM EXISTING PARTITION FILE")
    print("=" * 60)
    
    from utils.data_loader import DataLoader
    
    # Initialize loader
    loader = DataLoader()
    
    # Load users with partition info
    users = loader.load_presplit_data(partition_csv_path)
    
    # Load preprocessed dataset
    print(f"\n📂 Loading preprocessed time series data...")
    data_path = PROCESSED_DIR / "dataset.npz"
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("   Please run 1_preprocess.py first.")
        return
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    user_ids = data['user_ids']
    
    print(f"   Loaded X shape: {X.shape}")
    print(f"   Loaded y shape: {y.shape}")
    
    # Create mapping from user_id to index
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    
    # Separate by partition
    train_indices = []
    val_indices = []
    test_indices = []
    
    partition_mapping = {
        'train': train_indices,
        'val': val_indices,
        'validation': val_indices,  # Alternative name
        'test': test_indices
    }
    
    missing_users = []
    
    for user in users:
        user_id = user['user_id']
        partition = user['partition'].lower()
        
        if user_id not in user_id_to_idx:
            missing_users.append(user_id)
            continue
        
        idx = user_id_to_idx[user_id]
        
        if partition in partition_mapping:
            partition_mapping[partition].append(idx)
        else:
            print(f"⚠️  Unknown partition '{partition}' for user {user_id}")
    
    if missing_users:
        print(f"\n⚠️  Warning: {len(missing_users)} users in partition file not found in preprocessed data:")
        print(f"   {missing_users[:10]}" + (" ..." if len(missing_users) > 10 else ""))
    
    # Convert to arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Extract splits
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    ids_train = user_ids[train_indices]
    ids_val = user_ids[val_indices]
    ids_test = user_ids[test_indices]
    
    # Print summary
    print(f"\n📊 Split sizes based on partition file:")
    total = len(X)
    print(f"   Train: {len(X_train):4d} users ({len(X_train)/total*100:5.1f}%)")
    print(f"   Val:   {len(X_val):4d} users ({len(X_val)/total*100:5.1f}%)")
    print(f"   Test:  {len(X_test):4d} users ({len(X_test)/total*100:5.1f}%)")
    print(f"   Total: {total:4d} users")
    
    # Verify no data leakage
    train_set = set(ids_train)
    val_set = set(ids_val)
    test_set = set(ids_test)
    
    assert len(train_set & val_set) == 0, "Data leakage: Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Data leakage: Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Data leakage: Val and Test overlap!"
    
    print(f"\n✅ No data leakage detected (all splits are disjoint)")
    
    # Print class distribution
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    print(f"\n📊 Class distribution per split:")
    
    for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"\n   {split_name}:")
        unique, counts = np.unique(split_y, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = reverse_mapping.get(label, f"Unknown_{label}")
            percentage = (count / len(split_y)) * 100
            print(f"      {class_name} (label={label}): {count:3d} users ({percentage:5.1f}%)")
    
    # Save splits
    split_dir = PROCESSED_DIR / "splits"
    split_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving splits...")
    
    np.savez(split_dir / "train.npz", X=X_train, y=y_train, user_ids=ids_train)
    np.savez(split_dir / "val.npz", X=X_val, y=y_val, user_ids=ids_val)
    np.savez(split_dir / "test.npz", X=X_test, y=y_test, user_ids=ids_test)
    
    print(f"   ✓ Saved: {split_dir}/train.npz")
    print(f"   ✓ Saved: {split_dir}/val.npz")
    print(f"   ✓ Saved: {split_dir}/test.npz")
    
    # Save split info
    info_path = split_dir / "split_info.txt"
    
    with open(info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAIN/VAL/TEST SPLIT INFORMATION (FROM PARTITION FILE)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Source Partition File: {partition_csv_path}\n\n")
        f.write(f"Total Users: {total}\n\n")
        
        f.write("Actual Sizes:\n")
        f.write(f"  Train: {len(X_train):4d} users ({len(X_train)/total*100:5.1f}%)\n")
        f.write(f"  Val:   {len(X_val):4d} users ({len(X_val)/total*100:5.1f}%)\n")
        f.write(f"  Test:  {len(X_test):4d} users ({len(X_test)/total*100:5.1f}%)\n\n")
        
        f.write("Class Distribution:\n")
        f.write("-" * 60 + "\n")
        
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            f.write(f"\n{split_name}:\n")
            unique, counts = np.unique(split_y, return_counts=True)
            for label, count in zip(unique, counts):
                class_name = reverse_mapping.get(label, f"Unknown_{label}")
                percentage = (count / len(split_y)) * 100
                f.write(f"  {class_name:15s} (label={label}): {count:3d} users ({percentage:5.1f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("User IDs per split:\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Train users ({len(ids_train)}):\n")
        f.write(", ".join(ids_train[:20]))
        if len(ids_train) > 20:
            f.write(f", ... (and {len(ids_train)-20} more)")
        f.write("\n\n")
        
        f.write(f"Val users ({len(ids_val)}):\n")
        f.write(", ".join(ids_val[:20]))
        if len(ids_val) > 20:
            f.write(f", ... (and {len(ids_val)-20} more)")
        f.write("\n\n")
        
        f.write(f"Test users ({len(ids_test)}):\n")
        f.write(", ".join(ids_test[:20]))
        if len(ids_test) > 20:
            f.write(f", ... (and {len(ids_test)-20} more)")
        f.write("\n")
    
    print(f"   ✓ Saved: {info_path}")
    print(f"\n✅ Split creation from partition file complete!")


def create_train_val_test_split():
    """
    Create stratified train/validation/test splits
    
    Saves:
        - processed_data/splits/train.npz
        - processed_data/splits/val.npz
        - processed_data/splits/test.npz
        - processed_data/splits/split_info.txt
    """
    print("=" * 60)
    print("STEP 2: CREATE TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    
    # Create output directory
    split_dir = PROCESSED_DIR / "splits"
    split_dir.mkdir(exist_ok=True)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    data_path = PROCESSED_DIR / "dataset.npz"
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("   Please run 1_preprocess.py first.")
        return
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    user_ids = data['user_ids']
    
    print(f"   Loaded X shape: {X.shape}")
    print(f"   Loaded y shape: {y.shape}")
    print(f"   Number of users: {len(user_ids)}")
    
    # Print original class distribution
    print(f"\n📊 Original class distribution:")
    unique_labels, counts = np.unique(y, return_counts=True)
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    for label, count in zip(unique_labels, counts):
        class_name = reverse_mapping.get(label, f"Unknown_{label}")
        percentage = (count / len(y)) * 100
        print(f"   {class_name} (label={label}): {count} users ({percentage:.1f}%)")
    
    # Check if we have enough samples
    min_samples_per_class = min(counts)
    if min_samples_per_class < 3:
        print(f"\n⚠️  Warning: Class with only {min_samples_per_class} samples!")
        print("   This may cause issues with stratified splitting.")
        print("   Consider collecting more data or using different split ratios.")
    
    # ==================== SPLIT STRATEGY ====================
    # Split into train (60%) and temp (40%)
    # Then split temp into val (50% of temp = 20% total) and test (50% of temp = 20% total)
    
    print(f"\n🔀 Creating splits with ratios:")
    print(f"   Train: {TRAIN_RATIO*100:.0f}%")
    print(f"   Val:   {VAL_RATIO*100:.0f}%")
    print(f"   Test:  {TEST_RATIO*100:.0f}%")
    print(f"   Random seed: {RANDOM_SEED}")
    
    # First split: Train vs (Val + Test)
    try:
        X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
            X, y, user_ids,
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\n✅ First split (Train vs Temp):")
        print(f"   Train: {len(X_train)} users")
        print(f"   Temp:  {len(X_temp)} users")
        
    except ValueError as e:
        print(f"\n❌ Error during first split: {e}")
        print("   This usually means you don't have enough samples per class.")
        print("   Try reducing the number of classes or collecting more data.")
        return
    
    # Second split: Val vs Test (from temp)
    try:
        val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        
        X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
            X_temp, y_temp, ids_temp,
            test_size=(1 - val_ratio_of_temp),
            random_state=RANDOM_SEED,
            stratify=y_temp  # Maintain class distribution
        )
        
        print(f"\n✅ Second split (Val vs Test):")
        print(f"   Val:  {len(X_val)} users")
        print(f"   Test: {len(X_test)} users")
        
    except ValueError as e:
        print(f"\n❌ Error during second split: {e}")
        print("   This usually means you don't have enough samples per class.")
        return
    
    # ==================== VERIFY SPLITS ====================
    print(f"\n📊 Final split sizes:")
    total = len(X)
    print(f"   Train: {len(X_train):4d} users ({len(X_train)/total*100:5.1f}%)")
    print(f"   Val:   {len(X_val):4d} users ({len(X_val)/total*100:5.1f}%)")
    print(f"   Test:  {len(X_test):4d} users ({len(X_test)/total*100:5.1f}%)")
    print(f"   Total: {total:4d} users")
    
    # Verify no data leakage
    train_set = set(ids_train)
    val_set = set(ids_val)
    test_set = set(ids_test)
    
    assert len(train_set & val_set) == 0, "Data leakage: Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Data leakage: Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Data leakage: Val and Test overlap!"
    
    print(f"\n✅ No data leakage detected (all splits are disjoint)")
    
    # Print class distribution for each split
    print(f"\n📊 Class distribution per split:")
    
    for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"\n   {split_name}:")
        unique, counts = np.unique(split_y, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = reverse_mapping.get(label, f"Unknown_{label}")
            percentage = (count / len(split_y)) * 100
            print(f"      {class_name} (label={label}): {count:3d} users ({percentage:5.1f}%)")
    
    # ==================== SAVE SPLITS ====================
    print(f"\n💾 Saving splits...")
    
    # Save train split
    np.savez(
        split_dir / "train.npz",
        X=X_train,
        y=y_train,
        user_ids=ids_train
    )
    print(f"   ✓ Saved train split: {split_dir}/train.npz")
    
    # Save validation split
    np.savez(
        split_dir / "val.npz",
        X=X_val,
        y=y_val,
        user_ids=ids_val
    )
    print(f"   ✓ Saved val split: {split_dir}/val.npz")
    
    # Save test split
    np.savez(
        split_dir / "test.npz",
        X=X_test,
        y=y_test,
        user_ids=ids_test
    )
    print(f"   ✓ Saved test split: {split_dir}/test.npz")
    
    # ==================== SAVE SPLIT INFO ====================
    info_path = split_dir / "split_info.txt"
    
    with open(info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAIN/VAL/TEST SPLIT INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Total Users: {total}\n\n")
        
        f.write("Split Ratios:\n")
        f.write(f"  Train: {TRAIN_RATIO*100:.0f}%\n")
        f.write(f"  Val:   {VAL_RATIO*100:.0f}%\n")
        f.write(f"  Test:  {TEST_RATIO*100:.0f}%\n\n")
        
        f.write("Actual Sizes:\n")
        f.write(f"  Train: {len(X_train):4d} users ({len(X_train)/total*100:5.1f}%)\n")
        f.write(f"  Val:   {len(X_val):4d} users ({len(X_val)/total*100:5.1f}%)\n")
        f.write(f"  Test:  {len(X_test):4d} users ({len(X_test)/total*100:5.1f}%)\n\n")
        
        f.write("Class Distribution:\n")
        f.write("-" * 60 + "\n")
        
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            f.write(f"\n{split_name}:\n")
            unique, counts = np.unique(split_y, return_counts=True)
            for label, count in zip(unique, counts):
                class_name = reverse_mapping.get(label, f"Unknown_{label}")
                percentage = (count / len(split_y)) * 100
                f.write(f"  {class_name:15s} (label={label}): {count:3d} users ({percentage:5.1f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("User IDs per split:\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Train users ({len(ids_train)}):\n")
        f.write(", ".join(ids_train[:20]))
        if len(ids_train) > 20:
            f.write(f", ... (and {len(ids_train)-20} more)")
        f.write("\n\n")
        
        f.write(f"Val users ({len(ids_val)}):\n")
        f.write(", ".join(ids_val[:20]))
        if len(ids_val) > 20:
            f.write(f", ... (and {len(ids_val)-20} more)")
        f.write("\n\n")
        
        f.write(f"Test users ({len(ids_test)}):\n")
        f.write(", ".join(ids_test[:20]))
        if len(ids_test) > 20:
            f.write(f", ... (and {len(ids_test)-20} more)")
        f.write("\n")
    
    print(f"   ✓ Saved split info: {info_path}")
    
    print(f"\n✅ Split creation complete!")
    print(f"\n📁 Files created:")
    print(f"   - {split_dir}/train.npz")
    print(f"   - {split_dir}/val.npz")
    print(f"   - {split_dir}/test.npz")
    print(f"   - {split_dir}/split_info.txt")
    
    return {
        'train': (X_train, y_train, ids_train),
        'val': (X_val, y_val, ids_val),
        'test': (X_test, y_test, ids_test)
    }


def main():
    # Check for partition file argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--verify":
            from utils.data_loader import verify_splits
            verify_splits()
            return
        elif arg.startswith("--partition="):
            partition_file = arg.split("=")[1]
            if not Path(partition_file).exists():
                print(f"❌ Error: Partition file not found: {partition_file}")
                return
            create_splits_from_presplit(partition_file)
            return
        elif Path(arg).exists() and arg.endswith('.csv'):
            # Assume it's a partition file
            create_splits_from_presplit(arg)
            return
        else:
            print(f"❌ Error: Invalid argument or file not found: {arg}")
            print("\nUsage:")
            print("  python 2_create_splits.py                    # Create stratified splits")
            print("  python 2_create_splits.py partition.csv      # Use existing partition file")
            print("  python 2_create_splits.py --partition=file.csv")
            print("  python 2_create_splits.py --verify           # Verify splits")
            return
    
    # Default: create stratified splits
    create_train_val_test_split()
    
    print("\n" + "=" * 60)
    print("Additional usage options:")
    print("  python 2_create_splits.py partition.csv          # Use existing partition")
    print("  python 2_create_splits.py --partition=file.csv   # Use existing partition")
    print("  python 2_create_splits.py --verify               # Verify splits")
    print("=" * 60)


if __name__ == "__main__":
    main()