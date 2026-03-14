import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import PROCESSED_DIR, CLASS_MAPPING

def analyze_split_statistics():
    """
    Generate comprehensive statistics about the train/val/test splits
    """
    split_dir = PROCESSED_DIR / "splits"
    
    if not (split_dir / "train.npz").exists():
        print("❌ Splits not found. Run 2_create_splits.py first.")
        return
    
    # Load splits
    train_data = np.load(split_dir / "train.npz", allow_pickle=True)
    val_data = np.load(split_dir / "val.npz", allow_pickle=True)
    test_data = np.load(split_dir / "test.npz", allow_pickle=True)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    print("=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    
    # ==================== FEATURE STATISTICS ====================
    print("\n📊 Feature Statistics (Train Set):")
    print(f"   Mean: {X_train.mean():.6f}")
    print(f"   Std:  {X_train.std():.6f}")
    print(f"   Min:  {X_train.min():.6f}")
    print(f"   Max:  {X_train.max():.6f}")
    
    # Check for NaN or Inf
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()
    
    if nan_count > 0:
        print(f"\n⚠️  Warning: Found {nan_count} NaN values in training data!")
    if inf_count > 0:
        print(f"\n⚠️  Warning: Found {inf_count} Inf values in training data!")
    
    # ==================== SEQUENCE LENGTH ANALYSIS ====================
    print(f"\n📏 Sequence Analysis:")
    
    for split_name, X_split in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        # Count non-zero sequences (actual posts vs padding)
        non_zero_counts = []
        for sample in X_split:
            # Count rows that are not all zeros
            non_zero = np.any(sample != 0, axis=1).sum()
            non_zero_counts.append(non_zero)
        
        print(f"\n   {split_name}:")
        print(f"      Mean posts per user: {np.mean(non_zero_counts):.1f}")
        print(f"      Median posts per user: {np.median(non_zero_counts):.0f}")
        print(f"      Min posts: {np.min(non_zero_counts)}")
        print(f"      Max posts: {np.max(non_zero_counts)}")
    
    # ==================== CLASS BALANCE ====================
    print(f"\n⚖️  Class Balance Analysis:")
    
    total_samples = len(y_train) + len(y_val) + len(y_test)
    
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"\n   {split_name}:")
        unique, counts = np.unique(y_split, return_counts=True)
        
        for label, count in zip(unique, counts):
            class_name = reverse_mapping.get(label, f"Class_{label}")
            percentage = (count / len(y_split)) * 100
            print(f"      {class_name:15s}: {count:3d} ({percentage:5.1f}%)")
    
    # ==================== VISUALIZATION ====================
    print(f"\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Class distribution across splits
    ax = axes[0, 0]
    splits_names = ['Train', 'Val', 'Test']
    splits_data = [y_train, y_val, y_test]
    
    class_counts = {label: [] for label in CLASS_MAPPING.values()}
    
    for y_split in splits_data:
        unique, counts = np.unique(y_split, return_counts=True)
        count_dict = dict(zip(unique, counts))
        for label in CLASS_MAPPING.values():
            class_counts[label].append(count_dict.get(label, 0))
    
    x = np.arange(len(splits_names))
    width = 0.25
    
    for i, (label, counts) in enumerate(class_counts.items()):
        class_name = reverse_mapping.get(label, f"Class_{label}")
        ax.bar(x + i*width, counts, width, label=class_name)
    
    ax.set_xlabel('Split')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution Across Splits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(splits_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sequence length distribution
    ax = axes[0, 1]
    
    all_lengths = []
    for sample in X_train:
        non_zero = np.any(sample != 0, axis=1).sum()
        all_lengths.append(non_zero)
    
    ax.hist(all_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Posts')
    ax.set_ylabel('Number of Users')
    ax.set_title('Post Count Distribution (Train Set)')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(all_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_lengths):.1f}')
    ax.legend()
    
    # 3. Feature value distribution (first 100 samples)
    ax = axes[1, 0]
    
    # Sample random features from first timestep
    sample_features = X_train[:100, 0, :].flatten()
    ax.hist(sample_features, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Value Distribution (Sample)')
    ax.grid(True, alpha=0.3)
    
    # 4. Class proportions pie chart
    ax = axes[1, 1]
    
    total_counts = []
    labels = []
    
    for label in sorted(CLASS_MAPPING.values()):
        class_name = reverse_mapping.get(label, f"Class_{label}")
        count = np.sum(y_train == label) + np.sum(y_val == label) + np.sum(y_test == label)
        total_counts.append(count)
        labels.append(f"{class_name}\n({count} users)")
    
    ax.pie(total_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Overall Class Distribution')
    
    plt.tight_layout()
    
    # Save figure
    stats_dir = PROCESSED_DIR / "statistics"
    stats_dir.mkdir(exist_ok=True)
    
    fig_path = stats_dir / "split_statistics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {fig_path}")
    
    plt.show()
    
    print(f"\n✅ Statistics analysis complete!")


if __name__ == "__main__":
    analyze_split_statistics()