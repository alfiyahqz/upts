import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from config import DATASET_DIR, CSV_COLUMNS, CLASS_MAPPING
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """
    Comprehensive analysis of the dataset to understand:
    - Number of users per class
    - Post count distribution
    - Timestamp ranges
    - Text statistics
    """
    
    print("=" * 70)
    print(" " * 20 + "DATASET ANALYSIS")
    print("=" * 70)
    
    dataset_dir = Path(DATASET_DIR)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Storage for analysis
    all_users = []
    class_stats = defaultdict(lambda: {
        'n_users': 0,
        'n_posts': [],
        'users': []
    })
    
    errors = []
    
    # Iterate through all classes
    for class_name in CLASS_MAPPING.keys():
        class_dir = dataset_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Directory not found: {class_dir}")
            continue
        
        csv_files = list(class_dir.glob("*.csv"))
        print(f"\n📁 Analyzing {class_name}/")
        print(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Load CSV
                df = pd.read_csv(csv_file, quoting=1)
                
                if len(df) == 0:
                    errors.append(f"{csv_file.name}: Empty file")
                    continue
                
                # Get number of posts
                n_posts = len(df)
                
                # Get user_id
                if CSV_COLUMNS['user_id'] in df.columns:
                    user_id = str(df[CSV_COLUMNS['user_id']].iloc[0])
                else:
                    user_id = csv_file.stem
                
                # Store info
                user_info = {
                    'user_id': user_id,
                    'filename': csv_file.name,
                    'class': class_name,
                    'n_posts': n_posts,
                    'file_path': str(csv_file)
                }
                
                all_users.append(user_info)
                class_stats[class_name]['n_users'] += 1
                class_stats[class_name]['n_posts'].append(n_posts)
                class_stats[class_name]['users'].append(user_info)
                
            except Exception as e:
                errors.append(f"{csv_file.name}: {str(e)}")
    
    # Print errors if any
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    
    total_users = len(all_users)
    all_post_counts = [u['n_posts'] for u in all_users]
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total users: {total_users}")
    print(f"   Total posts: {sum(all_post_counts):,}")
    print(f"   Average posts per user: {np.mean(all_post_counts):.1f}")
    print(f"   Median posts per user: {np.median(all_post_counts):.0f}")
    print(f"   Min posts: {min(all_post_counts) if all_post_counts else 0}")
    print(f"   Max posts: {max(all_post_counts) if all_post_counts else 0}")
    
    # Class-wise statistics
    print("\n" + "=" * 70)
    print("CLASS-WISE STATISTICS")
    print("=" * 70)
    
    for class_name, stats in class_stats.items():
        n_users = stats['n_users']
        post_counts = stats['n_posts']
        
        if not post_counts:
            continue
        
        print(f"\n📌 {class_name}:")
        print(f"   Users: {n_users}")
        print(f"   Total posts: {sum(post_counts):,}")
        print(f"   Average posts: {np.mean(post_counts):.1f}")
        print(f"   Median posts: {np.median(post_counts):.0f}")
        print(f"   Std posts: {np.std(post_counts):.1f}")
        print(f"   Min posts: {min(post_counts)}")
        print(f"   Max posts: {max(post_counts)}")
        
        # Percentiles
        p25 = np.percentile(post_counts, 25)
        p50 = np.percentile(post_counts, 50)
        p75 = np.percentile(post_counts, 75)
        p90 = np.percentile(post_counts, 90)
        p95 = np.percentile(post_counts, 95)
        
        print(f"   Percentiles:")
        print(f"      25th: {p25:.0f}")
        print(f"      50th: {p50:.0f}")
        print(f"      75th: {p75:.0f}")
        print(f"      90th: {p90:.0f}")
        print(f"      95th: {p95:.0f}")
    
    # Post count distribution
    print("\n" + "=" * 70)
    print("POST COUNT DISTRIBUTION")
    print("=" * 70)
    
    # Create bins
    bins = [0, 10, 20, 50, 100, 200, 500, 1000, 5000, float('inf')]
    bin_labels = ['0-10', '11-20', '21-50', '51-100', '101-200', '201-500', '501-1000', '1001-5000', '5000+']
    
    print(f"\n{'Range':<15} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        
        count = sum(1 for pc in all_post_counts if lower < pc <= upper)
        percentage = (count / total_users * 100) if total_users > 0 else 0
        
        print(f"{bin_labels[i]:<15} {count:<10} {percentage:>6.2f}%")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR MIN/MAX POST THRESHOLDS")
    print("=" * 70)
    
    p10 = np.percentile(all_post_counts, 10)
    p25 = np.percentile(all_post_counts, 25)
    p75 = np.percentile(all_post_counts, 75)
    p90 = np.percentile(all_post_counts, 90)
    
    print(f"\n💡 Suggested thresholds based on your data:")
    print(f"\n   Conservative (keep most users):")
    print(f"      min_posts = {int(p10)}  (keeps ~90% of users)")
    print(f"      max_posts = {int(p90)}  (truncates top ~10%)")
    
    print(f"\n   Moderate (balance quality and quantity):")
    print(f"      min_posts = {int(p25)}  (keeps ~75% of users)")
    print(f"      max_posts = {int(p75)}  (truncates top ~25%)")
    
    print(f"\n   Strict (high quality):")
    print(f"      min_posts = 50  (ensures sufficient data)")
    print(f"      max_posts = 200 (computational efficiency)")
    
    # Show impact of different thresholds
    print(f"\n📊 Impact of different thresholds:")
    
    thresholds = [
        (5, None, "Very inclusive"),
        (10, None, "Minimal quality"),
        (20, None, "Low quality bar"),
        (50, None, "Moderate quality"),
        (10, 200, "Min 10, Max 200"),
        (20, 200, "Min 20, Max 200"),
        (50, 200, "Min 50, Max 200"),
    ]
    
    print(f"\n{'Threshold':<20} {'Users Kept':<15} {'% Kept':<10} {'Description':<20}")
    print("-" * 70)
    
    for min_p, max_p, desc in thresholds:
        if max_p is None:
            kept = sum(1 for pc in all_post_counts if pc >= min_p)
        else:
            kept = sum(1 for pc in all_post_counts if min_p <= pc <= max_p)
        
        pct = (kept / total_users * 100) if total_users > 0 else 0
        
        threshold_str = f"min={min_p}"
        if max_p:
            threshold_str += f", max={max_p}"
        
        print(f"{threshold_str:<20} {kept:<15} {pct:>6.2f}%    {desc:<20}")
    
    # Users with very few posts
    print("\n" + "=" * 70)
    print("USERS WITH VERY FEW POSTS (< 10)")
    print("=" * 70)
    
    few_posts_users = [u for u in all_users if u['n_posts'] < 10]
    
    if few_posts_users:
        print(f"\n⚠️  Found {len(few_posts_users)} users with < 10 posts:")
        print(f"\n{'User ID':<30} {'Class':<15} {'Posts':<10} {'Filename':<30}")
        print("-" * 90)
        
        for user in sorted(few_posts_users, key=lambda x: x['n_posts'])[:20]:
            print(f"{user['user_id']:<30} {user['class']:<15} {user['n_posts']:<10} {user['filename']:<30}")
        
        if len(few_posts_users) > 20:
            print(f"\n   ... and {len(few_posts_users) - 20} more")
    else:
        print("\n✅ All users have >= 10 posts")
    
    # Users with very many posts
    print("\n" + "=" * 70)
    print("USERS WITH VERY MANY POSTS (> 500)")
    print("=" * 70)
    
    many_posts_users = [u for u in all_users if u['n_posts'] > 500]
    
    if many_posts_users:
        print(f"\n📊 Found {len(many_posts_users)} users with > 500 posts:")
        print(f"\n{'User ID':<30} {'Class':<15} {'Posts':<10} {'Filename':<30}")
        print("-" * 90)
        
        for user in sorted(many_posts_users, key=lambda x: x['n_posts'], reverse=True)[:20]:
            print(f"{user['user_id']:<30} {user['class']:<15} {user['n_posts']:<10} {user['filename']:<30}")
        
        if len(many_posts_users) > 20:
            print(f"\n   ... and {len(many_posts_users) - 20} more")
    else:
        print("\n✅ No users have > 500 posts")
    
    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 70)
    
    create_visualizations(all_users, class_stats, all_post_counts)
    
    # Save detailed report
    save_detailed_report(all_users, class_stats)
    
    print("\n✅ Analysis complete!")
    
    return all_users, class_stats

def create_visualizations(all_users, class_stats, all_post_counts):
    """Create visualizations of the dataset"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall post count distribution (histogram)
    ax = axes[0, 0]
    ax.hist(all_post_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Posts', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('Distribution of Posts per User', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(all_post_counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_post_counts):.1f}')
    ax.axvline(np.median(all_post_counts), color='green', linestyle='--', 
               label=f'Median: {np.median(all_post_counts):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot by class
    ax = axes[0, 1]
    
    class_names = []
    class_data = []
    
    for class_name, stats in class_stats.items():
        if stats['n_posts']:
            class_names.append(class_name)
            class_data.append(stats['n_posts'])
    
    bp = ax.boxplot(class_data, labels=class_names, patch_artist=True, showmeans=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Posts', fontsize=12)
    ax.set_title('Posts per User by Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. User count by class
    ax = axes[1, 0]
    
    class_counts = [stats['n_users'] for stats in class_stats.values()]
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    
    bars = ax.bar(class_names, class_counts, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('Number of Users per Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    
    sorted_counts = np.sort(all_post_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    
    ax.plot(sorted_counts, cumulative, linewidth=2, color='darkblue')
    ax.set_xlabel('Number of Posts', fontsize=12)
    ax.set_ylabel('Cumulative Percentage of Users (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Posts per User', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    for pct in [25, 50, 75, 90]:
        idx = int(len(sorted_counts) * pct / 100)
        if idx < len(sorted_counts):
            val = sorted_counts[idx]
            ax.axhline(pct, color='red', linestyle='--', alpha=0.3)
            ax.axvline(val, color='red', linestyle='--', alpha=0.3)
            ax.text(val, pct + 2, f'{pct}%: {val:.0f} posts', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Saved visualization: analysis_results/dataset_analysis.png")
    
    # Show plot
    plt.show()

def save_detailed_report(all_users, class_stats):
    """Save detailed report to CSV and text files"""
    
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save user list to CSV
    df_users = pd.DataFrame(all_users)
    df_users = df_users.sort_values('n_posts', ascending=False)
    df_users.to_csv(output_dir / 'user_post_counts.csv', index=False)
    print(f"   ✓ Saved user details: analysis_results/user_post_counts.csv")
    
    # Save summary statistics
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("DATASET SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        all_post_counts = [u['n_posts'] for u in all_users]
        
        f.write(f"Total users: {len(all_users)}\n")
        f.write(f"Total posts: {sum(all_post_counts):,}\n")
        f.write(f"Average posts per user: {np.mean(all_post_counts):.2f}\n")
        f.write(f"Median posts per user: {np.median(all_post_counts):.0f}\n")
        f.write(f"Std deviation: {np.std(all_post_counts):.2f}\n")
        f.write(f"Min posts: {min(all_post_counts)}\n")
        f.write(f"Max posts: {max(all_post_counts)}\n\n")
        
        f.write("CLASS-WISE STATISTICS\n")
        f.write("-" * 70 + "\n\n")
        
        for class_name, stats in class_stats.items():
            post_counts = stats['n_posts']
            if not post_counts:
                continue
            
            f.write(f"{class_name}:\n")
            f.write(f"  Users: {stats['n_users']}\n")
            f.write(f"  Total posts: {sum(post_counts):,}\n")
            f.write(f"  Average: {np.mean(post_counts):.2f}\n")
            f.write(f"  Median: {np.median(post_counts):.0f}\n")
            f.write(f"  Std: {np.std(post_counts):.2f}\n")
            f.write(f"  Min: {min(post_counts)}\n")
            f.write(f"  Max: {max(post_counts)}\n\n")
    
    print(f"   ✓ Saved summary: analysis_results/summary_statistics.txt")

if __name__ == "__main__":
    all_users, class_stats = analyze_dataset()