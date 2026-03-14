import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def plot_cv_results(cv_results_dict, save_path=None, figsize=(14, 10)):
    """
    Visualize cross-validation results for multiple models
    
    Args:
        cv_results_dict: Dict of {model_name: cv_results}
        save_path: Path to save figure
    """
    n_models = len(cv_results_dict)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    models = list(cv_results_dict.keys())
    metrics = ['val_accuracy', 'val_f1', 'val_precision', 'val_recall']
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    for idx, (ax, metric, metric_name) in enumerate(zip(axes.flat, metrics, metric_names)):
        # Collect data for box plot
        data_to_plot = []
        labels = []
        
        for model in models:
            values = cv_results_dict[model]['summary'][metric]['values']
            data_to_plot.append(values)
            labels.append(model)
        
        # Box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} across Folds', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean values as text
        for i, model in enumerate(models):
            mean_val = cv_results_dict[model]['summary'][metric]['mean']
            ax.text(i + 1, mean_val, f'{mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Cross-Validation Results Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved CV visualization: {save_path}")
    
    return fig

def plot_fold_performance(cv_results, save_path=None, figsize=(12, 6)):
    """
    Plot performance across folds for a single model
    
    Args:
        cv_results: CV results for one model
        save_path: Path to save figure
    """
    fold_results = cv_results['fold_results']
    n_folds = len(fold_results)
    
    folds = [f"Fold {i}" for i in range(1, n_folds + 1)]
    
    metrics = {
        'Accuracy': [f['val_accuracy'] for f in fold_results],
        'Precision': [f['val_precision'] for f in fold_results],
        'Recall': [f['val_recall'] for f in fold_results],
        'F1-Score': [f['val_f1'] for f in fold_results]
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_folds)
    width = 0.2
    
    colors = ['steelblue', 'coral', 'seagreen', 'gold']
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric_name, 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f"{cv_results['model_name'].upper()} - Performance Across Folds", 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved fold performance plot: {save_path}")
    
    return fig

if __name__ == "__main__":
    # Example usage
    from config import RESULTS_DIR
    
    cv_dir = RESULTS_DIR / "cross_validation"
    
    if not cv_dir.exists():
        print("No CV results found. Run 6_cross_validation.py first.")
    else:
        # Load all CV results
        cv_results_dict = {}
        
        for model_dir in cv_dir.iterdir():
            if model_dir.is_dir():
                json_files = list(model_dir.glob("cv_*fold_results.json"))
                if json_files:
                    with open(json_files[0], 'r') as f:
                        cv_results_dict[model_dir.name] = json.load(f)
        
        if cv_results_dict:
            # Plot comparison
            plot_cv_results(cv_results_dict, save_path=cv_dir / "cv_comparison.png")
            
            # Plot individual models
            for model_name, results in cv_results_dict.items():
                plot_fold_performance(
                    results,
                    save_path=cv_dir / model_name / "fold_performance.png"
                )
            
            plt.show()
            print("\n✅ Visualizations complete!")