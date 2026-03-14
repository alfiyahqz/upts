import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, class_names=None):
        """
        Args:
            class_names: List of class names (e.g., ['depression', 'anxiety', 'stress'])
        """
        self.class_names = class_names
        
    def calculate_metrics(self, y_true, y_pred, y_probs=None):
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (n_samples, n_classes)
        
        Returns:
            metrics: Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Weighted metrics (accounts for class imbalance)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Macro metrics (treats all classes equally)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Per-class metrics
        precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'weighted': {
                'precision': float(precision_w),
                'recall': float(recall_w),
                'f1': float(f1_w)
            },
            'macro': {
                'precision': float(precision_m),
                'recall': float(recall_m),
                'f1': float(f1_m)
            },
            'per_class': {},
            'confusion_matrix': cm.tolist(),
            'support': support.tolist()
        }
        
        # Add per-class metrics
        n_classes = len(precision_pc)
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Class_{i}"
            metrics['per_class'][class_name] = {
                'precision': float(precision_pc[i]),
                'recall': float(recall_pc[i]),
                'f1': float(f1_pc[i]),
                'support': int(support[i])
            }
        
        # AUC if probabilities provided
        if y_probs is not None:
            try:
                if n_classes == 2:
                    # Binary classification
                    auc_score = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['auc'] = float(auc_score)
                else:
                    # Multi-class classification (one-vs-rest)
                    auc_score = roc_auc_score(
                        y_true, y_probs, multi_class='ovr', average='weighted'
                    )
                    metrics['auc'] = float(auc_score)
            except Exception as e:
                print(f"⚠️  Could not calculate AUC: {e}")
        
        return metrics
    
    def print_metrics(self, metrics, title="Evaluation Results"):
        """Print metrics in a readable format"""
        print("=" * 60)
        print(title)
        print("=" * 60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        
        if 'auc' in metrics:
            print(f"   AUC:      {metrics['auc']:.4f}")
        
        print(f"\n   Weighted Average (accounts for class imbalance):")
        print(f"      Precision: {metrics['weighted']['precision']:.4f}")
        print(f"      Recall:    {metrics['weighted']['recall']:.4f}")
        print(f"      F1-Score:  {metrics['weighted']['f1']:.4f}")
        
        print(f"\n   Macro Average (treats all classes equally):")
        print(f"      Precision: {metrics['macro']['precision']:.4f}")
        print(f"      Recall:    {metrics['macro']['recall']:.4f}")
        print(f"      F1-Score:  {metrics['macro']['f1']:.4f}")
        
        print(f"\n📋 Per-Class Metrics:")
        print(f"   {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("   " + "-" * 60)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"   {class_name:<15} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['f1']:<12.4f} "
                  f"{class_metrics['support']:<10d}")
        
        print("\n" + "=" * 60)
    
    def plot_confusion_matrix(self, cm, title="Confusion Matrix", 
                             save_path=None, figsize=(8, 6)):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix (numpy array or list)
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        cm = np.array(cm)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(len(cm)),
            yticklabels=self.class_names if self.class_names else range(len(cm)),
            ax=ax,
            cbar_kws={'label': 'Percentage'}
        )
        
        # Add counts
        for i in range(len(cm)):
            for j in range(len(cm)):
                text = ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                             ha="center", va="center", color="gray", fontsize=9)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved confusion matrix: {save_path}")
        
        return fig
    
    def plot_roc_curves(self, y_true, y_probs, title="ROC Curves", 
                       save_path=None, figsize=(10, 8)):
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_probs: Prediction probabilities (n_samples, n_classes)
            title: Plot title
            save_path: Path to save figure
        """
        n_classes = y_probs.shape[1]
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            ax.plot(
                fpr[i], tpr[i], 
                color=color, 
                lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})'
            )
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved ROC curves: {save_path}")
        
        return fig
    
    def plot_training_history(self, history, title="Training History", 
                             save_path=None, figsize=(12, 5)):
        """
        Plot training history (loss and accuracy)
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            title: Plot title
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved training history: {save_path}")
        
        return fig
    
    def plot_class_distribution(self, y_true, y_pred, title="Class Distribution", 
                               save_path=None, figsize=(10, 6)):
        """
        Plot comparison of true vs predicted class distributions
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        # Count classes
        true_counts = [np.sum(y_true == i) for i in range(len(self.class_names))]
        pred_counts = [np.sum(y_pred == i) for i in range(len(self.class_names))]
        
        # Plot bars
        ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='coral')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (true_c, pred_c) in enumerate(zip(true_counts, pred_counts)):
            ax.text(i - width/2, true_c + 0.5, str(true_c), 
                   ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, pred_c + 0.5, str(pred_c), 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved class distribution: {save_path}")
        
        return fig
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """
        Generate sklearn classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report
        
        Returns:
            report_str: Classification report as string
        """
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names if self.class_names else None,
            digits=4,
            zero_division=0
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"   ✓ Saved classification report: {save_path}")
        
        return report
    
    def compare_models(self, results_dict, metric='f1', save_path=None, 
                      figsize=(10, 6)):
        """
        Compare multiple models
        
        Args:
            results_dict: Dict of {model_name: metrics_dict}
            metric: Metric to compare ('accuracy', 'f1', 'precision', 'recall')
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(results_dict.keys())
        
        # Get metric values
        if metric == 'accuracy':
            values = [results_dict[m]['accuracy'] for m in models]
        else:
            values = [results_dict[m]['weighted'][metric] for m in models]
        
        # Sort by performance
        sorted_indices = np.argsort(values)[::-1]
        models = [models[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.barh(models, values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14, pad=15)
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved model comparison: {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test evaluator
    print("Testing ModelEvaluator...")
    
    # Dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
    y_probs = np.random.rand(10, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    
    class_names = ['Depression', 'Anxiety', 'Stress']
    
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_probs)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'])
    plt.show()
    
    print("\n✅ ModelEvaluator working correctly!")