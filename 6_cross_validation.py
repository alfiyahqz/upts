import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
import time
from config import (
    PROCESSED_DIR, 
    RESULTS_DIR, 
    NUM_CLASSES, 
    FEATURE_DIM,
    MODEL_LIST,
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
    CLASS_MAPPING
)
from utils.models import get_model
from utils.trainer import Trainer

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def perform_cross_validation(model_name, X, y, n_splits=5, device='cuda'):
    """
    Perform K-fold cross-validation for a single model
    
    Args:
        model_name: Name of the model
        X: Features (n_samples, seq_len, feature_dim)
        y: Labels (n_samples,)
        n_splits: Number of folds
        device: Device to use
    
    Returns:
        cv_results: Cross-validation results
    """
    print("=" * 60)
    print(f"CROSS-VALIDATION: {model_name.upper()} ({n_splits}-Fold)")
    print("=" * 60)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        print(f"\n   Train: {len(X_train_fold)} samples")
        print(f"   Val:   {len(X_val_fold)} samples")
        
        # Print class distribution
        reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
        print(f"\n   Class distribution:")
        for split_name, split_y in [('Train', y_train_fold), ('Val', y_val_fold)]:
            unique, counts = np.unique(split_y, return_counts=True)
            dist_str = ", ".join([
                f"{reverse_mapping.get(label, f'Class_{label}')}: {count}" 
                for label, count in zip(unique, counts)
            ])
            print(f"      {split_name}: {dist_str}")
        
        # Create fresh model for this fold
        model = get_model(model_name, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_classes=NUM_CLASSES
        )
        
        # Train
        fold_start_time = time.time()
        
        history = trainer.fit(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            epochs=200,  # Can adjust
            patience=15,
            verbose=False  # Less verbose for CV
        )
        
        fold_time = time.time() - fold_start_time
        
        # Evaluate
        metrics, preds, probs = trainer.evaluate(X_val_fold, y_val_fold)
        
        # Store results
        fold_result = {
            'fold': fold,
            'train_samples': len(X_train_fold),
            'val_samples': len(X_val_fold),
            'val_accuracy': float(metrics['accuracy']),
            'val_precision': float(metrics['precision']),
            'val_recall': float(metrics['recall']),
            'val_f1': float(metrics['f1']),
            'training_time': fold_time,
            'epochs_trained': len(history['train_loss'])
        }
        
        if 'auc' in metrics:
            fold_result['val_auc'] = float(metrics['auc'])
        
        fold_results.append(fold_result)
        fold_histories.append(history)
        
        # Print fold summary
        print(f"\n   ✅ Fold {fold} Complete:")
        print(f"      Accuracy:  {metrics['accuracy']:.4f}")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
        print(f"      F1-Score:  {metrics['f1']:.4f}")
        print(f"      Time:      {fold_time/60:.1f} minutes")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY: {model_name.upper()}")
    print(f"{'='*60}")
    
    metrics_to_aggregate = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
    if 'val_auc' in fold_results[0]:
        metrics_to_aggregate.append('val_auc')
    
    cv_summary = {}
    
    for metric in metrics_to_aggregate:
        values = [fold[metric] for fold in fold_results]
        cv_summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': [float(v) for v in values]
        }
    
    # Print summary
    print(f"\n📊 {n_splits}-Fold Cross-Validation Results:")
    print(f"   {'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"   {'-'*55}")
    
    for metric, stats in cv_summary.items():
        metric_name = metric.replace('val_', '').replace('_', ' ').title()
        print(f"   {metric_name:<15} "
              f"{stats['mean']:<10.4f} "
              f"{stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} "
              f"{stats['max']:<10.4f}")
    
    # Training time
    total_time = sum(fold['training_time'] for fold in fold_results)
    avg_time = total_time / n_splits
    
    print(f"\n⏱️  Training Time:")
    print(f"   Total:   {total_time/60:.1f} minutes")
    print(f"   Average: {avg_time/60:.1f} minutes per fold")
    
    cv_results = {
        'model_name': model_name,
        'n_splits': n_splits,
        'summary': cv_summary,
        'fold_results': fold_results,
        'total_training_time': total_time,
        'average_training_time': avg_time
    }
    
    return cv_results, fold_histories

def main():
    print("=" * 60)
    print("K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    # Load full dataset (not using pre-split)
    print(f"\n📂 Loading preprocessed data...")
    data_path = PROCESSED_DIR / "dataset.npz"
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("   Please run 1_preprocess.py first.")
        return
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    user_ids = data['user_ids']
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Total users: {len(user_ids)}")
    
    # Number of folds
    n_splits = 5
    print(f"\n🔄 Performing {n_splits}-fold cross-validation")
    
    # Create results directory
    cv_dir = RESULTS_DIR / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-validate each model
    all_cv_results = {}
    
    for model_name in MODEL_LIST:
        try:
            print(f"\n{'='*60}")
            print(f"Starting CV for: {model_name.upper()}")
            print(f"{'='*60}")
            
            cv_results, fold_histories = perform_cross_validation(
                model_name=model_name,
                X=X,
                y=y,
                n_splits=n_splits,
                device=device
            )
            
            all_cv_results[model_name] = cv_results
            
            # Save individual model results
            model_cv_dir = cv_dir / model_name
            model_cv_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CV results
            with open(model_cv_dir / f"cv_{n_splits}fold_results.json", 'w') as f:
                json.dump(cv_results, f, indent=2)
            
            print(f"\n💾 Saved results: {model_cv_dir}/cv_{n_splits}fold_results.json")
            
            # Save fold histories
            with open(model_cv_dir / f"cv_{n_splits}fold_histories.json", 'w') as f:
                # Convert to serializable format
                histories_serializable = [
                    {k: [float(v) for v in vals] for k, vals in hist.items()}
                    for hist in fold_histories
                ]
                json.dump(histories_serializable, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Error during CV for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==================== COMPARISON ====================
    if len(all_cv_results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (CROSS-VALIDATION)")
        print("=" * 60)
        
        # Create comparison table
        print(f"\n📊 Mean Performance Across {n_splits} Folds:")
        print(f"\n{'Model':<15} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 63)
        
        # Sort by F1 score
        sorted_models = sorted(
            all_cv_results.keys(),
            key=lambda x: all_cv_results[x]['summary']['val_f1']['mean'],
            reverse=True
        )
        
        for model_name in sorted_models:
            results = all_cv_results[model_name]['summary']
            print(f"{model_name:<15} "
                  f"{results['val_accuracy']['mean']:<12.4f} "
                  f"{results['val_f1']['mean']:<12.4f} "
                  f"{results['val_precision']['mean']:<12.4f} "
                  f"{results['val_recall']['mean']:<12.4f}")
        
        # Print with standard deviations
        print(f"\n📊 Performance with Standard Deviations:")
        print(f"\n{'Model':<15} {'Accuracy ± Std':<20} {'F1-Score ± Std':<20}")
        print("-" * 55)
        
        for model_name in sorted_models:
            results = all_cv_results[model_name]['summary']
            acc_mean = results['val_accuracy']['mean']
            acc_std = results['val_accuracy']['std']
            f1_mean = results['val_f1']['mean']
            f1_std = results['val_f1']['std']
            
            print(f"{model_name:<15} "
                  f"{acc_mean:.4f} ± {acc_std:.4f}      "
                  f"{f1_mean:.4f} ± {f1_std:.4f}")
        
        # Save comparison
        comparison = {}
        for model_name, results in all_cv_results.items():
            comparison[model_name] = {
                'accuracy_mean': results['summary']['val_accuracy']['mean'],
                'accuracy_std': results['summary']['val_accuracy']['std'],
                'f1_mean': results['summary']['val_f1']['mean'],
                'f1_std': results['summary']['val_f1']['std'],
                'precision_mean': results['summary']['val_precision']['mean'],
                'precision_std': results['summary']['val_precision']['std'],
                'recall_mean': results['summary']['val_recall']['mean'],
                'recall_std': results['summary']['val_recall']['std']
            }
        
        with open(cv_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Saved comparison: {cv_dir}/model_comparison.json")
        
        # Best model
        best_model = sorted_models[0]
        best_f1 = all_cv_results[best_model]['summary']['val_f1']['mean']
        best_std = all_cv_results[best_model]['summary']['val_f1']['std']
        
        print(f"\n🏆 Best Model: {best_model.upper()}")
        print(f"   F1-Score: {best_f1:.4f} ± {best_std:.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Completed {n_splits}-fold CV for {len(all_cv_results)} models")
    print(f"\n📁 Results saved to: {cv_dir}")
    print(f"\n   Individual model results:")
    for model_name in all_cv_results.keys():
        print(f"      - {cv_dir}/{model_name}/")
    
    if len(all_cv_results) > 1:
        print(f"\n   Comparison file:")
        print(f"      - {cv_dir}/model_comparison.json")
    
    print(f"\n✅ Cross-validation complete!")

if __name__ == "__main__":
    main()