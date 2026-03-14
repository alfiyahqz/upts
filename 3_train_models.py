import torch
import numpy as np
from pathlib import Path
import json
from config import (
    PROCESSED_DIR, 
    RESULTS_DIR, 
    NUM_CLASSES, 
    FEATURE_DIM,
    MODEL_LIST,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    CLASS_MAPPING
)
from utils.models import get_model
from utils.trainer import Trainer

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def load_splits():
    """Load train/val/test splits"""
    split_dir = PROCESSED_DIR / "splits"
    
    train_data = np.load(split_dir / "train.npz", allow_pickle=True)
    val_data = np.load(split_dir / "val.npz", allow_pickle=True)
    test_data = np.load(split_dir / "test.npz", allow_pickle=True)
    
    return {
        'train': (train_data['X'], train_data['y']),
        'val': (val_data['X'], val_data['y']),
        'test': (test_data['X'], test_data['y'])
    }

def train_single_model(model_name, X_train, y_train, X_val, y_val, 
                       save_dir, device='cuda'):
    """
    Train a single model
    
    Args:
        model_name: Name of the model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        save_dir: Directory to save results
        device: Device to use
    
    Returns:
        trainer: Trained model trainer
        metrics: Validation metrics
    """
    print("=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)
    
    # Create model
    model = get_model(model_name, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model: {model_name}")
    print(f"   Parameters: {n_params:,}")
    print(f"   Input shape: (batch, {X_train.shape[1]}, {X_train.shape[2]})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )
    
    # Train
    history = trainer.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        patience=15,
        verbose=True
    )
    
    # Evaluate on validation set
    print(f"\n📊 Evaluating on validation set...")
    val_metrics, val_preds, val_probs = trainer.evaluate(X_val, y_val)
    
    print(f"\n   Validation Results:")
    print(f"   Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall:    {val_metrics['recall']:.4f}")
    print(f"   F1-Score:  {val_metrics['f1']:.4f}")
    
    # Print per-class metrics
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    print(f"\n   Per-class metrics:")
    for i in range(NUM_CLASSES):
        class_name = reverse_mapping.get(i, f"Class_{i}")
        print(f"      {class_name:15s}: "
              f"P={val_metrics['precision_per_class'][i]:.4f} "
              f"R={val_metrics['recall_per_class'][i]:.4f} "
              f"F1={val_metrics['f1_per_class'][i]:.4f} "
              f"(n={val_metrics['support'][i]})")
    
    # Save model
    model_dir = save_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(model_dir / "model.pth")
    
    # Save history
    with open(model_dir / "history.json", 'w') as f:
        # Convert numpy types to native Python types
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    
    # Save validation metrics
    metrics_serializable = {
        'accuracy': float(val_metrics['accuracy']),
        'precision': float(val_metrics['precision']),
        'recall': float(val_metrics['recall']),
        'f1': float(val_metrics['f1']),
        'confusion_matrix': val_metrics['confusion_matrix'].tolist(),
        'per_class': {
            reverse_mapping.get(i, f"Class_{i}"): {
                'precision': float(val_metrics['precision_per_class'][i]),
                'recall': float(val_metrics['recall_per_class'][i]),
                'f1': float(val_metrics['f1_per_class'][i]),
                'support': int(val_metrics['support'][i])
            }
            for i in range(NUM_CLASSES)
        }
    }
    
    with open(model_dir / "val_metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\n💾 Saved results to: {model_dir}")
    
    return trainer, val_metrics

def main():
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\n📂 Loading data...")
    splits = load_splits()
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Create results directory
    results_dir = RESULTS_DIR / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    all_results = {}
    
    for model_name in MODEL_LIST:
        try:
            trainer, val_metrics = train_single_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                save_dir=results_dir,
                device=device
            )
            
            all_results[model_name] = {
                'val_accuracy': float(val_metrics['accuracy']),
                'val_f1': float(val_metrics['f1'])
            }
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    if all_results:
        print(f"\n📊 Model Comparison (Validation Set):")
        print(f"{'Model':<15} {'Accuracy':<12} {'F1-Score':<12}")
        print("-" * 40)
        
        for model_name in sorted(all_results.keys(), 
                                 key=lambda x: all_results[x]['val_f1'], 
                                 reverse=True):
            acc = all_results[model_name]['val_accuracy']
            f1 = all_results[model_name]['val_f1']
            print(f"{model_name:<15} {acc:<12.4f} {f1:<12.4f}")
        
        # Save summary
        with open(results_dir / "training_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Summary saved to: {results_dir}/training_summary.json")
    else:
        print("\n⚠️  No models were successfully trained")
    
    print(f"\n✅ Training pipeline complete!")
    print(f"\n📁 Results saved to: {results_dir}")

if __name__ == "__main__":
    main()