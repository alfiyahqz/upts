import torch
import numpy as np
from pathlib import Path
import json
from config import (
    PROCESSED_DIR, 
    RESULTS_DIR, 
    NUM_CLASSES, 
    FEATURE_DIM,
    CLASS_MAPPING,
    BATCH_SIZE
)
from utils.models import get_model
from utils.trainer import Trainer
from utils.evaluation import ModelEvaluator

def load_test_data():
    """Load test split"""
    split_dir = PROCESSED_DIR / "splits"
    test_data = np.load(split_dir / "test.npz", allow_pickle=True)
    return test_data['X'], test_data['y'], test_data['user_ids']

def evaluate_model(model_name, model_path, X_test, y_test, device='cuda'):
    """
    Evaluate a single trained model
    
    Args:
        model_name: Name of the model
        model_path: Path to saved model checkpoint
        X_test, y_test: Test data
        device: Device to use
    
    Returns:
        metrics: Evaluation metrics
        predictions: Model predictions
        probabilities: Prediction probabilities
    """
    print("=" * 60)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 60)
    
    # Create model
    model = get_model(model_name, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )
    
    # Load trained model
    print(f"\n📂 Loading model from: {model_path}")
    trainer.load_model(model_path)
    
    # Evaluate
    print(f"\n📊 Evaluating on test set...")
    metrics, predictions, probabilities = trainer.evaluate(X_test, y_test)
    
    return metrics, predictions, probabilities

def main():
    print("=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    # Load test data
    print(f"\n📂 Loading test data...")
    X_test, y_test, test_user_ids = load_test_data()
    
    print(f"   Test set: X={X_test.shape}, y={y_test.shape}")
    print(f"   Number of test users: {len(test_user_ids)}")
    
    # Class names
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    class_names = [reverse_mapping[i] for i in range(NUM_CLASSES)]
    
    print(f"\n   Classes: {class_names}")
    
    # Create evaluator
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Find trained models
    models_dir = RESULTS_DIR / "models"
    
    if not models_dir.exists():
        print(f"\n❌ Error: Models directory not found: {models_dir}")
        print("   Please run 3_train_models.py first.")
        return
    
    # Get list of trained models
    trained_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    if not trained_models:
        print(f"\n❌ Error: No trained models found in {models_dir}")
        return
    
    print(f"\n🔍 Found {len(trained_models)} trained models: {trained_models}")
    
    # Create evaluation results directory
    eval_dir = RESULTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = {}
    
    for model_name in trained_models:
        model_dir = models_dir / model_name
        model_path = model_dir / "model.pth"
        
        if not model_path.exists():
            print(f"\n⚠️  Skipping {model_name}: model.pth not found")
            continue
        
        try:
            # Evaluate
            metrics, predictions, probabilities = evaluate_model(
                model_name=model_name,
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                device=device
            )
            
            # Print metrics
            evaluator.print_metrics(
                metrics, 
                title=f"{model_name.upper()} - Test Set Results"
            )
            
            # Create model evaluation directory
            model_eval_dir = eval_dir / model_name
            model_eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(model_eval_dir / "test_metrics.json", 'w') as f:
                # Convert numpy types to native Python
                metrics_serializable = {
                    'accuracy': float(metrics['accuracy']),
                    'weighted': {
                        k: float(v) for k, v in metrics['weighted'].items()
                    },
                    'macro': {
                        k: float(v) for k, v in metrics['macro'].items()
                    },
                    'per_class': metrics['per_class'],
                    'confusion_matrix': metrics['confusion_matrix'],
                    'support': metrics['support']
                }
                if 'auc' in metrics:
                    metrics_serializable['auc'] = float(metrics['auc'])
                
                json.dump(metrics_serializable, f, indent=2)
            
            print(f"\n💾 Saved metrics: {model_eval_dir}/test_metrics.json")
            
            # Save predictions
            np.savez(
                model_eval_dir / "predictions.npz",
                y_true=y_test,
                y_pred=predictions,
                y_probs=probabilities,
                user_ids=test_user_ids
            )
            
            print(f"💾 Saved predictions: {model_eval_dir}/predictions.npz")
            
            # Generate classification report
            print(f"\n📋 Classification Report:")
            report = evaluator.generate_classification_report(
                y_test, 
                predictions,
                save_path=model_eval_dir / "classification_report.txt"
            )
            print(report)
            
            # Generate visualizations
            print(f"\n📊 Generating visualizations...")
            
            # Confusion matrix
            evaluator.plot_confusion_matrix(
                metrics['confusion_matrix'],
                title=f"{model_name.upper()} - Confusion Matrix (Test Set)",
                save_path=model_eval_dir / "confusion_matrix.png"
            )
            
            # ROC curves
            try:
                evaluator.plot_roc_curves(
                    y_test,
                    probabilities,
                    title=f"{model_name.upper()} - ROC Curves (Test Set)",
                    save_path=model_eval_dir / "roc_curves.png"
                )
            except Exception as e:
                print(f"   ⚠️  Could not plot ROC curves: {e}")
            
            # Class distribution
            evaluator.plot_class_distribution(
                y_test,
                predictions,
                title=f"{model_name.upper()} - Class Distribution (Test Set)",
                save_path=model_eval_dir / "class_distribution.png"
            )
            
            # Training history (if available)
            history_path = model_dir / "history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                evaluator.plot_training_history(
                    history,
                    title=f"{model_name.upper()} - Training History",
                    save_path=model_eval_dir / "training_history.png"
                )
            
            # Store results for comparison
            all_results[model_name] = metrics
            
            print(f"\n✅ Evaluation complete for {model_name}")
            print(f"   Results saved to: {model_eval_dir}")
            
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==================== MODEL COMPARISON ====================
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Print comparison table
        print(f"\n📊 Test Set Performance Comparison:")
        print(f"\n{'Model':<15} {'Accuracy':<12} {'Weighted F1':<12} {'Macro F1':<12}")
        print("-" * 52)
        
        # Sort by weighted F1
        sorted_models = sorted(
            all_results.keys(), 
            key=lambda x: all_results[x]['weighted']['f1'], 
            reverse=True
        )
        
        for model_name in sorted_models:
            metrics = all_results[model_name]
            print(f"{model_name:<15} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['weighted']['f1']:<12.4f} "
                  f"{metrics['macro']['f1']:<12.4f}")
        
        # Save comparison
        comparison = {}
        for model_name, metrics in all_results.items():
            comparison[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'weighted_f1': float(metrics['weighted']['f1']),
                'weighted_precision': float(metrics['weighted']['precision']),
                'weighted_recall': float(metrics['weighted']['recall']),
                'macro_f1': float(metrics['macro']['f1']),
                'macro_precision': float(metrics['macro']['precision']),
                'macro_recall': float(metrics['macro']['recall'])
            }
            if 'auc' in metrics:
                comparison[model_name]['auc'] = float(metrics['auc'])
        
        with open(eval_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Saved comparison: {eval_dir}/model_comparison.json")
        
        # Generate comparison plots
        print(f"\n📊 Generating comparison plots...")
        
        # Compare by accuracy
        evaluator.compare_models(
            all_results,
            metric='accuracy',
            save_path=eval_dir / "comparison_accuracy.png"
        )
        
        # Compare by F1
        evaluator.compare_models(
            all_results,
            metric='f1',
            save_path=eval_dir / "comparison_f1.png"
        )
        
        # Best model
        best_model = sorted_models[0]
        best_f1 = all_results[best_model]['weighted']['f1']
        
        print(f"\n🏆 Best Model: {best_model.upper()}")
        print(f"   Weighted F1: {best_f1:.4f}")
        print(f"   Accuracy:    {all_results[best_model]['accuracy']:.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Evaluated {len(all_results)} models")
    print(f"\n📁 Results saved to: {eval_dir}")
    print(f"\n   Individual model results:")
    for model_name in all_results.keys():
        print(f"      - {eval_dir}/{model_name}/")
    
    if len(all_results) > 1:
        print(f"\n   Comparison files:")
        print(f"      - {eval_dir}/model_comparison.json")
        print(f"      - {eval_dir}/comparison_accuracy.png")
        print(f"      - {eval_dir}/comparison_f1.png")
    
    print(f"\n✅ Evaluation pipeline complete!")

if __name__ == "__main__":
    main()