import numpy as np
import pandas as pd
from pathlib import Path
import json
from config import PROCESSED_DIR, RESULTS_DIR, CLASS_MAPPING

def analyze_misclassifications(model_name):
    """
    Analyze misclassified samples
    
    Args:
        model_name: Name of the model to analyze
    """
    print("=" * 60)
    print(f"ERROR ANALYSIS: {model_name.upper()}")
    print("=" * 60)
    
    # Load predictions
    eval_dir = RESULTS_DIR / "evaluation" / model_name
    pred_path = eval_dir / "predictions.npz"
    
    if not pred_path.exists():
        print(f"❌ Predictions not found: {pred_path}")
        return
    
    data = np.load(pred_path, allow_pickle=True)
    y_true = data['y_true']
    y_pred = data['y_pred']
    y_probs = data['y_probs']
    user_ids = data['user_ids']
    
    # Get class names
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    # Find misclassifications
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total samples: {len(y_true)}")
    print(f"   Correct: {len(y_true) - len(misclassified_idx)}")
    print(f"   Misclassified: {len(misclassified_idx)}")
    print(f"   Error rate: {len(misclassified_idx)/len(y_true)*100:.2f}%")
    
    if len(misclassified_idx) == 0:
        print("\n🎉 Perfect predictions! No errors to analyze.")
        return
    
    # Analyze misclassification patterns
    print(f"\n❌ Misclassification Patterns:")
    
    confusion_pairs = {}
    for idx in misclassified_idx:
        true_label = reverse_mapping[y_true[idx]]
        pred_label = reverse_mapping[y_pred[idx]]
        pair = f"{true_label} → {pred_label}"
        
        if pair not in confusion_pairs:
            confusion_pairs[pair] = []
        
        confusion_pairs[pair].append({
            'user_id': user_ids[idx],
            'confidence': float(y_probs[idx, y_pred[idx]])
        })
    
    # Print confusion pairs
    for pair, cases in sorted(confusion_pairs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n   {pair}: {len(cases)} cases")
        
        # Show top 3 most confident wrong predictions
        sorted_cases = sorted(cases, key=lambda x: x['confidence'], reverse=True)[:3]
        for case in sorted_cases:
            print(f"      - {case['user_id']} (confidence: {case['confidence']:.3f})")
    
    # Save detailed error analysis
    error_df = pd.DataFrame({
        'user_id': user_ids[misclassified_idx],
        'true_label': [reverse_mapping[y] for y in y_true[misclassified_idx]],
        'pred_label': [reverse_mapping[y] for y in y_pred[misclassified_idx]],
        'confidence': [y_probs[i, y_pred[i]] for i in misclassified_idx]
    })
    
    error_csv_path = eval_dir / "misclassified_samples.csv"
    error_df.to_csv(error_csv_path, index=False)
    
    print(f"\n💾 Saved detailed error analysis: {error_csv_path}")
    
    # Confidence analysis
    print(f"\n📊 Confidence Analysis (Misclassified Samples):")
    confidences = error_df['confidence'].values
    print(f"   Mean confidence: {np.mean(confidences):.3f}")
    print(f"   Median confidence: {np.median(confidences):.3f}")
    print(f"   Min confidence: {np.min(confidences):.3f}")
    print(f"   Max confidence: {np.max(confidences):.3f}")
    
    # High confidence errors (> 0.7)
    high_conf_errors = error_df[error_df['confidence'] > 0.7]
    if len(high_conf_errors) > 0:
        print(f"\n⚠️  High-confidence errors (confidence > 0.7): {len(high_conf_errors)}")
        print("   These are the most problematic misclassifications:")
        print(high_conf_errors.head(10).to_string(index=False))

def main():
    # Find evaluated models
    eval_dir = RESULTS_DIR / "evaluation"
    
    if not eval_dir.exists():
        print("❌ No evaluation results found. Run 4_evaluate.py first.")
        return
    
    evaluated_models = [d.name for d in eval_dir.iterdir() if d.is_dir()]
    
    if not evaluated_models:
        print("❌ No evaluated models found.")
        return
    
    print(f"Found {len(evaluated_models)} evaluated models: {evaluated_models}\n")
    
    for model_name in evaluated_models:
        analyze_misclassifications(model_name)
        print("\n")

if __name__ == "__main__":
    main()