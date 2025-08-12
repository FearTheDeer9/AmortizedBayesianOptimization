#!/usr/bin/env python3
"""
Analysis script for BC training results.

This script loads saved checkpoints and metrics to perform comprehensive
analysis of training performance, identifying problem areas and patterns.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from metrics_tracker import BCMetricsTracker
from visualize_metrics import generate_all_visualizations


def analyze_convergence(metrics_history: List[Dict]) -> Dict[str, Any]:
    """
    Analyze training convergence patterns.
    
    Args:
        metrics_history: List of epoch metrics
        
    Returns:
        Convergence analysis results
    """
    if not metrics_history:
        return {"status": "no_data"}
    
    # Extract validation accuracy if available
    val_accuracies = [m.get('val_accuracy', m.get('train_accuracy', 0)) 
                      for m in metrics_history]
    
    # Analyze convergence
    analysis = {
        "total_epochs": len(metrics_history),
        "best_epoch": int(np.argmax(val_accuracies)),
        "best_accuracy": float(max(val_accuracies)),
        "final_accuracy": float(val_accuracies[-1]) if val_accuracies else 0,
    }
    
    # Check for plateauing (no improvement in last 20% of training)
    if len(val_accuracies) > 10:
        recent_epochs = len(val_accuracies) // 5
        recent_best = max(val_accuracies[-recent_epochs:])
        overall_best = max(val_accuracies[:-recent_epochs]) if len(val_accuracies) > recent_epochs else 0
        
        if recent_best <= overall_best:
            analysis["plateaued"] = True
            analysis["plateau_epoch"] = len(val_accuracies) - recent_epochs
        else:
            analysis["plateaued"] = False
            analysis["improvement_in_last_20%"] = recent_best - overall_best
    
    # Check for overfitting
    if 'train_accuracy' in metrics_history[-1] and 'val_accuracy' in metrics_history[-1]:
        train_acc = metrics_history[-1]['train_accuracy']
        val_acc = metrics_history[-1]['val_accuracy']
        gap = train_acc - val_acc
        
        analysis["train_val_gap"] = float(gap)
        analysis["likely_overfitting"] = gap > 0.15
    
    return analysis


def identify_problem_variables(per_variable_stats: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Identify variables that are difficult to predict.
    
    Args:
        per_variable_stats: Per-variable performance statistics
        
    Returns:
        Analysis of problematic variables
    """
    if not per_variable_stats:
        return {"status": "no_data"}
    
    problem_vars = []
    strong_vars = []
    
    for var_name, stats in per_variable_stats.items():
        if stats['attempts'] > 0:
            accuracy = stats['correct'] / stats['attempts']
            
            if accuracy < 0.3:
                problem_vars.append({
                    "name": var_name,
                    "accuracy": accuracy,
                    "attempts": stats['attempts']
                })
            elif accuracy > 0.8:
                strong_vars.append({
                    "name": var_name,
                    "accuracy": accuracy,
                    "attempts": stats['attempts']
                })
    
    # Sort by accuracy
    problem_vars.sort(key=lambda x: x['accuracy'])
    strong_vars.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Calculate overall statistics
    all_accuracies = [stats['correct'] / stats['attempts'] 
                      for stats in per_variable_stats.values() 
                      if stats['attempts'] > 0]
    
    return {
        "problematic_variables": problem_vars[:5],  # Top 5 worst
        "strong_variables": strong_vars[:5],  # Top 5 best
        "mean_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0,
        "std_accuracy": float(np.std(all_accuracies)) if all_accuracies else 0,
        "n_variables": len(per_variable_stats),
        "n_problem_vars": len(problem_vars),
        "n_strong_vars": len(strong_vars)
    }


def analyze_confusion_patterns(confusion_matrices: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Analyze confusion matrix to identify systematic errors.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices by epoch
        
    Returns:
        Confusion pattern analysis
    """
    if not confusion_matrices:
        return {"status": "no_data"}
    
    # Get the latest confusion matrix
    latest_epoch = max(confusion_matrices.keys())
    cm_data = confusion_matrices[latest_epoch]
    cm = cm_data['matrix']
    labels = cm_data.get('labels', [f'Var{i}' for i in range(cm.shape[0])])
    
    # Identify most confused pairs
    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true": labels[i] if i < len(labels) else f"Var{i}",
                    "predicted": labels[j] if j < len(labels) else f"Var{j}",
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / (cm[i].sum() + 1e-10))
                })
    
    # Sort by confusion rate
    confused_pairs.sort(key=lambda x: x['rate'], reverse=True)
    
    # Calculate diagonal accuracy
    diagonal_sum = np.trace(cm)
    total_sum = cm.sum()
    overall_accuracy = diagonal_sum / total_sum if total_sum > 0 else 0
    
    return {
        "epoch": latest_epoch,
        "overall_accuracy": float(overall_accuracy),
        "most_confused_pairs": confused_pairs[:10],  # Top 10
        "n_classes": cm.shape[0]
    }


def analyze_embedding_quality(embeddings_history: Dict[int, List]) -> Dict[str, Any]:
    """
    Analyze embedding quality and evolution.
    
    Args:
        embeddings_history: Dictionary of embeddings by epoch
        
    Returns:
        Embedding quality analysis
    """
    if not embeddings_history:
        return {"status": "no_data"}
    
    epochs = sorted(embeddings_history.keys())
    diversity_scores = []
    
    for epoch in epochs:
        embeddings_list = embeddings_history[epoch]
        if not embeddings_list:
            continue
        
        # Concatenate embeddings
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Calculate diversity (average pairwise distance)
        n_samples = min(50, len(all_embeddings))
        if n_samples > 1:
            indices = np.random.choice(len(all_embeddings), n_samples, replace=False)
            sampled = all_embeddings[indices]
            
            distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(sampled[i] - sampled[j])
                    distances.append(dist)
            
            diversity = np.mean(distances) if distances else 0
            diversity_scores.append((epoch, diversity))
    
    # Check for embedding collapse
    if len(diversity_scores) > 1:
        initial_diversity = diversity_scores[0][1]
        final_diversity = diversity_scores[-1][1]
        diversity_change = (final_diversity - initial_diversity) / (initial_diversity + 1e-10)
        
        collapsed = diversity_change < -0.5  # >50% reduction
    else:
        diversity_change = 0
        collapsed = False
    
    return {
        "n_epochs_with_embeddings": len(diversity_scores),
        "diversity_scores": diversity_scores[:10],  # First 10 epochs
        "initial_diversity": float(diversity_scores[0][1]) if diversity_scores else 0,
        "final_diversity": float(diversity_scores[-1][1]) if diversity_scores else 0,
        "diversity_change_ratio": float(diversity_change),
        "likely_collapsed": collapsed
    }


def generate_analysis_report(metrics_file: Path, checkpoint_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report.
    
    Args:
        metrics_file: Path to metrics pickle file
        checkpoint_file: Optional path to model checkpoint
        
    Returns:
        Complete analysis report
    """
    print(f"Loading metrics from {metrics_file}")
    
    with open(metrics_file, 'rb') as f:
        data = pickle.load(f)
    
    report = {
        "metrics_file": str(metrics_file),
        "checkpoint_file": str(checkpoint_file) if checkpoint_file else None,
    }
    
    # Convergence analysis
    if 'epoch_metrics' in data:
        report["convergence"] = analyze_convergence(data['epoch_metrics'])
    
    # Problem variables
    if 'per_variable_stats' in data:
        report["variable_analysis"] = identify_problem_variables(data['per_variable_stats'])
    
    # Confusion patterns
    if 'confusion_matrices' in data:
        report["confusion_analysis"] = analyze_confusion_patterns(data['confusion_matrices'])
    
    # Embedding quality
    if 'embeddings_history' in data:
        report["embedding_analysis"] = analyze_embedding_quality(data['embeddings_history'])
    
    # Summary and recommendations
    report["summary"] = generate_summary_and_recommendations(report)
    
    return report


def generate_summary_and_recommendations(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary and actionable recommendations.
    
    Args:
        report: Analysis report
        
    Returns:
        Summary and recommendations
    """
    issues = []
    recommendations = []
    
    # Check convergence
    if 'convergence' in report:
        conv = report['convergence']
        if conv.get('plateaued'):
            issues.append(f"Training plateaued at epoch {conv['plateau_epoch']}")
            recommendations.append("Consider reducing learning rate or adding regularization")
        
        if conv.get('likely_overfitting'):
            gap = conv['train_val_gap']
            issues.append(f"Likely overfitting (train-val gap: {gap:.3f})")
            recommendations.append("Add dropout, weight decay, or collect more training data")
    
    # Check variable performance
    if 'variable_analysis' in report:
        var_anal = report['variable_analysis']
        if var_anal.get('n_problem_vars', 0) > 0:
            issues.append(f"{var_anal['n_problem_vars']} variables with <30% accuracy")
            recommendations.append("Analyze problematic variables for patterns")
            recommendations.append("Consider variable-specific augmentation or weighting")
    
    # Check embeddings
    if 'embedding_analysis' in report:
        emb_anal = report['embedding_analysis']
        if emb_anal.get('likely_collapsed'):
            issues.append("Embedding collapse detected")
            recommendations.append("Increase model capacity or add regularization")
            recommendations.append("Check for mode collapse in specific variable types")
    
    # Overall performance
    if 'convergence' in report:
        best_acc = report['convergence'].get('best_accuracy', 0)
        if best_acc < 0.5:
            issues.append(f"Low overall accuracy ({best_acc:.3f})")
            recommendations.append("Review data quality and labeling")
            recommendations.append("Consider simpler model architecture initially")
        elif best_acc > 0.9:
            recommendations.append("Model performing well - consider deployment")
    
    return {
        "identified_issues": issues,
        "recommendations": recommendations,
        "n_issues": len(issues)
    }


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze BC training results')
    
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to metrics pickle file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--output_dir', type=str,
                       default='debugging-bc-training/results/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate analysis report
    print("Generating analysis report...")
    report = generate_analysis_report(
        metrics_file,
        Path(args.checkpoint) if args.checkpoint else None
    )
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types."""
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        return obj
    
    # Save report as JSON
    report_file = output_dir / 'analysis_report.json'
    report_serializable = convert_to_json_serializable(report)
    with open(report_file, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    print(f"Saved analysis report to {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if 'summary' in report:
        summary = report['summary']
        
        if summary['identified_issues']:
            print("\nIdentified Issues:")
            for issue in summary['identified_issues']:
                print(f"  • {issue}")
        else:
            print("\nNo major issues identified!")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  → {rec}")
    
    if 'convergence' in report:
        conv = report['convergence']
        print(f"\nBest accuracy: {conv['best_accuracy']:.3f} (epoch {conv['best_epoch']})")
        print(f"Final accuracy: {conv['final_accuracy']:.3f}")
    
    if 'variable_analysis' in report:
        var_anal = report['variable_analysis']
        print(f"\nVariable performance: {var_anal['mean_accuracy']:.3f} ± {var_anal['std_accuracy']:.3f}")
        
        if var_anal['problematic_variables']:
            print("\nMost problematic variables:")
            for var in var_anal['problematic_variables'][:3]:
                print(f"  • {var['name']}: {var['accuracy']:.3f}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "="*60)
        print("Generating visualizations...")
        viz_dir = output_dir / 'plots'
        generate_all_visualizations(metrics_file, viz_dir)
        print(f"Visualizations saved to {viz_dir}")
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())