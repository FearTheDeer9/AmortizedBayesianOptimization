#!/usr/bin/env python3
"""
Plot F1 score trajectories from evaluation results.

This script reads the JSON output from full_evaluation.py and creates
plots showing how F1 scores improve with each intervention.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import seaborn as sns

def load_evaluation_results(json_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_f1_trajectories(results: Dict) -> Dict[str, List[List[float]]]:
    """Extract F1 trajectories for policy and baselines."""
    trajectories = {}
    
    # Extract policy trajectories
    policy_trajectories = []
    for episode in results['episodes']:
        f1_scores = episode.get('f1_scores', [])
        if f1_scores:
            policy_trajectories.append(f1_scores)
    trajectories['Policy'] = policy_trajectories
    
    # Extract baseline trajectories if available
    if 'baselines' in results:
        if 'random' in results['baselines']:
            random_trajectories = []
            for episode in results['baselines']['random']['episodes']:
                f1_scores = episode.get('f1_scores', [])
                if f1_scores:
                    random_trajectories.append(f1_scores)
            trajectories['Random'] = random_trajectories
        
        if 'oracle' in results['baselines']:
            oracle_trajectories = []
            for episode in results['baselines']['oracle']['episodes']:
                f1_scores = episode.get('f1_scores', [])
                if f1_scores:
                    oracle_trajectories.append(f1_scores)
            trajectories['Oracle'] = oracle_trajectories
    
    return trajectories

def plot_f1_trajectories(trajectories: Dict[str, List[List[float]]], save_path: Path = None):
    """Plot F1 trajectories with confidence intervals."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Colors for each method
    colors = {
        'Policy': '#2E86AB',  # Blue
        'Random': '#A23B72',  # Pink
        'Oracle': '#F18F01'   # Orange
    }
    
    for method_name, method_trajectories in trajectories.items():
        if not method_trajectories:
            continue
        
        # Pad trajectories to same length
        max_len = max(len(traj) for traj in method_trajectories)
        padded_trajectories = []
        for traj in method_trajectories:
            padded = traj + [traj[-1]] * (max_len - len(traj)) if traj else []
            padded_trajectories.append(padded)
        
        # Convert to numpy array
        traj_array = np.array(padded_trajectories)
        
        # Calculate mean and confidence interval
        mean_trajectory = np.mean(traj_array, axis=0)
        std_trajectory = np.std(traj_array, axis=0)
        n_episodes = len(padded_trajectories)
        
        # Standard error for confidence interval
        se_trajectory = std_trajectory / np.sqrt(n_episodes)
        ci_lower = mean_trajectory - 1.96 * se_trajectory
        ci_upper = mean_trajectory + 1.96 * se_trajectory
        
        # Plot
        x = np.arange(len(mean_trajectory))
        plt.plot(x, mean_trajectory, label=method_name, 
                color=colors.get(method_name, 'gray'), linewidth=2)
        plt.fill_between(x, ci_lower, ci_upper, 
                         color=colors.get(method_name, 'gray'), alpha=0.2)
    
    plt.xlabel('Intervention Number', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Convergence: Policy vs Baselines', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Add horizontal line at F1=1.0
    plt.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Perfect')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_structure_specific_trajectories(results: Dict, save_path: Path = None):
    """Plot F1 trajectories separated by structure type."""
    
    # Group episodes by structure type
    structure_groups = {}
    for episode in results['episodes']:
        structure = episode['scm_info'].get('structure_type', 'unknown')
        if structure not in structure_groups:
            structure_groups[structure] = []
        structure_groups[structure].append(episode.get('f1_scores', []))
    
    # Create subplots for each structure
    n_structures = len(structure_groups)
    fig, axes = plt.subplots(1, n_structures, figsize=(6*n_structures, 5))
    
    if n_structures == 1:
        axes = [axes]
    
    for idx, (structure, trajectories) in enumerate(structure_groups.items()):
        ax = axes[idx]
        
        # Plot trajectories
        for traj in trajectories:
            ax.plot(traj, alpha=0.3, color='blue')
        
        # Plot mean
        if trajectories:
            max_len = max(len(traj) for traj in trajectories)
            padded = []
            for traj in trajectories:
                padded.append(traj + [traj[-1]] * (max_len - len(traj)) if traj else [])
            mean_traj = np.mean(padded, axis=0)
            ax.plot(mean_traj, color='red', linewidth=2, label='Mean')
        
        ax.set_xlabel('Intervention Number')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{structure.capitalize()} Structure')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('F1 Trajectories by Structure Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def analyze_convergence_speed(trajectories: Dict[str, List[List[float]]]) -> Dict:
    """Analyze convergence speed metrics."""
    analysis = {}
    
    for method_name, method_trajectories in trajectories.items():
        if not method_trajectories:
            continue
        
        # Calculate interventions to reach F1 thresholds
        thresholds = [0.5, 0.7, 0.9]
        interventions_to_threshold = {t: [] for t in thresholds}
        
        for traj in method_trajectories:
            for threshold in thresholds:
                # Find first intervention where F1 >= threshold
                for i, f1 in enumerate(traj):
                    if f1 >= threshold:
                        interventions_to_threshold[threshold].append(i + 1)
                        break
                else:
                    # Never reached threshold
                    interventions_to_threshold[threshold].append(None)
        
        # Calculate statistics
        method_analysis = {}
        for threshold, counts in interventions_to_threshold.items():
            valid_counts = [c for c in counts if c is not None]
            if valid_counts:
                method_analysis[f'mean_to_f1_{threshold}'] = np.mean(valid_counts)
                method_analysis[f'std_to_f1_{threshold}'] = np.std(valid_counts)
                method_analysis[f'success_rate_f1_{threshold}'] = len(valid_counts) / len(counts)
            else:
                method_analysis[f'mean_to_f1_{threshold}'] = float('inf')
                method_analysis[f'std_to_f1_{threshold}'] = 0
                method_analysis[f'success_rate_f1_{threshold}'] = 0
        
        # Final F1 score
        final_f1s = [traj[-1] if traj else 0 for traj in method_trajectories]
        method_analysis['final_f1_mean'] = np.mean(final_f1s)
        method_analysis['final_f1_std'] = np.std(final_f1s)
        
        analysis[method_name] = method_analysis
    
    return analysis

def main():
    """Main function to run plotting from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot F1 trajectories from evaluation results")
    parser.add_argument('json_path', type=Path, 
                       help='Path to evaluation JSON file')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.json_path}")
    results = load_evaluation_results(args.json_path)
    
    # Extract trajectories
    trajectories = extract_f1_trajectories(results)
    print(f"Found trajectories for: {list(trajectories.keys())}")
    
    # Create plots
    args.output_dir.mkdir(exist_ok=True)
    
    # Main comparison plot
    plot_path = args.output_dir / "f1_trajectories.png"
    plot_f1_trajectories(trajectories, save_path=plot_path if not args.show else None)
    
    # Structure-specific plot
    structure_plot_path = args.output_dir / "f1_by_structure.png"
    plot_structure_specific_trajectories(results, 
                                        save_path=structure_plot_path if not args.show else None)
    
    # Analyze convergence speed
    print("\n" + "="*60)
    print("CONVERGENCE SPEED ANALYSIS")
    print("="*60)
    
    analysis = analyze_convergence_speed(trajectories)
    
    for method, metrics in analysis.items():
        print(f"\n{method}:")
        print(f"  Final F1: {metrics['final_f1_mean']:.3f} ± {metrics['final_f1_std']:.3f}")
        
        for threshold in [0.5, 0.7, 0.9]:
            mean_key = f'mean_to_f1_{threshold}'
            success_key = f'success_rate_f1_{threshold}'
            if mean_key in metrics:
                if metrics[mean_key] == float('inf'):
                    print(f"  To F1≥{threshold}: Never reached (0% success)")
                else:
                    print(f"  To F1≥{threshold}: {metrics[mean_key]:.1f} interventions "
                          f"({metrics[success_key]:.0%} success)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()