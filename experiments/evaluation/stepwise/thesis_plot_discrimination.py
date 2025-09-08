#!/usr/bin/env python3
"""
Discrimination ratio analysis for thesis evaluation.

This script analyzes the discrimination ratio (average probability for parents vs non-parents)
as a more principled metric than F1 score with arbitrary threshold.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def calculate_discrimination_ratio(episode_data: Dict) -> List[float]:
    """
    Calculate discrimination ratio trajectory from saved parent probabilities.
    
    Discrimination ratio = mean(P(parent)) - mean(P(non-parent))
    
    Args:
        episode_data: Episode data containing parent_probabilities and scm_info
        
    Returns:
        List of discrimination ratios at each intervention
    """
    discrimination_trajectory = []
    true_parents = set(episode_data['scm_info']['true_parents'])
    target = episode_data['scm_info']['target']
    
    for parent_probs in episode_data['parent_probabilities']:
        if not parent_probs:
            continue
            
        parent_probs_list = []
        non_parent_probs_list = []
        
        for var, prob in parent_probs.items():
            if var != target:  # Exclude target from analysis
                if var in true_parents:
                    parent_probs_list.append(prob)
                else:
                    non_parent_probs_list.append(prob)
        
        if parent_probs_list and non_parent_probs_list:
            avg_parent = np.mean(parent_probs_list)
            avg_non_parent = np.mean(non_parent_probs_list)
            discrimination = avg_parent - avg_non_parent
            discrimination_trajectory.append(discrimination)
        else:
            discrimination_trajectory.append(0.0)
    
    return discrimination_trajectory


def analyze_discrimination_by_structure(results: Dict) -> Dict:
    """
    Analyze discrimination ratio grouped by structure type.
    
    Args:
        results: Full evaluation results
        
    Returns:
        Dictionary with discrimination analysis by structure
    """
    structure_analysis = defaultdict(lambda: {
        'episodes': [],
        'discrimination_trajectories': [],
        'final_discriminations': [],
        'avg_parent_probs': [],
        'avg_non_parent_probs': []
    })
    
    for episode in results['episodes']:
        structure = episode['scm_info'].get('structure_type', 'unknown')
        discrimination = calculate_discrimination_ratio(episode)
        
        structure_analysis[structure]['episodes'].append(episode)
        structure_analysis[structure]['discrimination_trajectories'].append(discrimination)
        
        if discrimination:
            structure_analysis[structure]['final_discriminations'].append(discrimination[-1])
        
        # Calculate average probabilities for final intervention
        if episode['parent_probabilities']:
            final_probs = episode['parent_probabilities'][-1]
            true_parents = set(episode['scm_info']['true_parents'])
            target = episode['scm_info']['target']
            
            parent_probs = [p for v, p in final_probs.items() 
                          if v in true_parents and v != target]
            non_parent_probs = [p for v, p in final_probs.items() 
                              if v not in true_parents and v != target]
            
            if parent_probs:
                structure_analysis[structure]['avg_parent_probs'].append(np.mean(parent_probs))
            if non_parent_probs:
                structure_analysis[structure]['avg_non_parent_probs'].append(np.mean(non_parent_probs))
    
    return dict(structure_analysis)


def plot_discrimination_comparison(
    info_gain_results: Dict,
    parent_selection_results: Dict,
    output_dir: Path
) -> None:
    """
    Plot discrimination ratio comparison between different models.
    
    Args:
        info_gain_results: Results from info gain focused model
        parent_selection_results: Results from parent selection focused model
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Discrimination Ratio Analysis: Learning to Identify Parents', 
                 fontsize=14, fontweight='bold')
    
    # Colors for different methods
    colors = {
        'Info Gain Model': '#2E86AB',
        'Main Model': '#A23B72',
        'Random': '#808080',
        'Oracle': '#F18F01'
    }
    
    # Plot 1: Discrimination trajectories comparison
    ax1 = axes[0, 0]
    
    # Info gain model
    info_discriminations = []
    for episode in info_gain_results['episodes']:
        disc = calculate_discrimination_ratio(episode)
        info_discriminations.append(disc)
    
    # Main model
    main_discriminations = []
    for episode in parent_selection_results['episodes']:
        disc = calculate_discrimination_ratio(episode)
        main_discriminations.append(disc)
    
    # Plot trajectories with confidence intervals
    plot_trajectory_with_ci(info_discriminations, ax1, 
                           color=colors['Info Gain Model'], 
                           label='Info Gain Model')
    plot_trajectory_with_ci(main_discriminations, ax1,
                           color=colors['Main Model'],
                           label='Main Model')
    
    # Add baselines if available
    if 'baselines' in info_gain_results:
        if 'random' in info_gain_results['baselines']:
            random_discriminations = []
            for episode in info_gain_results['baselines']['random']['episodes']:
                disc = calculate_discrimination_ratio(episode)
                random_discriminations.append(disc)
            plot_trajectory_with_ci(random_discriminations, ax1,
                                   color=colors['Random'],
                                   label='Random', linestyle='--')
    
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Intervention Number')
    ax1.set_ylabel('Discrimination Ratio')
    ax1.set_title('Discrimination Ratio Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average parent vs non-parent probabilities
    ax2 = axes[0, 1]
    
    # Calculate final probabilities for both models
    models_data = {
        'Info Gain': analyze_discrimination_by_structure(info_gain_results),
        'Main': analyze_discrimination_by_structure(parent_selection_results)
    }
    
    x_pos = 0
    width = 0.35
    structures = set()
    for model_data in models_data.values():
        structures.update(model_data.keys())
    structures = sorted(structures)
    
    for i, structure in enumerate(structures):
        for j, (model_name, model_analysis) in enumerate(models_data.items()):
            if structure in model_analysis:
                data = model_analysis[structure]
                
                # Parent probabilities
                if data['avg_parent_probs']:
                    ax2.bar(x_pos + j*width, np.mean(data['avg_parent_probs']),
                           width, label=f'{model_name} (Parents)' if i == 0 else '',
                           color=colors[f'{model_name} Model'] if model_name == 'Info Gain' 
                                 else colors['Main Model'],
                           alpha=0.8)
                
                # Non-parent probabilities  
                if data['avg_non_parent_probs']:
                    ax2.bar(x_pos + j*width, np.mean(data['avg_non_parent_probs']),
                           width, label=f'{model_name} (Non-parents)' if i == 0 else '',
                           color=colors[f'{model_name} Model'] if model_name == 'Info Gain'
                                 else colors['Main Model'],
                           alpha=0.4, hatch='//')
        
        x_pos += 1
    
    ax2.set_xlabel('Structure Type')
    ax2.set_ylabel('Average Probability')
    ax2.set_title('Final Parent vs Non-Parent Probabilities')
    ax2.set_xticks(np.arange(len(structures)) + width/2)
    ax2.set_xticklabels(structures)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Discrimination distribution
    ax3 = axes[1, 0]
    
    final_disc_info = [d[-1] for d in info_discriminations if d]
    final_disc_main = [d[-1] for d in main_discriminations if d]
    
    data_to_plot = []
    labels = []
    
    if final_disc_info:
        data_to_plot.append(final_disc_info)
        labels.append('Info Gain Model')
    if final_disc_main:
        data_to_plot.append(final_disc_main)
        labels.append('Main Model')
    
    if data_to_plot:
        parts = ax3.violinplot(data_to_plot, showmeans=True, showmedians=True)
        ax3.set_xticks(range(1, len(labels) + 1))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Final Discrimination Ratio')
        ax3.set_title('Distribution of Final Discrimination Ratios')
        ax3.axhline(y=0, color='black', linestyle=':', alpha=0.3)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Discrimination Ratio Summary\n" + "="*30 + "\n\n"
    
    # Info gain model stats
    if final_disc_info:
        summary_text += f"Info Gain Model:\n"
        summary_text += f"  Mean: {np.mean(final_disc_info):.3f} ± {np.std(final_disc_info):.3f}\n"
        summary_text += f"  Max: {np.max(final_disc_info):.3f}\n"
        summary_text += f"  % Positive: {100 * np.mean(np.array(final_disc_info) > 0):.1f}%\n\n"
    
    # Main model stats
    if final_disc_main:
        summary_text += f"Main Model:\n"
        summary_text += f"  Mean: {np.mean(final_disc_main):.3f} ± {np.std(final_disc_main):.3f}\n"
        summary_text += f"  Max: {np.max(final_disc_main):.3f}\n"
        summary_text += f"  % Positive: {100 * np.mean(np.array(final_disc_main) > 0):.1f}%\n\n"
    
    # Interpretation
    summary_text += "Interpretation:\n"
    summary_text += "• Positive ratio = better parent discrimination\n"
    summary_text += "• Higher ratio = clearer parent identification\n"
    summary_text += "• Random baseline ≈ 0.0"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'discrimination_ratio_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved discrimination ratio analysis to: {output_path}")
    
    plt.close()


def plot_trajectory_with_ci(
    trajectories: List[List[float]],
    ax: plt.Axes,
    color: str = 'blue',
    label: str = '',
    linestyle: str = '-'
) -> None:
    """
    Plot trajectory with confidence interval.
    
    Args:
        trajectories: List of trajectories
        ax: Matplotlib axes
        color: Line color
        label: Line label
        linestyle: Line style
    """
    if not trajectories:
        return
    
    # Pad trajectories to same length
    max_len = max(len(t) for t in trajectories)
    padded = []
    for traj in trajectories:
        if len(traj) < max_len:
            padded.append(traj + [traj[-1]] * (max_len - len(traj)))
        else:
            padded.append(traj)
    
    traj_array = np.array(padded)
    mean_traj = np.mean(traj_array, axis=0)
    std_traj = np.std(traj_array, axis=0)
    sem_traj = std_traj / np.sqrt(len(trajectories))
    
    x = np.arange(len(mean_traj))
    ax.plot(x, mean_traj, color=color, linewidth=2, label=label, linestyle=linestyle)
    ax.fill_between(x, mean_traj - sem_traj, mean_traj + sem_traj,
                    alpha=0.2, color=color)


def main():
    """Main function to run discrimination ratio analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze discrimination ratio for thesis evaluation"
    )
    parser.add_argument('--info-gain-dir', type=Path, required=True,
                       help='Directory with info gain evaluation results')
    parser.add_argument('--parent-selection-dir', type=Path, required=True,
                       help='Directory with parent selection evaluation results')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    
    # Find latest evaluation files
    info_gain_files = sorted(args.info_gain_dir.glob('evaluation_*.json'))
    parent_selection_files = sorted(args.parent_selection_dir.glob('evaluation_*.json'))
    
    if not info_gain_files:
        print(f"No evaluation files found in {args.info_gain_dir}")
        return 1
    
    if not parent_selection_files:
        print(f"No evaluation files found in {args.parent_selection_dir}")
        return 1
    
    # Use latest files
    info_gain_file = info_gain_files[-1]
    parent_selection_file = parent_selection_files[-1]
    
    print(f"Using info gain results: {info_gain_file}")
    print(f"Using parent selection results: {parent_selection_file}")
    
    with open(info_gain_file, 'r') as f:
        info_gain_results = json.load(f)
    
    with open(parent_selection_file, 'r') as f:
        parent_selection_results = json.load(f)
    
    # Generate plots
    print("Generating discrimination ratio analysis...")
    plot_discrimination_comparison(
        info_gain_results,
        parent_selection_results,
        args.output_dir
    )
    
    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())