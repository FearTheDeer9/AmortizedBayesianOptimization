#!/usr/bin/env python3
"""
Combined thesis plots showing mixed results story.

Creates publication-quality 3-panel figure demonstrating:
1. Success in structure learning (discrimination ratio)
2. Success in parent selection (reducing search space)
3. Failure in target optimization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from thesis_plot_discrimination import calculate_discrimination_ratio

# Set publication quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'


def plot_thesis_mixed_results(
    info_gain_results: Dict,
    parent_selection_results: Dict,
    target_results: Dict,
    output_dir: Path
) -> None:
    """
    Create 3-panel thesis figure showing mixed results.
    
    Args:
        info_gain_results: Results focused on info gain
        parent_selection_results: Results focused on parent selection
        target_results: Results focused on target optimization
        output_dir: Directory to save plots
    """
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors for consistency
    policy_color = '#2E86AB'
    random_color = '#808080'
    oracle_color = '#F18F01'
    
    # ========== Panel 1: Discrimination Ratio (Structure Learning Success) ==========
    ax1 = axes[0]
    
    # Calculate discrimination trajectories
    policy_discriminations = []
    for episode in info_gain_results['episodes']:
        disc = calculate_discrimination_ratio(episode)
        policy_discriminations.append(disc)
    
    # Plot policy trajectory
    plot_mean_trajectory(policy_discriminations, ax1, 
                        color=policy_color, label='Policy')
    
    # Add random baseline if available
    if 'baselines' in info_gain_results and 'random' in info_gain_results['baselines']:
        random_discriminations = []
        for episode in info_gain_results['baselines']['random']['episodes']:
            disc = calculate_discrimination_ratio(episode)
            random_discriminations.append(disc)
        plot_mean_trajectory(random_discriminations, ax1,
                           color=random_color, label='Random', linestyle='--')
    
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Intervention Number')
    ax1.set_ylabel('Discrimination Ratio')
    ax1.set_title('(a) Structure Learning: SUCCESS', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-0.1, 0.5)
    
    # Add success indicator
    final_disc = np.mean([d[-1] for d in policy_discriminations if d])
    ax1.text(0.95, 0.95, f'Final: {final_disc:.2f}',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ========== Panel 2: Parent Selection Rate (Search Space Reduction Success) ==========
    ax2 = axes[1]
    
    # Calculate cumulative parent selection rates
    policy_parent_rates = []
    for episode in parent_selection_results['episodes']:
        selections = episode['parent_selections']
        cumulative_rate = np.cumsum(selections) / np.arange(1, len(selections) + 1)
        policy_parent_rates.append(cumulative_rate)
    
    # Plot policy trajectory
    plot_mean_trajectory(policy_parent_rates, ax2,
                        color=policy_color, label='Policy')
    
    # Add random baseline if available
    if 'baselines' in parent_selection_results and 'random' in parent_selection_results['baselines']:
        random_parent_rates = []
        for episode in parent_selection_results['baselines']['random']['episodes']:
            selections = episode['parent_selections']
            cumulative_rate = np.cumsum(selections) / np.arange(1, len(selections) + 1)
            random_parent_rates.append(cumulative_rate)
        plot_mean_trajectory(random_parent_rates, ax2,
                           color=random_color, label='Random', linestyle='--')
    
    # Add oracle baseline if available
    if 'baselines' in parent_selection_results and 'oracle' in parent_selection_results['baselines']:
        oracle_parent_rates = []
        for episode in parent_selection_results['baselines']['oracle']['episodes']:
            selections = episode['parent_selections']
            cumulative_rate = np.cumsum(selections) / np.arange(1, len(selections) + 1)
            oracle_parent_rates.append(cumulative_rate)
        plot_mean_trajectory(oracle_parent_rates, ax2,
                           color=oracle_color, label='Oracle', linestyle=':')
    
    # Add expected random rate line
    # Estimate from data structure
    if parent_selection_results['episodes']:
        num_vars = len(parent_selection_results['episodes'][0]['scm_info'].get('true_parents', [])) + 1
        num_parents = len(parent_selection_results['episodes'][0]['scm_info'].get('true_parents', []))
        expected_random = num_parents / (num_vars - 1)  # -1 for target
        ax2.axhline(y=expected_random, color='gray', linestyle=':', alpha=0.5,
                   label=f'Expected Random ({expected_random:.2f})')
    
    ax2.set_xlabel('Intervention Number')
    ax2.set_ylabel('Cumulative Parent Selection Rate')
    ax2.set_title('(b) Parent Selection: SUCCESS', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.05)
    
    # Add success indicator
    final_rate = np.mean([r[-1] for r in policy_parent_rates if len(r) > 0])
    ax2.text(0.95, 0.95, f'Final: {final_rate:.1%}',
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ========== Panel 3: Target Value Optimization (Failure) ==========
    ax3 = axes[2]
    
    # Normalize target values for comparison
    all_target_values = []
    
    # Get policy target values
    policy_targets = []
    for episode in target_results['episodes']:
        targets = episode['target_values']
        if targets:
            # Normalize to [0, 1] where 0 is best
            min_val = min(targets)
            max_val = max(targets)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in targets]
            else:
                normalized = [0.5] * len(targets)
            policy_targets.append(normalized)
    
    # Plot policy trajectory
    plot_mean_trajectory(policy_targets, ax3,
                        color=policy_color, label='Policy')
    
    # Add random baseline if available
    if 'baselines' in target_results and 'random' in target_results['baselines']:
        random_targets = []
        for episode in target_results['baselines']['random']['episodes']:
            targets = episode['target_values']
            if targets:
                min_val = min(targets)
                max_val = max(targets)
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in targets]
                else:
                    normalized = [0.5] * len(targets)
                random_targets.append(normalized)
        plot_mean_trajectory(random_targets, ax3,
                           color=random_color, label='Random', linestyle='--')
    
    # Add oracle baseline if available
    if 'baselines' in target_results and 'oracle' in target_results['baselines']:
        oracle_targets = []
        for episode in target_results['baselines']['oracle']['episodes']:
            targets = episode['target_values']
            if targets:
                min_val = min(targets)
                max_val = max(targets)
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in targets]
                else:
                    normalized = [0.5] * len(targets)
                oracle_targets.append(normalized)
        plot_mean_trajectory(oracle_targets, ax3,
                           color=oracle_color, label='Oracle', linestyle=':')
    
    ax3.set_xlabel('Intervention Number')
    ax3.set_ylabel('Normalized Target Value')
    ax3.set_title('(c) Target Optimization: FAILURE', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(-0.05, 1.05)
    ax3.invert_yaxis()  # Lower is better
    
    # Add failure indicator
    if policy_targets and random_targets:
        policy_final = np.mean([t[-1] for t in policy_targets if t])
        random_final = np.mean([t[-1] for t in random_targets if t])
        improvement = (random_final - policy_final) / random_final if random_final > 0 else 0
        
        indicator_text = f'vs Random: {improvement:.1%}'
        indicator_color = 'lightcoral' if improvement < 0.1 else 'lightyellow'
        
        ax3.text(0.95, 0.05, indicator_text,
                transform=ax3.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor=indicator_color, alpha=0.5))
    
    # Overall title
    fig.suptitle('Policy Evaluation: Mixed Results Across Objectives', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'thesis_mixed_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved thesis figure to: {output_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = output_dir / 'thesis_mixed_results.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF version to: {pdf_path}")
    
    plt.close()


def plot_mean_trajectory(
    trajectories: List[List[float]],
    ax: plt.Axes,
    color: str = 'blue',
    label: str = '',
    linestyle: str = '-'
) -> None:
    """
    Plot mean trajectory with confidence interval.
    
    Args:
        trajectories: List of trajectories
        ax: Matplotlib axes
        color: Line color
        label: Line label
        linestyle: Line style
    """
    if not trajectories:
        return
    
    # Filter out empty trajectories and find max length
    non_empty_trajectories = [t for t in trajectories if len(t) > 0]
    if not non_empty_trajectories:
        return
    
    # Pad trajectories to same length
    max_len = max(len(t) for t in non_empty_trajectories)
    padded = []
    for traj in non_empty_trajectories:
        if len(traj) < max_len:
            # Pad with last value
            padded.append(traj + [traj[-1]] * (max_len - len(traj)))
        else:
            padded.append(traj[:max_len])
    
    if not padded:
        return
    
    traj_array = np.array(padded)
    mean_traj = np.mean(traj_array, axis=0)
    sem_traj = np.std(traj_array, axis=0) / np.sqrt(len(padded))
    
    x = np.arange(len(mean_traj))
    ax.plot(x, mean_traj, color=color, linewidth=2, label=label, linestyle=linestyle)
    ax.fill_between(x, mean_traj - sem_traj, mean_traj + sem_traj,
                    alpha=0.15, color=color)


def create_summary_table(
    info_gain_results: Dict,
    parent_selection_results: Dict,
    target_results: Dict,
    output_path: Path
) -> None:
    """
    Create a summary table of key metrics.
    
    Args:
        info_gain_results: Info gain evaluation results
        parent_selection_results: Parent selection evaluation results
        target_results: Target optimization evaluation results
        output_path: Path to save summary table
    """
    summary = []
    summary.append("="*60)
    summary.append("THESIS EVALUATION SUMMARY")
    summary.append("="*60)
    summary.append("")
    
    # Structure Learning (Discrimination Ratio)
    summary.append("1. STRUCTURE LEARNING (Discrimination Ratio)")
    summary.append("-" * 40)
    
    policy_discriminations = []
    for episode in info_gain_results['episodes']:
        disc = calculate_discrimination_ratio(episode)
        if disc:
            policy_discriminations.append(disc[-1])
    
    if policy_discriminations:
        summary.append(f"Policy Final Discrimination: {np.mean(policy_discriminations):.3f} ± {np.std(policy_discriminations):.3f}")
    
    if 'baselines' in info_gain_results and 'random' in info_gain_results['baselines']:
        random_discriminations = []
        for episode in info_gain_results['baselines']['random']['episodes']:
            disc = calculate_discrimination_ratio(episode)
            if disc:
                random_discriminations.append(disc[-1])
        if random_discriminations:
            summary.append(f"Random Final Discrimination: {np.mean(random_discriminations):.3f} ± {np.std(random_discriminations):.3f}")
    
    summary.append("")
    
    # Parent Selection
    summary.append("2. PARENT SELECTION")
    summary.append("-" * 40)
    
    policy_rates = []
    for episode in parent_selection_results['episodes']:
        rate = np.mean(episode['parent_selections'])
        policy_rates.append(rate)
    
    if policy_rates:
        summary.append(f"Policy Parent Selection Rate: {np.mean(policy_rates):.1%} ± {np.std(policy_rates):.1%}")
    
    if 'baselines' in parent_selection_results and 'random' in parent_selection_results['baselines']:
        random_rates = []
        for episode in parent_selection_results['baselines']['random']['episodes']:
            rate = np.mean(episode['parent_selections'])
            random_rates.append(rate)
        if random_rates:
            summary.append(f"Random Parent Selection Rate: {np.mean(random_rates):.1%} ± {np.std(random_rates):.1%}")
    
    summary.append("")
    
    # Target Optimization
    summary.append("3. TARGET OPTIMIZATION")
    summary.append("-" * 40)
    
    policy_finals = []
    for episode in target_results['episodes']:
        if episode['target_values']:
            policy_finals.append(episode['target_values'][-1])
    
    if policy_finals:
        summary.append(f"Policy Final Target: {np.mean(policy_finals):.3f} ± {np.std(policy_finals):.3f}")
    
    if 'baselines' in target_results and 'random' in target_results['baselines']:
        random_finals = []
        for episode in target_results['baselines']['random']['episodes']:
            if episode['target_values']:
                random_finals.append(episode['target_values'][-1])
        if random_finals:
            summary.append(f"Random Final Target: {np.mean(random_finals):.3f} ± {np.std(random_finals):.3f}")
    
    summary.append("")
    summary.append("="*60)
    summary.append("CONCLUSION: Success in structure learning and parent selection,")
    summary.append("           but failure in target optimization.")
    summary.append("="*60)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"Saved summary table to: {output_path}")
    
    # Also print to console
    print('\n'.join(summary))


def main():
    """Main function to create combined thesis plots."""
    parser = argparse.ArgumentParser(
        description="Create combined thesis plots showing mixed results"
    )
    parser.add_argument('--info-gain-dir', type=Path, required=True,
                       help='Directory with info gain evaluation results')
    parser.add_argument('--parent-selection-dir', type=Path, required=True,
                       help='Directory with parent selection evaluation results')
    parser.add_argument('--target-dir', type=Path, required=True,
                       help='Directory with target optimization evaluation results')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Directory to save combined plots')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    
    # Find latest evaluation files
    info_gain_files = sorted(args.info_gain_dir.glob('evaluation_*.json'))
    parent_selection_files = sorted(args.parent_selection_dir.glob('evaluation_*.json'))
    target_files = sorted(args.target_dir.glob('evaluation_*.json'))
    
    if not info_gain_files:
        print(f"No evaluation files found in {args.info_gain_dir}")
        return 1
    
    if not parent_selection_files:
        print(f"No evaluation files found in {args.parent_selection_dir}")
        return 1
    
    if not target_files:
        print(f"No evaluation files found in {args.target_dir}")
        return 1
    
    # Use latest files
    info_gain_file = info_gain_files[-1]
    parent_selection_file = parent_selection_files[-1]
    target_file = target_files[-1]
    
    print(f"Using info gain results: {info_gain_file}")
    print(f"Using parent selection results: {parent_selection_file}")
    print(f"Using target optimization results: {target_file}")
    
    with open(info_gain_file, 'r') as f:
        info_gain_results = json.load(f)
    
    with open(parent_selection_file, 'r') as f:
        parent_selection_results = json.load(f)
    
    with open(target_file, 'r') as f:
        target_results = json.load(f)
    
    # Generate combined thesis plot
    print("Generating combined thesis figure...")
    plot_thesis_mixed_results(
        info_gain_results,
        parent_selection_results,
        target_results,
        args.output_dir
    )
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(
        info_gain_results,
        parent_selection_results,
        target_results,
        args.output_dir / 'thesis_evaluation_summary.txt'
    )
    
    print("Thesis plots complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())