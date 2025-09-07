#!/usr/bin/env python3
"""
Plot evaluation results for oracle surrogate and structure learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def plot_oracle_surrogate_results(json_path: Path, output_dir: Path):
    """
    Plot results from oracle surrogate evaluation.
    
    Compares policy performance with perfect vs learned surrogate.
    """
    print(f"\nPlotting oracle surrogate results from: {json_path}")
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Policy Performance with Oracle Surrogate (Perfect Parent Predictions)', fontsize=14, fontweight='bold')
    
    # Extract episode data
    episodes = data.get('episodes', [])
    if not episodes:
        print("No episode data found!")
        return
    
    # 1. Parent Selection Rate Comparison
    ax = axes[0, 0]
    parent_rates = [ep['summary']['parent_selection_rate'] for ep in episodes]
    
    # Calculate expected random rate (average across episodes)
    expected_random_rates = []
    for ep in episodes:
        num_parents = len(ep['scm_info']['true_parents'])
        num_vars = ep['scm_info']['num_variables']
        expected_random = num_parents / (num_vars - 1) if num_vars > 1 else 0
        expected_random_rates.append(expected_random)
    
    x_pos = np.arange(len(parent_rates))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, parent_rates, width, label='Policy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, expected_random_rates, width, label='Expected Random', color='gray', alpha=0.6)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Parent Selection Rate')
    ax.set_title('Parent Selection Rate per Episode')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(parent_rates))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # 2. F1 Score Progression
    ax = axes[0, 1]
    for i, ep in enumerate(episodes[:5]):  # Plot first 5 episodes
        f1_scores = ep.get('f1_scores', [])
        if f1_scores:
            ax.plot(f1_scores, label=f"Episode {i+1}", alpha=0.7)
    
    ax.set_xlabel('Intervention')
    ax.set_ylabel('F1 Score')
    ax.set_title('Structure Learning (F1) Over Interventions')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 3. Final F1 Distribution
    ax = axes[0, 2]
    final_f1s = [ep['summary']['final_f1'] for ep in episodes]
    
    ax.hist(final_f1s, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_f1s), color='red', linestyle='--', 
               label=f'Mean: {np.mean(final_f1s):.3f}')
    ax.set_xlabel('Final F1 Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final F1 Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance Summary Statistics
    ax = axes[1, 0]
    ax.axis('off')
    
    # Calculate summary statistics
    summary_stats = {
        'Parent Selection Rate': f"{np.mean(parent_rates):.2%} ± {np.std(parent_rates):.2%}",
        'Final F1 Score': f"{np.mean(final_f1s):.3f} ± {np.std(final_f1s):.3f}",
        'Best F1 Score': f"{np.max(final_f1s):.3f}",
        'Worst F1 Score': f"{np.min(final_f1s):.3f}",
        'Improvement over Random': f"{np.mean(parent_rates) / np.mean(expected_random_rates):.2f}x"
    }
    
    # Display as table
    table_data = [[k, v] for k, v in summary_stats.items()]
    table = ax.table(cellText=table_data, 
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # 5. Parent Selection by Variable Position
    ax = axes[1, 1]
    
    # Analyze which variables are being selected
    intervention_counts = {}
    for ep in episodes:
        for intervention in ep.get('interventions', []):
            if intervention and intervention[0]:  # (variable, value) tuple
                var = intervention[0]
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
    
    if intervention_counts:
        sorted_vars = sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        vars_list, counts = zip(*sorted_vars)
        
        # Check which are parents (this is approximate since parent status varies by episode)
        colors = []
        for var in vars_list:
            # Check if this variable is often a parent
            is_parent_count = sum(1 for ep in episodes if var in ep['scm_info']['true_parents'])
            if is_parent_count > len(episodes) / 2:
                colors.append('green')
            else:
                colors.append('gray')
        
        ax.barh(range(len(vars_list)), counts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(vars_list)))
        ax.set_yticklabels(vars_list)
        ax.set_xlabel('Selection Count')
        ax.set_title('Most Selected Variables')
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Often Parent'),
                          Patch(facecolor='gray', alpha=0.7, label='Rarely Parent')]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # 6. Compare with baselines if available
    ax = axes[1, 2]
    
    if 'baselines' in data:
        baseline_data = []
        
        # Policy results
        baseline_data.append({
            'Method': 'Policy',
            'Parent Rate': np.mean(parent_rates),
            'F1 Score': np.mean(final_f1s)
        })
        
        # Random baseline
        if 'random' in data['baselines']:
            random_episodes = data['baselines']['random'].get('episodes', [])
            if random_episodes:
                random_parent_rates = [ep['summary']['parent_selection_rate'] for ep in random_episodes]
                random_f1s = [ep['summary']['final_f1'] for ep in random_episodes]
                baseline_data.append({
                    'Method': 'Random',
                    'Parent Rate': np.mean(random_parent_rates),
                    'F1 Score': np.mean(random_f1s)
                })
        
        # Oracle baseline
        if 'oracle' in data['baselines']:
            oracle_episodes = data['baselines']['oracle'].get('episodes', [])
            if oracle_episodes:
                oracle_parent_rates = [ep['summary']['parent_selection_rate'] for ep in oracle_episodes]
                oracle_f1s = [ep['summary']['final_f1'] for ep in oracle_episodes]
                baseline_data.append({
                    'Method': 'Oracle',
                    'Parent Rate': np.mean(oracle_parent_rates),
                    'F1 Score': np.mean(oracle_f1s)
                })
        
        if baseline_data:
            df_baseline = pd.DataFrame(baseline_data)
            
            x = np.arange(len(df_baseline))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, df_baseline['Parent Rate'], width, 
                          label='Parent Selection Rate', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, df_baseline['F1 Score'], width, 
                          label='F1 Score', color='coral', alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_title('Comparison with Baselines')
            ax.set_xticks(x)
            ax.set_xticklabels(df_baseline['Method'])
            ax.legend()
            ax.set_ylim([0, 1.1])
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No baseline data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Baseline Comparison')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'oracle_surrogate_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.show()


def plot_structure_learning_results(csv_path: Path, output_dir: Path):
    """
    Plot results from structure learning comparison.
    
    Shows how different strategies help reveal structure.
    """
    print(f"\nPlotting structure learning results from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Structure Learning Comparison: Policy vs Random vs Oracle', 
                 fontsize=14, fontweight='bold')
    
    # 1. Final F1 Score by Strategy
    ax = axes[0, 0]
    sns.boxplot(data=df, x='strategy', y='final_f1', ax=ax, 
                order=['random', 'policy', 'oracle'])
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Final F1 Score')
    ax.set_title('Final F1 Score Distribution')
    ax.set_ylim([0, 1.05])
    
    # Add mean markers
    means = df.groupby('strategy')['final_f1'].mean()
    for i, strategy in enumerate(['random', 'policy', 'oracle']):
        if strategy in means.index:
            ax.plot(i, means[strategy], marker='D', color='red', markersize=8, 
                   label='Mean' if i == 0 else '')
    ax.legend()
    
    # 2. Interventions to F1 > 0.9
    ax = axes[0, 1]
    
    # Replace inf values with max interventions for visualization
    max_interventions = df['interventions_to_90'].replace(np.inf, np.nan).max()
    if pd.isna(max_interventions):
        max_interventions = 50  # Default max
    
    df_plot = df.copy()
    df_plot['interventions_to_90'] = df_plot['interventions_to_90'].replace(
        df['interventions_to_90'].max(), max_interventions + 5
    )
    
    sns.barplot(data=df_plot, x='strategy', y='interventions_to_90', 
                ax=ax, order=['random', 'policy', 'oracle'], 
                capsize=.2, errwidth=1.5)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Interventions to F1 > 0.9')
    ax.set_title('Sample Efficiency')
    ax.axhline(y=max_interventions, color='red', linestyle='--', alpha=0.5, 
               label=f'Max ({int(max_interventions)})')
    ax.legend()
    
    # 3. Performance by Structure Type
    ax = axes[0, 2]
    
    if 'structure' in df.columns:
        structures = df['structure'].unique()
        width = 0.25
        x = np.arange(len(structures))
        
        for i, strategy in enumerate(['random', 'policy', 'oracle']):
            means = []
            stds = []
            for struct in structures:
                data = df[(df['structure'] == struct) & (df['strategy'] == strategy)]['final_f1']
                means.append(data.mean() if len(data) > 0 else 0)
                stds.append(data.std() if len(data) > 0 else 0)
            
            ax.bar(x + i*width, means, width, label=strategy.capitalize(), 
                  yerr=stds, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Structure Type')
        ax.set_ylabel('Final F1 Score')
        ax.set_title('Performance by Structure Type')
        ax.set_xticks(x + width)
        ax.set_xticklabels(structures)
        ax.legend()
        ax.set_ylim([0, 1.1])
    
    # 4. Summary Statistics Table
    ax = axes[1, 0]
    ax.axis('off')
    
    summary = df.groupby('strategy').agg({
        'final_f1': ['mean', 'std'],
        'interventions_to_90': 'mean'
    }).round(3)
    
    table_data = []
    for strategy in ['random', 'policy', 'oracle']:
        if strategy in summary.index:
            row = [
                strategy.capitalize(),
                f"{summary.loc[strategy, ('final_f1', 'mean')]:.3f} ± {summary.loc[strategy, ('final_f1', 'std')]:.3f}",
                f"{summary.loc[strategy, ('interventions_to_90', 'mean')]:.1f}"
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Strategy', 'Final F1 (mean ± std)', 'Interventions to 0.9'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # 5. Improvement over Random
    ax = axes[1, 1]
    
    random_f1 = df[df['strategy'] == 'random']['final_f1'].mean()
    improvements = {}
    
    for strategy in ['policy', 'oracle']:
        strategy_f1 = df[df['strategy'] == strategy]['final_f1'].mean()
        improvements[strategy] = (strategy_f1 - random_f1) / random_f1 * 100 if random_f1 > 0 else 0
    
    strategies = list(improvements.keys())
    values = list(improvements.values())
    colors = ['steelblue' if v > 0 else 'coral' for v in values]
    
    bars = ax.bar(strategies, values, color=colors, alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Improvement over Random (%)')
    ax.set_title('Relative Performance vs Random Baseline')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:+.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, 3 if val > 0 else -15),
                   textcoords="offset points",
                   ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=11, fontweight='bold')
    
    # 6. Success Rate (reaching F1 > 0.9)
    ax = axes[1, 2]
    
    max_interventions_actual = df['interventions_to_90'].max()
    success_rates = {}
    
    for strategy in ['random', 'policy', 'oracle']:
        strategy_data = df[df['strategy'] == strategy]['interventions_to_90']
        # Count episodes that reached F1 > 0.9
        success_count = (strategy_data < max_interventions_actual).sum()
        total_count = len(strategy_data)
        success_rates[strategy] = success_count / total_count * 100 if total_count > 0 else 0
    
    strategies = list(success_rates.keys())
    rates = list(success_rates.values())
    
    bars = ax.bar(strategies, rates, color=['gray', 'steelblue', 'green'], alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('% Episodes Reaching F1 > 0.9')
    ax.set_ylim([0, 105])
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.annotate(f'{rate:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, rate),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'structure_learning_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    
    parser.add_argument('--oracle-results', type=Path,
                       help='Path to oracle surrogate evaluation JSON file')
    parser.add_argument('--structure-results', type=Path,
                       help='Path to structure learning CSV file')
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation_plots'),
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot oracle surrogate results if provided
    if args.oracle_results and args.oracle_results.exists():
        plot_oracle_surrogate_results(args.oracle_results, args.output_dir)
    elif args.oracle_results:
        print(f"Oracle results file not found: {args.oracle_results}")
    
    # Plot structure learning results if provided
    if args.structure_results and args.structure_results.exists():
        plot_structure_learning_results(args.structure_results, args.output_dir)
    elif args.structure_results:
        print(f"Structure learning results file not found: {args.structure_results}")
    
    if not args.oracle_results and not args.structure_results:
        print("Please provide at least one results file to plot.")
        print("\nExample usage:")
        print("  python plot_evaluation_results.py \\")
        print("    --oracle-results evaluation_results_oracle_test/evaluation_*.json \\")
        print("    --structure-results structure_learning_test/structure_learning_*.csv")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())