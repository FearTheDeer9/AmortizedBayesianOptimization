#!/bin/bash
# Full GRPO Training Experiment with Diverse SCMs
# This script runs comprehensive training and analysis

set -e  # Exit on error

echo "=========================================="
echo "GRPO TRAINING WITH DIVERSE SCMs (3-100)"
echo "=========================================="
echo ""

# Configuration
EPISODES=200
MIN_VARS=3
MAX_VARS=100
SEED=42
EXPERIMENT_NAME="grpo_diverse_full"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${EXPERIMENT_NAME}_${TIMESTAMP}"
RESULTS_DIR="results/diverse_training_fixed"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo "  Episodes: $EPISODES"
echo "  SCM size range: $MIN_VARS-$MAX_VARS variables"
echo "  Seed: $SEED"
echo "  Log directory: $LOG_DIR"
echo ""

# Function to run training with a specific seed
run_training() {
    local seed=$1
    local log_file="${LOG_DIR}/training_seed_${seed}.log"
    
    echo "Starting training with seed $seed..."
    echo "  Log file: $log_file"
    
    python experiments/joint-grpo-target-training/train_grpo_diverse_fixed.py \
        --episodes $EPISODES \
        --min-vars $MIN_VARS \
        --max-vars $MAX_VARS \
        --seed $seed \
        --wandb \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Training completed successfully for seed $seed"
    else
        echo "  ❌ Training failed for seed $seed"
        return 1
    fi
}

# Main training run
echo "=========================================="
echo "STARTING MAIN TRAINING RUN"
echo "=========================================="
run_training $SEED

# Optional: Run with multiple seeds for robustness analysis
if [ "$1" = "--multi-seed" ]; then
    echo ""
    echo "=========================================="
    echo "RUNNING MULTI-SEED ANALYSIS"
    echo "=========================================="
    
    for seed in 43 44 45; do
        echo ""
        run_training $seed
    done
    
    echo ""
    echo "Aggregating multi-seed results..."
    python -c "
import json
import numpy as np
from pathlib import Path
import pandas as pd

results_dir = Path('$RESULTS_DIR')
seeds = [42, 43, 44, 45]
all_results = []

for seed in seeds:
    # Find the most recent result file for this seed
    pattern = f'diverse_fixed_{MIN_VARS}to{MAX_VARS}_*.json'
    files = list(results_dir.glob(pattern))
    if files:
        with open(files[-1], 'r') as f:
            data = json.load(f)
            if 'analysis' in data and 'overall' in data['analysis']:
                all_results.append({
                    'seed': seed,
                    'mean_target': data['analysis']['overall']['mean_target'],
                    'best_target': data['analysis']['overall']['best_target'],
                    'parent_rate': data['analysis']['overall']['mean_parent_rate']
                })

if all_results:
    df = pd.DataFrame(all_results)
    print('\nMulti-seed Analysis:')
    print(f'  Mean target: {df[\"mean_target\"].mean():.3f} ± {df[\"mean_target\"].std():.3f}')
    print(f'  Best target: {df[\"best_target\"].mean():.3f} ± {df[\"best_target\"].std():.3f}')
    print(f'  Parent rate: {df[\"parent_rate\"].mean():.3f} ± {df[\"parent_rate\"].std():.3f}')
"
fi

# Generate analysis plots
echo ""
echo "=========================================="
echo "GENERATING ANALYSIS PLOTS"
echo "=========================================="

python -c "
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the most recent CSV file
results_dir = Path('$RESULTS_DIR')
csv_files = list(results_dir.glob('diverse_fixed_*.csv'))
if not csv_files:
    print('No CSV files found for analysis')
    exit(0)

csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
df = pd.read_csv(csv_file)

print(f'Analyzing {len(df)} episodes from {csv_file.name}')

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('GRPO Training Analysis - Diverse SCMs ($MIN_VARS-$MAX_VARS variables)', fontsize=16)

# 1. Learning curve
ax = axes[0, 0]
ax.plot(df['episode'], df['mean_target'], label='Mean Target', alpha=0.7)
ax.plot(df['episode'], df['best_target'], label='Best Target', alpha=0.7)
if len(df) >= 20:
    rolling_mean = df['mean_target'].rolling(20).mean()
    ax.plot(df['episode'], rolling_mean, 'r-', label='Rolling Avg (20)', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Target Value')
ax.set_title('Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Parent selection rate
ax = axes[0, 1]
ax.plot(df['episode'], df['parent_selection_rate'], alpha=0.7)
if len(df) >= 20:
    rolling_parent = df['parent_selection_rate'].rolling(20).mean()
    ax.plot(df['episode'], rolling_parent, 'g-', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Parent Selection Rate')
ax.set_title('Structure Learning Progress')
ax.grid(True, alpha=0.3)

# 3. Performance by SCM size
ax = axes[0, 2]
for category in ['small', 'medium', 'large']:
    cat_df = df[df['size_category'] == category]
    if not cat_df.empty:
        ax.scatter(cat_df['num_vars'], cat_df['mean_target'], 
                  label=f'{category.capitalize()} SCMs', alpha=0.5)
ax.set_xlabel('Number of Variables')
ax.set_ylabel('Mean Target Value')
ax.set_title('Performance by SCM Size')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Intervention value analysis
ax = axes[1, 0]
ax.plot(df['episode'], df['mean_abs_intervention_value'], alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Mean |Intervention Value|')
ax.set_title('Intervention Magnitude Over Time')
ax.grid(True, alpha=0.3)

# 5. Reward components
ax = axes[1, 1]
ax.plot(df['episode'], df['mean_reward'], alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Mean Reward')
ax.set_title('Reward Evolution')
ax.grid(True, alpha=0.3)

# 6. Convergence analysis
ax = axes[1, 2]
if 'target_std' in df.columns:
    ax.plot(df['episode'], df['target_std'], alpha=0.7)
    ax.set_ylabel('Target Std Dev')
    ax.set_title('Performance Stability')
else:
    # Alternative: Show parent vs non-parent interventions
    ax.bar(['Parent', 'Non-Parent'], 
           [df['parent_interventions'].sum(), df['non_parent_interventions'].sum()])
    ax.set_ylabel('Total Interventions')
    ax.set_title('Intervention Distribution')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('${LOG_DIR}/training_analysis.png', dpi=150, bbox_inches='tight')
print(f'Saved analysis plots to ${LOG_DIR}/training_analysis.png')

# Print summary statistics
print('\n' + '='*50)
print('TRAINING SUMMARY')
print('='*50)
print(f'Total episodes: {len(df)}')
print(f'Final mean target: {df.iloc[-1][\"mean_target\"]:.3f}')
print(f'Best achieved target: {df[\"best_target\"].min():.3f}')
print(f'Final parent selection rate: {df.iloc[-1][\"parent_selection_rate\"]:.2%}')

# Performance by category
for category in ['small', 'medium', 'large']:
    cat_df = df[df['size_category'] == category]
    if not cat_df.empty:
        print(f'\n{category.upper()} SCMs ({cat_df.iloc[0][\"num_vars\"]}-{cat_df.iloc[-1][\"num_vars\"]} vars):')
        print(f'  Episodes: {len(cat_df)}')
        print(f'  Mean target: {cat_df[\"mean_target\"].mean():.3f}')
        print(f'  Best target: {cat_df[\"best_target\"].min():.3f}')
" || echo "  ⚠️  Plot generation failed (matplotlib may not be installed)"

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "To monitor training with WandB:"
echo "  Visit: https://wandb.ai/your-username/causal-bayes-opt-grpo"
echo ""
echo "To analyze results:"
echo "  python analyze_grpo_results.py --results-dir $RESULTS_DIR"