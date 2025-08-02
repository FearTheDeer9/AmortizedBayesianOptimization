#!/bin/bash
# Comprehensive ACBO Evaluation Script
# Evaluates different policy-surrogate combinations with and without active learning

set -e  # Exit on error

echo "=============================================="
echo "COMPREHENSIVE ACBO EVALUATION"
echo "=============================================="
echo ""

# Check if checkpoint directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_directory>"
    echo "Example: $0 checkpoints/comprehensive_20250131_120000"
    echo ""
    echo "Looking for recent training directories..."
    LATEST_DIR=$(ls -td checkpoints/comprehensive_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Found: $LATEST_DIR"
        echo "Use this directory? (y/n)"
        read -r response
        if [ "$response" = "y" ]; then
            CHECKPOINT_DIR="$LATEST_DIR"
        else
            exit 1
        fi
    else
        echo "No comprehensive training directories found."
        echo "Please run train_comprehensive_acbo.sh first."
        exit 1
    fi
else
    CHECKPOINT_DIR="$1"
fi

# Verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Verify required checkpoints exist
BC_SURROGATE="$CHECKPOINT_DIR/bc_surrogate_final/checkpoint.pkl"
BC_POLICY="$CHECKPOINT_DIR/bc_final/checkpoint.pkl"
GRPO_POLICY="$CHECKPOINT_DIR/unified_grpo_final/checkpoint.pkl"

if [ ! -f "$BC_SURROGATE" ]; then
    echo "Error: BC surrogate checkpoint not found: $BC_SURROGATE"
    exit 1
fi

if [ ! -f "$BC_POLICY" ]; then
    echo "Error: BC policy checkpoint not found: $BC_POLICY"
    exit 1
fi

if [ ! -f "$GRPO_POLICY" ]; then
    echo "Error: GRPO policy checkpoint not found: $GRPO_POLICY"
    exit 1
fi

echo "Using checkpoints from: $CHECKPOINT_DIR"
echo ""

# Create output directories
OUTPUT_BASE="evaluation_results/comprehensive_$(date +%Y%m%d_%H%M%S)"
OUTPUT_NO_ACTIVE="$OUTPUT_BASE/no_active_learning"
OUTPUT_WITH_ACTIVE="$OUTPUT_BASE/with_active_learning"

mkdir -p "$OUTPUT_NO_ACTIVE"
mkdir -p "$OUTPUT_WITH_ACTIVE"

# Common evaluation parameters
N_SCMS=10
N_INTERVENTIONS=20
N_SAMPLES=10
SEED=42

echo "=============================================="
echo "1. Evaluating WITHOUT Active Learning"
echo "=============================================="
echo ""

poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$BC_SURROGATE" \
    --register_surrogate dummy dummy \
    --register_policy grpo "$GRPO_POLICY" \
    --register_policy bc "$BC_POLICY" \
    --evaluate_pairs grpo bc \
    --evaluate_pairs bc bc \
    --evaluate_pairs random bc \
    --include_baselines --baseline_surrogate bc \
    --surrogate_update_strategy none \
    --n_scms $N_SCMS \
    --n_obs 100 \
    --n_interventions $N_INTERVENTIONS \
    --n_samples $N_SAMPLES \
    --seed $SEED \
    --plot --plot_trajectories \
    --output_dir "$OUTPUT_NO_ACTIVE"

echo ""
echo "=============================================="
echo "2. Evaluating WITH Active Learning (BIC)"
echo "=============================================="
echo ""

poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$BC_SURROGATE" \
    --register_surrogate dummy dummy \
    --register_policy grpo "$GRPO_POLICY" \
    --register_policy bc "$BC_POLICY" \
    --evaluate_pairs grpo bc \
    --evaluate_pairs bc bc \
    --evaluate_pairs random bc \
    --include_baselines --baseline_surrogate bc \
    --surrogate_update_strategy bic \
    --n_scms $N_SCMS \
    --n_obs 100 \
    --n_interventions $N_INTERVENTIONS \
    --n_samples $N_SAMPLES \
    --seed $SEED \
    --plot --plot_trajectories \
    --output_dir "$OUTPUT_WITH_ACTIVE"

echo ""
echo "=============================================="
echo "3. Creating Comparison Analysis"
echo "=============================================="

# Create comparison summary
cat > "$OUTPUT_BASE/comparison_summary.md" << EOF
# Comprehensive ACBO Evaluation Results

## Evaluation Date: $(date)

### Checkpoints Used:
- BC Surrogate: $BC_SURROGATE
- BC Policy: $BC_POLICY  
- GRPO Policy: $GRPO_POLICY

### Evaluation Parameters:
- Number of SCMs: $N_SCMS
- Number of interventions: $N_INTERVENTIONS
- Samples per intervention: $N_SAMPLES
- Random seed: $SEED

### Methods Evaluated:
1. **GRPO + BC Surrogate**: RL policy with structure learning
2. **BC + BC Surrogate**: Full behavioral cloning approach
3. **Random + BC Surrogate**: Random baseline with structure learning
4. **Oracle + BC Surrogate**: Perfect knowledge baseline (upper bound)
5. **Random (no surrogate)**: Pure random baseline (lower bound)

### Key Comparisons:

#### Impact of Active Learning:
- Compare results in \`no_active_learning/\` vs \`with_active_learning/\`
- Look for improvements in F1 scores over time
- Check if target value optimization improves

#### Policy Comparison:
- GRPO vs BC vs Random vs Oracle
- Both with and without active learning

### Result Files:
- No Active Learning: $OUTPUT_NO_ACTIVE/
- With Active Learning: $OUTPUT_WITH_ACTIVE/

### Plots Generated:
1. **method_comparison.png**: Bar charts of mean improvement and F1 scores
2. **target_trajectories.png**: Target value evolution over interventions
3. **structure_trajectories.png**: F1 and SHD evolution
4. **scm_example_trajectories.png**: Individual SCM examples

EOF

# Create a Python script to generate comparative analysis
cat > "$OUTPUT_BASE/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""Analyze and compare active learning vs no active learning results."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(path):
    with open(path / 'evaluation_results.json', 'r') as f:
        return json.load(f)

# Load both result sets
no_active = load_results(Path('no_active_learning'))
with_active = load_results(Path('with_active_learning'))

# Compare key metrics
print("Comparative Analysis: Active Learning Impact")
print("=" * 60)

for method in ['grpo+bc', 'bc+bc', 'random+bc']:
    if method in no_active and method in with_active:
        no_active_metrics = no_active[method]['aggregate_metrics']
        with_active_metrics = with_active[method]['aggregate_metrics']
        
        print(f"\n{method.upper()}:")
        print(f"  Improvement - No Active: {no_active_metrics['mean_improvement']:.3f}")
        print(f"  Improvement - With Active: {with_active_metrics['mean_improvement']:.3f}")
        print(f"  Difference: {with_active_metrics['mean_improvement'] - no_active_metrics['mean_improvement']:.3f}")
        
        print(f"  F1 Score - No Active: {no_active_metrics['mean_f1']:.3f}")
        print(f"  F1 Score - With Active: {with_active_metrics['mean_f1']:.3f}")
        print(f"  Difference: {with_active_metrics['mean_f1'] - no_active_metrics['mean_f1']:.3f}")

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

methods = ['grpo+bc', 'bc+bc', 'random+bc']
x = np.arange(len(methods))
width = 0.35

# Improvement comparison
improvements_no = [no_active[m]['aggregate_metrics']['mean_improvement'] for m in methods if m in no_active]
improvements_with = [with_active[m]['aggregate_metrics']['mean_improvement'] for m in methods if m in with_active]

ax1.bar(x - width/2, improvements_no, width, label='No Active Learning', alpha=0.8)
ax1.bar(x + width/2, improvements_with, width, label='With Active Learning', alpha=0.8)
ax1.set_xlabel('Method')
ax1.set_ylabel('Mean Improvement')
ax1.set_title('Target Value Improvement Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()
ax1.grid(True, alpha=0.3)

# F1 score comparison
f1_no = [no_active[m]['aggregate_metrics']['mean_f1'] for m in methods if m in no_active]
f1_with = [with_active[m]['aggregate_metrics']['mean_f1'] for m in methods if m in with_active]

ax2.bar(x - width/2, f1_no, width, label='No Active Learning', alpha=0.8)
ax2.bar(x + width/2, f1_with, width, label='With Active Learning', alpha=0.8)
ax2.set_xlabel('Method')
ax2.set_ylabel('Mean F1 Score')
ax2.set_title('Structure Learning F1 Score Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('active_learning_comparison.png', dpi=150)
plt.close()

print("\nComparison plot saved to: active_learning_comparison.png")
EOF

chmod +x "$OUTPUT_BASE/analyze_results.py"

# Run the analysis
cd "$OUTPUT_BASE"
poetry run python analyze_results.py
cd - > /dev/null

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Key outputs:"
echo "  - No active learning results: $OUTPUT_NO_ACTIVE/"
echo "  - With active learning results: $OUTPUT_WITH_ACTIVE/"
echo "  - Comparison summary: $OUTPUT_BASE/comparison_summary.md"
echo "  - Active learning comparison plot: $OUTPUT_BASE/active_learning_comparison.png"
echo ""
echo "Next steps:"
echo "1. Review the comparison summary"
echo "2. Compare trajectory plots between active/no-active"
echo "3. Look for improvements in structure learning (F1 scores)"
echo "4. Check if active learning helps target optimization"
echo "=============================================="