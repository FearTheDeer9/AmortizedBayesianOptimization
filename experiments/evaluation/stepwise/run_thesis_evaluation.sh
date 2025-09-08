#!/bin/bash
# Thesis evaluation master script
# Runs all evaluations for demonstrating mixed results:
# - Success in info gain (discrimination ratio)
# - Success in parent selection
# - Failure in target optimization

set -e  # Exit on error

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${BASE_DIR}/thesis_results/policy"
EVAL_SCRIPT="${BASE_DIR}/experiments/evaluation/stepwise/full_evaluation.py"

# Checkpoint paths
POLICY_INFOGAIN="imperial-vm-checkpoints/grpo_enhanced_20250907_181212/final_policy.pkl"
POLICY_MAIN="imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl"
SURROGATE="imperial-vm-checkpoints/avici_style_20250907_034427/best_model.pkl"

# Common parameters
NUM_EPISODES=10  # Reduced for faster evaluation (was 20)
SEED=42

echo "=========================================="
echo "THESIS POLICY EVALUATION FRAMEWORK"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Create results directories
mkdir -p "${RESULTS_DIR}/info_gain"
mkdir -p "${RESULTS_DIR}/parent_selection"
mkdir -p "${RESULTS_DIR}/target_optimization"
mkdir -p "${RESULTS_DIR}/combined"

# Function to run evaluation
run_evaluation() {
    local name=$1
    local policy=$2
    local output_dir=$3
    local structures=$4
    local num_vars=$5
    local num_interventions=$6
    
    echo "----------------------------------------"
    echo "Running: ${name}"
    echo "Policy: ${policy}"
    echo "Output: ${output_dir}"
    echo "Structures: ${structures}"
    echo "Variables: ${num_vars}"
    echo "Interventions: ${num_interventions}"
    echo "----------------------------------------"
    
    cd "${BASE_DIR}"
    python "${EVAL_SCRIPT}" \
        --policy-path "${policy}" \
        --surrogate-path "${SURROGATE}" \
        --structures ${structures} \
        --num-vars ${num_vars} \
        --num-episodes ${NUM_EPISODES} \
        --num-interventions ${num_interventions} \
        --initial-observations 20 \
        --initial-interventions 10 \
        --baselines \
        --output-dir "${output_dir}" \
        --seed ${SEED}
    
    echo "✓ ${name} complete"
    echo ""
}

# 1. Info Gain Evaluation (Discrimination Ratio Focus)
echo "=========================================="
echo "1. INFO GAIN EVALUATION"
echo "=========================================="
# Test on training distribution (chain) with diverse sizes
run_evaluation \
    "Info Gain (Discrimination)" \
    "${POLICY_INFOGAIN}" \
    "${RESULTS_DIR}/info_gain" \
    "chain" \
    "8 15 30" \
    30

# 2. Parent Selection Evaluation
echo "=========================================="
echo "2. PARENT SELECTION EVALUATION"
echo "=========================================="
# Test on training distribution (chain) with diverse sizes
run_evaluation \
    "Parent Selection" \
    "${POLICY_MAIN}" \
    "${RESULTS_DIR}/parent_selection" \
    "chain" \
    "8 15 30" \
    30

# 3. Target Optimization Evaluation
echo "=========================================="
echo "3. TARGET OPTIMIZATION EVALUATION"
echo "=========================================="
# Test on training distribution (chain) with diverse sizes
run_evaluation \
    "Target Optimization" \
    "${POLICY_MAIN}" \
    "${RESULTS_DIR}/target_optimization" \
    "chain" \
    "8 15 30" \
    40

# 4. Generate Plots
echo "=========================================="
echo "4. GENERATING PLOTS"
echo "=========================================="

cd "${BASE_DIR}"

# Generate discrimination ratio plots
echo "Generating discrimination ratio analysis..."
python experiments/evaluation/stepwise/thesis_plot_discrimination.py \
    --info-gain-dir "${RESULTS_DIR}/info_gain" \
    --parent-selection-dir "${RESULTS_DIR}/parent_selection" \
    --output-dir "${RESULTS_DIR}/combined"

# Generate combined thesis figure
echo "Generating combined thesis figure..."
python experiments/evaluation/stepwise/thesis_plot_combined.py \
    --info-gain-dir "${RESULTS_DIR}/info_gain" \
    --parent-selection-dir "${RESULTS_DIR}/parent_selection" \
    --target-dir "${RESULTS_DIR}/target_optimization" \
    --output-dir "${RESULTS_DIR}/combined"

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Key outputs:"
echo "  - Info gain results: ${RESULTS_DIR}/info_gain/"
echo "  - Parent selection results: ${RESULTS_DIR}/parent_selection/"
echo "  - Target optimization results: ${RESULTS_DIR}/target_optimization/"
echo "  - Combined plots: ${RESULTS_DIR}/combined/"
echo ""
echo "Main thesis figure: ${RESULTS_DIR}/combined/thesis_mixed_results.png"