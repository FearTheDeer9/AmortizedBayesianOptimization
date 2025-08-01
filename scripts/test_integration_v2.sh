#!/bin/bash

# Integration test for refactored evaluation script
# This tests the full loop: train -> save checkpoints -> evaluate with new script

echo "=========================================="
echo "INTEGRATION TEST: Train + Evaluate V2"
echo "=========================================="
echo ""

# Create test checkpoint directory
CHECKPOINT_BASE="checkpoints/test_v2"
mkdir -p $CHECKPOINT_BASE

# 1. Train BC Surrogate (minimal episodes for testing)
echo "1. Training BC Surrogate..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 5 \
    --batch_size 32 \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --surrogate_heads 8 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --max_demos 5 \
    --seed 42

# Check if surrogate was created
if [ ! -d "${CHECKPOINT_BASE}/bc_surrogate_final" ]; then
    echo "ERROR: BC surrogate checkpoint not created!"
    exit 1
fi

echo ""
echo "2. Training BC Policy..."
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 5 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --max_demos 5 \
    --seed 43

# Check if policy was created
if [ ! -d "${CHECKPOINT_BASE}/bc_final" ]; then
    echo "ERROR: BC policy checkpoint not created!"
    exit 1
fi

echo ""
echo "3. Training GRPO..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 5 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --scm_type mixed \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --seed 44

# Check if GRPO was created
if [ ! -d "${CHECKPOINT_BASE}/unified_grpo_final" ]; then
    echo "ERROR: GRPO checkpoint not created!"
    exit 1
fi

echo ""
echo "=========================================="
echo "TRAINING COMPLETE! Testing evaluation..."
echo "=========================================="
echo ""

# 4. Test evaluation with new v2 script
echo "4. Testing evaluation with refactored script..."

# Test various combinations
poetry run python scripts/evaluate_acbo_methods_v2.py \
    --register_surrogate bc_surrogate ${CHECKPOINT_BASE}/bc_surrogate_final \
    --register_surrogate dummy dummy \
    --register_policy bc_policy ${CHECKPOINT_BASE}/bc_final \
    --register_policy grpo_policy ${CHECKPOINT_BASE}/unified_grpo_final \
    --include_baselines \
    --baseline_surrogate dummy \
    --evaluate_pairs bc_policy bc_surrogate \
    --evaluate_pairs bc_policy dummy \
    --evaluate_pairs grpo_policy bc_surrogate \
    --evaluate_pairs grpo_policy dummy \
    --evaluate_pairs random bc_surrogate \
    --n_scms 2 \
    --n_interventions 3 \
    --output_dir evaluation_results/test_v2 \
    --plot

# Check if results were created
if [ ! -f "evaluation_results/test_v2/evaluation_results.json" ]; then
    echo "ERROR: Evaluation results not created!"
    exit 1
fi

echo ""
echo "=========================================="
echo "INTEGRATION TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: evaluation_results/test_v2/"
echo ""
echo "Key files to check:"
echo "  - ${CHECKPOINT_BASE}/bc_surrogate_final/ (surrogate checkpoint)"
echo "  - ${CHECKPOINT_BASE}/bc_final/ (policy checkpoint)"
echo "  - ${CHECKPOINT_BASE}/unified_grpo_final/ (GRPO checkpoint)"
echo "  - evaluation_results/test_v2/evaluation_results.json"
echo "  - evaluation_results/test_v2/method_comparison.png"
echo ""
echo "The test should show:"
echo "  - Different results for policies with bc_surrogate vs dummy surrogate"
echo "  - BC surrogate should give non-uniform probabilities"
echo "  - All methods should evaluate without errors"
echo "=========================================="