#!/bin/bash
# Quick test of the comprehensive workflow with minimal episodes

set -e

echo "Testing comprehensive ACBO workflow..."
echo ""

# Create test checkpoint directory
TEST_DIR="checkpoints/test_comprehensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "1. Testing BC surrogate training (5 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 5 \
    --batch_size 32 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 42

echo ""
echo "2. Testing BC policy training (5 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 5 \
    --batch_size 32 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 43

echo ""
echo "3. Testing GRPO training WITH surrogate (5 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo_with_surrogate \
    --episodes 5 \
    --batch_size 16 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 44

echo ""
echo "4. Testing GRPO training WITHOUT surrogate (5 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 5 \
    --batch_size 16 \
    --checkpoint_dir "$TEST_DIR" \
    --seed 45

echo ""
echo "5. Testing evaluation without active learning..."
poetry run python scripts/evaluate_acbo_methods_v2.py \
    --register_surrogate bc "$TEST_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo_with_surrogate "$TEST_DIR/unified_grpo_final/checkpoint.pkl" \
    --register_policy grpo_no_surrogate "$TEST_DIR/grpo_no_surrogate_final/checkpoint.pkl" \
    --register_policy bc "$TEST_DIR/bc_final/checkpoint.pkl" \
    --evaluate_pairs grpo_with_surrogate bc \
    --evaluate_pairs grpo_no_surrogate bc \
    --evaluate_pairs bc bc \
    --n_scms 2 \
    --n_interventions 5 \
    --surrogate_update_strategy none \
    --plot --plot_trajectories \
    --output_dir "$TEST_DIR/eval_no_active"

echo ""
echo "6. Testing evaluation with active learning..."
poetry run python scripts/evaluate_acbo_methods_v2.py \
    --register_surrogate bc "$TEST_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo_with_surrogate "$TEST_DIR/unified_grpo_final/checkpoint.pkl" \
    --register_policy grpo_no_surrogate "$TEST_DIR/grpo_no_surrogate_final/checkpoint.pkl" \
    --evaluate_pairs grpo_with_surrogate bc \
    --evaluate_pairs grpo_no_surrogate bc \
    --n_scms 2 \
    --n_interventions 5 \
    --surrogate_update_strategy bic \
    --plot --plot_trajectories \
    --output_dir "$TEST_DIR/eval_with_active"

echo ""
echo "=============================================="
echo "TEST COMPLETE!"
echo "=============================================="
echo ""
echo "Test results saved to: $TEST_DIR"
echo ""
echo "If all tests passed, you can run the full workflow:"
echo "  1. ./train_comprehensive_acbo.sh"
echo "  2. ./evaluate_comprehensive.sh <checkpoint_dir>"
echo ""

# Cleanup option
echo "Delete test directory? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    rm -rf "$TEST_DIR"
    echo "Test directory deleted."
fi