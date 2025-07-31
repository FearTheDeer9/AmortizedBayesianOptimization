#!/bin/bash

# Validation Script for 5-Channel Integration
# Trains models and evaluates with/without surrogates to validate integration

echo "=========================================="
echo "5-CHANNEL INTEGRATION VALIDATION"
echo "=========================================="
echo ""

# Create checkpoint directory
CHECKPOINT_BASE="checkpoints/validation"
mkdir -p $CHECKPOINT_BASE

# 1. Train GRPO (minimal episodes for validation)
echo "1. Training GRPO policy (100 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 100 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --scm_type mixed \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --seed 42

echo ""
echo "2. Training BC Policy (100 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --seed 43

echo ""
echo "3. Training BC Surrogate (100 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 100 \
    --batch_size 32 \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --surrogate_heads 8 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --seed 44

echo ""
echo "=========================================="
echo "TRAINING COMPLETE! Starting evaluation..."
echo "=========================================="
echo ""

# 4. Evaluate all combinations using model registry
echo "4. Evaluating policy-surrogate pairs..."
poetry run python scripts/evaluate_acbo_methods.py \
    --register_policy grpo_policy ${CHECKPOINT_BASE}/unified_grpo_final \
    --register_policy bc_policy ${CHECKPOINT_BASE}/bc_final \
    --register_surrogate bc_surrogate ${CHECKPOINT_BASE}/bc_surrogate_final \
    --register_surrogate none dummy \
    --evaluate_pairs grpo_policy none \
    --evaluate_pairs grpo_policy bc_surrogate \
    --evaluate_pairs bc_policy none \
    --evaluate_pairs bc_policy bc_surrogate \
    --n_scms 5 \
    --n_interventions 10 \
    --plot \
    --output_dir evaluation_results/validation

echo ""
echo "=========================================="
echo "VALIDATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: evaluation_results/validation/"
echo "Check the following:"
echo "  - evaluation_results/validation/evaluation_results.json"
echo "  - evaluation_results/validation/improvement_comparison.png"
echo "  - evaluation_results/validation/target_trajectories.png"
echo ""
echo "Look for improvements when using surrogates:"
echo "  - grpo_policy+none vs grpo_policy+bc_surrogate"
echo "  - bc_policy+none vs bc_policy+bc_surrogate"
echo "=========================================="