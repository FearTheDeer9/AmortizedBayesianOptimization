#!/bin/bash

# Comprehensive ACBO Training Script
# This script trains all ACBO models with longer episodes for better convergence

echo "=========================================="
echo "COMPREHENSIVE ACBO TRAINING"
echo "=========================================="
echo ""

# Create checkpoint directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_BASE="checkpoints/long_training_${TIMESTAMP}"
mkdir -p $CHECKPOINT_BASE

# 1. Train GRPO with surrogate (joint training)
echo "1. Training GRPO with surrogate (5000 episodes)..."
echo "This trains both policy and surrogate jointly"
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 5000 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --use_surrogate \
    --surrogate_lr 5e-4 \
    --surrogate_layers 6 \
    --surrogate_hidden_dim 256 \
    --scm_type mixed \
    --min_vars 3 \
    --max_vars 8 \
    --checkpoint_dir "${CHECKPOINT_BASE}/grpo_with_surrogate" \
    --seed 42

echo ""
echo "2. Training BC Policy from Oracle demonstrations (2000 episodes)..."
echo "This trains a policy to mimic the oracle"
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type policy \
    --episodes 2000 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --hidden_dim 512 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --scm_type mixed \
    --min_vars 3 \
    --max_vars 8 \
    --checkpoint_dir "${CHECKPOINT_BASE}/bc_policy" \
    --seed 43

echo ""
echo "3. Training BC Surrogate from demonstrations (2000 episodes)..."
echo "This trains a surrogate for structure learning"
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type surrogate \
    --episodes 2000 \
    --batch_size 64 \
    --surrogate_lr 1e-4 \
    --surrogate_layers 6 \
    --surrogate_hidden_dim 256 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --scm_type mixed \
    --min_vars 3 \
    --max_vars 8 \
    --checkpoint_dir "${CHECKPOINT_BASE}/bc_surrogate" \
    --seed 44

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "Checkpoints saved to: $CHECKPOINT_BASE"
echo ""
echo "To evaluate these models, run:"
echo "poetry run python scripts/evaluate_acbo_methods.py \\"
echo "    --grpo ${CHECKPOINT_BASE}/grpo_with_surrogate/checkpoint_final.pkl \\"
echo "    --bc ${CHECKPOINT_BASE}/bc_policy/checkpoint_final.pkl \\"
echo "    --n_scms 20 \\"
echo "    --n_interventions 30 \\"
echo "    --use_active_learning \\"
echo "    --output_dir evaluation_results/long_training_${TIMESTAMP}"
echo "=========================================="