#!/bin/bash

# Focused ACBO Training Script
# More reasonable training times while still achieving good performance

echo "=========================================="
echo "FOCUSED ACBO TRAINING"
echo "=========================================="
echo ""

# Create checkpoint directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_BASE="checkpoints/focused_${TIMESTAMP}"
mkdir -p $CHECKPOINT_BASE

# 1. Train GRPO with surrogate (focused)
echo "1. Training GRPO with surrogate (1500 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1500 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --use_surrogate \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --scm_type mixed \
    --checkpoint_dir "${CHECKPOINT_BASE}/grpo" \
    --seed 42

echo ""
echo "2. Training BC Policy (1000 episodes)..."
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type policy \
    --episodes 1000 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "${CHECKPOINT_BASE}/bc" \
    --seed 43

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "Checkpoints saved to: $CHECKPOINT_BASE"
echo ""
echo "Quick evaluation command:"
echo "poetry run python scripts/evaluate_acbo_methods.py \\"
echo "    --grpo ${CHECKPOINT_BASE}/grpo/checkpoint_final.pkl \\"
echo "    --bc ${CHECKPOINT_BASE}/bc/checkpoint_final.pkl \\"
echo "    --n_scms 10 --n_interventions 20 --use_active_learning"
echo "=========================================="