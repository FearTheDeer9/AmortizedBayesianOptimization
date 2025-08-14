#!/bin/bash
# Train a BC surrogate model for structure learning

echo "Training BC Surrogate Model"
echo "==========================="

python scripts/main/train.py \
  --method surrogate \
  --episodes 100 \
  --demo_path expert_demonstrations/raw/raw_demonstrations \
  --max_demos 50 \
  --encoder_type node_feature \
  --surrogate_hidden_dim 128 \
  --surrogate_layers 4 \
  --surrogate_heads 8 \
  --surrogate_lr 1e-3 \
  --batch_size 32 \
  --seed 42

echo "Training complete! Checkpoint saved to checkpoints/bc_surrogate_final"