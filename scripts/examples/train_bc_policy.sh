#!/bin/bash
# Train a BC policy from expert demonstrations

echo "Training BC Policy Model"
echo "========================"

python scripts/main/train.py \
  --method bc \
  --episodes 100 \
  --demo_path expert_demonstrations/raw/raw_demonstrations \
  --max_demos 50 \
  --architecture alternating_attention \
  --use_permutation \
  --label_smoothing 0.1 \
  --hidden_dim 256 \
  --learning_rate 3e-4 \
  --batch_size 32 \
  --seed 42

echo "Training complete! Checkpoint saved to checkpoints/bc_final"