#!/bin/bash
# Train a GRPO policy with optional surrogate

echo "Training GRPO Policy"
echo "===================="

# Check if surrogate checkpoint exists
SURROGATE_CHECKPOINT="checkpoints/bc_surrogate_final"

if [ -f "$SURROGATE_CHECKPOINT" ]; then
    echo "Found pre-trained surrogate at $SURROGATE_CHECKPOINT"
    echo "Training GRPO with structure learning enabled"
    
    python scripts/main/train.py \
      --method grpo \
      --episodes 1000 \
      --scm_type mixed \
      --min_vars 3 \
      --max_vars 8 \
      --use_surrogate \
      --surrogate_checkpoint $SURROGATE_CHECKPOINT \
      --hidden_dim 256 \
      --learning_rate 3e-4 \
      --batch_size 32 \
      --seed 42
else
    echo "No pre-trained surrogate found"
    echo "Training GRPO without structure learning"
    
    python scripts/main/train.py \
      --method grpo \
      --episodes 1000 \
      --scm_type mixed \
      --min_vars 3 \
      --max_vars 8 \
      --hidden_dim 256 \
      --learning_rate 3e-4 \
      --batch_size 32 \
      --seed 42
fi

echo "Training complete! Checkpoint saved to checkpoints/unified_grpo_final"