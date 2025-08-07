#!/bin/bash
# Run the FIXED enhanced trainer with better logging

python debugging-bc-training/enhanced_bc_trainer_fixed.py \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --hidden_dim 256 \
    --save_embeddings_every 5 \
    --output_dir debugging-bc-training/results/