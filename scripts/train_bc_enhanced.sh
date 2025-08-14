#!/bin/bash
# Train BC with permutation and label smoothing

python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 10 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --max_demos 100 \
    --checkpoint_dir checkpoints_enhanced \
    --use_permutation \
    --label_smoothing 0.1 \
    --permutation_seed 42