#!/bin/bash
# Quick test with 2 SCMs, 5 interventions

OUTPUT_DIR="evaluation_results/test_12_pairs"
mkdir -p $OUTPUT_DIR

echo "Testing 12-pair evaluation setup..."

poetry run python scripts/evaluate_acbo_methods.py \
    --register_policy bc checkpoints/validation/bc_final \
    --register_policy grpo checkpoints/validation/grpo_final \
    --register_surrogate bc_static checkpoints/validation/bc_surrogate_final \
    --register_surrogate bc_active checkpoints/validation/bc_surrogate_final \
    --register_surrogate untrained active_learning \
    --evaluate_pairs random bc_static \
    --evaluate_pairs oracle bc_static \
    --evaluate_pairs bc bc_static \
    --evaluate_pairs grpo bc_static \
    --n_scms 2 \
    --n_interventions 5 \
    --n_obs 50 \
    --n_samples 20 \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Test complete!"