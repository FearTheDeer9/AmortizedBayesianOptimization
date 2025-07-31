#!/bin/bash
# Test active learning surrogates

OUTPUT_DIR="evaluation_results/active_learning_test"
mkdir -p $OUTPUT_DIR

echo "Testing active learning evaluation..."

poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc_static checkpoints/validation/bc_surrogate_final \
    --register_surrogate bc_active checkpoints/validation/bc_surrogate_final \
    --register_surrogate untrained active_learning \
    --evaluate_pairs random bc_static \
    --evaluate_pairs random bc_active \
    --evaluate_pairs random untrained \
    --n_scms 1 \
    --n_interventions 10 \
    --n_obs 50 \
    --n_samples 50 \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Evaluation complete! Results in $OUTPUT_DIR"