#!/bin/bash
# Full evaluation with 12 policy-surrogate pairs

OUTPUT_DIR="evaluation_results/full_12_pairs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Running full evaluation with 12 policy-surrogate pairs..."
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
poetry run python scripts/evaluate_acbo_methods.py \
    --register_policy bc checkpoints/validation/bc_final \
    --register_policy grpo checkpoints/validation/grpo_final \
    --register_surrogate bc_static checkpoints/validation/bc_surrogate_final \
    --register_surrogate bc_active checkpoints/validation/bc_surrogate_final \
    --register_surrogate untrained active_learning \
    --evaluate_pairs random bc_static \
    --evaluate_pairs random bc_active \
    --evaluate_pairs random untrained \
    --evaluate_pairs oracle bc_static \
    --evaluate_pairs oracle bc_active \
    --evaluate_pairs oracle untrained \
    --evaluate_pairs bc bc_static \
    --evaluate_pairs bc bc_active \
    --evaluate_pairs bc untrained \
    --evaluate_pairs grpo bc_static \
    --evaluate_pairs grpo bc_active \
    --evaluate_pairs grpo untrained \
    --n_scms 10 \
    --n_interventions 20 \
    --n_obs 100 \
    --n_samples 50 \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Evaluation complete! Results saved to $OUTPUT_DIR"