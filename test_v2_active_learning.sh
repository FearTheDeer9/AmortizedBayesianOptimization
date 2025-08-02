#!/bin/bash
# Test script for v2 evaluation with active learning

set -e

echo "Testing evaluate_acbo_methods_v2.py with active learning..."

# Test 1: No updates (baseline)
echo -e "\n=== Test 1: Random policy with dummy surrogate + no updates ==="
poetry run python scripts/evaluate_acbo_methods_v2.py \
    --register_surrogate dummy dummy \
    --evaluate_pairs random dummy \
    --surrogate_update_strategy none \
    --n_scms 2 \
    --n_interventions 5 \
    --output_dir test_v2_active_results/dummy_none

# Test 2: BC surrogate with BIC updates (if checkpoint exists)
BC_CHECKPOINT="checkpoints/bc_surrogate_final/checkpoint.pkl"
if [ -f "$BC_CHECKPOINT" ]; then
    echo -e "\n=== Test 2: Random policy with BC surrogate + BIC updates ==="
    poetry run python scripts/evaluate_acbo_methods_v2.py \
        --register_surrogate bc "$BC_CHECKPOINT" \
        --evaluate_pairs random bc \
        --surrogate_update_strategy bic \
        --n_scms 2 \
        --n_interventions 5 \
        --output_dir test_v2_active_results/bc_bic
else
    echo -e "\n=== Skipping Test 2: BC checkpoint not found at $BC_CHECKPOINT ==="
fi

echo -e "\n=== All tests completed successfully! ==="
echo "Results saved to test_v2_active_results/"