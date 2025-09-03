#!/bin/bash

# Clean test script showing only channel inputs/outputs

echo "Testing channel inputs/outputs for VM checkpoint"
echo "================================================="

# Test with minimal SCMs and show only first 3 steps
poetry run python experiments/evaluation/multi_scm_evaluation.py \
    --policy imperial-vm-checkpoints/checkpoints/vm_scalefree_infogain_12hr/policy_phase_31.pkl \
    --n-scms 1 \
    --structure-types fork \
    --num-variables-options 4 \
    --debug 2>&1 | grep -v "WARNING:" | grep -v "INFO:" | grep -v "src.causal_bayes_opt"

echo ""
echo "Test complete!"