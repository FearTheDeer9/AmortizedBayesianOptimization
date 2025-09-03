#!/bin/bash

# Test script for VM checkpoints with debug output

echo "Testing VM checkpoints with enhanced debug output"
echo "=================================================="

# Set JAX to suppress compilation messages
export JAX_LOG_LEVEL=WARNING

# Test the infogain checkpoint with latest phase file
echo ""
echo "Testing vm_scalefree_infogain_12hr checkpoint (phase 31)..."
echo "-------------------------------------------------"
poetry run python experiments/evaluation/multi_scm_evaluation.py \
    --policy imperial-vm-checkpoints/checkpoints/vm_scalefree_infogain_12hr/policy_phase_31.pkl \
    --n-scms 2 \
    --structure-types fork chain \
    --num-variables-options 3 4 \
    --debug

echo ""
echo "Testing vm_scalefree_target_12hr checkpoint (latest phase)..."
echo "-----------------------------------------------"
# Find the latest phase for target checkpoint
LATEST_TARGET=$(ls imperial-vm-checkpoints/checkpoints/vm_scalefree_target_12hr/policy_phase_*.pkl | sort -V | tail -1)
echo "Using checkpoint: $LATEST_TARGET"
poetry run python experiments/evaluation/multi_scm_evaluation.py \
    --policy "$LATEST_TARGET" \
    --n-scms 2 \
    --structure-types fork chain \
    --num-variables-options 3 4 \
    --debug

echo ""
echo "Test complete!"