#!/bin/bash
# Quick test to ensure core functionality works after archiving

set -e

echo "Testing core functionality after archiving..."
echo ""

# Test 1: Train a small BC surrogate
echo "1. Testing BC surrogate training..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 2 \
    --batch_size 32 \
    --checkpoint_dir "test_post_archive" \
    --max_demos 2 \
    --seed 200 > /dev/null 2>&1
echo "✓ BC surrogate training works"

# Test 2: Test evaluation script (renamed)
echo "2. Testing evaluation script..."
poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "test_post_archive/bc_surrogate_final/checkpoint.pkl" \
    --evaluate_pairs random dummy \
    --n_scms 1 \
    --n_interventions 2 \
    --output_dir "test_post_archive/eval" > /dev/null 2>&1
echo "✓ Evaluation script works"

# Clean up
rm -rf test_post_archive

echo ""
echo "All tests passed! Core functionality preserved after archiving."