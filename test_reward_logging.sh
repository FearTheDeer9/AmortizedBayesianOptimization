#!/bin/bash
# Test script to verify reward computation logging

set -e

echo "Testing reward computation logging..."
echo ""

# Create test checkpoint directory
TEST_DIR="checkpoints/test_reward_logging_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "1. Training small BC surrogate..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 2 \
    --batch_size 32 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 2 \
    --seed 42 2>&1 | grep -v "INFO:src" | grep -v "DEBUG"

echo ""
echo "2. Training GRPO WITH surrogate (should show info gain rewards)..."
echo "========================================================"
poetry run python scripts/train_acbo_methods.py \
    --method grpo_with_surrogate \
    --episodes 1 \
    --batch_size 16 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 2 \
    --seed 44 \
    --n_interventions 3 2>&1 | grep -E "\[SURROGATE\]|\[REWARD\]|Episode 0|Step [0-9]|Final reward" || true

echo ""
echo "========================================================"
echo ""
echo "3. Training GRPO WITHOUT surrogate (should show no info gain)..."
echo "========================================================"
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1 \
    --batch_size 16 \
    --checkpoint_dir "$TEST_DIR" \
    --seed 45 \
    --n_interventions 3 2>&1 | grep -E "\[SURROGATE\]|\[REWARD\]|Episode 0|Step [0-9]|Final reward" || true

echo ""
echo "========================================================"
echo "Test complete. Check output above for:"
echo "  - [SURROGATE] logs showing posterior updates with surrogate"
echo "  - [REWARD] logs showing info gain > 0 with surrogate"
echo "  - [REWARD] logs showing info gain = 0 without surrogate"