#!/bin/bash
# Comprehensive test script to verify functionality before cleanup
# This ensures we don't break anything during the cleanup process

set -e  # Exit on error

echo "=========================================="
echo "PRE-CLEANUP FUNCTIONALITY TEST"
echo "=========================================="
echo ""
echo "This script tests the current pipeline to establish a baseline"
echo "before making any cleanup changes."
echo ""

# Create test directory
TEST_DIR="test_results/pre_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Test results will be saved to: $TEST_DIR"
echo ""

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 passed"
    else
        echo "✗ $1 failed"
        exit 1
    fi
}

# Test 1: Train BC Surrogate
echo "1. Testing BC Surrogate Training..."
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 5 \
    --batch_size 32 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 100 > "$TEST_DIR/train_surrogate.log" 2>&1
check_status "BC Surrogate Training"

# Test 2: Train BC Policy
echo "2. Testing BC Policy Training..."
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 5 \
    --batch_size 32 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 101 > "$TEST_DIR/train_bc.log" 2>&1
check_status "BC Policy Training"

# Test 3: Train GRPO with Surrogate
echo "3. Testing GRPO with Surrogate Training..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo_with_surrogate \
    --episodes 5 \
    --batch_size 16 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$TEST_DIR" \
    --max_demos 5 \
    --seed 102 > "$TEST_DIR/train_grpo_with_surrogate.log" 2>&1
check_status "GRPO with Surrogate Training"

# Test 4: Train GRPO without Surrogate
echo "4. Testing GRPO without Surrogate Training..."
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 5 \
    --batch_size 16 \
    --checkpoint_dir "$TEST_DIR" \
    --seed 103 > "$TEST_DIR/train_grpo_no_surrogate.log" 2>&1
check_status "GRPO without Surrogate Training"

# Test 5: Evaluate with v2 script (no active learning)
echo "5. Testing Evaluation (no active learning)..."
poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$TEST_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo "$TEST_DIR/unified_grpo_final/checkpoint.pkl" \
    --register_policy grpo_no_surrogate "$TEST_DIR/grpo_no_surrogate_final/checkpoint.pkl" \
    --register_policy bc "$TEST_DIR/bc_final/checkpoint.pkl" \
    --evaluate_pairs grpo bc \
    --evaluate_pairs grpo_no_surrogate bc \
    --evaluate_pairs bc bc \
    --n_scms 2 \
    --n_interventions 5 \
    --surrogate_update_strategy none \
    --output_dir "$TEST_DIR/eval_no_active" > "$TEST_DIR/eval_no_active.log" 2>&1
check_status "Evaluation without Active Learning"

# Test 6: Evaluate with v2 script (with active learning)
echo "6. Testing Evaluation (with active learning)..."
poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$TEST_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo "$TEST_DIR/unified_grpo_final/checkpoint.pkl" \
    --evaluate_pairs grpo bc \
    --n_scms 2 \
    --n_interventions 5 \
    --surrogate_update_strategy bic \
    --output_dir "$TEST_DIR/eval_with_active" > "$TEST_DIR/eval_with_active.log" 2>&1
check_status "Evaluation with Active Learning"

# Test 7: Check that checkpoints can be loaded
echo "7. Testing Checkpoint Loading..."
poetry run python -c "
import sys
sys.path.append('.')
from pathlib import Path
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint

# Test loading each checkpoint
checkpoints = [
    '$TEST_DIR/bc_surrogate_final/checkpoint.pkl',
    '$TEST_DIR/bc_final/checkpoint.pkl',
    '$TEST_DIR/unified_grpo_final/checkpoint.pkl',
    '$TEST_DIR/grpo_no_surrogate_final/checkpoint.pkl'
]

for cp_path in checkpoints:
    try:
        cp = load_checkpoint(Path(cp_path))
        print(f'✓ Loaded {cp_path}: {cp[\"model_type\"]} ({cp[\"model_subtype\"]})')
    except Exception as e:
        print(f'✗ Failed to load {cp_path}: {e}')
        sys.exit(1)
"
check_status "Checkpoint Loading"

# Test 8: Test comprehensive training script
echo "8. Testing Comprehensive Training Script..."
if [ -f "train_comprehensive_acbo.sh" ]; then
    # Just test that it starts without errors (kill after 10 seconds)
    timeout 10 ./train_comprehensive_acbo.sh > "$TEST_DIR/train_comprehensive.log" 2>&1 || true
    if grep -q "Training BC Surrogate" "$TEST_DIR/train_comprehensive.log"; then
        echo "✓ Comprehensive Training Script passed"
    else
        echo "✗ Comprehensive Training Script failed"
        exit 1
    fi
else
    echo "⚠ Comprehensive training script not found, skipping"
fi

# Create summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "All tests passed! Current functionality verified."
echo ""
echo "Checkpoint Summary:"
ls -la "$TEST_DIR"/*_final/checkpoint.pkl 2>/dev/null | awk '{print "  - " $9 ": " $5 " bytes"}'
echo ""
echo "Next steps:"
echo "1. Create a git branch for cleanup work"
echo "2. Run Phase 2: Archive deprecated code"
echo ""

# Save test configuration for comparison after cleanup
cat > "$TEST_DIR/test_config.json" << EOF
{
  "test_date": "$(date)",
  "python_version": "$(poetry run python --version)",
  "test_episodes": 5,
  "test_scms": 2,
  "test_interventions": 5,
  "models_trained": [
    "bc_surrogate",
    "bc_policy", 
    "grpo_with_surrogate",
    "grpo_without_surrogate"
  ],
  "evaluation_modes": [
    "no_active_learning",
    "with_active_learning"
  ]
}
EOF

echo "Test configuration saved to: $TEST_DIR/test_config.json"