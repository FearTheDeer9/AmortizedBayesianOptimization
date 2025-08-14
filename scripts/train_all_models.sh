#!/bin/bash
# Train all ACBO models with the new data format fixes
# Run with: bash scripts/train_all_models.sh

echo "=============================================="
echo "Training ALL ACBO Models"
echo "Starting at: $(date)"
echo "=============================================="

# Ensure we're using the correct Python environment
# Try to detect and use poetry if available
if command -v poetry &> /dev/null; then
    echo "Using poetry environment"
    PYTHON_CMD="poetry run python"
else
    echo "Poetry not found, using system python"
    PYTHON_CMD="python"
fi

# Alternatively, if you have a specific conda/venv environment, activate it:
# source /path/to/venv/bin/activate
# or
# conda activate your_env_name

# Create checkpoint directory if it doesn't exist
mkdir -p checkpoints

# 1. Train BC Surrogate (Structure Learning)
echo ""
echo "=============================================="
echo "[1/4] Training BC Surrogate Model"
echo "=============================================="
$PYTHON_CMD scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 1000 \
    --encoder_type node_feature \
    --learning_rate 1e-3 \
    --batch_size 32

# 2. Train BC Policy (Acquisition)
echo ""
echo "=============================================="
echo "[2/4] Training BC Policy Model"
echo "=============================================="
$PYTHON_CMD scripts/train_acbo_methods.py \
    --method bc \
    --episodes 500 \
    --architecture alternating_attention \
    --learning_rate 3e-4 \
    --batch_size 32

# 3. Train GRPO without Surrogate
echo ""
echo "=============================================="
echo "[3/4] Training GRPO (No Surrogate)"
echo "=============================================="
$PYTHON_CMD scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1000 \
    --architecture alternating_attention \
    --learning_rate 3e-4 \
    --batch_size 32

# 4. Train GRPO with Surrogate (uses the surrogate we just trained)
echo ""
echo "=============================================="
echo "[4/4] Training GRPO with Surrogate"
echo "=============================================="
$PYTHON_CMD scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1000 \
    --architecture alternating_attention \
    --use_surrogate \
    --surrogate_checkpoint checkpoints/bc_surrogate_final \
    --learning_rate 3e-4 \
    --batch_size 32

echo ""
echo "=============================================="
echo "All Training Complete!"
echo "Finished at: $(date)"
echo "=============================================="
echo ""
echo "Checkpoints saved:"
echo "  - BC Surrogate: checkpoints/bc_surrogate_final"
echo "  - BC Policy: checkpoints/bc_final"
echo "  - GRPO (no surrogate): checkpoints/grpo_no_surrogate_final"
echo "  - GRPO (with surrogate): checkpoints/unified_grpo_final"
echo ""
echo "Next step: Run evaluation with:"
echo "  $PYTHON_CMD scripts/evaluate_acbo_methods_v2.py --include_baselines ..."