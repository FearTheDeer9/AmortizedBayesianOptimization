#!/bin/bash
# Complete ACBO training and evaluation pipeline

echo "======================================"
echo "Complete ACBO Pipeline"
echo "======================================"
echo ""

# Step 1: Train Surrogate
echo "Step 1/4: Training Surrogate Model"
echo "-----------------------------------"
bash scripts/examples/train_surrogate.sh
echo ""

# Step 2: Train BC Policy
echo "Step 2/4: Training BC Policy"
echo "-----------------------------------"
bash scripts/examples/train_bc_policy.sh
echo ""

# Step 3: Train GRPO Policy
echo "Step 3/4: Training GRPO Policy"
echo "-----------------------------------"
bash scripts/examples/train_grpo.sh
echo ""

# Step 4: Evaluate All Methods
echo "Step 4/4: Evaluating All Methods"
echo "-----------------------------------"
bash scripts/examples/evaluate_comprehensive.sh
echo ""

echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Checkpoints saved to: checkpoints/"
echo "Results saved to: evaluation_results/"