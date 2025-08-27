#!/bin/bash

# Test the joint training orchestrator with a quick test configuration

echo "Testing Joint Training Orchestrator"
echo "==================================="
echo ""
echo "This will run a quick 2-minute test with 1 minute per phase"
echo ""

# Run orchestrator with quick test config
python experiments/joint-training/train_joint_orchestrator.py \
    --config experiments/joint-training/configs/quick_test_orchestrator.json

echo ""
echo "Test complete. Check experiments/joint-training/checkpoints/quick_test/ for results."