#!/bin/bash
# Test structure learning evaluation

echo "Testing Structure Learning Evaluation"
echo "======================================"

# Test structure learning comparison
echo -e "\nComparing intervention strategies for structure discovery..."
python experiments/evaluation/stepwise/structure_learning_evaluation.py \
  --policy-path imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl \
  --surrogate-path checkpoints/avici_runs/avici_style_20250903_154909/best_model.pkl \
  --structures chain \
  --num-vars 5 \
  --num-episodes 5 \
  --num-interventions 30 \
  --output-dir structure_learning_test

echo -e "\nTest completed! Check structure_learning_test/ for results."