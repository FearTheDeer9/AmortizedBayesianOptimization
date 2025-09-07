#!/bin/bash
# Test oracle surrogate evaluation

echo "Testing Oracle Surrogate Evaluation"
echo "===================================="

# Test 1: Policy with oracle surrogate on chain graphs
echo -e "\n1. Testing policy with ORACLE surrogate (perfect parent predictions)..."
python experiments/evaluation/stepwise/full_evaluation.py \
  --policy-path imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl \
  --surrogate-path checkpoints/avici_runs/avici_style_20250903_154909/best_model.pkl \
  --oracle-surrogate \
  --structures chain \
  --num-episodes 5 \
  --num-interventions 30 \
  --initial-observations 20 \
  --initial-interventions 10 \
  --num-vars 5 \
  --output-dir evaluation_results_oracle_test \
  --baselines

echo -e "\nTest completed! Check evaluation_results_oracle_test/ for results."