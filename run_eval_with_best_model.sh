#!/bin/bash
# Run evaluation with best_model.pkl instead of checkpoint_step_1000.pkl

echo "Running evaluation with BEST surrogate model..."

python experiments/evaluation/stepwise/full_evaluation.py \
  --policy-path imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl \
  --surrogate-path checkpoints/avici_runs/avici_style_20250903_154909/best_model.pkl \
  --structures chain \
  --num-episodes 10 \
  --num-interventions 30 \
  --initial-observations 20 \
  --initial-interventions 10 \
  --num-vars 5 \
  --output-dir evaluation_results_chain_best_surrogate \
  --baselines