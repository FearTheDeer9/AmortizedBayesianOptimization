#!/bin/bash
# Comprehensive evaluation of trained models

echo "Comprehensive ACBO Evaluation"
echo "============================="

# Check which checkpoints exist
BC_POLICY="checkpoints/bc_final"
GRPO_POLICY="checkpoints/unified_grpo_final"
SURROGATE="checkpoints/bc_surrogate_final"

EVAL_ARGS="--include_baselines --n_scms 10 --n_interventions 20 --n_samples 10 --plot --plot_trajectories"

# Register surrogate if it exists
if [ -f "$SURROGATE" ]; then
    echo "Found surrogate checkpoint: $SURROGATE"
    EVAL_ARGS="$EVAL_ARGS --register_surrogate trained $SURROGATE --baseline_surrogate trained"
else
    echo "No surrogate checkpoint found, using dummy surrogate"
    EVAL_ARGS="$EVAL_ARGS --baseline_surrogate dummy"
fi

# Register and evaluate BC policy if it exists
if [ -f "$BC_POLICY" ]; then
    echo "Found BC policy checkpoint: $BC_POLICY"
    EVAL_ARGS="$EVAL_ARGS --register_policy bc $BC_POLICY"
    if [ -f "$SURROGATE" ]; then
        EVAL_ARGS="$EVAL_ARGS --evaluate_pairs bc trained"
    fi
fi

# Register and evaluate GRPO policy if it exists
if [ -f "$GRPO_POLICY" ]; then
    echo "Found GRPO policy checkpoint: $GRPO_POLICY"
    EVAL_ARGS="$EVAL_ARGS --register_policy grpo $GRPO_POLICY"
    if [ -f "$SURROGATE" ]; then
        EVAL_ARGS="$EVAL_ARGS --evaluate_pairs grpo trained"
    fi
fi

echo "Running evaluation with arguments:"
echo "python scripts/main/evaluate.py $EVAL_ARGS"
echo ""

python scripts/main/evaluate.py $EVAL_ARGS

echo ""
echo "Evaluation complete! Results saved to evaluation_results/"