#!/bin/bash
# Run training progression evaluation with specified parameters

echo "=============================================="
echo "EVALUATING SURROGATE TRAINING PROGRESSION"
echo "=============================================="

# Default configuration
POLICY_PATH="checkpoints/grpo_single_scm/single_scm_scale_free_30vars_20250903_212403/final_policy.pkl"
SURROGATE_DIR="local_simple_10hr/fork_and_chain"
OUTPUT_DIR="evaluation_results/training_progression"

# Evaluation parameters
OBSERVATIONS=1000
INTERVENTIONS=500
MIN_COEFFICIENT=0.5
NUM_EPISODES=10

# Surrogate phases to test (equally spaced: 1, 11, 21, 31)
PHASES="1 11 21 31"

# Graph sizes to test
GRAPH_SIZES="3 8 15 30 50 100"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the evaluation
echo "Configuration:"
echo "  Policy: $POLICY_PATH"
echo "  Surrogate directory: $SURROGATE_DIR"
echo "  Phases: $PHASES"
echo "  Graph sizes: $GRAPH_SIZES"
echo "  Observations: $OBSERVATIONS"
echo "  Interventions: $INTERVENTIONS"
echo "  Min coefficient: $MIN_COEFFICIENT"
echo "  Episodes: $NUM_EPISODES"
echo ""

# Check if policy exists
if [ ! -f "$POLICY_PATH" ]; then
    echo "❌ Error: Policy checkpoint not found at $POLICY_PATH"
    echo "Available policies in grpo_single_scm:"
    ls -la checkpoints/grpo_single_scm/*/final_policy.pkl 2>/dev/null
    exit 1
fi

# Check if surrogate directory exists
if [ ! -d "$SURROGATE_DIR" ]; then
    echo "❌ Error: Surrogate directory not found at $SURROGATE_DIR"
    exit 1
fi

# Run the Python evaluation script
python experiments/evaluation/test_training_progression.py \
    --policy-path "$POLICY_PATH" \
    --surrogate-dir "$SURROGATE_DIR" \
    --observations $OBSERVATIONS \
    --interventions $INTERVENTIONS \
    --min-coefficient $MIN_COEFFICIENT \
    --graph-sizes $GRAPH_SIZES \
    --surrogate-phases $PHASES \
    --num-episodes $NUM_EPISODES \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    
    # Run analysis if results exist
    if ls $OUTPUT_DIR/combined_results_*.json 1> /dev/null 2>&1; then
        echo ""
        echo "Running analysis on results..."
        python experiments/evaluation/analyze_training_progression.py \
            --results-dir "$OUTPUT_DIR"
    fi
else
    echo ""
    echo "❌ Evaluation failed. Please check the error messages above."
    exit 1
fi