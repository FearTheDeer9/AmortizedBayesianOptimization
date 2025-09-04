#!/bin/bash
# Run convergence tests on fixed graph structures

echo "========================================="
echo "Fixed Graph Convergence Test"
echo "========================================="
echo ""
echo "This script will:"
echo "1. Create 5 fixed graph types with zero noise"
echo "2. Train GRPO policies on each graph"
echo "3. Generate convergence visualizations"
echo ""

# Change to project root
cd ../..

# Optional: Run with fewer interventions for quick test
INTERVENTIONS=${1:-50}  # Default 50, can override with argument

echo "Running with $INTERVENTIONS interventions per graph"
echo ""

# Create output directory
mkdir -p thesis_results/fixed_graph_convergence

# Run the test (without training first to just test setup)
echo "Testing setup (no training)..."
python experiments/policy-only-training/test_fixed_graph_convergence.py \
    --interventions $INTERVENTIONS \
    --output-dir thesis_results/fixed_graph_convergence \
    --skip-training

echo ""
echo "========================================="
echo "Setup test complete!"
echo "========================================="
echo ""
echo "To run full training on all 5 graph types:"
echo "  python experiments/policy-only-training/test_fixed_graph_convergence.py --interventions 100"
echo ""
echo "To run training on a single graph type:"
echo "  python experiments/policy-only-training/train_grpo_single_scm_with_surrogate.py --scm-type fixed_fork --interventions 100"
echo ""
echo "Available fixed graph types:"
echo "  - fixed_fork: Multiple causes → one effect"
echo "  - fixed_true_fork: One cause → multiple effects"
echo "  - fixed_chain: Sequential chain"
echo "  - fixed_scale_free: Hub-based network"
echo "  - fixed_random: Mixed positive/negative coefficients"