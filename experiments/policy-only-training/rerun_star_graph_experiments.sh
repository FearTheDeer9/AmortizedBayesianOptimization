#!/bin/bash

# Script to re-run star graph experiments for non-converged and missing configurations
# Created based on analysis of existing results

echo "================================================"
echo "Star Graph Experiment Re-run Script"
echo "================================================"

# Navigate to project root
cd /Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt-worktree

echo ""
echo "Step 1: Cleaning up non-noise configurations for re-run..."
echo "--------------------------------------------------------"

# Delete missing 3-node configs (to ensure clean run)
echo "Removing missing 3-node configurations..."
rm -f thesis_results/star_graph_convergence/raw_data/star_3nodes_L1.0*.json
rm -f thesis_results/star_graph_convergence/raw_data/star_3nodes_L2.0*.json
rm -f thesis_results/star_graph_convergence/raw_data/star_3nodes_L5.0*.json

# Delete ALL 15 and 20 node results for re-run with new setup
echo "Removing all 15-node configurations (for re-run with new setup)..."
rm -f thesis_results/star_graph_convergence/raw_data/star_15nodes_*.json

echo "Removing all 20-node configurations (for re-run with new setup)..."
rm -f thesis_results/star_graph_convergence/raw_data/star_20nodes_*.json

echo ""
echo "Step 2: Re-running non-noise experiments..."
echo "--------------------------------------------------------"
echo "This will run:"
echo "  - 3 missing 3-node configurations"
echo "  - 5 configurations for 15 nodes"
echo "  - 5 configurations for 20 nodes"
echo "  Total: 13 configurations"
echo ""

python experiments/policy-only-training/test_star_graph_convergence_benchmark.py --workers 4

echo ""
echo "Step 3: Cleaning up non-converged noise=0.5 configurations..."
echo "--------------------------------------------------------"
echo "Removing non-converged noise configurations..."

# Delete non-converged noise=0.5 configurations
rm -f thesis_results/star_graph_convergence_noise_0.5/raw_data/star_10nodes_L2.0*.json
rm -f thesis_results/star_graph_convergence_noise_0.5/raw_data/star_10nodes_L3.0*.json
rm -f thesis_results/star_graph_convergence_noise_0.5/raw_data/star_15nodes_L3.0*.json
rm -f thesis_results/star_graph_convergence_noise_0.5/raw_data/star_20nodes_L2.0*.json
rm -f thesis_results/star_graph_convergence_noise_0.5/raw_data/star_20nodes_L5.0*.json

echo ""
echo "Step 4: Re-running noise=0.5 experiments..."
echo "--------------------------------------------------------"
echo "This will run 5 non-converged configurations"
echo ""

python experiments/policy-only-training/test_star_graph_convergence_benchmark.py --noise-std 0.5 --workers 4

echo ""
echo "================================================"
echo "Re-run complete!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  - thesis_results/star_graph_convergence/ (non-noise)"
echo "  - thesis_results/star_graph_convergence_noise_0.5/ (with noise)"