#!/usr/bin/env python3
"""
Quick test of the experiment infrastructure.
Tests just a few small graphs to validate the approach works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from scripts.erdos_renyi_scaling_experiment import run_single_experiment, ScalingExperimentConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_experiment():
    """Test basic experiment functionality."""
    logger.info("Testing basic experiment functionality")
    
    # Create minimal config
    config = ScalingExperimentConfig(
        min_nodes=5,
        max_nodes=6,  # Just test 5-6 nodes
        edge_probability=0.3,
        n_intervention_steps=5,  # Reduced steps
        n_observational_samples=20,  # Reduced samples
        learning_rate=1e-3,
        n_runs_per_config=1,  # Single run
        base_random_seed=42
    )
    
    results = []
    
    # Test one experiment of each type
    for graph_size in [5, 6]:
        for method in ['static_surrogate', 'learning_surrogate']:
            try:
                logger.info(f"Testing {method} on {graph_size} nodes")
                result = run_single_experiment(graph_size, method, 0, config)
                results.append(result)
                logger.info(f"✅ Success: F1={result.final_f1_score:.3f}, time={result.runtime_seconds:.1f}s")
            except Exception as e:
                logger.error(f"❌ Failed {method} on {graph_size} nodes: {e}")
    
    # Print summary
    if results:
        logger.info(f"\n📊 Test Summary:")
        for result in results:
            logger.info(f"  {result.method} ({result.graph_size} nodes): F1={result.final_f1_score:.3f}, improvement={result.target_improvement:.3f}")
    
    return results

if __name__ == "__main__":
    test_basic_experiment()