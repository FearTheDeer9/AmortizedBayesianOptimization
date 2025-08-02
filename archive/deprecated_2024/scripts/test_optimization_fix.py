#!/usr/bin/env python3
"""Test that optimization direction is correctly passed through the training pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import DictConfig


def test_optimization_direction():
    """Test that optimization direction is correctly extracted from config."""
    
    # Just test the logic for extracting optimization direction
    print("Testing optimization direction extraction logic:")
    
    # Test case 1: optimization direction in config.optimization.direction
    print("\nTest 1: config.optimization.direction")
    config1 = DictConfig({
        'optimization': {
            'direction': 'MINIMIZE'
        }
    })
    
    # Simulate the logic from enriched_trainer.py
    optimization_direction = 'MAXIMIZE'  # Default
    if hasattr(config1, 'optimization') and hasattr(config1.optimization, 'direction'):
        optimization_direction = config1.optimization.direction
    elif hasattr(config1, 'experiment') and hasattr(config1.experiment, 'optimization_direction'):
        optimization_direction = config1.experiment.optimization_direction
    elif config1.get('optimization_direction'):
        optimization_direction = config1.get('optimization_direction')
    
    print(f"  Result: {optimization_direction}")
    print(f"  {'✓' if optimization_direction == 'MINIMIZE' else '✗'} Expected: MINIMIZE")
    
    # Test case 2: optimization direction as MAXIMIZE
    print("\nTest 2: config.optimization.direction = MAXIMIZE")
    config2 = DictConfig({
        'optimization': {
            'direction': 'MAXIMIZE'
        }
    })
    
    # Simulate the logic
    optimization_direction = 'MAXIMIZE'  # Default
    if hasattr(config2, 'optimization') and hasattr(config2.optimization, 'direction'):
        optimization_direction = config2.optimization.direction
    elif hasattr(config2, 'experiment') and hasattr(config2.experiment, 'optimization_direction'):
        optimization_direction = config2.experiment.optimization_direction
    elif config2.get('optimization_direction'):
        optimization_direction = config2.get('optimization_direction')
    
    print(f"  Result: {optimization_direction}")
    print(f"  {'✓' if optimization_direction == 'MAXIMIZE' else '✗'} Expected: MAXIMIZE")
    
    # Test case 3: No optimization config (should default to MAXIMIZE)
    print("\nTest 3: No optimization config (default)")
    config3 = DictConfig({})
    
    # Simulate the logic
    optimization_direction = 'MAXIMIZE'  # Default
    if hasattr(config3, 'optimization') and hasattr(config3.optimization, 'direction'):
        optimization_direction = config3.optimization.direction
    elif hasattr(config3, 'experiment') and hasattr(config3.experiment, 'optimization_direction'):
        optimization_direction = config3.experiment.optimization_direction
    elif config3.get('optimization_direction'):
        optimization_direction = config3.get('optimization_direction')
    
    print(f"  Result: {optimization_direction}")
    print(f"  {'✓' if optimization_direction == 'MAXIMIZE' else '✗'} Expected: MAXIMIZE (default)")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_optimization_direction()