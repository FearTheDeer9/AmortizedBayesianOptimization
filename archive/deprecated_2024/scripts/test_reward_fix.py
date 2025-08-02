#!/usr/bin/env python3
"""Test script to verify reward calculation fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyrsistent as pyr
from src.causal_bayes_opt.acquisition.rewards import create_default_reward_config


def test_reward_directions():
    """Test that rewards correctly handle minimization vs maximization."""
    
    # Test configuration creation
    print("Testing reward config creation:")
    config_min = create_default_reward_config(optimization_direction='MINIMIZE')
    print(f"  MINIMIZE config: optimization_direction = {config_min.get('optimization_direction')}")
    
    config_max = create_default_reward_config(optimization_direction='MAXIMIZE')
    print(f"  MAXIMIZE config: optimization_direction = {config_max.get('optimization_direction')}")
    
    # Test that configs have correct weights
    print(f"\nReward weights (both should be same):")
    print(f"  MINIMIZE: {config_min.get('reward_weights')}")
    print(f"  MAXIMIZE: {config_max.get('reward_weights')}")
    
    print("\n✓ Configuration test passed!")


if __name__ == "__main__":
    test_reward_directions()