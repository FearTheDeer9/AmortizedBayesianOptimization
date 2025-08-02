#!/usr/bin/env python3
"""
Inspect actual expert demonstration data to find the issue.
"""

import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp


def inspect_demo_file():
    """Inspect the structure of an expert demonstration file."""
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    
    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return
    
    print(f"Loading {demo_file}")
    
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nFile contains: {type(data)}")
    print(f"Attributes: {dir(data)}")
    
    # Check if it's a DemonstrationBatch
    if hasattr(data, 'demonstrations'):
        print(f"\nDemonstrations length: {len(data.demonstrations)}")
        demos = data.demonstrations
    elif isinstance(data, list):
        demos = data
    else:
        print("Unknown data format")
        return
    
    if len(demos) > 0:
        first_item = demos[0]
        print(f"\nFirst item type: {type(first_item)}")
        print(f"First item attributes: {dir(first_item)}")
        
        # Check specific attributes
        if hasattr(first_item, 'trajectory'):
            print(f"\nTrajectory type: {type(first_item.trajectory)}")
            print(f"Trajectory attributes: {dir(first_item.trajectory)}")
            
            if hasattr(first_item.trajectory, 'states'):
                print(f"Number of states: {len(first_item.trajectory.states)}")
                if len(first_item.trajectory.states) > 0:
                    print(f"First state type: {type(first_item.trajectory.states[0])}")
                    print(f"First state attributes: {dir(first_item.trajectory.states[0])}")
        
        # Check parent posterior
        if hasattr(first_item, 'parent_posterior'):
            print(f"\nParent posterior type: {type(first_item.parent_posterior)}")
            posterior = first_item.parent_posterior
            
            if isinstance(posterior, dict):
                print(f"Parent posterior keys: {list(posterior.keys())}")
                
                # Check if it has 'posterior_distribution' key
                if 'posterior_distribution' in posterior:
                    print(f"\nPosterior distribution found!")
                    post_dist = posterior['posterior_distribution']
                    print(f"Posterior distribution type: {type(post_dist)}")
                    print(f"Posterior distribution length: {len(post_dist)}")
                    
                    if isinstance(post_dist, dict):
                        # Check values
                        probs_list = list(post_dist.values())
                        parent_sets = list(post_dist.keys())
                        
                        print(f"\nFirst few parent sets: {parent_sets[:3]}")
                        print(f"First few probs: {probs_list[:5]}")
                        print(f"Sum of probs: {sum(probs_list)}")
                        print(f"Min/Max prob: {min(probs_list)}/{max(probs_list)}")
                        
                        # Check for astronomical values
                        if max(probs_list) > 1000:
                            print(f"\nWARNING: Astronomical probability values detected!")
                            print(f"Probs > 1000: {[p for p in probs_list if p > 1000][:10]}")
                            
                        # Check if probabilities sum to 1
                        prob_sum = sum(probs_list)
                        if abs(prob_sum - 1.0) > 0.01:
                            print(f"\nWARNING: Probabilities don't sum to 1.0! Sum = {prob_sum}")
                
                # Check posterior history
                if 'posterior_history' in posterior:
                    print(f"\nPosterior history found with {len(posterior['posterior_history'])} entries")
                    if len(posterior['posterior_history']) > 0:
                        first_entry = posterior['posterior_history'][0]
                        print(f"First history entry: {first_entry}")
            
            elif hasattr(posterior, 'parent_probs'):
                # Handle structured posterior object
                print(f"Parent probs type: {type(posterior.parent_probs)}")
                print(f"Parent probs length: {len(posterior.parent_probs)}")
        
        # Check other relevant attributes
        if hasattr(first_item, 'target_variable'):
            print(f"\nTarget variable: {first_item.target_variable}")
        
        if hasattr(first_item, 'n_nodes'):
            print(f"Number of nodes: {first_item.n_nodes}")
            
        if hasattr(first_item, 'observational_samples'):
            obs = first_item.observational_samples
            print(f"\nObservational samples type: {type(obs)}")
            if hasattr(obs, 'shape'):
                print(f"Observational samples shape: {obs.shape}")
            elif hasattr(obs, '__len__'):
                print(f"Observational samples length: {len(obs)}")


if __name__ == "__main__":
    inspect_demo_file()