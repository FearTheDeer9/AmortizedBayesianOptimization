#!/usr/bin/env python3
"""
Final 20-Node Validation Test

Quick validation with properly scaled parameters based on our analysis:
- 500 samples (close to recommended ~400-600 range)
- 15 bootstrap samples  
- Comprehensive intervention coverage
"""

import sys
import os
import numpy as np
import time

# Add external PARENT_SCALE to path
sys.path.insert(0, 'external/parent_scale')
sys.path.insert(0, 'external')

# Import original PARENT_SCALE components
try:
    from parent_scale.posterior_model.model import DoublyRobustModel, Data
    DOUBLY_ROBUST_AVAILABLE = True
    print("✅ Original DoublyRobustModel imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DoublyRobustModel: {e}")
    DOUBLY_ROBUST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class TestGraph:
    """Simple graph for testing."""
    
    def __init__(self, variables, edges, target):
        self.variables = variables
        self.edges = edges  
        self.target = target
        self.parents = self._build_parents()
    
    def _build_parents(self):
        parents = {var: [] for var in self.variables}
        for parent, child in self.edges:
            parents[child].append(parent)
        return parents


def test_20_node_final():
    """Final test with optimal parameters."""
    
    if not DOUBLY_ROBUST_AVAILABLE:
        print("❌ Cannot run test")
        return
    
    print("FINAL 20-NODE VALIDATION")
    print("=" * 40)
    print("Using optimal scaling parameters")
    print()
    
    # Create 20-node chain: X0 -> X1 -> ... -> X19
    variables = [f"X{i}" for i in range(20)]
    edges = [(variables[i], variables[i+1]) for i in range(19)]
    target = variables[-1]
    true_parents = [variables[18]]
    
    graph = TestGraph(variables, edges, target)
    
    print(f"Graph: Chain X0->X1->...->X19")
    print(f"Target: {target}")
    print(f"True parents: {true_parents}")
    
    # Generate optimal data
    np.random.seed(42)
    n_obs, n_int = 425, 75  # Total 500 samples
    
    print(f"Data: {n_obs} obs + {n_int} int = {n_obs + n_int} total")
    
    # Observational data
    obs_data = []
    for _ in range(n_obs):
        sample = np.zeros(20)
        for i in range(20):
            if i == 0:
                sample[i] = np.random.normal(0, 0.5)
            else:
                sample[i] = sample[i-1] + np.random.normal(0, 0.1)
        obs_data.append(sample)
    
    # Interventional data - comprehensive coverage
    int_data = []
    int_nodes = []
    
    # Intervene on X0-X18 (all non-target variables)
    interventions_per_var = n_int // 19  # ~4 per variable
    
    for var_idx in range(19):  # X0 to X18
        for _ in range(interventions_per_var):
            intervention_val = np.random.normal(0, 1)
            
            sample = np.zeros(20)
            intervention_indicator = np.zeros(20)
            intervention_indicator[var_idx] = 1
            
            for i in range(20):
                if i == var_idx:
                    sample[i] = intervention_val
                elif i == 0:
                    sample[i] = np.random.normal(0, 0.5)
                else:
                    sample[i] = sample[i-1] + np.random.normal(0, 0.1)
            
            int_data.append(sample)
            int_nodes.append(intervention_indicator)
    
    # Combine data
    all_samples = np.vstack([obs_data, int_data[:n_int]])
    all_nodes = np.vstack([np.zeros((n_obs, 20)), int_nodes[:n_int]])
    
    # Create model with optimal parameters
    num_bootstraps = 15
    print(f"Bootstrap samples: {num_bootstraps}")
    
    data = Data(samples=all_samples, nodes=all_nodes)
    model = DoublyRobustModel(
        graph=graph,
        topological_order=graph.variables,
        target=graph.target,
        num_bootstraps=num_bootstraps
    )
    
    print(f"\nRunning inference...")
    start_time = time.time()
    
    try:
        estimate = model.run_method(data)
        inference_time = time.time() - start_time
        
        print(f"✅ Completed in {inference_time:.1f}s")
        
        if model.prob_estimate:
            # Get top parent sets
            sorted_estimates = sorted(model.prob_estimate.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop parent sets:")
            for i, (parent_set, prob) in enumerate(sorted_estimates[:3]):
                parent_list = list(parent_set) if parent_set else ["∅"]
                print(f"  {i+1}. {parent_list}: {prob:.3f}")
            
            # Most likely
            most_likely = sorted_estimates[0]
            detected = set(most_likely[0]) if most_likely[0] else set()
            true_set = set(true_parents)
            
            # Calculate accuracy
            if detected == true_set:
                accuracy = 1.0
                status = "✅ PERFECT"
            else:
                intersection = len(detected.intersection(true_set))
                union = len(detected.union(true_set))
                accuracy = intersection / union if union > 0 else 0.0
                if accuracy >= 0.8:
                    status = "✅ EXCELLENT"
                elif accuracy >= 0.6:
                    status = "✅ GOOD"
                else:
                    status = "⚠️ NEEDS WORK"
            
            print(f"\nResults:")
            print(f"  Detected: {detected}")
            print(f"  True: {true_set}")
            print(f"  Accuracy: {accuracy:.3f} {status}")
            print(f"  Confidence: {most_likely[1]:.3f}")
            
            print(f"\n🎯 FINAL ASSESSMENT:")
            if accuracy >= 0.8:
                print("  ✅ SUCCESS: 20-node performance achieved!")
                print("  ✅ Ready for ACBO integration")
                print("  ✅ Data scaling approach validated")
            elif accuracy >= 0.6:
                print("  ✅ GOOD: Reasonable 20-node performance")
                print("  ✅ May need minor tuning for optimal results")
            else:
                print("  ⚠️ PARTIAL: Further optimization needed")
            
            return accuracy >= 0.7
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_20_node_final()
    
    if success:
        print(f"\n🚀 CONCLUSION:")
        print("Neural doubly robust scales to 20 nodes with proper data scaling!")
        print("Occam's razor solution confirmed: more data + training = better performance.")
    else:
        print(f"\n📊 CONCLUSION:")
        print("Need to investigate further or increase resources.")