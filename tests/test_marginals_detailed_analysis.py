"""
Detailed analysis of marginals to understand their true nature.
"""

import json
import numpy as np
from pathlib import Path


def analyze_all_marginals_in_file(results_path):
    """Analyze all marginals across all methods and runs."""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("=== COMPREHENSIVE MARGINALS ANALYSIS ===\n")
    
    # Analyze each method
    for method_name, results in data["method_results"].items():
        print(f"\n--- Method: {method_name} ---")
        
        all_marginals_found = False
        marginal_sums = []
        marginal_counts = []
        variable_sets = []
        
        for run_idx, result in enumerate(results):
            learning_history = result.get("detailed_results", {}).get("learning_history", [])
            
            # Collect all non-empty marginals
            for step_data in learning_history:
                if "marginals" in step_data and step_data["marginals"]:
                    marginals = step_data["marginals"]
                    marginal_sum = sum(marginals.values())
                    marginal_sums.append(marginal_sum)
                    marginal_counts.append(len(marginals))
                    variable_sets.append(set(marginals.keys()))
                    all_marginals_found = True
                    
                    # Show first few examples
                    if len(marginal_sums) <= 3:
                        print(f"  Run {run_idx+1}, Step {step_data['step']}:")
                        print(f"    Variables: {sorted(marginals.keys())}")
                        print(f"    Values: {marginals}")
                        print(f"    Sum: {marginal_sum:.6f}")
        
        if all_marginals_found:
            # Analyze patterns
            print(f"\n  Statistics across all steps:")
            print(f"    Total marginals found: {len(marginal_sums)}")
            print(f"    Sum range: [{min(marginal_sums):.6f}, {max(marginal_sums):.6f}]")
            print(f"    Average sum: {np.mean(marginal_sums):.6f}")
            print(f"    Variable count range: [{min(marginal_counts)}, {max(marginal_counts)}]")
            
            # Check if all variable sets are the same
            unique_var_sets = []
            for var_set in variable_sets:
                if var_set not in unique_var_sets:
                    unique_var_sets.append(var_set)
            
            print(f"    Unique variable sets: {len(unique_var_sets)}")
            for i, var_set in enumerate(unique_var_sets[:3]):  # Show first 3
                print(f"      Set {i+1}: {sorted(var_set)}")
            
            # Determine type
            avg_sum = np.mean(marginal_sums)
            sum_variance = np.var(marginal_sums)
            
            print(f"\n  CONCLUSION:")
            if 0.95 < avg_sum < 1.05 and sum_variance < 0.01:
                print(f"    Likely node selection probabilities (softmax output)")
                print(f"    Reason: Sums very close to 1.0 with low variance")
            elif avg_sum > 1.5 or avg_sum < 0.5:
                print(f"    Likely parent marginal probabilities")
                print(f"    Reason: Sums significantly different from 1.0")
            else:
                print(f"    AMBIGUOUS - could be either type")
                print(f"    Need to check variable coverage and SCM structure")
                
                # Additional check: Look at metadata
                metadata = result.get("metadata", {})
                scm_name = metadata.get("scm_name", "unknown")
                scm_type = metadata.get("scm_type", "unknown") 
                print(f"    SCM: {scm_name} (type: {scm_type})")
        else:
            print(f"  No non-empty marginals found")


def check_scm_structure_info(results_path):
    """Check SCM structure information to understand expected parent counts."""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("\n\n=== SCM STRUCTURE INFORMATION ===\n")
    
    # Collect unique SCMs used
    scm_info = {}
    
    for method_name, results in data["method_results"].items():
        for result in results:
            metadata = result.get("metadata", {})
            scm_name = metadata.get("scm_name", "unknown")
            scm_type = metadata.get("scm_type", "unknown")
            
            # Look for true parent information
            detailed = result.get("detailed_results", {})
            true_parents = None
            
            # Try to find true parents in various places
            if "true_parents" in result:
                true_parents = result["true_parents"]
            elif "true_parents" in detailed:
                true_parents = detailed["true_parents"]
            
            if scm_name not in scm_info:
                scm_info[scm_name] = {
                    "type": scm_type,
                    "true_parents": true_parents,
                    "methods": []
                }
            
            if method_name not in scm_info[scm_name]["methods"]:
                scm_info[scm_name]["methods"].append(method_name)
    
    # Display SCM information
    for scm_name, info in scm_info.items():
        print(f"SCM: {scm_name}")
        print(f"  Type: {info['type']}")
        print(f"  True parents: {info['true_parents']}")
        print(f"  Used by methods: {', '.join(info['methods'])}")
        
        # Analyze expected marginals
        if info['true_parents']:
            n_parents = len(info['true_parents']) if isinstance(info['true_parents'], list) else 0
            print(f"  Expected parent count: {n_parents}")
            
            # For a 3-variable SCM with 2 parents, parent marginals could sum to ~1.0
            # if one parent has high prob and other has low prob
            if n_parents == 2:
                print(f"  WARNING: With 2 parents, parent marginals could sum near 1.0!")


def check_for_enriched_policy_marginals(results_path):
    """Check if enriched policy method stores different marginals."""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("\n\n=== ENRICHED POLICY MARGINALS CHECK ===\n")
    
    if "Trained Policy + Learning" in data["method_results"]:
        results = data["method_results"]["Trained Policy + Learning"]
        
        for run_idx, result in enumerate(results[:1]):  # Just check first run
            learning_history = result.get("detailed_results", {}).get("learning_history", [])
            
            print(f"Run {run_idx+1}:")
            print(f"  Total steps: {len(learning_history)}")
            
            # Check if marginals exist
            steps_with_marginals = 0
            for step_data in learning_history:
                if "marginals" in step_data:
                    steps_with_marginals += 1
                    if steps_with_marginals == 1:  # Show first example
                        print(f"  First step with marginals: {step_data.get('step', 'unknown')}")
                        print(f"  Marginals present: {'marginals' in step_data}")
                        print(f"  Marginals value: {step_data.get('marginals', 'NOT FOUND')}")
            
            print(f"  Steps with marginals field: {steps_with_marginals}")
            
            # Check what other fields exist
            if learning_history:
                first_step = learning_history[0]
                print(f"\n  Available fields in learning history:")
                for key in sorted(first_step.keys()):
                    print(f"    - {key}")


if __name__ == "__main__":
    results_path = Path("/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt/results/evaluation_single_checkpoint_20250723_124106/grpo_quick_minimize_20250723_101252_fixed/comparison_results.json")
    
    if results_path.exists():
        analyze_all_marginals_in_file(results_path)
        check_scm_structure_info(results_path)
        check_for_enriched_policy_marginals(results_path)
    else:
        print(f"Results file not found: {results_path}")