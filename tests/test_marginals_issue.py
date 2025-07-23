"""
Test to demonstrate the marginals issue in learning history.

This test shows that the 'marginals' field in learning history contains
node selection probabilities (from GRPO policy) rather than parent edge
probabilities needed for F1 computation.
"""

import json
import numpy as np
from pathlib import Path


def analyze_marginals_in_results():
    """Analyze marginals in existing results to show the issue."""
    
    # Load a recent results file
    results_path = Path("/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt/results/evaluation_single_checkpoint_20250723_124106/grpo_quick_minimize_20250723_101252_fixed/comparison_results.json")
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("=== MARGINALS ANALYSIS ===\n")
    
    # Analyze Random + Learning method results
    if "Random + Learning" in data["method_results"]:
        results = data["method_results"]["Random + Learning"]
        
        for i, result in enumerate(results):
            print(f"\n--- Run {i+1} ---")
            
            learning_history = result["detailed_results"]["learning_history"]
            
            # Check first few steps with marginals
            for step_data in learning_history[:5]:
                if "marginals" in step_data and step_data["marginals"]:
                    step = step_data["step"]
                    marginals = step_data["marginals"]
                    
                    print(f"\nStep {step}:")
                    print(f"  Marginals: {marginals}")
                    
                    # Check if they sum to ~1 (node selection probs)
                    marginal_sum = sum(marginals.values())
                    print(f"  Sum of marginals: {marginal_sum:.6f}")
                    
                    # Check if values are in [0, 1]
                    all_in_range = all(0 <= v <= 1 for v in marginals.values())
                    print(f"  All values in [0,1]: {all_in_range}")
                    
                    # Analyze distribution
                    values = list(marginals.values())
                    if len(values) > 1:
                        max_val = max(values)
                        min_val = min(values)
                        print(f"  Max value: {max_val:.6f}")
                        print(f"  Min value: {min_val:.6f}")
                        print(f"  Looks like node selection probs: {marginal_sum > 0.95 and marginal_sum < 1.05}")
            
            # Check F1 scores
            f1_scores = result["detailed_results"].get("f1_scores", [])
            print(f"\nF1 scores: {f1_scores[:5]}...")
            print(f"All F1 scores are 0: {all(score == 0.0 for score in f1_scores)}")
            
            break  # Just analyze first run


def demonstrate_correct_parent_marginals():
    """Show what parent marginals should look like."""
    
    print("\n\n=== WHAT PARENT MARGINALS SHOULD LOOK LIKE ===\n")
    
    # Example SCM: Y has parents X1 and X2
    true_parents = ["X1", "X2"]
    all_variables = ["X0", "X1", "X2", "X3"]
    target = "Y"
    
    print(f"True graph: {true_parents} → {target}")
    print(f"All variables: {all_variables}")
    
    # Example parent marginals (what we should be storing)
    parent_marginals = {
        "X0": 0.1,   # Low probability of being parent
        "X1": 0.8,   # High probability (true parent)
        "X2": 0.75,  # High probability (true parent)  
        "X3": 0.05   # Low probability of being parent
    }
    
    print(f"\nParent marginals (P(X→Y)):")
    for var, prob in parent_marginals.items():
        is_parent = var in true_parents
        print(f"  P({var}→{target}) = {prob:.2f} {'✓ (true parent)' if is_parent else ''}")
    
    print(f"\nSum of parent marginals: {sum(parent_marginals.values()):.2f}")
    print("Note: Parent marginals do NOT sum to 1.0!")
    print("Each represents independent probability of edge existence")
    
    # Show how F1 would be computed
    threshold = 0.5
    predicted_parents = [var for var, prob in parent_marginals.items() if prob > threshold]
    
    true_positives = len(set(predicted_parents) & set(true_parents))
    false_positives = len(set(predicted_parents) - set(true_parents))
    false_negatives = len(set(true_parents) - set(predicted_parents))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nWith threshold {threshold}:")
    print(f"  Predicted parents: {predicted_parents}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")


def show_node_selection_vs_parent_probs():
    """Clearly show the difference between node selection and parent probabilities."""
    
    print("\n\n=== NODE SELECTION vs PARENT PROBABILITIES ===\n")
    
    print("1. Node Selection Probabilities (what we currently store):")
    print("   - Output of GRPO policy for choosing intervention target")
    print("   - Softmax distribution over nodes")
    print("   - MUST sum to 1.0")
    print("   - Example: {'X0': 0.02, 'X2': 0.98} means 98% chance to intervene on X2")
    
    print("\n2. Parent Marginal Probabilities (what we need for F1):")
    print("   - P(Xi → Target) from structure learning")
    print("   - Independent probabilities for each edge")
    print("   - Do NOT sum to 1.0")
    print("   - Example: {'X0': 0.1, 'X1': 0.9, 'X2': 0.85} means X1 and X2 likely parents")
    
    print("\n3. Why current F1 computation fails:")
    print("   - We're treating node selection probs as parent probs")
    print("   - Node with highest selection prob gets treated as only parent")
    print("   - Real parent structure is completely ignored")


if __name__ == "__main__":
    analyze_marginals_in_results()
    demonstrate_correct_parent_marginals()
    show_node_selection_vs_parent_probs()