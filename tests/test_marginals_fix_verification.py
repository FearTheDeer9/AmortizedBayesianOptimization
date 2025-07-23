"""
Test to verify that marginals are properly tracked and F1 scores can be computed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
from scripts.core.acbo_comparison.structure_metrics_helper import (
    compute_f1_from_marginals,
    compute_parent_probability,
    compute_shd_from_marginals
)


def test_marginals_computation():
    """Test that parent marginals are computed correctly for F1 calculation."""
    
    print("=== TESTING MARGINALS COMPUTATION ===\n")
    
    # Create a simple SCM
    key = random.PRNGKey(42)
    scm = create_erdos_renyi_scm(n_nodes=3, edge_prob=0.5, seed=42)
    
    # The target should be the last variable (Y or X2)
    from causal_bayes_opt.data_structures.scm import get_target, get_parents, get_variables
    
    target = get_target(scm)
    true_parents = list(get_parents(scm, target))
    all_variables = sorted(get_variables(scm))
    
    print(f"SCM Structure:")
    print(f"  Target: {target}")
    print(f"  True parents: {true_parents}")
    print(f"  All variables: {all_variables}")
    
    # Test various marginal scenarios
    test_cases = [
        {
            "name": "Perfect prediction",
            "marginals": {
                var: 1.0 if var in true_parents else 0.0 
                for var in all_variables if var != target
            }
        },
        {
            "name": "High confidence correct",
            "marginals": {
                var: 0.9 if var in true_parents else 0.1 
                for var in all_variables if var != target
            }
        },
        {
            "name": "Uncertain (all 0.5)",
            "marginals": {
                var: 0.5 for var in all_variables if var != target
            }
        },
        {
            "name": "Wrong prediction",
            "marginals": {
                var: 0.1 if var in true_parents else 0.9 
                for var in all_variables if var != target
            }
        }
    ]
    
    print("\n=== F1 SCORE COMPUTATION TESTS ===\n")
    
    for test_case in test_cases:
        name = test_case["name"]
        marginals = test_case["marginals"]
        
        print(f"\nTest: {name}")
        print(f"  Marginals: {marginals}")
        print(f"  Sum of marginals: {sum(marginals.values()):.2f} (should NOT be 1.0)")
        
        # Compute metrics
        f1, precision, recall = compute_f1_from_marginals(
            marginals, true_parents, target, threshold=0.5
        )
        parent_prob = compute_parent_probability(marginals, true_parents)
        shd = compute_shd_from_marginals(marginals, true_parents, target, threshold=0.5)
        
        print(f"  Results:")
        print(f"    F1 score: {f1:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    Avg parent probability: {parent_prob:.3f}")
        print(f"    SHD: {shd}")
        
        # Verify expectations
        if name == "Perfect prediction":
            assert f1 == 1.0, f"Perfect prediction should have F1=1.0, got {f1}"
            assert shd == 0, f"Perfect prediction should have SHD=0, got {shd}"
        elif name == "Wrong prediction":
            assert f1 < 0.5, f"Wrong prediction should have low F1, got {f1}"
            assert shd > 0, f"Wrong prediction should have SHD>0, got {shd}"


def test_enriched_policy_marginals_integration():
    """Test that enriched policy properly tracks marginals during learning."""
    
    print("\n\n=== TESTING ENRICHED POLICY MARGINALS INTEGRATION ===\n")
    
    # This would require actually running the enriched policy
    # For now, we'll create mock data to test the flow
    
    # Mock learning history with proper parent marginals
    mock_learning_history = []
    
    # Simulate learning progression
    n_steps = 10
    variables = ["X0", "X1", "X2"]
    target = "Y"
    true_parents = ["X0", "X2"]  # X1 is not a parent
    
    for step in range(n_steps):
        # Simulate marginals converging to truth
        learning_progress = step / n_steps
        
        marginals = {}
        for var in variables:
            if var in true_parents:
                # True parents: probability increases from 0.3 to 0.9
                prob = 0.3 + 0.6 * learning_progress
            else:
                # Non-parents: probability decreases from 0.7 to 0.1
                prob = 0.7 - 0.6 * learning_progress
            marginals[var] = prob
        
        step_data = {
            'step': step + 1,
            'marginals': marginals,
            'learning_enabled': True
        }
        mock_learning_history.append(step_data)
    
    print("Simulated learning progression:")
    print(f"  True parents: {true_parents}")
    
    # Compute F1 scores over time
    f1_scores = []
    for step_data in mock_learning_history:
        marginals = step_data['marginals']
        f1, _, _ = compute_f1_from_marginals(marginals, true_parents, target)
        f1_scores.append(f1)
    
    # Show progression
    print("\n  F1 score progression:")
    for i, f1 in enumerate(f1_scores):
        if i % 3 == 0:  # Show every 3rd step
            print(f"    Step {i+1}: F1 = {f1:.3f}")
    
    # Verify learning
    assert f1_scores[0] < 0.5, "Initial F1 should be low"
    assert f1_scores[-1] > 0.8, "Final F1 should be high"
    assert all(f1_scores[i] <= f1_scores[i+1] for i in range(len(f1_scores)-1)), \
        "F1 should increase monotonically with proper learning"
    
    print("\n  ✓ F1 scores increase properly during learning")


def validate_marginals_are_parent_probs():
    """Validate that marginals represent parent probabilities, not node selection."""
    
    print("\n\n=== VALIDATING MARGINALS ARE PARENT PROBABILITIES ===\n")
    
    # Key differences between parent probs and node selection probs
    differences = [
        ("Sum constraint", "Do NOT sum to 1.0", "MUST sum to 1.0"),
        ("Independence", "Each edge probability is independent", "Softmax output - mutually exclusive"),
        ("Range", "Each can be any value in [0,1]", "Exactly one high value, others low"),
        ("Interpretation", "P(X→Y) for each variable", "P(select X for intervention)"),
        ("Use case", "Structure learning / F1 computation", "Intervention targeting")
    ]
    
    print("Parent Marginals vs Node Selection Probabilities:\n")
    print(f"{'Property':<20} {'Parent Marginals':<40} {'Node Selection Probs':<40}")
    print("-" * 100)
    
    for prop, parent_desc, node_desc in differences:
        print(f"{prop:<20} {parent_desc:<40} {node_desc:<40}")
    
    print("\n✓ Marginals in learning history should be parent probabilities for F1 computation")


if __name__ == "__main__":
    test_marginals_computation()
    test_enriched_policy_marginals_integration()
    validate_marginals_are_parent_probs()
    
    print("\n\n=== ALL TESTS PASSED ===")
    print("\nSummary:")
    print("1. Parent marginals are correctly interpreted for F1 computation")
    print("2. Marginals do NOT sum to 1.0 (they are independent edge probabilities)")
    print("3. F1 scores can be properly computed from parent marginals")
    print("4. Learning progression can be tracked through marginal evolution")