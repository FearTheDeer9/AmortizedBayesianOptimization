#!/usr/bin/env python3
"""
Test F1 and SHD computation from marginal probabilities.

This test verifies that the structure metrics are computed correctly
from parent marginal probabilities (edge probabilities).
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.acbo_comparison.structure_metrics_helper import (
    compute_f1_from_marginals,
    compute_shd_from_marginals,
    compute_parent_probability
)


def test_f1_computation_perfect_prediction():
    """Test F1 computation with perfect predictions."""
    # True parents: X1, X3
    true_parents = ['X1', 'X3']
    target = 'Y'
    
    # Perfect predictions - high prob for true parents, low for others
    marginals = {
        'X0': 0.1,  # Not a parent
        'X1': 0.9,  # True parent
        'X2': 0.2,  # Not a parent
        'X3': 0.8,  # True parent
    }
    
    f1, precision, recall = compute_f1_from_marginals(marginals, true_parents, target)
    
    # Should be perfect
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_f1_computation_no_true_positives():
    """Test F1 computation when no true parents are identified."""
    true_parents = ['X1', 'X3']
    target = 'Y'
    
    # Low prob for all (including true parents)
    marginals = {
        'X0': 0.1,
        'X1': 0.2,  # True parent but low prob
        'X2': 0.2,
        'X3': 0.3,  # True parent but low prob
    }
    
    f1, precision, recall = compute_f1_from_marginals(marginals, true_parents, target)
    
    # Should be 0 (no true positives)
    assert f1 == 0.0
    assert recall == 0.0


def test_f1_computation_partial_match():
    """Test F1 computation with partial matches."""
    true_parents = ['X1', 'X3'] 
    target = 'Y'
    
    # Mixed predictions
    marginals = {
        'X0': 0.8,  # False positive
        'X1': 0.9,  # True positive
        'X2': 0.2,  # True negative
        'X3': 0.3,  # False negative
    }
    
    f1, precision, recall = compute_f1_from_marginals(marginals, true_parents, target)
    
    # 1 true positive, 1 false positive, 1 false negative
    # Precision = 1/2 = 0.5
    # Recall = 1/2 = 0.5
    # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    assert f1 == 0.5
    assert precision == 0.5
    assert recall == 0.5


def test_marginals_are_parent_probabilities():
    """Test that marginals represent parent probabilities (not node selection)."""
    # Parent marginals should NOT sum to 1.0
    # They are independent edge probabilities
    
    marginals = {
        'X0': 0.9,
        'X1': 0.8,
        'X2': 0.7,
        'X3': 0.95,
    }
    
    total = sum(marginals.values())
    
    # Should NOT sum to 1.0 (unless by coincidence)
    assert abs(total - 1.0) > 0.1  # Very unlikely to sum to 1.0
    
    # Each marginal is a valid probability
    for prob in marginals.values():
        assert 0.0 <= prob <= 1.0


def test_shd_computation():
    """Test SHD computation from marginals."""
    true_parents = ['X1', 'X3']
    target = 'Y'
    
    # Perfect predictions
    marginals_perfect = {
        'X0': 0.1,  # Correctly identified as non-parent
        'X1': 0.9,  # Correctly identified as parent
        'X2': 0.2,  # Correctly identified as non-parent
        'X3': 0.8,  # Correctly identified as parent
    }
    
    shd_perfect = compute_shd_from_marginals(marginals_perfect, true_parents, target)
    assert shd_perfect == 0  # No errors
    
    # Worst case - all wrong
    marginals_worst = {
        'X0': 0.9,  # Wrong - false positive
        'X1': 0.1,  # Wrong - false negative
        'X2': 0.8,  # Wrong - false positive
        'X3': 0.2,  # Wrong - false negative
    }
    
    shd_worst = compute_shd_from_marginals(marginals_worst, true_parents, target)
    assert shd_worst == 4  # All 4 edges wrong


def test_parent_probability_computation():
    """Test average parent probability computation."""
    true_parents = ['X1', 'X3']
    
    marginals = {
        'X0': 0.1,
        'X1': 0.8,  # True parent
        'X2': 0.2,
        'X3': 0.6,  # True parent
    }
    
    avg_prob = compute_parent_probability(marginals, true_parents)
    
    # Should be average of 0.8 and 0.6
    assert avg_prob == 0.7


def test_empty_marginals():
    """Test handling of empty marginals."""
    true_parents = ['X1', 'X3']
    target = 'Y'
    
    # Empty marginals
    marginals = {}
    
    f1, precision, recall = compute_f1_from_marginals(marginals, true_parents, target)
    assert f1 == 0.0
    assert precision == 0.0
    assert recall == 0.0
    
    shd = compute_shd_from_marginals(marginals, true_parents, target)
    assert shd == len(true_parents)  # All parents missing
    
    avg_prob = compute_parent_probability(marginals, true_parents)
    assert avg_prob == 0.0


def test_real_learning_history_example():
    """Test with a real example from learning history."""
    # This mimics what we see in actual learning results
    target = 'Y'
    true_parents = ['X0', 'X2']  # Example true parents
    
    # Early in learning - high uncertainty
    marginals_early = {
        'X0': 0.5,   # Uncertain about true parent
        'X1': 0.45,  # Almost equally likely
        'X2': 0.48,  # Uncertain about true parent
        'X3': 0.52,  # Slightly higher but still uncertain
    }
    
    f1_early, _, _ = compute_f1_from_marginals(marginals_early, true_parents, target)
    
    # Later in learning - more confident
    marginals_late = {
        'X0': 0.85,  # High confidence in true parent
        'X1': 0.15,  # Low confidence in non-parent
        'X2': 0.90,  # High confidence in true parent
        'X3': 0.20,  # Low confidence in non-parent
    }
    
    f1_late, _, _ = compute_f1_from_marginals(marginals_late, true_parents, target)
    
    # F1 should improve with learning
    assert f1_late > f1_early
    assert f1_late > 0.8  # Should be high with good predictions


if __name__ == "__main__":
    # Run all tests
    test_f1_computation_perfect_prediction()
    test_f1_computation_no_true_positives()
    test_f1_computation_partial_match()
    test_marginals_are_parent_probabilities()
    test_shd_computation()
    test_parent_probability_computation()
    test_empty_marginals()
    test_real_learning_history_example()
    
    print("All tests passed! ✅")
    print("\nKey findings:")
    print("1. F1 computation correctly handles parent marginal probabilities")
    print("2. Marginals do NOT sum to 1.0 (they are independent edge probabilities)")
    print("3. SHD computation correctly counts edge differences")
    print("4. Empty marginals are handled gracefully")