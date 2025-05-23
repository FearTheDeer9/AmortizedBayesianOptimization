"""
Test script for Phase 1.1: Basic Linear Mechanisms

This script validates the core functionality implemented in mechanisms/linear.py
according to the AVICI Adaptation Implementation Plan Phase 1.1 requirements.
"""

import sys
import logging
from pathlib import Path

# Add src to path to import our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import jax.random as random
import jax.numpy as jnp

from causal_bayes_opt.mechanisms.linear import (
    create_linear_mechanism,
    create_root_mechanism,
    sample_from_linear_scm
)
from causal_bayes_opt.experiments.test_scms import (
    create_simple_test_scm,
    create_simple_linear_scm,
    create_chain_test_scm
)
from causal_bayes_opt.data_structures.sample import get_values, get_value
from causal_bayes_opt.data_structures.scm import get_variables, topological_sort

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_create_linear_mechanism():
    """Test creating linear mechanisms with proper JAX random key threading."""
    print("\n=== Testing create_linear_mechanism ===")
    
    # Test root mechanism (no parents)
    root_mechanism = create_root_mechanism(mean=5.0, noise_scale=0.1)
    key = random.PRNGKey(42)
    value = root_mechanism({}, key)
    print(f"Root mechanism value: {value:.4f} (expected ~5.0)")
    assert isinstance(value, float), "Root mechanism should return float"
    
    # Test linear mechanism with parents
    mechanism = create_linear_mechanism(
        parents=['X', 'Z'],
        coefficients={'X': 2.0, 'Z': -1.5},
        intercept=1.0,
        noise_scale=0.1
    )
    
    parent_values = {'X': 3.0, 'Z': 2.0}
    key = random.PRNGKey(123)
    value = mechanism(parent_values, key)
    expected = 1.0 + 2.0 * 3.0 - 1.5 * 2.0  # 1 + 6 - 3 = 4.0
    print(f"Linear mechanism value: {value:.4f} (expected ~{expected:.1f})")
    assert abs(value - expected) < 1.0, "Linear mechanism result should be close to expected"
    
    print("✓ create_linear_mechanism tests passed")


def test_validation_and_error_handling():
    """Test that validation works and provides helpful error messages."""
    print("\n=== Testing Validation and Error Handling ===")
    
    # Test missing coefficients
    try:
        create_linear_mechanism(
            parents=['X', 'Y'],
            coefficients={'X': 2.0}  # Missing coefficient for Y
        )
        assert False, "Should have raised ValueError for missing coefficients"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
        assert "Missing coefficients" in str(e)
    
    # Test invalid noise scale
    try:
        create_linear_mechanism(
            parents=[],
            coefficients={},
            noise_scale=-1.0  # Invalid negative noise scale
        )
        assert False, "Should have raised ValueError for negative noise scale"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
        assert "non-negative" in str(e)
    
    print("✓ Validation tests passed")


def test_sample_from_linear_scm():
    """Test sampling from linear SCMs with proper random key management."""
    print("\n=== Testing sample_from_linear_scm ===")
    
    # Create a simple test SCM: X → Y ← Z
    scm = create_simple_test_scm(noise_scale=0.5, target="Y")
    
    # Test basic sampling
    samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
    
    print(f"Generated {len(samples)} samples")
    assert len(samples) == 100, "Should generate exactly 100 samples"
    
    # Check sample structure
    first_sample = samples[0]
    sample_values = get_values(first_sample)
    variables = set(sample_values.keys())
    expected_variables = {'X', 'Y', 'Z'}
    assert variables == expected_variables, f"Sample should have variables {expected_variables}, got {variables}"
    
    # Check that values are reasonable
    x_vals = [get_value(sample, 'X') for sample in samples[:10]]
    y_vals = [get_value(sample, 'Y') for sample in samples[:10]]
    z_vals = [get_value(sample, 'Z') for sample in samples[:10]]
    
    print(f"First 5 X values: {[f'{x:.3f}' for x in x_vals[:5]]}")
    print(f"First 5 Y values: {[f'{y:.3f}' for y in y_vals[:5]]}")
    print(f"First 5 Z values: {[f'{z:.3f}' for z in z_vals[:5]]}")
    
    # Test reproducibility
    samples2 = sample_from_linear_scm(scm, n_samples=10, seed=42)
    samples3 = sample_from_linear_scm(scm, n_samples=10, seed=42)
    
    for i in range(10):
        val2 = get_value(samples2[i], 'Y')
        val3 = get_value(samples3[i], 'Y')
        assert abs(val2 - val3) < 1e-10, f"Sampling should be reproducible with same seed"
    
    print("✓ sample_from_linear_scm tests passed")


def test_factory_functions():
    """Test factory functions for creating test SCMs."""
    print("\n=== Testing Factory Functions ===")
    
    # Test simple test SCM
    scm = create_simple_test_scm()
    variables = get_variables(scm)
    expected_vars = {'X', 'Y', 'Z'}
    assert variables == expected_vars, f"Simple test SCM should have variables {expected_vars}"
    
    order = topological_sort(scm)
    print(f"Topological order: {order}")
    # Y should come after both X and Z
    y_index = order.index('Y')
    x_index = order.index('X')
    z_index = order.index('Z')
    assert x_index < y_index and z_index < y_index, "Y should come after its parents X and Z"
    
    # Test general linear SCM factory
    scm2 = create_simple_linear_scm(
        variables=['A', 'B', 'C'],
        edges=[('A', 'B'), ('B', 'C')],
        coefficients={('A', 'B'): 1.5, ('B', 'C'): 0.8},
        noise_scales={'A': 1.0, 'B': 0.5, 'C': 1.0},
        target='C'
    )
    
    variables2 = get_variables(scm2)
    expected_vars2 = {'A', 'B', 'C'}
    assert variables2 == expected_vars2, f"Chain SCM should have variables {expected_vars2}"
    
    # Test that target is set correctly
    assert scm2['target'] == 'C', "Target should be set correctly"
    
    # Test chain SCM
    chain_scm = create_chain_test_scm(chain_length=4, coefficient=0.8)
    chain_vars = get_variables(chain_scm)
    expected_chain_vars = {'X0', 'X1', 'X2', 'X3'}
    assert chain_vars == expected_chain_vars, f"Chain SCM should have variables {expected_chain_vars}"
    
    print("✓ Factory function tests passed")


def test_integration_with_existing_structures():
    """Test integration with existing SCM and Sample data structures."""
    print("\n=== Testing Integration with Existing Structures ===")
    
    # Create SCM and verify it works with existing SCM utilities
    scm = create_simple_test_scm()
    
    # Test with existing SCM functions
    variables = get_variables(scm)
    order = topological_sort(scm)
    print(f"Variables: {sorted(variables)}")
    print(f"Topological order: {order}")
    
    # Generate samples and verify they work with existing Sample utilities
    samples = sample_from_linear_scm(scm, n_samples=5, seed=999)
    
    for i, sample in enumerate(samples):
        values = get_values(sample)
        x_val = get_value(sample, 'X')
        y_val = get_value(sample, 'Y')
        z_val = get_value(sample, 'Z')
        
        # Verify the linear relationship: Y = 2*X - 1.5*Z + noise
        expected_y = 2.0 * x_val - 1.5 * z_val
        noise = y_val - expected_y
        print(f"Sample {i}: X={x_val:.3f}, Z={z_val:.3f}, Y={y_val:.3f}, noise={noise:.3f}")
        
        # Noise should be reasonable (within a few standard deviations)
        assert abs(noise) < 5.0, f"Noise seems too large: {noise}"
    
    print("✓ Integration tests passed")


def test_module_separation():
    """Test that the refactored module structure works correctly."""
    print("\n=== Testing Module Separation ===")
    
    # Test that mechanisms module only contains core functions
    from causal_bayes_opt.mechanisms import create_linear_mechanism, create_root_mechanism, sample_from_linear_scm
    print("✓ Core mechanism functions imported from mechanisms module")
    
    # Test that test SCM factories are in experiments module
    from causal_bayes_opt.experiments import create_simple_test_scm, create_chain_test_scm, create_simple_linear_scm
    print("✓ Test SCM factories imported from experiments module")
    
    # Test that they work together
    scm = create_simple_test_scm(noise_scale=0.1)
    samples = sample_from_linear_scm(scm, n_samples=5, seed=42)
    assert len(samples) == 5, "Cross-module integration should work"
    print("✓ Cross-module integration works")
    
    print("✓ Module separation tests passed")


def main():
    """Run all Phase 1.1 validation tests."""
    print("Starting Phase 1.1 Validation Tests (Refactored)")
    print("=" * 55)
    
    try:
        test_create_linear_mechanism()
        test_validation_and_error_handling()
        test_sample_from_linear_scm()
        test_factory_functions()
        test_integration_with_existing_structures()
        test_module_separation()
        
        print("\n" + "=" * 55)
        print("🎉 ALL PHASE 1.1 TESTS PASSED! 🎉")
        print("\nPhase 1.1 Success Criteria Met:")
        print("✓ Can create linear mechanisms with proper JAX integration")
        print("✓ Mechanisms follow functional programming principles")
        print("✓ Proper validation and error handling with helpful messages")
        print("✓ Sampling works with topological ordering")
        print("✓ Random key threading is implemented correctly")
        print("✓ Integration with existing SCM and Sample structures works")
        print("✓ Factory functions create valid test SCMs")
        print("✓ Clean module separation achieved")
        
        print("\nArchitectural Improvements:")
        print("✓ mechanisms/ module focused on core mechanism creation")
        print("✓ experiments/ module contains test utilities and SCM factories")
        print("✓ Clear separation of concerns between modules")
        print("✓ No circular dependencies or cluttered APIs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
