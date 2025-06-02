"""
Test suite for Phase 1.2: AVICI Data Format Bridge

Run with: pytest tests/test_integration/test_avici_data_bridge.py -v
"""

import pytest
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.avici_integration import (
    samples_to_avici_format,
    create_training_batch,
    validate_data_conversion,
    analyze_avici_data,
)
from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.data_structures.sample import (
    create_observational_sample,
    create_interventional_sample,
    get_values,
    get_value
)


@pytest.fixture
def simple_observational_samples():
    """Fixture providing simple observational samples for testing."""
    return [
        create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0}),
        create_observational_sample({'X': 2.0, 'Y': 3.0, 'Z': 1.0}),
        create_observational_sample({'X': 0.0, 'Y': 1.0, 'Z': 2.0})
    ]


@pytest.fixture
def mixed_samples():
    """Fixture providing mixed observational and interventional samples."""
    return [
        create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0}),
        create_interventional_sample(
            values={'X': 5.0, 'Y': 2.0, 'Z': 3.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        ),
        create_interventional_sample(
            values={'X': 1.0, 'Y': 2.0, 'Z': 10.0},
            intervention_type='perfect', 
            targets=frozenset(['Z'])
        ),
        create_interventional_sample(
            values={'X': 5.0, 'Y': 2.0, 'Z': 10.0},
            intervention_type='perfect',
            targets=frozenset(['X', 'Z'])
        )
    ]


@pytest.fixture
def variable_order():
    """Standard variable order for testing."""
    return ['X', 'Y', 'Z']


@pytest.fixture
def test_scm():
    """Fixture providing a test SCM."""
    return create_simple_test_scm(noise_scale=0.5, target="Y")


class TestBasicConversion:
    """Test basic data conversion functionality."""
    
    def test_conversion_shape(self, simple_observational_samples, variable_order):
        """Test that conversion produces correct tensor shape."""
        avici_data = samples_to_avici_format(
            samples=simple_observational_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        expected_shape = (3, 3, 3)  # n_samples, n_vars, n_channels
        assert avici_data.shape == expected_shape
    
    def test_values_channel(self, simple_observational_samples, variable_order):
        """Test that values channel contains correct values."""
        avici_data = samples_to_avici_format(
            samples=simple_observational_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        expected_values = jnp.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0], 
            [0.0, 1.0, 2.0]
        ])
        
        assert jnp.allclose(avici_data[:, :, 0], expected_values)
    
    def test_observational_intervention_channel(self, simple_observational_samples, variable_order):
        """Test that intervention channel is zero for observational data."""
        avici_data = samples_to_avici_format(
            samples=simple_observational_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        expected_interventions = jnp.zeros((3, 3))
        assert jnp.allclose(avici_data[:, :, 1], expected_interventions)
    
    def test_target_channel(self, simple_observational_samples, variable_order):
        """Test that target channel correctly identifies target variable."""
        avici_data = samples_to_avici_format(
            samples=simple_observational_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        # Y is at index 1 in ['X', 'Y', 'Z']
        expected_targets = jnp.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        assert jnp.allclose(avici_data[:, :, 2], expected_targets)


class TestInterventionIndicators:
    """Test intervention indicator functionality."""
    
    def test_intervention_indicators(self, mixed_samples, variable_order):
        """Test that intervention indicators are correctly set."""
        avici_data = samples_to_avici_format(
            samples=mixed_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        intervention_channel = avici_data[:, :, 1]
        
        # Sample 0: observational (no interventions)
        assert jnp.allclose(intervention_channel[0], jnp.array([0.0, 0.0, 0.0]))
        
        # Sample 1: intervention on X
        assert jnp.allclose(intervention_channel[1], jnp.array([1.0, 0.0, 0.0]))
        
        # Sample 2: intervention on Z  
        assert jnp.allclose(intervention_channel[2], jnp.array([0.0, 0.0, 1.0]))
        
        # Sample 3: interventions on X and Z
        assert jnp.allclose(intervention_channel[3], jnp.array([1.0, 0.0, 1.0]))


class TestTargetConditioning:
    """Test target conditioning functionality."""
    
    @pytest.mark.parametrize("target_var,expected_idx", [
        ('X', 0),
        ('Y', 1), 
        ('Z', 2)
    ])
    def test_target_conditioning(self, simple_observational_samples, variable_order, target_var, expected_idx):
        """Test target conditioning for different target variables."""
        avici_data = samples_to_avici_format(
            samples=simple_observational_samples,
            variable_order=variable_order,
            target_variable=target_var,
            standardize=False
        )
        
        target_channel = avici_data[:, :, 2]
        
        # Create expected target indicators
        expected = jnp.zeros((3, 3))
        expected = expected.at[:, expected_idx].set(1.0)
        
        assert jnp.allclose(target_channel, expected)


class TestStandardization:
    """Test data standardization functionality."""
    
    def test_standardization_effect(self, variable_order):
        """Test that standardization normalizes data appropriately."""
        # Create samples with known values for easy testing
        samples = [
            create_observational_sample({'X': 0.0, 'Y': 10.0, 'Z': 100.0}),
            create_observational_sample({'X': 2.0, 'Y': 20.0, 'Z': 200.0}),
            create_observational_sample({'X': 4.0, 'Y': 30.0, 'Z': 300.0})
        ]
        
        # Test without standardization
        avici_data_raw = samples_to_avici_format(samples, variable_order, 'Y', standardize=False)
        values_raw = avici_data_raw[:, :, 0]
        
        # Test with standardization
        avici_data_std = samples_to_avici_format(samples, variable_order, 'Y', standardize=True)
        values_std = avici_data_std[:, :, 0]
        
        # Check that standardized values have approximately zero mean and unit variance
        for j in range(3):  # For each variable
            var_values = values_std[:, j]
            assert abs(jnp.mean(var_values)) < 1e-6
            assert abs(jnp.std(var_values) - 1.0) < 1e-6
        
        # Check that raw values are unchanged
        expected_raw = jnp.array([
            [0.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 30.0, 300.0]
        ])
        assert jnp.allclose(values_raw, expected_raw)


class TestValidation:
    """Test data conversion validation."""
    
    def test_valid_conversion(self, mixed_samples, variable_order):
        """Test that validation passes for correct conversion."""
        avici_data = samples_to_avici_format(
            samples=mixed_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        is_valid = validate_data_conversion(
            original_samples=mixed_samples,
            converted_data=avici_data,
            variable_order=variable_order,
            target_variable='Y'
        )
        
        assert is_valid
    
    def test_corrupted_data_detection(self, mixed_samples, variable_order):
        """Test that validation detects corrupted data."""
        avici_data = samples_to_avici_format(
            samples=mixed_samples,
            variable_order=variable_order,
            target_variable='Y',
            standardize=False
        )
        
        # Corrupt target indicator
        corrupted_data = avici_data.at[0, 1, 2].set(0.5)  # Should be 1.0 for target
        
        is_valid = validate_data_conversion(
            original_samples=mixed_samples,
            converted_data=corrupted_data,
            variable_order=variable_order,
            target_variable='Y'
        )
        
        assert not is_valid


class TestTrainingBatch:
    """Test training batch creation."""
    
    def test_batch_structure(self, test_scm):
        """Test that training batch has correct structure."""
        samples = sample_from_linear_scm(test_scm, n_samples=20, seed=42)
        
        batch = create_training_batch(test_scm, samples, target_variable="Y")
        
        # Check required keys
        required_keys = ['x', 'g', 'is_count_data', 'target_variable', 'variable_order']
        for key in required_keys:
            assert key in batch
    
    def test_batch_shapes(self, test_scm):
        """Test that batch tensors have correct shapes."""
        samples = sample_from_linear_scm(test_scm, n_samples=20, seed=42)
        
        batch = create_training_batch(test_scm, samples, target_variable="Y")
        
        x_data = batch['x']
        g_matrix = batch['g']
        
        assert x_data.shape == (20, 3, 3)  # n_samples, n_vars, n_channels
        assert g_matrix.shape == (3, 3)    # n_vars x n_vars adjacency matrix
    
    def test_ground_truth_adjacency(self, test_scm):
        """Test that ground truth adjacency matrix is correct."""
        samples = sample_from_linear_scm(test_scm, n_samples=10, seed=42)
        
        batch = create_training_batch(test_scm, samples, target_variable="Y")
        
        g_matrix = batch['g']
        variable_order = batch['variable_order']
        
        x_idx = variable_order.index('X')
        y_idx = variable_order.index('Y')
        z_idx = variable_order.index('Z')
        
        # For X → Y ← Z structure
        assert g_matrix[x_idx, y_idx] == 1.0  # X → Y
        assert g_matrix[z_idx, y_idx] == 1.0  # Z → Y
        assert g_matrix[y_idx, x_idx] == 0.0  # Not Y → X
        
        # No self-loops
        for i in range(3):
            assert g_matrix[i, i] == 0.0


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_empty_variable_order(self):
        """Test error for empty variable order."""
        samples = [create_observational_sample({'X': 1.0, 'Y': 2.0})]
        
        with pytest.raises(ValueError, match="Variable order cannot be empty"):
            samples_to_avici_format(samples, [], 'Y')
    
    def test_invalid_target_variable(self, simple_observational_samples, variable_order):
        """Test error for invalid target variable."""
        with pytest.raises(ValueError, match="not in variable_order"):
            samples_to_avici_format(simple_observational_samples, variable_order, 'W')
    
    def test_empty_samples(self, variable_order):
        """Test error for empty samples."""
        with pytest.raises(ValueError, match="Samples list cannot be empty"):
            samples_to_avici_format([], variable_order, 'Y')
    
    def test_missing_variables_in_samples(self, variable_order):
        """Test error for samples missing variables."""
        samples_missing = [create_observational_sample({'X': 1.0, 'Y': 2.0})]  # Missing Z
        
        with pytest.raises(ValueError, match="missing variables"):
            samples_to_avici_format(samples_missing, variable_order, 'Y')


class TestIntegrationWithSCM:
    """Test integration with existing SCM structures."""
    
    def test_scm_integration(self, test_scm):
        """Test integration with SCM structures and sampling."""
        # Generate samples using existing sampling function
        samples = sample_from_linear_scm(test_scm, n_samples=50, seed=999)
        
        # Convert to AVICI format
        variable_order = sorted(test_scm['variables'])
        avici_data = samples_to_avici_format(samples, variable_order, test_scm['target'])
        
        # Basic shape checks
        assert avici_data.shape == (50, 3, 3)
        
        # Validate conversion
        is_valid = validate_data_conversion(samples, avici_data, variable_order, test_scm['target'])
        assert is_valid
        
        # Create training batch
        batch = create_training_batch(test_scm, samples, test_scm['target'])
        assert batch['x'].shape == (50, 3, 3)
        assert batch['target_variable'] == test_scm['target']
    
    def test_data_analysis(self, test_scm):
        """Test data analysis functionality."""
        samples = sample_from_linear_scm(test_scm, n_samples=30, seed=42)
        variable_order = sorted(test_scm['variables'])
        
        avici_data = samples_to_avici_format(samples, variable_order, test_scm['target'])
        
        analysis = analyze_avici_data(avici_data, variable_order)
        
        assert analysis["structure"]['n_samples'] == 30
        assert analysis["structure"]['n_variables'] == 3
        assert analysis["structure"]['variable_order'] == variable_order
        assert analysis['targets']['target_variable'] == test_scm['target']
        
        # For observational data, no interventions
        assert analysis['interventions']['total_interventions'] == 0
        assert analysis['interventions']['samples_with_interventions'] == 0


@pytest.mark.integration
class TestPhase12SuccessCriteria:
    """Integration tests for Phase 1.2 success criteria."""
    
    def test_all_success_criteria(self, test_scm):
        """Test all Phase 1.2 success criteria together."""
        # Create comprehensive test data
        samples = sample_from_linear_scm(test_scm, n_samples=100, seed=42)
        
        # Add some interventional samples
        interventional_samples = [
            create_interventional_sample(
                values={'X': 5.0, 'Y': 2.0, 'Z': 3.0},
                intervention_type='perfect',
                targets=frozenset(['X'])
            ),
            create_interventional_sample(
                values={'X': 1.0, 'Y': 2.0, 'Z': 10.0},
                intervention_type='perfect',
                targets=frozenset(['Z'])
            ),
        ]
        all_samples = samples + interventional_samples
        
        variable_order = sorted(test_scm['variables'])
        target_variable = test_scm['target']
        
        # Criterion 1: Can convert SCM + samples to AVICI format [N, d, 3]
        avici_data = samples_to_avici_format(all_samples, variable_order, target_variable)
        expected_shape = (102, 3, 3)  # 100 obs + 2 interventional, 3 vars, 3 channels
        assert avici_data.shape == expected_shape
        
        # Criterion 2: Data conversion preserves all information
        is_valid = validate_data_conversion(all_samples, avici_data, variable_order, target_variable)
        assert is_valid
        
        # Criterion 3: Format compatible with modified AVICI input
        assert avici_data.dtype == jnp.float32
        assert len(avici_data.shape) == 3
        assert avici_data.shape[2] == 3
        
        # Criterion 4: Target conditioning channel correctly populated
        target_channel = avici_data[:, :, 2]
        target_idx = variable_order.index(target_variable)
        
        # Check target indicators for a few samples
        for i in [0, 50, 101]:  # Check beginning, middle, end
            for j in range(len(variable_order)):
                expected = 1.0 if j == target_idx else 0.0
                assert abs(target_channel[i, j] - expected) < 1e-6
        
        # Criterion 5: Validation framework ensures information preservation
        # Test with corrupted data - validation should fail
        corrupted_data = avici_data.at[0, 0, 1].set(0.5)  # Corrupt intervention indicator
        is_valid_corrupted = validate_data_conversion(all_samples, corrupted_data, variable_order, target_variable)
        assert not is_valid_corrupted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])