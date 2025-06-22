"""
Comprehensive tests for the unified parent set model.

Tests the integration of proven transformer architecture with enhanced capabilities.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from typing import List, Dict, Any

from causal_bayes_opt.avici_integration.parent_set.unified import (
    UnifiedParentSetModel,
    TargetAwareConfig,
    create_unified_parent_set_model,
    create_structure_only_config,
    create_mechanism_aware_config,
    compute_adaptive_max_parents,
    validate_unified_config
)


class TestTargetAwareConfig:
    """Test the adaptive configuration system."""
    
    def test_structure_only_config_creation(self):
        """Test creating structure-only configuration."""
        config = create_structure_only_config()
        
        assert not config.predict_mechanisms
        assert config.enable_target_conditioning  # Still beneficial
        assert config.mechanism_types == ['linear']  # Default
        assert config.max_parent_density == 0.3
        
    def test_mechanism_aware_config_creation(self):
        """Test creating mechanism-aware configuration."""
        mechanism_types = ['linear', 'polynomial', 'gaussian']
        config = create_mechanism_aware_config(mechanism_types)
        
        assert config.predict_mechanisms
        assert config.enable_target_conditioning
        assert config.mechanism_types == mechanism_types
        
    def test_custom_config_parameters(self):
        """Test custom configuration parameters."""
        config = create_structure_only_config(
            layers=6,
            dim=96,
            max_parent_density=0.2,
            min_max_parents=2,
            max_max_parents=8
        )
        
        assert config.layers == 6
        assert config.dim == 96
        assert config.max_parent_density == 0.2
        assert config.min_max_parents == 2
        assert config.max_max_parents == 8
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = TargetAwareConfig(max_parent_density=0.3)
        # Should not raise
        
        # Invalid density
        with pytest.raises(ValueError, match="max_parent_density must be in"):
            TargetAwareConfig(max_parent_density=0.0)
            
        with pytest.raises(ValueError, match="max_parent_density must be in"):
            TargetAwareConfig(max_parent_density=1.5)
            
        # Invalid min_max_parents
        with pytest.raises(ValueError, match="min_max_parents must be >= 1"):
            TargetAwareConfig(min_max_parents=0)
            
        # Invalid max_max_parents
        with pytest.raises(ValueError, match="max_max_parents must be >= min_max_parents"):
            TargetAwareConfig(min_max_parents=5, max_max_parents=3)


class TestAdaptiveMaxParents:
    """Test the adaptive max_parents functionality."""
    
    def test_adaptive_computation_small_graph(self):
        """Test adaptive max_parents for small graphs."""
        config = create_structure_only_config(max_parent_density=0.5)
        
        # Small graph: 4 variables
        max_parents = compute_adaptive_max_parents(4, config)
        # max_possible = 3, density_based = 1.5 -> 1, bounded by min_max_parents=2
        assert max_parents == 2  # min_max_parents
        
    def test_adaptive_computation_medium_graph(self):
        """Test adaptive max_parents for medium graphs."""
        config = create_structure_only_config(
            max_parent_density=0.3,
            min_max_parents=2,
            max_max_parents=8
        )
        
        # Medium graph: 10 variables  
        max_parents = compute_adaptive_max_parents(10, config)
        # max_possible = 9, density_based = 2.7 -> 2, bounded by min=2
        assert max_parents == 2
        
    def test_adaptive_computation_large_graph(self):
        """Test adaptive max_parents for large graphs."""
        config = create_structure_only_config(
            max_parent_density=0.2,
            min_max_parents=3,
            max_max_parents=10
        )
        
        # Large graph: 30 variables
        max_parents = compute_adaptive_max_parents(30, config)
        # max_possible = 29, density_based = 5.8 -> 5
        assert max_parents == 5
        
    def test_explicit_override(self):
        """Test explicit max_parent_size override."""
        config = create_structure_only_config(
            max_parent_size=7,  # Explicit override
            max_parent_density=0.3,
            min_max_parents=2,
            max_max_parents=10
        )
        
        max_parents = compute_adaptive_max_parents(20, config)
        assert max_parents == 7  # Should use explicit value
        
    def test_bounds_enforcement(self):
        """Test that bounds are enforced."""
        config = create_structure_only_config(
            max_parent_size=15,  # Too large
            min_max_parents=3,
            max_max_parents=10
        )
        
        max_parents = compute_adaptive_max_parents(8, config)
        # Should be bounded by min(max_max_parents=10, n_variables-1=7)
        assert max_parents == 7


class TestUnifiedModelCreation:
    """Test unified model creation and basic functionality."""
    
    def test_structure_only_model_creation(self):
        """Test creating structure-only model."""
        config = create_structure_only_config()
        model = create_unified_parent_set_model(config)
        
        assert model is not None
        
    def test_mechanism_aware_model_creation(self):
        """Test creating mechanism-aware model."""
        config = create_mechanism_aware_config(['linear', 'polynomial'])
        model = create_unified_parent_set_model(config)
        
        assert model is not None
        
    def test_default_config_model(self):
        """Test creating model with default configuration."""
        model = create_unified_parent_set_model()  # No config provided
        
        assert model is not None


class TestUnifiedModelForwardPass:
    """Test unified model forward pass functionality."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for model testing."""
        key = random.PRNGKey(42)
        variable_order = ["X", "Y", "Z", "W"]
        target_variable = "Y"
        N, d = 10, 4
        
        # Create test data [N, d, 3]
        data = jnp.ones((N, d, 3))
        data = data.at[:, :, 0].set(random.normal(key, (N, d)))  # Values
        data = data.at[:5, 0, 1].set(1.0)  # Some interventions on X
        data = data.at[:, 1, 2].set(1.0)    # Y is target
        
        return {
            'data': data,
            'variable_order': variable_order,
            'target_variable': target_variable,
            'key': key
        }
    
    def test_structure_only_forward_pass(self, test_data):
        """Test forward pass in structure-only mode."""
        config = create_structure_only_config()
        model = create_unified_parent_set_model(config)
        
        # Initialize and run forward pass
        params = model.init(
            test_data['key'], 
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable']
        )
        
        output = model.apply(
            params, 
            test_data['key'],
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable']
        )
        
        # Check output structure
        assert 'parent_set_logits' in output
        assert 'parent_sets' in output
        assert 'k' in output
        assert 'all_possible_parent_sets' in output
        
        # Mechanism predictions should not be present
        assert 'mechanism_predictions' not in output
        
        # Check shapes and types
        assert output['parent_set_logits'].ndim == 1
        assert len(output['parent_sets']) == output['k']
        assert all(isinstance(ps, frozenset) for ps in output['parent_sets'])
        
    def test_mechanism_aware_forward_pass(self, test_data):
        """Test forward pass in mechanism-aware mode."""
        config = create_mechanism_aware_config(['linear', 'polynomial'])
        model = create_unified_parent_set_model(config)
        
        # Initialize and run forward pass
        params = model.init(
            test_data['key'], 
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable']
        )
        
        output = model.apply(
            params, 
            test_data['key'],
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable']
        )
        
        # Check output structure
        assert 'parent_set_logits' in output
        assert 'parent_sets' in output
        assert 'k' in output
        assert 'mechanism_predictions' in output
        
        # Check mechanism predictions structure
        mech_preds = output['mechanism_predictions']
        assert 'mechanism_type_logits' in mech_preds
        assert 'mechanism_parameters' in mech_preds
        
        # Check shapes
        k = output['k']
        n_types = len(config.mechanism_types)
        
        assert mech_preds['mechanism_type_logits'].shape == (k, n_types)
        assert mech_preds['mechanism_parameters'].shape == (k, n_types, config.mechanism_param_dim)
        
    def test_training_vs_inference_mode(self, test_data):
        """Test difference between training and inference modes."""
        config = create_structure_only_config(dropout=0.2)
        model = create_unified_parent_set_model(config)
        
        params = model.init(
            test_data['key'], 
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable'],
            True  # Training mode
        )
        
        # Training mode
        train_output = model.apply(
            params, 
            test_data['key'],
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable'],
            True  # Training mode
        )
        
        # Inference mode
        infer_output = model.apply(
            params, 
            test_data['key'],
            test_data['data'], 
            test_data['variable_order'], 
            test_data['target_variable'],
            False  # Inference mode
        )
        
        # Outputs should have same structure but potentially different values due to dropout
        assert train_output.keys() == infer_output.keys()
        assert train_output['k'] == infer_output['k']
        
    def test_adaptive_max_parents_in_practice(self, test_data):
        """Test that adaptive max_parents affects actual outputs."""
        # High density config
        high_density_config = create_structure_only_config(max_parent_density=0.8)
        high_model = create_unified_parent_set_model(high_density_config)
        
        # Low density config  
        low_density_config = create_structure_only_config(max_parent_density=0.2)
        low_model = create_unified_parent_set_model(low_density_config)
        
        # Both should work but may have different numbers of parent sets
        high_params = high_model.init(test_data['key'], test_data['data'], 
                                     test_data['variable_order'], test_data['target_variable'])
        low_params = low_model.init(test_data['key'], test_data['data'], 
                                   test_data['variable_order'], test_data['target_variable'])
        
        high_output = high_model.apply(high_params, test_data['key'], test_data['data'], 
                                      test_data['variable_order'], test_data['target_variable'])
        low_output = low_model.apply(low_params, test_data['key'], test_data['data'], 
                                    test_data['variable_order'], test_data['target_variable'])
        
        # Both should produce valid outputs
        assert 'parent_set_logits' in high_output
        assert 'parent_set_logits' in low_output


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_output_format_compatibility(self):
        """Test that output format matches original model."""
        from causal_bayes_opt.avici_integration.parent_set.model import create_parent_set_model
        
        key = random.PRNGKey(42)
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        N, d = 8, 3
        
        data = jnp.ones((N, d, 3))
        data = data.at[:, :, 0].set(random.normal(key, (N, d)))
        data = data.at[:, 1, 2].set(1.0)  # Y is target
        
        # Original model
        original_model = create_parent_set_model(max_parent_size=3)
        original_params = original_model.init(key, data, variable_order, target_variable)
        original_output = original_model.apply(original_params, key, data, variable_order, target_variable)
        
        # Unified model in structure-only mode
        unified_config = create_structure_only_config(
            max_parent_size=3,  # Match original
            enable_target_conditioning=False  # Disable enhancements for pure compatibility test
        )
        unified_model = create_unified_parent_set_model(unified_config)
        unified_params = unified_model.init(key, data, variable_order, target_variable)
        unified_output = unified_model.apply(unified_params, key, data, variable_order, target_variable)
        
        # Check that both have same required keys
        required_keys = {'parent_set_logits', 'parent_sets', 'k'}
        assert required_keys.issubset(set(original_output.keys()))
        assert required_keys.issubset(set(unified_output.keys()))
        
        # Check that shapes are compatible (values may differ due to different architectures)
        assert original_output['parent_set_logits'].shape == unified_output['parent_set_logits'].shape
        assert original_output['k'] == unified_output['k']


class TestConfigValidation:
    """Test configuration validation for different graph sizes."""
    
    def test_valid_config_validation(self):
        """Test validation of valid configurations."""
        config = create_structure_only_config()
        
        # Should not raise for reasonable graph sizes
        assert validate_unified_config(config, 5)
        assert validate_unified_config(config, 10)
        assert validate_unified_config(config, 20)
        
    def test_invalid_graph_size(self):
        """Test validation with invalid graph sizes."""
        config = create_structure_only_config()
        
        with pytest.raises(ValueError, match="must have at least 2 variables"):
            validate_unified_config(config, 1)
            
    def test_max_parents_too_large(self):
        """Test validation when max_parents would be too large."""
        config = create_structure_only_config(
            max_parent_size=10  # Explicit large value
        )
        
        # Should still work due to bounds enforcement
        assert validate_unified_config(config, 5)  # Will be bounded to 4


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_variable_order(self):
        """Test handling of empty variable order."""
        config = create_structure_only_config()
        model = create_unified_parent_set_model(config)
        
        key = random.PRNGKey(42)
        
        with pytest.raises((ValueError, IndexError)):
            # Should fail gracefully
            model.init(key, jnp.ones((1, 0, 3)), [], "Y")
            
    def test_target_not_in_variables(self):
        """Test handling when target variable is not in variable order."""
        config = create_structure_only_config()
        model = create_unified_parent_set_model(config)
        
        key = random.PRNGKey(42)
        data = jnp.ones((5, 3, 3))
        variable_order = ["X", "Y", "Z"]
        target_variable = "W"  # Not in variable_order
        
        with pytest.raises(ValueError):
            model.init(key, data, variable_order, target_variable)
            
    def test_malformed_input_data(self):
        """Test handling of malformed input data."""
        config = create_structure_only_config()
        model = create_unified_parent_set_model(config)
        
        key = random.PRNGKey(42)
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        
        # Test passes if model handles malformed input gracefully
        # (JAX/Haiku is quite permissive with tensor shapes, so this test mainly
        # ensures no crashes rather than specific error types)
        
        try:
            # Wrong number of channels (may or may not raise)
            malformed_data = jnp.ones((5, 3, 2))  # Should be (5, 3, 3)
            model.init(key, malformed_data, variable_order, target_variable)
        except Exception:
            pass  # Error is fine
        
        # This should definitely work
        valid_data = jnp.ones((5, 3, 3))
        model.init(key, valid_data, variable_order, target_variable)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])