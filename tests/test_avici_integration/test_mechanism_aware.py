#!/usr/bin/env python3
"""
Test suite for mechanism-aware parent set prediction models.

Tests the ModularParentSetModel with configurable mechanism prediction capabilities,
ensuring backward compatibility with structure-only mode and proper functionality
in enhanced mechanism-aware mode.

Following TDD approach as outlined in the architecture enhancement pivot.
"""

import pytest
from typing import List, Dict, Any, FrozenSet
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import pyrsistent as pyr

from src.causal_bayes_opt.avici_integration.parent_set.mechanism_aware import (
    ModularParentSetModel,
    MechanismAwareConfig,
    MechanismPrediction,
    create_modular_parent_set_model,
    predict_with_mechanisms,
    MechanismType
)
from src.causal_bayes_opt.avici_integration.parent_set.posterior import (
    ParentSetPosterior,
    create_parent_set_posterior
)


@dataclass(frozen=True)
class MechanismTestCase:
    """Test case for mechanism-aware prediction."""
    name: str
    n_vars: int
    max_parents: int
    mechanism_types: List[str]
    predict_mechanisms: bool
    expected_structure_shape: tuple
    expected_mechanism_keys: List[str]


class TestMechanismAwareConfig:
    """Test mechanism-aware configuration system."""
    
    def test_default_config_structure_only(self):
        """Test default config is structure-only mode."""
        config = MechanismAwareConfig()
        
        assert config.predict_mechanisms is False
        assert config.mechanism_types == ["linear"]
        assert config.max_parents == 5
        assert config.hidden_dim == 128
        assert config.n_layers == 8
    
    def test_enhanced_config_with_mechanisms(self):
        """Test enhanced config with mechanism prediction."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear", "polynomial", "gaussian"],
            max_parents=3,
            hidden_dim=256
        )
        
        assert config.predict_mechanisms is True
        assert config.mechanism_types == ["linear", "polynomial", "gaussian"]
        assert config.max_parents == 3
        assert config.hidden_dim == 256
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations should not raise
        MechanismAwareConfig(mechanism_types=["linear"])
        MechanismAwareConfig(mechanism_types=["linear", "polynomial", "gaussian", "neural"])
        
        # Invalid configurations should raise
        with pytest.raises(ValueError, match="mechanism_types cannot be empty"):
            MechanismAwareConfig(mechanism_types=[])
        
        with pytest.raises(ValueError, match="Unknown mechanism type"):
            MechanismAwareConfig(mechanism_types=["invalid_type"])
        
        with pytest.raises(ValueError, match="max_parents must be positive"):
            MechanismAwareConfig(max_parents=0)


class TestMechanismType:
    """Test mechanism type enumeration and utilities."""
    
    def test_mechanism_type_enum(self):
        """Test mechanism type enumeration."""
        assert MechanismType.LINEAR == "linear"
        assert MechanismType.POLYNOMIAL == "polynomial"
        assert MechanismType.GAUSSIAN == "gaussian"
        assert MechanismType.NEURAL == "neural"
    
    def test_get_all_mechanism_types(self):
        """Test getting all supported mechanism types."""
        from src.causal_bayes_opt.avici_integration.parent_set.mechanism_aware import get_all_mechanism_types
        
        all_types = get_all_mechanism_types()
        expected = ["linear", "polynomial", "gaussian", "neural"]
        
        assert all_types == expected
    
    def test_validate_mechanism_types(self):
        """Test mechanism type validation."""
        from src.causal_bayes_opt.avici_integration.parent_set.mechanism_aware import validate_mechanism_types
        
        # Valid types
        assert validate_mechanism_types(["linear"]) is True
        assert validate_mechanism_types(["linear", "polynomial"]) is True
        assert validate_mechanism_types(["linear", "polynomial", "gaussian", "neural"]) is True
        
        # Invalid types
        assert validate_mechanism_types([]) is False
        assert validate_mechanism_types(["invalid"]) is False
        assert validate_mechanism_types(["linear", "invalid"]) is False


class TestMechanismPrediction:
    """Test mechanism prediction data structure."""
    
    def test_mechanism_prediction_creation(self):
        """Test creating mechanism prediction objects."""
        pred = MechanismPrediction(
            parent_set=frozenset({"X", "Y"}),
            mechanism_type="linear",
            parameters={"coefficients": {"X": 0.5, "Y": -0.3}, "intercept": 1.0},
            confidence=0.85
        )
        
        assert pred.parent_set == frozenset({"X", "Y"})
        assert pred.mechanism_type == "linear"
        assert pred.parameters["coefficients"]["X"] == 0.5
        assert pred.confidence == 0.85
    
    def test_mechanism_prediction_immutable(self):
        """Test that mechanism prediction is immutable."""
        pred = MechanismPrediction(
            parent_set=frozenset({"X"}),
            mechanism_type="linear",
            parameters={"intercept": 0.0},
            confidence=0.9
        )
        
        # Should not be able to modify
        with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
            pred.confidence = 0.8


class TestModularParentSetModel:
    """Test modular parent set model with configurable mechanism prediction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
        self.n_vars = 4
        self.batch_size = 8
        
        # Create test data in AVICI format [N, d, 3]
        self.test_data = jnp.ones((self.batch_size, self.n_vars, 3))
        self.variable_order = ["X", "Y", "Z", "W"]
        self.target_variable = "Y"
        self.target_idx = 1
    
    def test_structure_only_mode(self):
        """Test structure-only mode (backward compatibility)."""
        config = MechanismAwareConfig(predict_mechanisms=False)
        
        def model_fn(x, variable_order, target_variable):
            model = ModularParentSetModel(config)
            return model(x, variable_order, target_variable)
        
        net = hk.transform(model_fn)
        params = net.init(self.key, self.test_data, self.variable_order, self.target_variable)
        
        # Forward pass
        output = net.apply(params, self.key, self.test_data, self.variable_order, self.target_variable)
        
        # Structure-only mode should only return parent set logits
        assert "parent_set_logits" in output
        assert "mechanism_predictions" not in output
        
        # Check output shape [k] where k is number of possible parent sets
        parent_set_logits = output["parent_set_logits"]
        assert len(parent_set_logits.shape) == 1
        assert parent_set_logits.shape[0] > 0  # Should have some parent sets
    
    def test_mechanism_aware_mode(self):
        """Test enhanced mechanism-aware mode."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear", "polynomial"],
            max_parents=3
        )
        
        def model_fn(x, variable_order, target_variable):
            model = ModularParentSetModel(config)
            return model(x, variable_order, target_variable)
        
        net = hk.transform(model_fn)
        params = net.init(self.key, self.test_data, self.variable_order, self.target_variable)
        
        # Forward pass
        output = net.apply(params, self.key, self.test_data, self.variable_order, self.target_variable)
        
        # Enhanced mode should return both structure and mechanism predictions
        assert "parent_set_logits" in output
        assert "mechanism_predictions" in output
        
        # Check mechanism predictions structure
        mechanism_preds = output["mechanism_predictions"]
        assert "mechanism_type_logits" in mechanism_preds
        assert "mechanism_parameters" in mechanism_preds
        
        # Shapes should match number of parent sets
        k = output["parent_set_logits"].shape[0]
        assert mechanism_preds["mechanism_type_logits"].shape[0] == k
        assert mechanism_preds["mechanism_parameters"].shape[0] == k
    
    def test_feature_flag_switching(self):
        """Test switching between modes via feature flag."""
        # Test data
        data = self.test_data
        var_order = self.variable_order
        target = self.target_variable
        
        # Structure-only model
        config_struct = MechanismAwareConfig(predict_mechanisms=False)
        
        def struct_model_fn(x, variable_order, target_variable):
            model = ModularParentSetModel(config_struct)
            return model(x, variable_order, target_variable)
        
        struct_net = hk.transform(struct_model_fn)
        struct_params = struct_net.init(self.key, data, var_order, target)
        struct_output = struct_net.apply(struct_params, self.key, data, var_order, target)
        
        # Enhanced model  
        config_enhanced = MechanismAwareConfig(predict_mechanisms=True)
        
        def enhanced_model_fn(x, variable_order, target_variable):
            model = ModularParentSetModel(config_enhanced)
            return model(x, variable_order, target_variable)
        
        enhanced_net = hk.transform(enhanced_model_fn)
        enhanced_params = enhanced_net.init(self.key, data, var_order, target)
        enhanced_output = enhanced_net.apply(enhanced_params, self.key, data, var_order, target)
        
        # Structure predictions should exist in both
        assert "parent_set_logits" in struct_output
        assert "parent_set_logits" in enhanced_output
        
        # Mechanism predictions should only exist in enhanced
        assert "mechanism_predictions" not in struct_output
        assert "mechanism_predictions" in enhanced_output
    
    def test_different_mechanism_types(self):
        """Test model with different mechanism type configurations."""
        test_cases = [
            (["linear"], 1),
            (["linear", "polynomial"], 2),
            (["linear", "polynomial", "gaussian"], 3),
            (["linear", "polynomial", "gaussian", "neural"], 4)
        ]
        
        for mechanism_types, expected_n_types in test_cases:
            config = MechanismAwareConfig(
                predict_mechanisms=True,
                mechanism_types=mechanism_types
            )
            
            def model_fn(x, variable_order, target_variable):
                model = ModularParentSetModel(config)
                return model(x, variable_order, target_variable)
            
            net = hk.transform(model_fn)
            params = net.init(self.key, self.test_data, self.variable_order, self.target_variable)
            output = net.apply(params, self.key, self.test_data, self.variable_order, self.target_variable)
            
            # Check mechanism type predictions
            mech_preds = output["mechanism_predictions"]
            mechanism_type_logits = mech_preds["mechanism_type_logits"]
            
            # Should have logits for each mechanism type for each parent set
            k = output["parent_set_logits"].shape[0]
            assert mechanism_type_logits.shape == (k, expected_n_types)
    
    def test_parameter_dimensions(self):
        """Test mechanism parameter prediction dimensions."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear", "polynomial"],
            max_parents=2
        )
        
        def model_fn(x, variable_order, target_variable):
            model = ModularParentSetModel(config)
            return model(x, variable_order, target_variable)
        
        net = hk.transform(model_fn)
        params = net.init(self.key, self.test_data, self.variable_order, self.target_variable)
        output = net.apply(params, self.key, self.test_data, self.variable_order, self.target_variable)
        
        # Check parameter predictions
        mech_preds = output["mechanism_predictions"]
        mech_params = mech_preds["mechanism_parameters"]
        
        k = output["parent_set_logits"].shape[0]
        
        # Should predict parameters for each parent set and mechanism type
        # Shape: [k, n_mechanism_types, param_dim]
        # Where param_dim depends on max_parents and mechanism complexity
        assert len(mech_params.shape) == 3
        assert mech_params.shape[0] == k
        assert mech_params.shape[1] == 2  # linear + polynomial
        assert mech_params.shape[2] > 0  # Some parameter dimension


class TestFactoryFunctions:
    """Test factory functions for creating mechanism-aware models."""
    
    def test_create_modular_parent_set_model_structure_only(self):
        """Test factory function for structure-only model."""
        config = MechanismAwareConfig(predict_mechanisms=False)
        
        net = create_modular_parent_set_model(config)
        
        # Should create a valid Haiku transformed function
        assert hasattr(net, 'init')
        assert hasattr(net, 'apply')
        
        # Test initialization
        key = random.PRNGKey(42)
        data = jnp.ones((5, 3, 3))
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        
        params = net.init(key, data, variable_order, target_variable)
        output = net.apply(params, key, data, variable_order, target_variable)
        
        assert "parent_set_logits" in output
        assert "mechanism_predictions" not in output
    
    def test_create_modular_parent_set_model_enhanced(self):
        """Test factory function for enhanced model."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear", "polynomial"]
        )
        
        net = create_modular_parent_set_model(config)
        
        # Test initialization and forward pass
        key = random.PRNGKey(42)
        data = jnp.ones((5, 3, 3))
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        
        params = net.init(key, data, variable_order, target_variable)
        output = net.apply(params, key, data, variable_order, target_variable)
        
        assert "parent_set_logits" in output
        assert "mechanism_predictions" in output


class TestPredictWithMechanisms:
    """Test high-level prediction function with mechanism awareness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
        self.data = jnp.ones((10, 4, 3))
        self.variable_order = ["X", "Y", "Z", "W"]
        self.target_variable = "Y"
    
    def test_predict_with_mechanisms_structure_only(self):
        """Test prediction function in structure-only mode."""
        config = MechanismAwareConfig(predict_mechanisms=False)
        net = create_modular_parent_set_model(config)
        params = net.init(self.key, self.data, self.variable_order, self.target_variable)
        
        posterior = predict_with_mechanisms(
            net, params, self.data, self.variable_order, self.target_variable
        )
        
        # Should return standard ParentSetPosterior
        assert isinstance(posterior, ParentSetPosterior)
        assert posterior.target_variable == self.target_variable
        assert len(posterior.parent_set_probs) > 0
        assert posterior.uncertainty >= 0.0
        
        # Should not have mechanism predictions in metadata
        metadata = posterior.metadata
        assert "mechanism_predictions" not in metadata
    
    def test_predict_with_mechanisms_enhanced(self):
        """Test prediction function in enhanced mode."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear", "polynomial"]
        )
        net = create_modular_parent_set_model(config)
        params = net.init(self.key, self.data, self.variable_order, self.target_variable)
        
        posterior = predict_with_mechanisms(
            net, params, self.data, self.variable_order, self.target_variable, config
        )
        
        # Should return enhanced ParentSetPosterior with mechanism info
        assert isinstance(posterior, ParentSetPosterior)
        assert posterior.target_variable == self.target_variable
        
        # Should have mechanism predictions in metadata
        metadata = posterior.metadata
        assert "mechanism_predictions" in metadata
        
        mechanism_preds = metadata["mechanism_predictions"]
        assert isinstance(mechanism_preds, list)
        assert len(mechanism_preds) > 0
        
        # Each prediction should be a MechanismPrediction
        for pred in mechanism_preds:
            assert isinstance(pred, MechanismPrediction)
            assert pred.mechanism_type in ["linear", "polynomial"]
            assert isinstance(pred.parent_set, frozenset)
            assert 0.0 <= pred.confidence <= 1.0
    
    def test_mechanism_parameter_extraction(self):
        """Test extraction of mechanism parameters from model output."""
        config = MechanismAwareConfig(
            predict_mechanisms=True,
            mechanism_types=["linear"],
            max_parents=2
        )
        net = create_modular_parent_set_model(config)
        params = net.init(self.key, self.data, self.variable_order, self.target_variable)
        
        posterior = predict_with_mechanisms(
            net, params, self.data, self.variable_order, self.target_variable, config
        )
        
        # Check mechanism predictions
        mechanism_preds = posterior.metadata["mechanism_predictions"]
        
        for pred in mechanism_preds:
            assert pred.mechanism_type == "linear"
            
            # Linear mechanism should have meaningful parameters
            assert "parameters" in pred.parameters or len(pred.parameters) > 0
            
            # Confidence should be reasonable
            assert 0.0 <= pred.confidence <= 1.0


class TestIntegrationWithExistingSystem:
    """Test integration with existing parent set prediction infrastructure."""
    
    def test_backward_compatibility_with_existing_api(self):
        """Test that structure-only mode is compatible with existing API."""
        # This test ensures that existing code continues to work
        config = MechanismAwareConfig(predict_mechanisms=False)
        net = create_modular_parent_set_model(config)
        
        key = random.PRNGKey(42)
        data = jnp.ones((8, 3, 3))
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        
        # Should work with existing prediction pipeline
        params = net.init(key, data, variable_order, target_variable)
        posterior = predict_with_mechanisms(net, params, data, variable_order, target_variable)
        
        # Should produce standard posterior format
        assert isinstance(posterior, ParentSetPosterior)
        
        # Should be compatible with existing analysis functions
        from src.causal_bayes_opt.avici_integration.parent_set.posterior import (
            get_most_likely_parents,
            get_marginal_parent_probabilities,
            summarize_posterior
        )
        
        # These functions should work without modification
        most_likely = get_most_likely_parents(posterior)
        marginals = get_marginal_parent_probabilities(posterior, variable_order)
        summary = summarize_posterior(posterior)
        
        assert isinstance(most_likely, list)
        assert isinstance(marginals, dict)
        assert isinstance(summary, dict)
    
    def test_enhanced_mode_extends_existing_functionality(self):
        """Test that enhanced mode extends rather than replaces existing functionality."""
        config = MechanismAwareConfig(predict_mechanisms=True)
        net = create_modular_parent_set_model(config)
        
        key = random.PRNGKey(42)
        data = jnp.ones((8, 3, 3))
        variable_order = ["X", "Y", "Z"]
        target_variable = "Y"
        
        params = net.init(key, data, variable_order, target_variable)
        posterior = predict_with_mechanisms(net, params, data, variable_order, target_variable)
        
        # Should still work with all existing functions
        from src.causal_bayes_opt.avici_integration.parent_set.posterior import (
            get_most_likely_parents,
            get_marginal_parent_probabilities,
            summarize_posterior
        )
        
        most_likely = get_most_likely_parents(posterior)
        marginals = get_marginal_parent_probabilities(posterior, variable_order)
        summary = summarize_posterior(posterior)
        
        # But should also have enhanced information
        assert "mechanism_predictions" in posterior.metadata
        
        # Summary should mention mechanism awareness
        # (This would be implemented in the enhanced summarize_posterior)
        assert isinstance(summary, dict)


if __name__ == "__main__":
    pytest.main([__file__])