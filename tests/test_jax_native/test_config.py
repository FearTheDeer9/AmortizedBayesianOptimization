"""
Tests for JAX-Native Configuration System

Validates static configuration creation, validation, and JAX compatibility.
"""

import pytest
import jax.numpy as jnp

from causal_bayes_opt.jax_native.config import (
    JAXConfig, create_jax_config, validate_jax_config, create_test_config
)


class TestJAXConfig:
    """Test JAX configuration creation and validation."""
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = JAXConfig(
            n_vars=3,
            target_idx=1,
            max_samples=100,
            max_history=50,
            variable_names=('X', 'Y', 'Z'),
            mechanism_types=(0, 1, 2)
        )
        
        assert config.n_vars == 3
        assert config.target_idx == 1
        assert config.max_samples == 100
        assert config.max_history == 50
        assert config.variable_names == ('X', 'Y', 'Z')
        assert config.mechanism_types == (0, 1, 2)
    
    def test_config_validation_n_vars(self):
        """Test validation of n_vars parameter."""
        with pytest.raises(ValueError, match="n_vars must be positive"):
            JAXConfig(
                n_vars=0,
                target_idx=0,
                max_samples=100,
                max_history=50,
                variable_names=(),
                mechanism_types=()
            )
    
    def test_config_validation_target_idx(self):
        """Test validation of target_idx parameter."""
        with pytest.raises(ValueError, match="target_idx must be in"):
            JAXConfig(
                n_vars=3,
                target_idx=3,  # Out of range
                max_samples=100,
                max_history=50,
                variable_names=('X', 'Y', 'Z'),
                mechanism_types=(0, 1, 2)
            )
    
    def test_config_validation_variable_names_length(self):
        """Test validation of variable names length."""
        with pytest.raises(ValueError, match="variable_names length"):
            JAXConfig(
                n_vars=3,
                target_idx=1,
                max_samples=100,
                max_history=50,
                variable_names=('X', 'Y'),  # Wrong length
                mechanism_types=(0, 1, 2)
            )
    
    def test_config_validation_mechanism_types_length(self):
        """Test validation of mechanism types length."""
        with pytest.raises(ValueError, match="mechanism_types length"):
            JAXConfig(
                n_vars=3,
                target_idx=1,
                max_samples=100,
                max_history=50,
                variable_names=('X', 'Y', 'Z'),
                mechanism_types=(0, 1)  # Wrong length
            )
    
    def test_config_validation_unique_names(self):
        """Test validation of unique variable names."""
        with pytest.raises(ValueError, match="Variable names must be unique"):
            JAXConfig(
                n_vars=3,
                target_idx=1,
                max_samples=100,
                max_history=50,
                variable_names=('X', 'X', 'Z'),  # Duplicate names
                mechanism_types=(0, 1, 2)
            )
    
    def test_config_validation_negative_mechanism_types(self):
        """Test validation of non-negative mechanism types."""
        with pytest.raises(ValueError, match="All mechanism types must be non-negative"):
            JAXConfig(
                n_vars=3,
                target_idx=1,
                max_samples=100,
                max_history=50,
                variable_names=('X', 'Y', 'Z'),
                mechanism_types=(0, -1, 2)  # Negative type
            )
    
    def test_get_variable_name(self):
        """Test getting variable name by index."""
        config = create_test_config()
        
        assert config.get_variable_name(0) == 'X'
        assert config.get_variable_name(1) == 'Y'
        assert config.get_variable_name(2) == 'Z'
        
        with pytest.raises(ValueError, match="Index .* out of range"):
            config.get_variable_name(3)
    
    def test_get_target_name(self):
        """Test getting target variable name."""
        config = create_test_config()
        assert config.get_target_name() == 'Y'
    
    def test_get_non_target_indices(self):
        """Test getting non-target variable indices."""
        config = create_test_config()
        non_target_indices = config.get_non_target_indices()
        
        assert non_target_indices == (0, 2)  # X and Z indices
        assert config.target_idx not in non_target_indices
    
    def test_create_target_mask(self):
        """Test creating target variable mask."""
        config = create_test_config()
        target_mask = config.create_target_mask()
        
        assert target_mask.shape == (config.n_vars,)
        assert target_mask.dtype == bool
        assert target_mask[config.target_idx] == True
        assert jnp.sum(target_mask) == 1
    
    def test_create_non_target_mask(self):
        """Test creating non-target variable mask."""
        config = create_test_config()
        non_target_mask = config.create_non_target_mask()
        
        assert non_target_mask.shape == (config.n_vars,)
        assert non_target_mask.dtype == bool
        assert non_target_mask[config.target_idx] == False
        assert jnp.sum(non_target_mask) == config.n_vars - 1


class TestCreateJAXConfig:
    """Test the create_jax_config helper function."""
    
    def test_create_config_basic(self):
        """Test basic configuration creation."""
        config = create_jax_config(
            variable_names=['A', 'B', 'C'],
            target_variable='B'
        )
        
        assert config.n_vars == 3
        assert config.target_idx == 1
        assert config.variable_names == ('A', 'B', 'C')
        assert config.get_target_name() == 'B'
    
    def test_create_config_with_params(self):
        """Test configuration creation with all parameters."""
        config = create_jax_config(
            variable_names=['X1', 'X2', 'Y'],
            target_variable='Y',
            max_samples=500,
            max_history=200,
            mechanism_types=[0, 1, 2],
            feature_dim=4
        )
        
        assert config.max_samples == 500
        assert config.max_history == 200
        assert config.mechanism_types == (0, 1, 2)
        assert config.feature_dim == 4
    
    def test_create_config_target_not_found(self):
        """Test error when target variable not in list."""
        with pytest.raises(ValueError, match="Target variable .* not found"):
            create_jax_config(
                variable_names=['A', 'B', 'C'],
                target_variable='D'  # Not in list
            )
    
    def test_create_config_default_mechanism_types(self):
        """Test default mechanism types creation."""
        config = create_jax_config(
            variable_names=['A', 'B'],
            target_variable='B'
        )
        
        assert config.mechanism_types == (4, 4)  # Default 'unknown' type


class TestValidateJAXConfig:
    """Test the validate_jax_config function."""
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = create_test_config()
        validate_jax_config(config)  # Should not raise
    
    def test_validate_max_samples_too_large(self):
        """Test validation of max_samples upper limit."""
        config = JAXConfig(
            n_vars=3,
            target_idx=1,
            max_samples=200000,  # Too large
            max_history=50,
            variable_names=('X', 'Y', 'Z'),
            mechanism_types=(0, 1, 2)
        )
        
        with pytest.raises(ValueError, match="max_samples too large"):
            validate_jax_config(config)
    
    def test_validate_max_history_larger_than_samples(self):
        """Test validation that max_history <= max_samples."""
        config = JAXConfig(
            n_vars=3,
            target_idx=1,
            max_samples=100,
            max_history=200,  # Larger than max_samples
            variable_names=('X', 'Y', 'Z'),
            mechanism_types=(0, 1, 2)
        )
        
        with pytest.raises(ValueError, match="max_history .* cannot exceed max_samples"):
            validate_jax_config(config)
    
    def test_validate_n_vars_too_large(self):
        """Test validation of n_vars upper limit."""
        large_names = tuple(f'X{i}' for i in range(2000))
        large_types = tuple(0 for _ in range(2000))
        
        config = JAXConfig(
            n_vars=2000,  # Too large
            target_idx=0,
            max_samples=100,
            max_history=50,
            variable_names=large_names,
            mechanism_types=large_types
        )
        
        with pytest.raises(ValueError, match="n_vars too large"):
            validate_jax_config(config)


class TestJAXCompatibility:
    """Test JAX compatibility of configuration operations."""
    
    def test_mask_operations_jax_compatible(self):
        """Test that mask operations are JAX-compatible."""
        config = create_test_config()
        
        # Test that the mask creation methods work and produce JAX arrays
        target_mask = config.create_target_mask()
        non_target_mask = config.create_non_target_mask()
        
        # These should be JAX arrays that can be used in compiled functions
        import jax
        
        @jax.jit
        def test_mask_operations(target_mask, non_target_mask):
            # Test operations on the masks
            combined = target_mask | non_target_mask
            intersection = target_mask & non_target_mask
            return combined, intersection
        
        combined, intersection = test_mask_operations(target_mask, non_target_mask)
        
        assert target_mask.shape == (config.n_vars,)
        assert non_target_mask.shape == (config.n_vars,)
        assert jnp.sum(target_mask) == 1
        assert jnp.sum(non_target_mask) == config.n_vars - 1
        assert jnp.all(combined)  # Should be all True
        assert not jnp.any(intersection)  # Should be all False
    
    def test_config_immutability(self):
        """Test that configuration is truly immutable."""
        config = create_test_config()
        
        # These should all fail
        with pytest.raises(Exception):  # dataclass is frozen
            config.n_vars = 5
        
        with pytest.raises(Exception):
            config.target_idx = 2
        
        with pytest.raises(Exception):
            config.variable_names = ('A', 'B', 'C')