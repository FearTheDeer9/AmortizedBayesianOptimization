"""
Integration layer for continuous parent set modeling.

This module provides factory functions and compatibility wrappers for
integrating continuous parent set models with the existing ACBO pipeline.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Any, Callable, Optional, List, Union

from .model import ContinuousParentSetPredictionModel
from .structure import DifferentiableStructureLearning, StructureLearningLoss
from .sampling import DifferentiableParentSampling, ContinuousToDiscrete


def create_continuous_surrogate_model(config: Dict[str, Any]) -> Callable:
    """
    Factory function for creating continuous surrogate model.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Haiku-compatible model function
    """
    def model_fn(data: jnp.ndarray, 
                 target_variable: int, 
                 is_training: bool = True) -> jnp.ndarray:
        """
        Continuous surrogate model function.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Target variable index
            is_training: Training mode flag
            
        Returns:
            Parent probabilities [d]
        """
        model = ContinuousParentSetPredictionModel(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            key_size=config.get('key_size', 32),
            dropout=config.get('dropout', 0.1)
        )
        
        outputs = model(data, target_variable, is_training)
        parent_probs = outputs['parent_probabilities']
        
        # Apply optional post-processing
        if config.get('use_temperature_scaling', False):
            temperature = config.get('temperature', 1.0)
            parent_probs = jax.nn.softmax(jnp.log(parent_probs + 1e-8) / temperature)
        
        return parent_probs
    
    return model_fn


def create_structure_learning_model(config: Dict[str, Any]) -> Callable:
    """
    Factory function for creating structure learning model.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Haiku-compatible structure learning function
    """
    def structure_fn(data: jnp.ndarray, 
                    is_training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Structure learning model function.
        
        Args:
            data: Intervention data [N, d, 3]
            is_training: Training mode flag
            
        Returns:
            Dictionary with structure learning outputs
        """
        n_vars = data.shape[1]
        
        model = DifferentiableStructureLearning(
            n_vars=n_vars,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            acyclicity_penalty_weight=config.get('acyclicity_penalty_weight', 1.0)
        )
        
        parent_probs = model(data, is_training)
        
        # Compute additional metrics
        results = {
            'parent_probabilities': parent_probs,
            'structure_entropy': model.compute_structure_entropy(parent_probs),
            'acyclicity_penalty': model.compute_acyclicity_penalty(parent_probs)
        }
        
        # Add topological ordering if requested
        if config.get('compute_topological_order', False):
            results['topological_order'] = model.get_topological_order(parent_probs)
        
        # Add binary adjacency matrix if requested
        if config.get('compute_adjacency', False):
            threshold = config.get('adjacency_threshold', 0.5)
            results['adjacency_matrix'] = model.get_adjacency_matrix(parent_probs, threshold)
        
        return results
    
    return structure_fn


class ContinuousParentSetCompatibilityWrapper:
    """
    Compatibility wrapper for continuous parent set model.
    
    Provides a drop-in replacement for discrete parent set models
    while maintaining the same interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_fn = create_continuous_surrogate_model(config)
        self.transformed = hk.transform(self.model_fn)
        self.params = None
        # Note: converter will be created as needed (not as Haiku module)
    
    def init(self, rng_key: jnp.ndarray, data: jnp.ndarray, target_variable: int):
        """Initialize model parameters."""
        self.params = self.transformed.init(rng_key, data, target_variable)
    
    def predict_parent_sets(self,
                          data: jnp.ndarray,
                          target_variable: Union[int, str],
                          variable_order: Optional[List[str]] = None,
                          rng_key: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
        """
        Predict parent sets with backward compatibility.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Target variable (index or name)
            variable_order: Variable names (for compatibility)
            rng_key: Random key (optional)
            
        Returns:
            Dictionary compatible with discrete parent set format
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call init() first.")
        
        # Convert target variable to index if needed
        if isinstance(target_variable, str):
            if variable_order is None:
                raise ValueError("variable_order required when target_variable is string")
            target_idx = variable_order.index(target_variable)
        else:
            target_idx = target_variable
        
        # Get continuous predictions
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        outputs = self.transformed.apply(self.params, rng_key, data, target_idx)
        parent_probs = outputs['parent_probabilities']
        
        # Convert to legacy format
        legacy_format = convert_to_legacy_format(
            parent_probs, 
            variable_order,
            top_k=self.config.get('top_k_parents', 5)
        )
        
        # Add additional compatibility fields
        legacy_format.update({
            'target_variable': target_variable,
            'prediction_type': 'continuous',
            'model_config': self.config
        })
        
        return legacy_format
    
    def predict_uncertainty(self,
                          data: jnp.ndarray,
                          target_variable: Union[int, str],
                          variable_order: Optional[List[str]] = None,
                          rng_key: Optional[jnp.ndarray] = None) -> float:
        """Predict uncertainty for target variable."""
        prediction = self.predict_parent_sets(data, target_variable, variable_order, rng_key)
        return prediction['uncertainty']
    
    def get_marginal_probabilities(self,
                                 data: jnp.ndarray,
                                 variable_order: Optional[List[str]] = None,
                                 rng_key: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """Get marginal parent probabilities for all variables."""
        if self.params is None:
            raise ValueError("Model not initialized. Call init() first.")
        
        n_vars = data.shape[1]
        if variable_order is None:
            variable_order = [f"X{i}" for i in range(n_vars)]
        
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        marginal_probs = {}
        
        for i, var_name in enumerate(variable_order):
            outputs = self.transformed.apply(self.params, rng_key, data, i)
            parent_probs = outputs['parent_probabilities']
            # Use maximum probability as marginal probability
            marginal_probs[var_name] = float(jnp.max(parent_probs))
        
        return marginal_probs


class ContinuousACBOIntegration:
    """Integration utilities for ACBO pipeline."""
    
    @staticmethod
    def create_acquisition_compatible_model(config: Dict[str, Any]) -> Callable:
        """
        Create model compatible with acquisition policy network.
        
        Args:
            config: Model configuration
            
        Returns:
            Model function compatible with AcquisitionState
        """
        def acbo_model_fn(data: jnp.ndarray, 
                         target_variable: int,
                         return_uncertainty: bool = True) -> Dict[str, jnp.ndarray]:
            """
            ACBO-compatible model function.
            
            Args:
                data: Intervention data [N, d, 3]
                target_variable: Target variable index
                return_uncertainty: Whether to return uncertainty estimates
                
            Returns:
                Dictionary with parent probabilities and uncertainty
            """
            model = ContinuousParentSetPredictionModel(
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 4),
                dropout=config.get('dropout', 0.1)
            )
            
            outputs = model(data, target_variable, is_training=False)
            parent_probs = outputs['parent_probabilities']
            
            results = {
                'parent_probabilities': parent_probs,
                'marginal_probability': jnp.max(parent_probs)
            }
            
            if return_uncertainty:
                uncertainty = model.compute_uncertainty(parent_probs)
                results['uncertainty'] = uncertainty
                results['uncertainty_bits'] = uncertainty / jnp.log(2.0)  # Convert to bits
            
            return results
        
        return acbo_model_fn
    
    @staticmethod
    def create_experience_buffer_compatible_sampler(config: Dict[str, Any]) -> Callable:
        """
        Create sampler compatible with experience buffer.
        
        Args:
            config: Sampler configuration
            
        Returns:
            Sampling function for experience collection
        """
        def sampling_fn(parent_probs: jnp.ndarray,
                       rng_key: jnp.ndarray,
                       sampling_strategy: str = "gumbel_softmax") -> jnp.ndarray:
            """
            Sample parent sets for experience collection.
            
            Args:
                parent_probs: Parent probabilities [d]
                rng_key: Random key
                sampling_strategy: Sampling strategy to use
                
            Returns:
                Sampled parent set representation
            """
            sampler = DifferentiableParentSampling()
            
            if sampling_strategy == "gumbel_softmax":
                temperature = config.get('temperature', 1.0)
                return sampler.gumbel_softmax_sample(parent_probs, rng_key, temperature)
            elif sampling_strategy == "straight_through":
                return sampler.straight_through_sample(parent_probs, rng_key)
            elif sampling_strategy == "top_k":
                k = config.get('k', 3)
                temperature = config.get('temperature', 1.0)
                return sampler.top_k_relaxed_sample(parent_probs, rng_key, k, temperature)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        return sampling_fn


def create_continuous_pipeline_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration for continuous parent set pipeline.
    
    Args:
        base_config: Base configuration
        
    Returns:
        Enhanced configuration for continuous models
    """
    config = base_config.copy()
    
    # Model architecture
    config.setdefault('hidden_dim', 128)
    config.setdefault('num_layers', 4)
    config.setdefault('num_heads', 8)
    config.setdefault('key_size', 32)
    config.setdefault('dropout', 0.1)
    
    # Training parameters
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('weight_decay', 1e-4)
    config.setdefault('gradient_clip_norm', 1.0)
    
    # Structure learning
    config.setdefault('use_acyclicity_constraint', True)
    config.setdefault('acyclicity_penalty_weight', 1.0)
    config.setdefault('sparsity_penalty_weight', 0.1)
    
    # Sampling
    config.setdefault('sampling_strategy', 'gumbel_softmax')
    config.setdefault('temperature', 1.0)
    config.setdefault('temperature_annealing', True)
    config.setdefault('initial_temperature', 5.0)
    config.setdefault('final_temperature', 0.1)
    config.setdefault('annealing_steps', 10000)
    
    # Compatibility
    config.setdefault('top_k_parents', 5)
    config.setdefault('compute_topological_order', False)
    config.setdefault('compute_adjacency', False)
    config.setdefault('adjacency_threshold', 0.5)
    
    return config


def convert_to_legacy_format(parent_probs: jnp.ndarray,
                           variable_names: Optional[List[str]] = None,
                           top_k: int = 5) -> Dict[str, Any]:
    """
    Convert continuous probabilities to legacy discrete format.
    
    Args:
        parent_probs: Parent probabilities [d]
        variable_names: List of variable names (optional)
        top_k: Number of top parent sets to return
        
    Returns:
        Dictionary compatible with legacy discrete parent set format
    """
    d = parent_probs.shape[0]
    
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(d)]
    
    # Get top-k parent indices and probabilities
    top_k_probs, top_k_indices = jax.lax.top_k(parent_probs, min(top_k, d))
    
    # Create parent set entries
    parent_sets = []
    for i in range(len(top_k_indices)):
        parent_idx = int(top_k_indices[i])
        parent_prob = float(top_k_probs[i])
        
        parent_set = {
            'parents': frozenset([variable_names[parent_idx]]),
            'probability': parent_prob,
            'log_probability': float(jnp.log(parent_prob + 1e-8)),
            'parent_indices': [parent_idx]
        }
        parent_sets.append(parent_set)
    
    # Compute uncertainty measures
    entropy = float(-jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8)))
    max_prob = float(jnp.max(parent_probs))
    
    return {
        'parent_probabilities': parent_probs,
        'parent_sets': parent_sets,
        'uncertainty': entropy,
        'confidence': max_prob,
        'num_variables': d,
        'variable_names': variable_names
    }


def validate_continuous_model_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration for continuous parent set models.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ['hidden_dim', 'num_layers']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    
    if config['hidden_dim'] <= 0:
        raise ValueError("hidden_dim must be positive")
    
    if config['num_layers'] <= 0:
        raise ValueError("num_layers must be positive")
    
    if 'dropout' in config and not (0.0 <= config['dropout'] <= 1.0):
        raise ValueError("dropout must be between 0.0 and 1.0")
    
    if 'temperature' in config and config['temperature'] <= 0:
        raise ValueError("temperature must be positive")
    
    print("✓ Continuous model configuration is valid")