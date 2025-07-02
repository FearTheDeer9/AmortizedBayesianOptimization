"""
Enhanced Surrogate Model Factory for GRPO Training.

This module provides factory functions for creating enhanced surrogate models
that use continuous parent set prediction for scalable causal structure learning.
These models are designed to work with GRPO training and provide the
structure learning backbone for enhanced ACBO.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import pyrsistent as pyr

from .continuous.model import ContinuousParentSetPredictionModel
from .continuous.factory import create_continuous_parent_set_config, create_continuous_parent_set_model
from .continuous.sampling import sample_gumbel_softmax_parent_sets
from .parent_set.factory import create_parent_set_config
from ..data_structures import ExperienceBuffer
import pyrsistent as pyr

logger = logging.getLogger(__name__)


class EnhancedSurrogateModel(hk.Module):
    """
    Enhanced surrogate model using continuous parent set prediction.
    
    Integrates:
    - Continuous parent set models for O(d) scaling vs O(2^d) discrete
    - Differentiable structure learning with Gumbel-Softmax sampling
    - JAX-native implementation for GRPO training compatibility
    """
    
    def __init__(self,
                 # Model architecture parameters
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 use_continuous: bool = True,
                 # Continuous model parameters
                 temperature: float = 1.0,
                 straight_through: bool = True,
                 # Training parameters
                 dropout: float = 0.1,
                 name: str = "EnhancedSurrogateModel"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_continuous = use_continuous
        self.temperature = temperature
        self.straight_through = straight_through
        self.dropout = dropout
    
    def __call__(self,
                 data: jnp.ndarray,          # [N, d, 3] intervention data
                 variable_order: List[str],  # Variable ordering
                 target_variable: str,       # Target variable name
                 is_training: bool = True    # Training mode flag
                 ) -> Dict[str, jnp.ndarray]:
        """
        Enhanced surrogate model forward pass.
        
        Args:
            data: Intervention data in standard format [N, d, 3]
            variable_order: List of variable names
            target_variable: Target variable name
            is_training: Training mode flag
            
        Returns:
            Dictionary containing:
            - 'parent_probabilities': [d] - Continuous parent probabilities
            - 'structure_logits': [d] - Raw structure logits
            - 'posterior_samples': [K, d] - Sampled parent sets (if sampling enabled)
            - 'log_likelihood': [] - Model log likelihood
            - 'kl_regularization': [] - KL regularization term
        """
        N, d, channels = data.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"
        
        # Find target variable index
        if target_variable not in variable_order:
            raise ValueError(f"Target variable '{target_variable}' not in variable_order")
        target_idx = variable_order.index(target_variable)
        
        if self.use_continuous:
            # Use continuous parent set prediction
            return self._continuous_forward(data, variable_order, target_idx, is_training)
        else:
            # Fallback to simpler discrete model for comparison
            return self._discrete_forward(data, variable_order, target_idx, is_training)
    
    def _continuous_forward(self,
                          data: jnp.ndarray,
                          variable_order: List[str],
                          target_idx: int,
                          is_training: bool) -> Dict[str, jnp.ndarray]:
        """
        Forward pass using continuous parent set prediction.
        
        Args:
            data: Intervention data [N, d, 3]
            variable_order: Variable ordering
            target_idx: Target variable index
            is_training: Training mode flag
            
        Returns:
            Continuous model outputs
        """
        # Create continuous parent set model
        continuous_model = ContinuousParentSetPredictionModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout if is_training else 0.0
        )
        
        # Forward pass
        parent_probs = continuous_model(data, target_idx, is_training)  # [d]
        
        # Structure logits are the same as parent probabilities for continuous model
        structure_logits = parent_probs  # [d]
        
        # Sample parent sets using Gumbel-Softmax if training
        if is_training:
            # Sample multiple parent sets for regularization
            key = hk.next_rng_key()
            posterior_samples = sample_gumbel_softmax_parent_sets(
                structure_logits,
                temperature=self.temperature,
                straight_through=self.straight_through,
                n_samples=5,  # Multiple samples for training
                key=key
            )  # [5, d]
        else:
            # Use deterministic parent probabilities for inference
            posterior_samples = jnp.where(
                parent_probs > 0.5, 1.0, 0.0
            )[None, :]  # [1, d]
        
        # Compute log likelihood based on data fit
        log_likelihood = self._compute_data_log_likelihood(
            data, parent_probs, target_idx
        )
        
        # Compute KL regularization to encourage sparsity
        kl_regularization = self._compute_kl_regularization(parent_probs)
        
        return {
            'parent_probabilities': parent_probs,
            'structure_logits': structure_logits,
            'posterior_samples': posterior_samples,
            'log_likelihood': log_likelihood,
            'kl_regularization': kl_regularization
        }
    
    def _discrete_forward(self,
                        data: jnp.ndarray,
                        variable_order: List[str],
                        target_idx: int,
                        is_training: bool) -> Dict[str, jnp.ndarray]:
        """
        Fallback discrete forward pass for comparison.
        
        Args:
            data: Intervention data [N, d, 3]
            variable_order: Variable ordering
            target_idx: Target variable index
            is_training: Training mode flag
            
        Returns:
            Discrete model outputs
        """
        N, d, _ = data.shape
        
        # Simple discrete model: just compute correlations as proxy
        values = data[:, :, 0]  # [N, d]
        target_values = values[:, target_idx]  # [N]
        
        # Compute correlations with target
        parent_probs = []
        for var_idx in range(d):
            if var_idx == target_idx:
                parent_probs.append(0.0)  # Cannot be parent of itself
            else:
                var_values = values[:, var_idx]
                # Simple correlation as parent probability
                correlation = jnp.corrcoef(var_values, target_values)[0, 1]
                correlation = jnp.nan_to_num(correlation, 0.0)
                prob = jax.nn.sigmoid(correlation * 3.0)  # Scale correlation
                parent_probs.append(prob)
        
        parent_probs = jnp.array(parent_probs)
        structure_logits = parent_probs  # Same as probabilities for discrete
        
        # Create dummy posterior samples
        posterior_samples = jnp.where(
            parent_probs > 0.5, 1.0, 0.0
        )[None, :]  # [1, d]
        
        # Simple log likelihood
        log_likelihood = jnp.sum(jnp.log(parent_probs + 1e-8))
        
        # Simple sparsity regularization
        kl_regularization = jnp.sum(parent_probs)  # Encourage sparsity
        
        return {
            'parent_probabilities': parent_probs,
            'structure_logits': structure_logits,
            'posterior_samples': posterior_samples,
            'log_likelihood': log_likelihood,
            'kl_regularization': kl_regularization
        }
    
    def _compute_data_log_likelihood(self,
                                   data: jnp.ndarray,
                                   parent_probs: jnp.ndarray,
                                   target_idx: int) -> jnp.ndarray:
        """
        Compute data log likelihood given parent probabilities.
        
        Args:
            data: Intervention data [N, d, 3]
            parent_probs: Parent probabilities [d]
            target_idx: Target variable index
            
        Returns:
            Log likelihood scalar
        """
        N, d, _ = data.shape
        values = data[:, :, 0]  # [N, d]
        interventions = data[:, :, 1]  # [N, d]
        
        target_values = values[:, target_idx]  # [N]
        
        # Compute predicted target values based on parent probabilities
        predicted_targets = jnp.zeros(N)
        
        for var_idx in range(d):
            if var_idx != target_idx:
                # Weight contribution by parent probability
                contribution = parent_probs[var_idx] * values[:, var_idx]
                predicted_targets += contribution
        
        # Simple Gaussian likelihood
        residuals = target_values - predicted_targets
        log_likelihood = -0.5 * jnp.mean(residuals ** 2)
        
        return log_likelihood
    
    def _compute_kl_regularization(self,
                                 parent_probs: jnp.ndarray,
                                 prior_sparsity: float = 0.1) -> jnp.ndarray:
        """
        Compute KL regularization to encourage sparsity.
        
        Args:
            parent_probs: Parent probabilities [d]
            prior_sparsity: Prior probability of being a parent
            
        Returns:
            KL regularization term
        """
        # KL divergence from Bernoulli(prior_sparsity) prior
        kl_term = (
            parent_probs * jnp.log(parent_probs / prior_sparsity + 1e-8) +
            (1 - parent_probs) * jnp.log((1 - parent_probs) / (1 - prior_sparsity) + 1e-8)
        )
        
        return jnp.sum(kl_term)


class EnhancedSurrogateFactory:
    """Factory for creating enhanced surrogate models with various configurations."""
    
    @staticmethod
    def create_enhanced_surrogate_model(config: Dict[str, Any]) -> Callable:
        """
        Create enhanced surrogate model function for GRPO training.
        
        Args:
            config: Configuration dictionary with parameters:
                - hidden_dim: Hidden dimension size
                - num_layers: Number of model layers
                - num_heads: Number of attention heads
                - use_continuous: Whether to use continuous parent set prediction
                - temperature: Gumbel-Softmax temperature
                - straight_through: Whether to use straight-through gradients
                - dropout: Dropout rate
                
        Returns:
            Haiku-compatible surrogate model function
        """
        def surrogate_fn(data: jnp.ndarray,
                        variable_order: List[str],
                        target_variable: str,
                        is_training: bool = True) -> Dict[str, jnp.ndarray]:
            """
            Enhanced surrogate model function.
            
            Args:
                data: Intervention data [N, d, 3]
                variable_order: Variable ordering
                target_variable: Target variable name
                is_training: Training mode flag
                
            Returns:
                Surrogate model outputs
            """
            model = EnhancedSurrogateModel(
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 3),
                num_heads=config.get('num_heads', 8),
                use_continuous=config.get('use_continuous', True),
                temperature=config.get('temperature', 1.0),
                straight_through=config.get('straight_through', True),
                dropout=config.get('dropout', 0.1)
            )
            
            return model(data, variable_order, target_variable, is_training)
        
        return surrogate_fn
    
    @staticmethod
    def create_enhanced_config(
        model_complexity: str = "full",
        use_continuous: bool = True,
        performance_mode: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Create enhanced surrogate configuration.
        
        Args:
            model_complexity: "full", "medium", or "simple"
            use_continuous: Whether to use continuous parent set prediction
            performance_mode: "fast", "balanced", or "quality"
            
        Returns:
            Configuration dictionary
        """
        # Base configuration
        config = {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'use_continuous': use_continuous,
            'temperature': 1.0,
            'straight_through': True,
            'dropout': 0.1
        }
        
        # Model complexity adjustments
        if model_complexity == "full":
            config.update({
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 8,
            })
        elif model_complexity == "medium":
            config.update({
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4,
            })
        elif model_complexity == "simple":
            config.update({
                'hidden_dim': 32,
                'num_layers': 1,
                'num_heads': 2,
            })
        else:
            raise ValueError(f"Unknown model_complexity: {model_complexity}")
        
        # Performance mode adjustments
        if performance_mode == "fast":
            config.update({
                'hidden_dim': min(64, config['hidden_dim']),
                'num_layers': max(1, config['num_layers'] - 1),
                'temperature': 2.0,  # Higher temperature for faster sampling
            })
        elif performance_mode == "quality":
            config.update({
                'hidden_dim': max(256, config['hidden_dim']),
                'num_layers': config['num_layers'] + 1,
                'temperature': 0.5,  # Lower temperature for more precise sampling
            })
        # "balanced" uses default values
        
        return config
    
    @staticmethod
    def create_grpo_compatible_surrogate(
        variables: List[str],
        target_variable: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Create GRPO-compatible enhanced surrogate model.
        
        Args:
            variables: List of variable names
            target_variable: Target variable name
            config: Optional configuration dictionary
            
        Returns:
            Tuple of (surrogate_function, final_config)
        """
        if config is None:
            config = EnhancedSurrogateFactory.create_enhanced_config()
        
        # Add problem-specific information
        n_vars = len(variables)
        target_idx = variables.index(target_variable)
        
        final_config = {
            **config,
            'n_vars': n_vars,
            'target_idx': target_idx,
            'variables': variables,
            'target_variable': target_variable
        }
        
        def grpo_surrogate_fn(data: jnp.ndarray,
                             is_training: bool = True) -> Dict[str, jnp.ndarray]:
            """
            GRPO-compatible surrogate function.
            
            Args:
                data: Intervention data [N, d, 3]
                is_training: Training mode flag
                
            Returns:
                Surrogate outputs for GRPO training
            """
            surrogate_fn = EnhancedSurrogateFactory.create_enhanced_surrogate_model(final_config)
            return surrogate_fn(data, variables, target_variable, is_training)
        
        return grpo_surrogate_fn, final_config


def create_enhanced_surrogate_for_grpo(
    variables: List[str],
    target_variable: str,
    model_complexity: str = "full",
    use_continuous: bool = True,
    performance_mode: str = "balanced"
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Convenience function to create enhanced surrogate for GRPO training.
    
    Args:
        variables: List of variable names
        target_variable: Target variable name
        model_complexity: "full", "medium", or "simple"
        use_continuous: Whether to use continuous parent set prediction
        performance_mode: "fast", "balanced", or "quality"
        
    Returns:
        Tuple of (surrogate_function, configuration)
    """
    config = EnhancedSurrogateFactory.create_enhanced_config(
        model_complexity=model_complexity,
        use_continuous=use_continuous,
        performance_mode=performance_mode
    )
    
    return EnhancedSurrogateFactory.create_grpo_compatible_surrogate(
        variables, target_variable, config
    )


def create_enhanced_surrogate_from_buffer(
    buffer: ExperienceBuffer,
    variables: List[str],
    target_variable: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Callable, jnp.ndarray]:
    """
    Create enhanced surrogate from experience buffer.
    
    Args:
        buffer: Experience buffer with intervention data
        variables: List of variable names
        target_variable: Target variable name
        config: Optional configuration
        
    Returns:
        Tuple of (surrogate_function, training_data)
    """
    # Extract training data from buffer
    samples = buffer.get_all_samples()
    if not samples:
        raise ValueError("Experience buffer is empty")
    
    # Convert samples to training data format
    training_data = _convert_samples_to_training_data(samples, variables)
    
    # Create surrogate model
    surrogate_fn, final_config = create_enhanced_surrogate_for_grpo(
        variables, target_variable, **config if config else {}
    )
    
    return surrogate_fn, training_data


def _convert_samples_to_training_data(
    samples: List[pyr.PMap],
    variables: List[str]
) -> jnp.ndarray:
    """
    Convert experience buffer samples to training data format.
    
    Args:
        samples: List of samples from experience buffer
        variables: Variable names for ordering
        
    Returns:
        Training data in [N, d, 3] format
    """
    N = len(samples)
    d = len(variables)
    
    # Initialize data tensor
    data = jnp.zeros((N, d, 3))
    
    for i, sample in enumerate(samples):
        # Extract values, interventions, and targets
        values = []
        interventions = []
        targets = []
        
        for var_name in variables:
            # Get variable value
            if hasattr(sample, 'values') and var_name in sample.values:
                value = sample.values[var_name]
            else:
                value = 0.0  # Default value
            values.append(value)
            
            # Check if intervened
            intervention = 0.0
            if hasattr(sample, 'intervention') and sample.intervention:
                if var_name in sample.intervention.get('values', {}):
                    intervention = 1.0
            interventions.append(intervention)
            
            # Target indicator (for now, simple heuristic)
            target = 0.0  # Could be enhanced with actual target information
            targets.append(target)
        
        # Fill data tensor
        data = data.at[i, :, 0].set(jnp.array(values))
        data = data.at[i, :, 1].set(jnp.array(interventions))
        data = data.at[i, :, 2].set(jnp.array(targets))
    
    return data


def validate_enhanced_surrogate_integration() -> bool:
    """
    Validate enhanced surrogate integration with basic test.
    
    Returns:
        True if integration is working correctly
    """
    try:
        # Create test configuration
        variables = ['A', 'B', 'C', 'D']
        target_variable = 'D'
        
        surrogate_fn, config = create_enhanced_surrogate_for_grpo(
            variables, target_variable, model_complexity="medium"
        )
        
        # Test basic functionality
        key = jax.random.PRNGKey(42)
        N, d = 10, len(variables)
        
        # Create dummy training data
        dummy_data = jax.random.normal(key, (N, d, 3))
        
        # Test forward pass
        def test_forward(data):
            return surrogate_fn(data, is_training=False)
        
        # Initialize and test
        transformed = hk.transform(test_forward)
        params = transformed.init(key, dummy_data)
        outputs = transformed.apply(params, key, dummy_data)
        
        # Validate outputs
        required_keys = ['parent_probabilities', 'structure_logits', 'log_likelihood']
        for key in required_keys:
            if key not in outputs:
                logger.error(f"Missing output key: {key}")
                return False
        
        # Check shapes
        if outputs['parent_probabilities'].shape != (d,):
            logger.error(f"Invalid parent_probabilities shape: {outputs['parent_probabilities'].shape}")
            return False
        
        if not jnp.isscalar(outputs['log_likelihood']):
            logger.error(f"log_likelihood should be scalar: {outputs['log_likelihood'].shape}")
            return False
        
        logger.info("Enhanced surrogate integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced surrogate integration validation failed: {e}")
        return False


# Export key functions for integration
__all__ = [
    'EnhancedSurrogateModel',
    'EnhancedSurrogateFactory',
    'create_enhanced_surrogate_for_grpo',
    'create_enhanced_surrogate_from_buffer',
    'validate_enhanced_surrogate_integration'
]