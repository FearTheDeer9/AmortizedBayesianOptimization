"""
Differentiable Parent Set Sampling.

This module implements differentiable sampling techniques for converting
continuous parent probabilities to discrete parent set selections.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from typing import Tuple, Union


class DifferentiableParentSampling(hk.Module):
    """
    Differentiable sampling for parent set selection.
    
    Provides multiple sampling strategies for converting continuous parent
    probabilities to discrete parent sets while maintaining differentiability.
    """
    
    def __init__(self, name: str = "DifferentiableParentSampling"):
        super().__init__(name=name)
    
    def gumbel_softmax_sample(self,
                            parent_probs: jnp.ndarray,  # [d] parent probabilities
                            rng_key: jnp.ndarray,       # Random key
                            temperature: float = 1.0    # Temperature for Gumbel-Softmax
                            ) -> jnp.ndarray:           # [d] relaxed one-hot sample
        """
        Differentiable sampling using Gumbel-Softmax trick.
        
        Args:
            parent_probs: Parent probabilities [d]
            temperature: Temperature parameter (lower = more discrete)
            rng_key: Random key for sampling
            
        Returns:
            Relaxed one-hot sample [d] that sums to 1.0
        """
        # Sample Gumbel noise using Haiku's RNG
        hk_rng_key = hk.next_rng_key()
        uniform = random.uniform(hk_rng_key, parent_probs.shape, minval=1e-8, maxval=1.0)
        gumbel_noise = -jnp.log(-jnp.log(uniform))
        
        # Add noise to log probabilities
        logits = jnp.log(parent_probs + 1e-8) + gumbel_noise
        
        # Apply softmax with temperature
        relaxed_sample = jax.nn.softmax(logits / temperature)
        
        return relaxed_sample
    
    def straight_through_sample(self,
                              parent_probs: jnp.ndarray,  # [d] parent probabilities
                              rng_key: jnp.ndarray        # Random key
                              ) -> jnp.ndarray:           # [d] one-hot sample
        """
        Straight-through estimator for discrete sampling.
        
        Forward pass: discrete sampling
        Backward pass: continuous gradients
        
        Args:
            parent_probs: Parent probabilities [d]
            rng_key: Random key for sampling
            
        Returns:
            One-hot sample [d] with exactly one element = 1.0
        """
        # Sample discrete parent index using Haiku's RNG
        hk_rng_key = hk.next_rng_key()
        parent_idx = random.categorical(hk_rng_key, jnp.log(parent_probs + 1e-8))
        
        # Create one-hot vector
        one_hot = jnp.zeros_like(parent_probs)
        one_hot = one_hot.at[parent_idx].set(1.0)
        
        # Straight-through estimator: use continuous probabilities for gradients
        return jax.lax.stop_gradient(one_hot - parent_probs) + parent_probs
    
    def top_k_relaxed_sample(self,
                           parent_probs: jnp.ndarray,  # [d] parent probabilities
                           rng_key: jnp.ndarray,       # Random key
                           k: int = 3,                 # Number of parents to select
                           temperature: float = 1.0    # Temperature for relaxation
                           ) -> jnp.ndarray:           # [d] relaxed k-hot sample
        """
        Sample top-k parents with relaxed selection.
        
        Args:
            parent_probs: Parent probabilities [d]
            k: Number of parents to select
            temperature: Temperature for relaxation
            rng_key: Random key for sampling
            
        Returns:
            Relaxed k-hot sample [d] where top-k elements are emphasized
        """
        d = parent_probs.shape[0]
        k = min(k, d)
        
        # Use Gumbel-Softmax to get relaxed samples
        relaxed_sample = self.gumbel_softmax_sample(parent_probs, rng_key, temperature)
        
        # Soft top-k selection
        _, top_k_indices = jax.lax.top_k(relaxed_sample, k)
        
        # Create soft k-hot mask
        mask = jnp.zeros(d)
        mask = mask.at[top_k_indices].set(1.0)
        
        # Apply mask with some relaxation
        k_hot_sample = relaxed_sample * mask
        
        # Renormalize
        k_hot_sample = k_hot_sample / (jnp.sum(k_hot_sample) + 1e-8)
        
        return k_hot_sample
    
    def adaptive_temperature_sample(self,
                                  parent_probs: jnp.ndarray,  # [d] parent probabilities
                                  rng_key: jnp.ndarray,       # Random key
                                  step: int,                  # Training step
                                  initial_temp: float = 5.0,  # Initial temperature
                                  final_temp: float = 0.1,    # Final temperature
                                  anneal_steps: int = 10000   # Annealing steps
                                  ) -> jnp.ndarray:           # [d] sample with adaptive temperature
        """
        Sample with adaptive temperature annealing.
        
        Temperature starts high (more exploration) and decreases over time
        (more exploitation/discrete behavior).
        
        Args:
            parent_probs: Parent probabilities [d]
            step: Current training step
            initial_temp: Starting temperature
            final_temp: Final temperature
            anneal_steps: Number of steps for annealing
            rng_key: Random key for sampling
            
        Returns:
            Temperature-annealed sample [d]
        """
        # Linear annealing schedule
        progress = jnp.clip(step / anneal_steps, 0.0, 1.0)
        temperature = initial_temp * (1 - progress) + final_temp * progress
        
        return self.gumbel_softmax_sample(parent_probs, rng_key, temperature)


class ContinuousToDiscrete(hk.Module):
    """
    Utilities for converting continuous parent probabilities to discrete formats.
    
    Provides backward compatibility with existing discrete parent set interfaces.
    """
    
    def __init__(self, name: str = "ContinuousToDiscrete"):
        super().__init__(name=name)
    
    def get_deterministic_parent_set(self,
                                   parent_probs: jnp.ndarray,  # [d] parent probabilities
                                   max_parents: int = 3        # Maximum number of parents
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get deterministic parent set by selecting top-k highest probabilities.
        
        Args:
            parent_probs: Parent probabilities [d]
            max_parents: Maximum number of parents to include
            
        Returns:
            Tuple of (parent_indices, parent_probabilities)
        """
        d = parent_probs.shape[0]
        k = min(max_parents, d)
        
        # Get top-k parents
        top_k_probs, top_k_indices = jax.lax.top_k(parent_probs, k)
        
        return top_k_indices, top_k_probs
    
    def sample_multiple_parent_sets(self,
                                  parent_probs: jnp.ndarray,  # [d] parent probabilities
                                  rng_key: jnp.ndarray,       # Random key
                                  num_samples: int = 10,      # Number of samples
                                  max_parents: int = 3        # Max parents per set
                                  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample multiple discrete parent sets from continuous probabilities.
        
        Args:
            parent_probs: Parent probabilities [d]
            num_samples: Number of parent sets to sample
            max_parents: Maximum parents per set
            rng_key: Random key for sampling
            
        Returns:
            Tuple of (sampled_sets, set_probabilities) where:
            - sampled_sets: [num_samples, max_parents] parent indices (-1 for padding)
            - set_probabilities: [num_samples] probability of each sampled set
        """
        d = parent_probs.shape[0]
        keys = random.split(rng_key, num_samples)
        
        sampler = DifferentiableParentSampling()
        
        def sample_single_set(key):
            # Sample using straight-through estimator
            one_hot = sampler.straight_through_sample(parent_probs, key)
            
            # Convert to parent set format
            parent_indices = jnp.where(one_hot > 0.5, 
                                     jnp.arange(d), 
                                     -1)  # -1 for non-parents
            
            # Keep only actual parents and pad to max_parents
            actual_parents = parent_indices[parent_indices >= 0]
            padded_parents = jnp.pad(actual_parents, 
                                   (0, max_parents - len(actual_parents)), 
                                   constant_values=-1)[:max_parents]
            
            # Compute set probability
            set_prob = jnp.prod(jnp.where(one_hot > 0.5, 
                                        parent_probs, 
                                        1.0 - parent_probs))
            
            return padded_parents, set_prob
        
        # Vectorize across samples
        sampled_sets, set_probs = jax.vmap(sample_single_set)(keys)
        
        return sampled_sets, set_probs
    
    def convert_to_legacy_format(self,
                               parent_probs: jnp.ndarray,    # [d] parent probabilities
                               variable_names: list = None,  # Variable names (optional)
                               top_k: int = 5                # Number of top parent sets
                               ) -> dict:
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


class ParentSetEnsemble(hk.Module):
    """Ensemble methods for robust parent set prediction."""
    
    def __init__(self, 
                 num_models: int = 5,
                 name: str = "ParentSetEnsemble"):
        super().__init__(name=name)
        self.num_models = num_models
    
    def ensemble_predict(self,
                        data: jnp.ndarray,         # [N, d, 3] intervention data
                        target_variable: int,      # Target variable index
                        rng_key: jnp.ndarray,      # Random key
                        dropout_rate: float = 0.1  # Dropout for diversity
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict parent probabilities using model ensemble.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Target variable index
            rng_key: Random key
            dropout_rate: Dropout rate for ensemble diversity
            
        Returns:
            Tuple of (mean_probabilities, uncertainty_estimates)
        """
        from .model import ContinuousParentSetPredictionModel
        
        keys = random.split(rng_key, self.num_models)
        
        def single_model_predict(key):
            model = ContinuousParentSetPredictionModel(dropout=dropout_rate)
            return model(data, target_variable, is_training=True)  # Keep dropout on
        
        # Get predictions from all models
        all_predictions = jax.vmap(single_model_predict)(keys)  # [num_models, d]
        
        # Compute ensemble statistics
        mean_probs = jnp.mean(all_predictions, axis=0)  # [d]
        std_probs = jnp.std(all_predictions, axis=0)    # [d]
        
        return mean_probs, std_probs


def sample_gumbel_softmax_parent_sets(
    parent_probs: jnp.ndarray,
    rng_key: jnp.ndarray,
    temperature: float = 1.0,
    num_samples: int = 1
) -> jnp.ndarray:
    """
    Sample parent sets using Gumbel-Softmax differentiable sampling.
    
    This is a standalone function for compatibility with existing enhanced surrogate code.
    
    Args:
        parent_probs: Parent probabilities [d] or [batch_size, d]
        rng_key: Random key for sampling
        temperature: Temperature for Gumbel-Softmax (lower = more discrete)
        num_samples: Number of samples to generate
        
    Returns:
        Sampled parent sets [num_samples, d] or [batch_size, num_samples, d]
    """
    # Handle both single and batch inputs
    if parent_probs.ndim == 1:
        # Single parent probability vector
        d = parent_probs.shape[0]
        keys = random.split(rng_key, num_samples)
        
        def sample_single(key):
            # Sample Gumbel noise
            uniform = random.uniform(key, (d,), minval=1e-8, maxval=1.0)
            gumbel_noise = -jnp.log(-jnp.log(uniform))
            
            # Add noise to log probabilities
            logits = jnp.log(parent_probs + 1e-8) + gumbel_noise
            
            # Apply softmax with temperature
            return jax.nn.softmax(logits / temperature)
        
        # Vectorize over samples
        samples = jax.vmap(sample_single)(keys)
        
        if num_samples == 1:
            return samples[0]  # Return [d] instead of [1, d]
        return samples
    
    else:
        # Batch parent probability vectors
        batch_size, d = parent_probs.shape
        keys = random.split(rng_key, batch_size * num_samples)
        keys = keys.reshape(batch_size, num_samples, -1)
        
        def sample_batch_single(batch_probs, batch_keys):
            def sample_single(key):
                # Sample Gumbel noise
                uniform = random.uniform(key, (d,), minval=1e-8, maxval=1.0)
                gumbel_noise = -jnp.log(-jnp.log(uniform))
                
                # Add noise to log probabilities
                logits = jnp.log(batch_probs + 1e-8) + gumbel_noise
                
                # Apply softmax with temperature
                return jax.nn.softmax(logits / temperature)
            
            return jax.vmap(sample_single)(batch_keys)
        
        # Vectorize over batch
        samples = jax.vmap(sample_batch_single)(parent_probs, keys)
        
        if num_samples == 1:
            return samples[:, 0, :]  # Return [batch_size, d] instead of [batch_size, 1, d]
        return samples


def validate_gumbel_softmax_sampling() -> bool:
    """
    Validate Gumbel-Softmax sampling functionality.
    
    Returns:
        True if sampling is working correctly, False otherwise
    """
    try:
        # Test single parent probability vector
        key = jax.random.PRNGKey(42)
        parent_probs = jnp.array([0.8, 0.3, 0.1, 0.6, 0.2])
        
        # Test single sample
        sample = sample_gumbel_softmax_parent_sets(
            parent_probs, key, temperature=1.0, num_samples=1
        )
        
        if sample.shape != (5,):
            print(f"Unexpected single sample shape: {sample.shape}")
            return False
        
        # Test multiple samples
        key, subkey = jax.random.split(key)
        samples = sample_gumbel_softmax_parent_sets(
            parent_probs, subkey, temperature=1.0, num_samples=3
        )
        
        if samples.shape != (3, 5):
            print(f"Unexpected multiple samples shape: {samples.shape}")
            return False
        
        # Test batch sampling
        key, subkey = jax.random.split(key)
        batch_probs = jnp.array([[0.8, 0.3, 0.1], [0.2, 0.7, 0.4]])
        batch_samples = sample_gumbel_softmax_parent_sets(
            batch_probs, subkey, temperature=1.0, num_samples=2
        )
        
        if batch_samples.shape != (2, 2, 3):
            print(f"Unexpected batch samples shape: {batch_samples.shape}")
            return False
        
        # Check that samples are approximately probability vectors (sum to ~1)
        sample_sums = jnp.sum(samples, axis=1)
        if not jnp.allclose(sample_sums, 1.0, atol=1e-6):
            print(f"Samples don't sum to 1.0: {sample_sums}")
            return False
        
        # Check finite values
        if not jnp.all(jnp.isfinite(samples)):
            print("Found non-finite values in samples")
            return False
        
        print("Gumbel-Softmax sampling validation passed")
        return True
        
    except Exception as e:
        print(f"Gumbel-Softmax sampling validation failed: {e}")
        return False