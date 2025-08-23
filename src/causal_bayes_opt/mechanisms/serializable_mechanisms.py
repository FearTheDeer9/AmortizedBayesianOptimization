"""
Serializable mechanism implementations for causal models.

This module provides pickleable mechanism classes that can be used
with multiprocessing, replacing the closure-based mechanisms.
"""

from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as random


class LinearMechanism:
    """
    Serializable linear mechanism class.
    
    Implements: Y = intercept + sum(coeff_i * X_i) + noise
    """
    
    def __init__(self, coefficients: Dict[str, float], intercept: float, noise_scale: float,
                 output_bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize linear mechanism.
        
        Args:
            coefficients: Mapping from parent variable names to coefficients
            intercept: Constant term in the linear equation
            noise_scale: Standard deviation of Gaussian noise
            output_bounds: Optional (min, max) bounds for output values
        """
        self.coefficients = dict(coefficients)  # Store as regular dict for pickling
        self.intercept = float(intercept)
        self.noise_scale = float(noise_scale)
        self.output_bounds = output_bounds
        self.parent_names = list(coefficients.keys())
        self.coeff_array = jnp.array([coefficients[name] for name in self.parent_names])
    
    def __call__(self, parent_values: Dict[str, float], noise_key: jax.Array) -> float:
        """
        Compute linear mechanism value.
        
        Args:
            parent_values: Dictionary mapping parent names to their values
            noise_key: JAX random key for noise generation
            
        Returns:
            Computed value for this variable
        """
        # Extract parent values in the same order as coefficients
        parent_vals = jnp.array([parent_values[name] for name in self.parent_names])
        
        # Linear combination: sum(coeff_i * X_i)
        linear_term = jnp.dot(self.coeff_array, parent_vals)
        
        # Add intercept
        result = linear_term + self.intercept
        
        # Add Gaussian noise
        if self.noise_scale > 0:
            noise = random.normal(noise_key) * self.noise_scale
            result = result + noise
        
        # Apply output bounds if specified
        if self.output_bounds is not None:
            result = jnp.clip(result, self.output_bounds[0], self.output_bounds[1])
            
        return float(result)
    
    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        return {
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'noise_scale': self.noise_scale,
            'output_bounds': self.output_bounds
        }
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from unpickling."""
        self.coefficients = state['coefficients']
        self.intercept = state['intercept']
        self.noise_scale = state['noise_scale']
        self.output_bounds = state.get('output_bounds')  # Use get for backward compatibility
        self.parent_names = list(self.coefficients.keys())
        self.coeff_array = jnp.array([self.coefficients[name] for name in self.parent_names])


class RootMechanism:
    """
    Serializable root mechanism class.
    
    Implements: Y = mean + noise
    """
    
    def __init__(self, mean: float, noise_scale: float, 
                 output_bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize root mechanism.
        
        Args:
            mean: Mean value for the root variable
            noise_scale: Standard deviation of Gaussian noise
            output_bounds: Optional (min, max) bounds for output values
        """
        self.mean = float(mean)
        self.noise_scale = float(noise_scale)
        self.output_bounds = output_bounds
    
    def __call__(self, parent_values: Dict[str, float], noise_key: jax.Array) -> float:
        """
        Compute root mechanism value.
        
        Args:
            parent_values: Should be empty for root variables
            noise_key: JAX random key for noise generation
            
        Returns:
            Computed value for this variable
        """
        if parent_values:
            import logging
            logging.getLogger(__name__).warning(
                f"Root mechanism called with non-empty parent values: {parent_values}"
            )
        
        noise = random.normal(noise_key) * self.noise_scale
        result = self.mean + noise
        
        # Apply output bounds if specified
        if self.output_bounds is not None:
            result = jnp.clip(result, self.output_bounds[0], self.output_bounds[1])
            
        return float(result)
    
    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        return {
            'mean': self.mean,
            'noise_scale': self.noise_scale,
            'output_bounds': self.output_bounds
        }
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from unpickling."""
        self.mean = state['mean']
        self.noise_scale = state['noise_scale']
        self.output_bounds = state.get('output_bounds')  # Use get for backward compatibility