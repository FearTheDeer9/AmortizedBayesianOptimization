"""
JAX-Native Sample Buffer

Provides a circular buffer implementation using pure JAX tensors with no
dictionary operations or Python loops in the hot paths.

Key features:
- Fixed-size tensor storage for optimal JAX compilation
- Circular buffer semantics with JAX operations
- Immutable interface (returns new state rather than mutating)
- Pure functions for all operations
"""

from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

from .config import JAXConfig


@dataclass(frozen=True)
class JAXSampleBuffer:
    """
    Immutable sample buffer using pure JAX tensors.
    
    Stores samples in fixed-size tensors with circular buffer semantics.
    All operations are pure functions that return new buffer instances.
    
    Args:
        values: Variable values [max_samples, n_vars]
        interventions: Intervention indicators [max_samples, n_vars] 
        targets: Target values [max_samples]
        valid_mask: Valid sample indicators [max_samples]
        write_idx: Current write position in circular buffer
        n_samples: Number of valid samples stored
        config: Static configuration
    """
    
    values: jnp.ndarray           # [max_samples, n_vars] - variable values
    interventions: jnp.ndarray    # [max_samples, n_vars] - intervention indicators
    targets: jnp.ndarray          # [max_samples] - target values
    valid_mask: jnp.ndarray       # [max_samples] - validity indicators
    write_idx: int                # Current write position (circular)
    n_samples: int                # Number of valid samples
    config: JAXConfig             # Static configuration
    
    def __post_init__(self) -> None:
        """Validate buffer consistency."""
        # Shape validation
        expected_shape = (self.config.max_samples, self.config.n_vars)
        if self.values.shape != expected_shape:
            raise ValueError(f"values shape {self.values.shape} != {expected_shape}")
        
        if self.interventions.shape != expected_shape:
            raise ValueError(f"interventions shape {self.interventions.shape} != {expected_shape}")
        
        target_shape = (self.config.max_samples,)
        if self.targets.shape != target_shape:
            raise ValueError(f"targets shape {self.targets.shape} != {target_shape}")
        
        if self.valid_mask.shape != target_shape:
            raise ValueError(f"valid_mask shape {self.valid_mask.shape} != {target_shape}")
        
        # Index validation
        if not (0 <= self.write_idx < self.config.max_samples):
            raise ValueError(f"write_idx {self.write_idx} out of range")
        
        if not (0 <= self.n_samples <= self.config.max_samples):
            raise ValueError(f"n_samples {self.n_samples} out of range")
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.n_samples == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.n_samples == self.config.max_samples
    
    def get_latest_samples(self, n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get the n most recent samples.
        
        Args:
            n: Number of samples to retrieve
            
        Returns:
            Tuple of (values, interventions, targets) for latest n samples
        """
        if n <= 0:
            empty_shape = (0, self.config.n_vars)
            return (
                jnp.zeros(empty_shape),
                jnp.zeros(empty_shape), 
                jnp.zeros((0,))
            )
        
        # Use JAX operations to extract latest samples
        return get_latest_samples_jax(
            self.values, self.interventions, self.targets,
            self.valid_mask, self.write_idx, self.n_samples, n
        )


def create_empty_jax_buffer(config: JAXConfig) -> JAXSampleBuffer:
    """
    Create an empty JAX sample buffer.
    
    Args:
        config: Static configuration
        
    Returns:
        Empty buffer ready for sample addition
    """
    return JAXSampleBuffer(
        values=jnp.zeros((config.max_samples, config.n_vars)),
        interventions=jnp.zeros((config.max_samples, config.n_vars)),
        targets=jnp.zeros((config.max_samples,)),
        valid_mask=jnp.zeros((config.max_samples,), dtype=bool),
        write_idx=0,
        n_samples=0,
        config=config
    )


def add_sample_jax(
    buffer: JAXSampleBuffer,
    variable_values: jnp.ndarray,  # [n_vars]
    intervention_mask: jnp.ndarray,  # [n_vars] boolean
    target_value: float
) -> JAXSampleBuffer:
    """
    Add a sample to the buffer using pure JAX operations.
    
    Args:
        buffer: Current buffer state
        variable_values: Values for all variables [n_vars]
        intervention_mask: Boolean mask indicating interventions [n_vars]
        target_value: Target variable value
        
    Returns:
        New buffer with sample added
    """
    # Use JAX-compiled tensor operations for performance
    new_values, new_interventions, new_targets, new_valid_mask, new_write_idx, new_n_samples = (
        add_sample_to_tensors_jax(
            buffer.values, buffer.interventions, buffer.targets, buffer.valid_mask,
            buffer.write_idx, buffer.n_samples, buffer.config.max_samples,
            variable_values, intervention_mask, target_value
        )
    )
    
    return JAXSampleBuffer(
        values=new_values,
        interventions=new_interventions,
        targets=new_targets,
        valid_mask=new_valid_mask,
        write_idx=int(new_write_idx),
        n_samples=int(new_n_samples),
        config=buffer.config
    )


def get_latest_samples_jax(
    values: jnp.ndarray,        # [max_samples, n_vars]
    interventions: jnp.ndarray, # [max_samples, n_vars]
    targets: jnp.ndarray,       # [max_samples]
    valid_mask: jnp.ndarray,    # [max_samples]
    write_idx: int,
    n_samples: int,
    n_latest: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extract the n_latest most recent samples using JAX operations.
    
    Args:
        values, interventions, targets, valid_mask: Buffer tensors
        write_idx: Current write position
        n_samples: Number of valid samples
        n_latest: Number of latest samples to extract
        
    Returns:
        Tuple of (values, interventions, targets) for latest samples
    """
    # Use the JAX-compiled extraction function
    return extract_latest_samples_jax(
        values, interventions, targets, valid_mask,
        write_idx, n_samples, n_latest
    )


def get_intervention_matrix_jax(
    buffer: JAXSampleBuffer,
    max_history: int
) -> jnp.ndarray:
    """
    Extract intervention matrix for the latest samples.
    
    Args:
        buffer: Sample buffer
        max_history: Maximum number of samples to include
        
    Returns:
        Intervention matrix [min(max_history, n_samples), n_vars]
    """
    values, interventions, targets = buffer.get_latest_samples(max_history)
    return interventions


def compute_buffer_statistics_jax(buffer: JAXSampleBuffer) -> dict:
    """
    Compute buffer statistics using JAX operations.
    
    Args:
        buffer: Sample buffer
        
    Returns:
        Dictionary with buffer statistics
    """
    # Count interventions per variable
    if buffer.n_samples > 0:
        # Get valid samples only
        valid_interventions = jnp.where(
            buffer.valid_mask[:, None], 
            buffer.interventions, 
            0.0
        )
        intervention_counts = jnp.sum(valid_interventions, axis=0)
        
        # Total interventions (excluding target variable)
        non_target_mask = buffer.config.create_non_target_mask()
        total_interventions = jnp.sum(intervention_counts * non_target_mask)
        
        # Coverage statistics
        variables_explored = jnp.sum(intervention_counts > 0)
        coverage_rate = variables_explored / buffer.config.n_vars
        
    else:
        intervention_counts = jnp.zeros(buffer.config.n_vars)
        total_interventions = 0.0
        coverage_rate = 0.0
    
    return {
        'total_samples': buffer.n_samples,
        'total_interventions': float(total_interventions),
        'intervention_counts': intervention_counts,
        'coverage_rate': float(coverage_rate),
        'buffer_utilization': buffer.n_samples / buffer.config.max_samples
    }


# JAX-compiled tensor operations for performance-critical paths
@jax.jit
def add_sample_to_tensors_jax(
    values_array: jnp.ndarray,       # [max_samples, n_vars] 
    interventions_array: jnp.ndarray, # [max_samples, n_vars]
    targets_array: jnp.ndarray,       # [max_samples]
    valid_mask: jnp.ndarray,          # [max_samples]
    write_idx: int,
    n_samples: int,
    max_samples: int,
    new_values: jnp.ndarray,          # [n_vars]
    new_interventions: jnp.ndarray,   # [n_vars]
    new_target: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-compiled function to add sample to tensor arrays."""
    # Update tensors at write position
    new_values_array = values_array.at[write_idx].set(new_values)
    new_interventions_array = interventions_array.at[write_idx].set(
        new_interventions.astype(jnp.float32)
    )
    new_targets_array = targets_array.at[write_idx].set(new_target)
    new_valid_mask = valid_mask.at[write_idx].set(True)
    
    # Update circular buffer indices
    new_write_idx = (write_idx + 1) % max_samples
    new_n_samples = jnp.minimum(n_samples + 1, max_samples)
    
    return (
        new_values_array, new_interventions_array, new_targets_array, 
        new_valid_mask, new_write_idx, new_n_samples
    )


def extract_latest_samples_jax(
    values: jnp.ndarray,        # [max_samples, n_vars]
    interventions: jnp.ndarray, # [max_samples, n_vars]
    targets: jnp.ndarray,       # [max_samples]
    valid_mask: jnp.ndarray,    # [max_samples]
    write_idx: int,
    n_samples: int,
    n_latest: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-compiled function to extract latest samples with fixed output size."""
    max_samples = values.shape[0]
    n_vars = values.shape[1]
    
    # Create fixed-size output arrays
    output_values = jnp.zeros((n_latest, n_vars))
    output_interventions = jnp.zeros((n_latest, n_vars))
    output_targets = jnp.zeros((n_latest,))
    
    # Only proceed if we have samples and want samples
    def extract():
        # Calculate how many samples we can actually get
        actual_n = jnp.minimum(n_latest, n_samples)
        
        # Create indices going backwards from write position
        indices = jnp.arange(n_latest)
        # Circular buffer indices
        buffer_indices = (write_idx - 1 - indices) % max_samples
        
        # Mask for valid entries (only first actual_n are valid)
        valid_entries = indices < actual_n
        
        # Extract with masking
        extracted_values = jnp.where(
            valid_entries[:, None],
            values[buffer_indices],
            0.0
        )
        extracted_interventions = jnp.where(
            valid_entries[:, None], 
            interventions[buffer_indices],
            0.0
        )
        extracted_targets = jnp.where(
            valid_entries,
            targets[buffer_indices],
            0.0
        )
        
        return extracted_values, extracted_interventions, extracted_targets
    
    def return_zeros():
        return output_values, output_interventions, output_targets
    
    return jax.lax.cond(
        (n_samples > 0) & (n_latest > 0),
        extract,
        return_zeros
    )


def create_test_buffer() -> JAXSampleBuffer:
    """Create a test buffer with sample data for unit testing."""
    from .config import create_test_config
    
    config = create_test_config()
    buffer = create_empty_jax_buffer(config)
    
    # Add some test samples
    key = jax.random.PRNGKey(42)
    
    for i in range(5):
        key, subkey = jax.random.split(key)
        values = jax.random.normal(subkey, (config.n_vars,))
        interventions = jnp.zeros(config.n_vars, dtype=bool).at[i % (config.n_vars - 1)].set(True)
        target_val = float(values[config.target_idx])
        
        buffer = add_sample_jax(buffer, values, interventions, target_val)
    
    return buffer