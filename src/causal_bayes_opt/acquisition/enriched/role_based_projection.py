"""
Role-based projection module for addressing variable embedding collapse.

This module implements role-aware projections that give different embeddings
to variables based on their causal role (target, intervention, other) while
maintaining variable-agnosticism and permutation invariance.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RoleBasedProjection(hk.Module):
    """
    Role-based projection that applies different linear transformations
    based on variable roles in the causal query.
    
    This maintains variable-agnosticism while providing the differentiation
    needed to prevent embedding collapse. Variables are projected differently
    based on whether they are:
    - Target variable (cannot be intervened on)
    - Intervention variable (if applicable)
    - Other variables (to be marginalized over)
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 name: str = "RoleBasedProjection"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.w_init = w_init or hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 enriched_history: jnp.ndarray,  # [T, n_vars, n_channels]
                 target_mask: Optional[jnp.ndarray] = None,  # [n_vars] binary mask
                 intervention_mask: Optional[jnp.ndarray] = None,  # [n_vars] binary mask
                 ) -> jnp.ndarray:  # [T, n_vars, hidden_dim]
        """
        Apply role-based projection to enriched history.
        
        Args:
            enriched_history: Multi-channel temporal input [T, n_vars, n_channels]
            target_mask: Binary mask indicating target variable [n_vars]
            intervention_mask: Binary mask indicating intervention variables [n_vars]
            
        Returns:
            Projected features [T, n_vars, hidden_dim]
        """
        T, n_vars, n_channels = enriched_history.shape
        
        # Create role masks if not provided
        if target_mask is None:
            # If no target mask provided, assume no target (all zeros)
            target_mask = jnp.zeros(n_vars, dtype=jnp.float32)
        else:
            target_mask = target_mask.astype(jnp.float32)
            
        if intervention_mask is None:
            # If no intervention mask provided, assume no interventions
            intervention_mask = jnp.zeros(n_vars, dtype=jnp.float32)
        else:
            intervention_mask = intervention_mask.astype(jnp.float32)
        
        # Ensure target and intervention masks don't overlap
        # (a variable cannot be both target and intervention)
        intervention_mask = intervention_mask * (1 - target_mask)
        
        # Create mask for "other" variables (neither target nor intervention)
        other_mask = (1 - target_mask) * (1 - intervention_mask)
        
        # Flatten temporal and variable dimensions for projection
        flat_history = enriched_history.reshape(T * n_vars, n_channels)
        
        # Create three different projections
        target_projection = hk.Linear(
            self.hidden_dim, 
            w_init=self.w_init, 
            name="target_projection"
        )
        intervention_projection = hk.Linear(
            self.hidden_dim, 
            w_init=self.w_init, 
            name="intervention_projection"
        )
        other_projection = hk.Linear(
            self.hidden_dim, 
            w_init=self.w_init, 
            name="other_projection"
        )
        
        # Apply projections
        proj_target = target_projection(flat_history)  # [T*n_vars, hidden_dim]
        proj_intervention = intervention_projection(flat_history)  # [T*n_vars, hidden_dim]
        proj_other = other_projection(flat_history)  # [T*n_vars, hidden_dim]
        
        # Expand role masks to match flattened shape
        # Repeat each mask value T times for temporal dimension
        target_mask_flat = jnp.repeat(target_mask, T)  # [T*n_vars]
        intervention_mask_flat = jnp.repeat(intervention_mask, T)  # [T*n_vars]
        other_mask_flat = jnp.repeat(other_mask, T)  # [T*n_vars]
        
        # Combine projections based on roles
        # Each variable gets exactly one projection based on its role
        combined_projection = (
            proj_target * target_mask_flat[:, None] +
            proj_intervention * intervention_mask_flat[:, None] +
            proj_other * other_mask_flat[:, None]
        )
        
        # Reshape back to temporal structure
        projected = combined_projection.reshape(T, n_vars, self.hidden_dim)
        
        return projected
    
    def get_role_statistics(self,
                           target_mask: Optional[jnp.ndarray] = None,
                           intervention_mask: Optional[jnp.ndarray] = None,
                           n_vars: int = None) -> dict:
        """
        Get statistics about role assignments for debugging.
        
        Args:
            target_mask: Binary mask indicating target variable
            intervention_mask: Binary mask indicating intervention variables
            n_vars: Number of variables (required if masks not provided)
            
        Returns:
            Dictionary with role counts
        """
        if target_mask is None and n_vars is None:
            raise ValueError("Either target_mask or n_vars must be provided")
            
        if target_mask is None:
            target_mask = jnp.zeros(n_vars, dtype=jnp.float32)
        else:
            target_mask = target_mask.astype(jnp.float32)
            
        if intervention_mask is None:
            n_vars = target_mask.shape[0]
            intervention_mask = jnp.zeros(n_vars, dtype=jnp.float32)
        else:
            intervention_mask = intervention_mask.astype(jnp.float32)
        
        # Ensure no overlap
        intervention_mask = intervention_mask * (1 - target_mask)
        other_mask = (1 - target_mask) * (1 - intervention_mask)
        
        return {
            'n_target': jnp.sum(target_mask),
            'n_intervention': jnp.sum(intervention_mask),
            'n_other': jnp.sum(other_mask),
            'total': target_mask.shape[0]
        }


def create_target_mask_from_index(target_idx: int, n_vars: int) -> jnp.ndarray:
    """
    Helper function to create target mask from variable index.
    
    Args:
        target_idx: Index of target variable
        n_vars: Total number of variables
        
    Returns:
        Binary mask [n_vars] with 1 at target position
    """
    return jnp.array([1.0 if i == target_idx else 0.0 for i in range(n_vars)])


def create_intervention_mask_from_history(enriched_history: jnp.ndarray) -> jnp.ndarray:
    """
    Create intervention mask from enriched history by checking intervention channel.
    
    Args:
        enriched_history: Enriched history tensor [T, n_vars, n_channels]
                         where channel 1 is intervention indicators
                         
    Returns:
        Binary mask [n_vars] indicating which variables have been intervened on
    """
    # Channel 1 contains intervention indicators
    # Check if any variable has been intervened on across time
    intervention_indicators = enriched_history[:, :, 1]  # [T, n_vars]
    
    # A variable is considered "intervened" if it has any interventions in history
    has_interventions = jnp.any(intervention_indicators > 0, axis=0)  # [n_vars]
    
    return has_interventions.astype(jnp.float32)


def test_role_based_projection():
    """Test function to verify role-based projection behavior."""
    import jax.random as random
    
    # Test setup
    key = random.PRNGKey(42)
    T, n_vars, n_channels = 10, 5, 5
    hidden_dim = 32
    
    # Create mock enriched history
    enriched_history = random.normal(key, (T, n_vars, n_channels))
    
    # Create role masks
    target_mask = create_target_mask_from_index(0, n_vars)  # Variable 0 is target
    intervention_mask = jnp.array([0., 0., 1., 0., 0.])  # Variable 2 is intervened
    
    # Initialize and apply projection
    def forward_fn(history):
        proj = RoleBasedProjection(hidden_dim)
        return proj(history, target_mask, intervention_mask)
    
    # Transform to Haiku
    forward = hk.transform(forward_fn)
    params = forward.init(random.PRNGKey(43), enriched_history)
    projected = forward.apply(params, random.PRNGKey(44), enriched_history)
    
    print(f"Input shape: {enriched_history.shape}")
    print(f"Output shape: {projected.shape}")
    print(f"Target variable: 0")
    print(f"Intervention variable: 2")
    print(f"Other variables: 1, 3, 4")
    
    # Verify permutation invariance for "other" variables
    # Swap variables 1 and 3 (both are "other" variables)
    history_swapped = enriched_history.copy()
    history_swapped = history_swapped.at[:, 1, :].set(enriched_history[:, 3, :])
    history_swapped = history_swapped.at[:, 3, :].set(enriched_history[:, 1, :])
    
    projected_swapped = forward.apply(params, random.PRNGKey(44), history_swapped)
    
    # Check that swapped "other" variables produce swapped outputs
    print(f"\nPermutation invariance check:")
    print(f"Projection difference for swapped 'other' variables: "
          f"{jnp.max(jnp.abs(projected[:, 1, :] - projected_swapped[:, 3, :]))}")
    
    return projected


if __name__ == "__main__":
    # Run test
    test_role_based_projection()