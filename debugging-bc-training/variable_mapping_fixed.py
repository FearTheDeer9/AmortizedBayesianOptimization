"""
Unified variable mapping utilities for ACBO.
FIXED VERSION: Uses numerical sorting for variables to ensure consistent indexing.

This module provides a single source of truth for variable naming and ordering
across all ACBO components. It replaces the scattered hardcoded logic with
a consistent interface.

Key principles:
- Variable names should be explicitly passed, never inferred
- Consistent ordering (numerical instead of alphabetical)
- Support for both names and indices
- Clear conversion between representations
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our numerical sorting utility
from numerical_sort_utils import numerical_sort_variables

logger = logging.getLogger(__name__)


class VariableMapper:
    """
    Centralized variable mapping for consistent naming across models.
    
    This class provides:
    - Consistent variable ordering (numerical)
    - Name to index mapping
    - Index to name mapping
    - Validation of variable names
    """
    
    def __init__(self, variables: List[str], target_variable: Optional[str] = None):
        """
        Initialize variable mapper.
        
        Args:
            variables: List of variable names (will be sorted numerically)
            target_variable: Optional target variable (must be in variables)
            
        Raises:
            ValueError: If target not in variables or variables empty
        """
        if not variables:
            raise ValueError("Variables list cannot be empty")
            
        # Store numerically sorted variable order (single source of truth)
        # FIXED: Use numerical sorting instead of alphabetical
        self.variables = numerical_sort_variables(list(set(variables)))
        self.n_vars = len(self.variables)
        
        # Log the sorting for debugging
        alphabetical = sorted(set(variables))
        if alphabetical != self.variables:
            logger.debug(f"Variable order changed by numerical sorting:")
            logger.debug(f"  Alphabetical: {alphabetical}")
            logger.debug(f"  Numerical:    {self.variables}")
        
        # Create bidirectional mappings
        self.name_to_idx = {var: i for i, var in enumerate(self.variables)}
        self.idx_to_name = {i: var for i, var in enumerate(self.variables)}
        
        # Store target info if provided
        if target_variable is not None:
            if target_variable not in self.variables:
                raise ValueError(f"Target '{target_variable}' not in variables: {self.variables}")
            self.target_variable = target_variable
            self.target_idx = self.name_to_idx[target_variable]
        else:
            self.target_variable = None
            self.target_idx = None
    
    def get_index(self, variable_name: str) -> int:
        """Get index for variable name."""
        if variable_name not in self.name_to_idx:
            raise ValueError(f"Unknown variable: {variable_name}")
        return self.name_to_idx[variable_name]
    
    def get_name(self, index: int) -> str:
        """Get variable name for index."""
        if index < 0 or index >= self.n_vars:
            raise ValueError(f"Index {index} out of range [0, {self.n_vars})")
        return self.idx_to_name[index]
    
    def get_non_target_indices(self) -> List[int]:
        """Get indices of all non-target variables."""
        if self.target_idx is None:
            return list(range(self.n_vars))
        return [i for i in range(self.n_vars) if i != self.target_idx]
    
    def get_non_target_names(self) -> List[str]:
        """Get names of all non-target variables."""
        if self.target_variable is None:
            return self.variables.copy()
        return [var for var in self.variables if var != self.target_variable]
    
    def create_name_mask(self, selected_names: List[str]) -> jnp.ndarray:
        """
        Create binary mask for selected variable names.
        
        Args:
            selected_names: List of variable names to mark as 1
            
        Returns:
            Binary mask array of shape [n_vars]
        """
        mask = jnp.zeros(self.n_vars)
        for name in selected_names:
            if name in self.name_to_idx:
                mask = mask.at[self.name_to_idx[name]].set(1.0)
        return mask
    
    def create_index_mask(self, selected_indices: List[int]) -> jnp.ndarray:
        """
        Create binary mask for selected variable indices.
        
        Args:
            selected_indices: List of indices to mark as 1
            
        Returns:
            Binary mask array of shape [n_vars]
        """
        mask = jnp.zeros(self.n_vars)
        for idx in selected_indices:
            if 0 <= idx < self.n_vars:
                mask = mask.at[idx].set(1.0)
        return mask
    
    def extract_marginal_probs(self, posterior: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract marginal probabilities in consistent order.
        
        Args:
            posterior: Posterior dict with 'marginal_parent_probs' key
            
        Returns:
            Dict mapping variable names to probabilities
        """
        if 'marginal_parent_probs' not in posterior:
            logger.warning("No marginal_parent_probs in posterior")
            return {var: 0.0 for var in self.variables}
        
        marginals = posterior['marginal_parent_probs']
        
        # Ensure all variables have values (default to 0)
        result = {}
        for var in self.variables:
            if var in marginals:
                result[var] = float(marginals[var])
            else:
                result[var] = 0.0
                
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mapper state to dictionary for serialization."""
        return {
            'variables': self.variables,
            'target_variable': self.target_variable,
            'target_idx': self.target_idx,
            'n_vars': self.n_vars
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VariableMapper':
        """Create mapper from dictionary."""
        return cls(
            variables=data['variables'],
            target_variable=data.get('target_variable')
        )


def create_mapper_from_buffer(buffer: Any, target_variable: Optional[str] = None) -> VariableMapper:
    """
    Create variable mapper from experience buffer.
    
    Args:
        buffer: Experience buffer with samples
        target_variable: Optional target variable
        
    Returns:
        Configured VariableMapper instance
    """
    # Get variables from buffer coverage
    # FIXED: Use numerical sorting
    variables = numerical_sort_variables(list(buffer.get_variable_coverage()))
    
    if not variables:
        raise ValueError("No variables found in buffer")
    
    return VariableMapper(variables, target_variable)


def create_mapper_from_scm(scm: Any, target_variable: Optional[str] = None) -> VariableMapper:
    """
    Create variable mapper from SCM.
    
    Args:
        scm: Structural causal model
        target_variable: Optional target variable (defaults to SCM target)
        
    Returns:
        Configured VariableMapper instance
    """
    from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
    
    # Get variables from SCM
    variables = list(get_variables(scm))
    
    # Use SCM target if not specified
    if target_variable is None:
        target_variable = get_target(scm)
    
    return VariableMapper(variables, target_variable)


def update_surrogate_wrapper(
    net: Any,
    params: Any,
    variable_mapper: VariableMapper
) -> Any:
    """
    Create an updated surrogate function wrapper that uses explicit variable mapping.
    
    This replaces the hardcoded variable inference logic with explicit mapping.
    
    Args:
        net: Haiku transformed model
        params: Model parameters
        variable_mapper: Configured variable mapper
        
    Returns:
        Function that takes (tensor, target_var) and returns posterior dict
    """
    def surrogate_fn(tensor: jnp.ndarray, target_var: str) -> Dict[str, Any]:
        # Validate tensor shape
        n_vars = tensor.shape[1]
        if n_vars != variable_mapper.n_vars:
            raise ValueError(f"Tensor has {n_vars} vars but mapper expects {variable_mapper.n_vars}")
        
        # Get target index from mapper
        target_idx = variable_mapper.get_index(target_var)
        
        # Get prediction (no dropout during inference)
        rng = jax.random.PRNGKey(0)
        outputs = net.apply(params, rng, tensor, target_idx, False)
        
        # Extract parent probabilities
        parent_probs = outputs['parent_probabilities']
        
        # Create output dictionary with proper variable names
        marginal_probs = {}
        for i, var in enumerate(variable_mapper.variables):
            if i != target_idx:
                marginal_probs[var] = float(parent_probs[i])
            else:
                marginal_probs[var] = 0.0
        
        # Compute entropy
        safe_probs = jnp.maximum(parent_probs, 1e-10)
        entropy = -jnp.sum(parent_probs * jnp.log(safe_probs))
        
        return {
            'marginal_parent_probs': marginal_probs,
            'parent_probabilities': parent_probs,
            'entropy': float(entropy),
            'attention_logits': outputs.get('attention_logits'),
            'model_type': 'continuous',
            'variable_order': variable_mapper.variables  # Include for debugging
        }
    
    return surrogate_fn