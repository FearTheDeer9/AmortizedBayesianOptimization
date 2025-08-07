"""
Wrapper classes for BC surrogates to implement the SurrogateInterface.

This module provides clean wrappers around the raw BC surrogate functions
returned by create_bc_surrogate, ensuring they properly implement the
SurrogateInterface protocol.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import jax.numpy as jnp
import logging

from .surrogate_interface import SurrogateInterface

logger = logging.getLogger(__name__)


class BCSurrogateWrapper(SurrogateInterface):
    """
    Wrapper for BC surrogate functions to implement SurrogateInterface.
    
    This wraps the raw predict/update functions returned by create_bc_surrogate
    into a clean interface that follows our protocol.
    """
    
    def __init__(self, 
                 predict_fn: Callable,
                 update_fn: Optional[Callable] = None,
                 name: str = "BC_Surrogate",
                 is_active: bool = False,
                 net: Optional[Any] = None,
                 params: Optional[Any] = None,
                 opt_state: Optional[Any] = None):
        """
        Initialize wrapper around BC surrogate functions.
        
        Args:
            predict_fn: The prediction function from create_bc_surrogate
            update_fn: Optional update function for active learning
            name: Name for logging
            is_active: Whether this supports active learning
            net: Neural network model (needed for recreating predict_fn after updates)
            params: Initial model parameters
            opt_state: Initial optimizer state
        """
        self._predict_fn = predict_fn
        self._update_fn = update_fn
        self._name = name
        self._is_active = is_active
        
        # Track state for active learning
        self._net = net
        self._params = params
        self._opt_state = opt_state
        
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Call the wrapped predict function."""
        return self._predict_fn(tensor, target, variables)
    
    def update(self, samples: List[Any], posterior: Any, 
               target: str = None, variables: List[str] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Update surrogate with new data (for active learning).
        
        Args:
            samples: New intervention samples
            posterior: Current posterior beliefs
            target: Target variable (optional)
            variables: All variables (optional)
            
        Returns:
            Updated params and metrics dict
        """
        if self._update_fn is None:
            return self._params, {"error": "No update function available"}
        
        if self._params is None or self._opt_state is None:
            return self._params, {"error": "No params/opt_state to update"}
        
        # Call the update function with the expected signature
        # Based on the code, update_fn expects: (params, opt_state, posterior, samples, vars, target)
        new_params, new_opt_state, metrics = self._update_fn(
            self._params,
            self._opt_state, 
            posterior,
            samples,
            variables or [],  # Provide empty list if not given
            target or "Y"     # Default to Y if not given
        )
        
        # Update internal state
        self._params = new_params
        self._opt_state = new_opt_state
        
        # Recreate predict function with updated params
        if self._net is not None and new_params is not None:
            # Create new predict function with updated params
            def new_predict_fn(tensor: jnp.ndarray, target: str, vars: List[str]) -> Dict[str, Any]:
                """Predict using updated parameters."""
                vars_to_use = vars if vars else variables
                target_idx = vars_to_use.index(target) if target in vars_to_use else len(vars_to_use) - 1
                
                # Apply model with new params
                output = self._net.apply(new_params, None, tensor, target_idx, False)
                
                # Extract parent probabilities
                if isinstance(output, dict) and 'parent_probabilities' in output:
                    parent_probs = output['parent_probabilities']
                else:
                    parent_probs = output
                
                # Convert to marginal probabilities
                posterior = {}
                for i, var in enumerate(vars_to_use):
                    if i < len(parent_probs):
                        posterior[var] = float(parent_probs[i])
                    else:
                        posterior[var] = 0.0
                
                # Calculate entropy
                probs = jnp.array(list(posterior.values()))
                probs = probs[probs > 0]  # Remove zeros
                entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
                
                return {
                    'parent_probs': posterior,
                    'entropy': float(entropy),
                    'confidence': float(1.0 - entropy / jnp.log(len(vars_to_use)))
                }
            
            self._predict_fn = new_predict_fn
        
        return new_params, metrics
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def surrogate_type(self) -> str:
        return "bc_active" if self._is_active else "bc_static"


class FunctionSurrogateWrapper(SurrogateInterface):
    """
    Generic wrapper for function-based surrogates.
    
    This wraps any callable that follows the (tensor, target, variables) -> dict
    signature into a proper SurrogateInterface.
    """
    
    def __init__(self,
                 predict_fn: Callable,
                 name: str = "Function_Surrogate",
                 surrogate_type: str = "function"):
        """
        Initialize wrapper around a prediction function.
        
        Args:
            predict_fn: Function with signature (tensor, target, variables) -> dict
            name: Name for logging
            surrogate_type: Type identifier
        """
        self._predict_fn = predict_fn
        self._name = name
        self._surrogate_type = surrogate_type
        
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Call the wrapped predict function."""
        return self._predict_fn(tensor, target, variables)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def surrogate_type(self) -> str:
        return self._surrogate_type


def wrap_bc_surrogate(predict_fn: Callable,
                      update_fn: Optional[Callable] = None,
                      is_active: bool = False) -> BCSurrogateWrapper:
    """
    Convenience function to wrap BC surrogate functions.
    
    Args:
        predict_fn: Prediction function from create_bc_surrogate
        update_fn: Optional update function
        is_active: Whether active learning is enabled
        
    Returns:
        BCSurrogateWrapper instance
    """
    name = "BC_Active" if is_active else "BC_Static"
    return BCSurrogateWrapper(predict_fn, update_fn, name, is_active)