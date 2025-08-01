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
                 is_active: bool = False):
        """
        Initialize wrapper around BC surrogate functions.
        
        Args:
            predict_fn: The prediction function from create_bc_surrogate
            update_fn: Optional update function for active learning
            name: Name for logging
            is_active: Whether this supports active learning
        """
        self._predict_fn = predict_fn
        self._update_fn = update_fn
        self._name = name
        self._is_active = is_active
        
        # Track state for active learning
        self._params = None
        self._opt_state = None
        
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Call the wrapped predict function."""
        return self._predict_fn(tensor, target, variables)
    
    def update(self, samples: List[Any], posterior: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Update surrogate with new data (for active learning).
        
        Note: This requires the update function to be provided and
        the params/opt_state to be tracked.
        """
        if self._update_fn is None:
            return None, {}
        
        # TODO: Implement proper state tracking for active learning
        # For now, return empty metrics
        return None, {"skipped": True}
    
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