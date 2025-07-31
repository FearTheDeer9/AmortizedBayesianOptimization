"""
Unified interface for all surrogate models in ACBO evaluation.

This module provides a consistent interface for static and active learning
surrogates, replacing the previous patchy approach with multiple conditionals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import jax.numpy as jnp
from pathlib import Path

from ..avici_integration.parent_set.posterior import ParentSetPosterior


class SurrogateInterface(ABC):
    """
    Base interface for all surrogate models.
    
    This provides a consistent API for:
    - Static surrogates (e.g., pre-trained BC)
    - Active learning surrogates (learning from scratch)
    - Hybrid surrogates (pre-trained + active learning)
    """
    
    @abstractmethod
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """
        Predict parent posterior for the target variable.
        
        Args:
            tensor: [T, n_vars, 3] tensor in 3-channel format
            target: Target variable name
            variables: List of variable names in tensor order
            
        Returns:
            Posterior dictionary or ParentSetPosterior object
        """
        pass
    
    def update(self, samples: List[Any], posterior: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Update surrogate with new data (for active learning).
        
        Args:
            samples: New samples to learn from
            posterior: Current posterior prediction
            
        Returns:
            Tuple of (updated_state, metrics_dict)
            
        Note:
            Default implementation is no-op for static surrogates.
        """
        return None, {}
    
    @property
    def is_active(self) -> bool:
        """Whether this surrogate supports active learning."""
        return False
    
    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        return self.__class__.__name__


class StaticBCSurrogate(SurrogateInterface):
    """Static BC surrogate (no active learning)."""
    
    def __init__(self, checkpoint_path: Path):
        """
        Initialize from BC checkpoint.
        
        Args:
            checkpoint_path: Path to BC surrogate checkpoint
        """
        from ..evaluation.model_interfaces import create_bc_surrogate
        self._predict_fn, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)
    
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Use pre-trained BC model for prediction."""
        return self._predict_fn(tensor, target, variables)
    
    @property
    def name(self) -> str:
        return "BC_Static"


class ActiveBCSurrogate(SurrogateInterface):
    """BC surrogate with active learning enabled."""
    
    def __init__(self, checkpoint_path: Path, learning_rate: float = 1e-4):
        """
        Initialize from BC checkpoint with active learning.
        
        Args:
            checkpoint_path: Path to BC surrogate checkpoint
            learning_rate: Learning rate for updates
        """
        from ..evaluation.model_interfaces import create_bc_surrogate
        self._predict_fn, self._update_fn = create_bc_surrogate(
            checkpoint_path, allow_updates=True, learning_rate=learning_rate
        )
        self._params = None  # Will be set by evaluator
        self._opt_state = None
    
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Use BC model with current parameters."""
        return self._predict_fn(tensor, target, variables)
    
    def update(self, samples: List[Any], posterior: Any) -> Tuple[Any, Dict[str, float]]:
        """Update BC parameters with new data."""
        if self._update_fn is None:
            return None, {}
        
        # Note: Update function needs to be implemented in model_interfaces
        # For now, return empty metrics
        return None, {"loss": 0.0}
    
    @property
    def is_active(self) -> bool:
        return True
    
    @property
    def name(self) -> str:
        return "BC_Active"


class PureActiveSurrogate(SurrogateInterface):
    """Pure active learning surrogate (no pre-training)."""
    
    def __init__(self, scm: Any, learning_rate: float = 1e-3, seed: int = 42):
        """
        Initialize active learning from scratch.
        
        Args:
            scm: The SCM (for variable names)
            learning_rate: Learning rate for updates
            seed: Random seed
        """
        from ..evaluation.model_interfaces import create_learning_surrogate
        self._predict_fn, self._update_fn = create_learning_surrogate(
            scm, learning_rate=learning_rate, seed=seed
        )
    
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Use current learned model."""
        return self._predict_fn(tensor, target, variables)
    
    def update(self, samples: List[Any], posterior: Any) -> Tuple[Any, Dict[str, float]]:
        """Update with new data."""
        # Active learning update
        # Note: Implementation depends on active learning module
        return None, {"loss": 0.0}
    
    @property
    def is_active(self) -> bool:
        return True
    
    @property
    def name(self) -> str:
        return "Pure_Active"


class DummySurrogate(SurrogateInterface):
    """Dummy surrogate for baseline evaluation (uniform probabilities)."""
    
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """Return uniform probabilities."""
        # Create uniform marginal probabilities
        marginals = {}
        for var in variables:
            if var != target:
                marginals[var] = 0.5  # Uniform probability
            else:
                marginals[var] = 0.0  # Target can't be its own parent
        
        return {
            'marginal_parent_probs': marginals,
            'entropy': 1.0,
            'model_type': 'dummy'
        }
    
    @property
    def name(self) -> str:
        return "Dummy"


def create_surrogate(surrogate_type: str, **kwargs) -> SurrogateInterface:
    """
    Factory function to create surrogates.
    
    Args:
        surrogate_type: Type of surrogate ('static_bc', 'active_bc', 'pure_active', 'dummy')
        **kwargs: Additional arguments for the specific surrogate
        
    Returns:
        SurrogateInterface instance
    """
    if surrogate_type == 'static_bc':
        return StaticBCSurrogate(kwargs['checkpoint_path'])
    elif surrogate_type == 'active_bc':
        return ActiveBCSurrogate(kwargs['checkpoint_path'], kwargs.get('learning_rate', 1e-4))
    elif surrogate_type == 'pure_active':
        return PureActiveSurrogate(kwargs['scm'], kwargs.get('learning_rate', 1e-3), kwargs.get('seed', 42))
    elif surrogate_type == 'dummy':
        return DummySurrogate()
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")