"""
Unified interface for all surrogate models in ACBO evaluation.

This module provides a consistent interface for static and active learning
surrogates, replacing the previous patchy approach with multiple conditionals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import jax.numpy as jnp
from pathlib import Path

from ..avici_integration.parent_set.posterior import ParentSetPosterior

if TYPE_CHECKING:
    from ..utils.update_functions import UpdateFunction


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
    
    @property
    @abstractmethod
    def surrogate_type(self) -> str:
        """
        Type identifier for the surrogate.
        
        Returns one of: 'dummy', 'bc_static', 'bc_active', 'pure_active', etc.
        This is used for clean type checking without hacky try/except blocks.
        """
        pass
    
    def get_update_function(self) -> Optional['UpdateFunction']:
        """
        Get the update function if this surrogate supports updates.
        
        Returns:
            UpdateFunction instance or None if updates not supported
        """
        return None
    
    @property
    def supports_updates(self) -> bool:
        """Whether this surrogate supports parameter updates."""
        return self.get_update_function() is not None


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
    
    @property
    def surrogate_type(self) -> str:
        return "bc_static"


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
    
    @property
    def surrogate_type(self) -> str:
        return "bc_active"


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
    
    @property
    def surrogate_type(self) -> str:
        return "pure_active"


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
    
    @property
    def surrogate_type(self) -> str:
        return "dummy"


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


class ActiveLearningSurrogateWrapper(SurrogateInterface):
    """
    Wrapper that adds active learning capabilities to any surrogate.
    
    This wrapper:
    - Manages params and opt_state internally
    - Provides UpdateFunction integration
    - Supports both 'none' and 'bic' update strategies
    - Works with any base surrogate (even dummy)
    """
    
    def __init__(
        self,
        base_surrogate: SurrogateInterface,
        update_strategy: str = "none",
        learning_rate: float = 1e-3,
        min_samples: int = 5,
        net: Optional[Any] = None,
        initial_params: Optional[Any] = None,
        initial_opt_state: Optional[Any] = None,
        seed: int = 42
    ):
        """
        Initialize active learning wrapper.
        
        Args:
            base_surrogate: Base surrogate to wrap
            update_strategy: Update strategy ('none' or 'bic')
            learning_rate: Learning rate for updates
            min_samples: Minimum samples needed for update
            net: Haiku network (required for 'bic' strategy)
            initial_params: Initial parameters (optional)
            initial_opt_state: Initial optimizer state (optional)
            seed: Random seed
        """
        self.base_surrogate = base_surrogate
        self.update_strategy = update_strategy
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.net = net
        self.seed = seed
        
        # Initialize params and optimizer
        if update_strategy == "bic":
            if net is None:
                # Try to extract net from base surrogate if it's a BC surrogate
                if hasattr(base_surrogate, '_net'):
                    self.net = base_surrogate._net
                else:
                    raise ValueError("BIC update strategy requires a network")
            
            # Initialize optimizer
            import optax
            self.optimizer = optax.adam(learning_rate)
            
            # Initialize params
            if initial_params is not None:
                self.params = initial_params
            elif hasattr(base_surrogate, '_params'):
                self.params = base_surrogate._params
            else:
                # Initialize fresh params
                import jax.random as random
                import jax.numpy as jnp
                key = random.PRNGKey(seed)
                dummy_data = jnp.zeros((10, 10, 3))  # Dummy data
                self.params = self.net.init(key, dummy_data, 0, False)
            
            # Initialize optimizer state
            if initial_opt_state is not None:
                self.opt_state = initial_opt_state
            else:
                self.opt_state = self.optimizer.init(self.params)
        else:
            # No-op strategy
            self.params = initial_params
            self.opt_state = initial_opt_state
            self.optimizer = None
        
        # Create update function
        self._create_update_function()
    
    def _create_update_function(self):
        """Create the update function based on strategy."""
        from ..utils.update_functions import create_update_function
        
        if self.update_strategy == "bic" and self.net is not None:
            self.update_fn = create_update_function(
                strategy="bic",
                net=self.net,
                learning_rate=self.learning_rate,
                min_samples=self.min_samples
            )
        else:
            self.update_fn = create_update_function(strategy="no_op")
    
    def predict(self, tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """
        Make predictions using current parameters.
        
        For BIC strategy with a network, uses the network with current params.
        Otherwise, delegates to base surrogate.
        """
        if self.update_strategy == "bic" and self.net is not None and self.params is not None:
            # Use network with current params
            import jax.random as random
            key = random.PRNGKey(self.seed)
            
            # Find target index
            target_idx = variables.index(target) if target in variables else 0
            
            # Forward pass
            outputs = self.net.apply(self.params, key, tensor, target_idx, False)
            
            # Convert to expected format
            if isinstance(outputs, dict) and 'marginal_parent_probs' in outputs:
                return outputs
            else:
                # Extract marginals from parent set predictions if available
                from ..avici_integration.parent_set.posterior import (
                    create_parent_set_posterior, get_marginal_parent_probabilities
                )
                
                if 'parent_sets' in outputs and 'parent_set_probs' in outputs:
                    posterior = create_parent_set_posterior(
                        target_variable=target,
                        parent_sets=outputs['parent_sets'],
                        probabilities=outputs['parent_set_probs'],
                        metadata={'source': 'active_learning'}
                    )
                    
                    marginals = get_marginal_parent_probabilities(posterior, variables)
                    return {
                        'marginal_parent_probs': marginals,
                        'entropy': float(posterior.uncertainty),
                        'model_type': 'active_bic',
                        'posterior': posterior
                    }
                else:
                    # Fallback to base surrogate
                    return self.base_surrogate.predict(tensor, target, variables)
        else:
            # Delegate to base surrogate
            return self.base_surrogate.predict(tensor, target, variables)
    
    def update(self, samples: List[Any], posterior: Any) -> Tuple[Any, Dict[str, float]]:
        """Update wrapper for compatibility."""
        # This method is for the old interface compatibility
        # The actual updates happen through get_update_function()
        return None, {"skipped": True, "reason": "use_update_function"}
    
    def get_update_function(self) -> Optional['UpdateFunction']:
        """Get the update function for universal evaluator."""
        if self.update_strategy != "none":
            return self.update_fn
        return None
    
    @property
    def is_active(self) -> bool:
        """Whether this surrogate supports active learning."""
        return self.update_strategy != "none"
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        base_name = self.base_surrogate.name
        if self.update_strategy == "none":
            return f"{base_name}_wrapped"
        return f"{base_name}_active_{self.update_strategy}"
    
    @property
    def surrogate_type(self) -> str:
        """Type identifier."""
        base_type = self.base_surrogate.surrogate_type
        if self.update_strategy == "none":
            return base_type
        return f"{base_type}_active_{self.update_strategy}"
    
    @property
    def supports_updates(self) -> bool:
        """Whether this surrogate supports updates."""
        return self.update_strategy != "none"
    
    def get_params(self) -> Optional[Any]:
        """Get current parameters for evaluator."""
        return self.params
    
    def get_opt_state(self) -> Optional[Any]:
        """Get current optimizer state for evaluator."""
        return self.opt_state
    
    def set_params(self, params: Any) -> None:
        """Update parameters after external update."""
        self.params = params
    
    def set_opt_state(self, opt_state: Any) -> None:
        """Update optimizer state after external update."""
        self.opt_state = opt_state