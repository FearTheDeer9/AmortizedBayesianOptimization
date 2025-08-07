"""
Centralized registry for managing surrogates in ACBO evaluation.

This module provides a clean registry pattern for surrogate management,
replacing the scattered conditional logic with a unified approach.
"""

from typing import Dict, Optional, Union, Callable
from pathlib import Path
import logging

from .surrogate_interface import SurrogateInterface, DummySurrogate, create_surrogate
from .surrogate_wrappers import BCSurrogateWrapper, FunctionSurrogateWrapper, wrap_bc_surrogate
from ..evaluation.model_interfaces import create_bc_surrogate

logger = logging.getLogger(__name__)


class SurrogateRegistry:
    """
    Central registry for all surrogates used in evaluation.
    
    This provides:
    - Consistent registration and retrieval
    - Automatic wrapping of function-based surrogates
    - Clear logging of surrogate types
    - Factory methods for common surrogate types
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._surrogates: Dict[str, SurrogateInterface] = {}
        self._default_surrogate: Optional[SurrogateInterface] = None
        
    def register(self, name: str, surrogate: Union[SurrogateInterface, Callable, str, Path]) -> None:
        """
        Register a surrogate with the given name.
        
        Args:
            name: Name to register under
            surrogate: Can be:
                - SurrogateInterface instance
                - Callable with (tensor, target, variables) signature
                - String identifier ('dummy', 'none', 'active_learning')
                - Path to checkpoint file
        """
        logger.info(f"Registering surrogate '{name}'")
        
        # Handle different input types
        if isinstance(surrogate, SurrogateInterface):
            # Already a proper surrogate
            self._surrogates[name] = surrogate
            logger.info(f"  Registered {surrogate.name} (type: {surrogate.surrogate_type})")
            
        elif callable(surrogate):
            # Raw function - wrap it
            wrapped = FunctionSurrogateWrapper(surrogate, name=name)
            self._surrogates[name] = wrapped
            logger.info(f"  Registered function surrogate '{name}'")
            
        elif isinstance(surrogate, str):
            # String identifier
            if surrogate.lower() in ['dummy', 'none']:
                self._surrogates[name] = DummySurrogate()
                logger.info(f"  Registered dummy surrogate (uniform probabilities)")
            elif surrogate.lower() == 'active_learning':
                # Mark for special handling - no actual surrogate yet
                self._surrogates[name] = 'active_learning_placeholder'
                logger.info(f"  Registered placeholder for pure active learning")
            else:
                # Assume it's a path
                self._register_from_path(name, Path(surrogate))
                
        elif isinstance(surrogate, Path):
            self._register_from_path(name, surrogate)
            
        else:
            raise ValueError(f"Unknown surrogate type: {type(surrogate)}")
    
    def _register_from_path(self, name: str, path: Path) -> None:
        """Register a surrogate from a checkpoint path."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Determine if active learning is requested
        is_active = name.endswith('_active')
        
        try:
            # Use the load_bc_surrogate method which handles active learning properly
            wrapped = self.load_bc_surrogate(path, is_active=is_active)
            self._surrogates[name] = wrapped
            logger.info(f"  Loaded BC surrogate from {path} (active: {is_active})")
        except Exception as e:
            logger.error(f"  Failed to load surrogate from {path}: {e}")
            raise
    
    def get(self, name: str) -> Optional[SurrogateInterface]:
        """
        Get a registered surrogate by name.
        
        Args:
            name: Name to look up
            
        Returns:
            SurrogateInterface or None if not found
        """
        return self._surrogates.get(name)
    
    def get_or_default(self, name: str) -> SurrogateInterface:
        """
        Get a surrogate or return the default.
        
        Args:
            name: Name to look up
            
        Returns:
            SurrogateInterface (registered one or default)
            
        Raises:
            ValueError if no surrogate found and no default set
        """
        surrogate = self._surrogates.get(name)
        if surrogate is not None:
            return surrogate
        
        if self._default_surrogate is not None:
            logger.warning(f"Surrogate '{name}' not found, using default")
            return self._default_surrogate
        
        raise ValueError(f"Surrogate '{name}' not found and no default set")
    
    def set_default(self, surrogate: Union[SurrogateInterface, str]) -> None:
        """
        Set the default surrogate to use when a requested one isn't found.
        
        Args:
            surrogate: SurrogateInterface or name of registered surrogate
        """
        if isinstance(surrogate, str):
            surrogate = self.get_or_default(surrogate)
        self._default_surrogate = surrogate
        logger.info(f"Set default surrogate to {surrogate.name}")
    
    def list_registered(self) -> Dict[str, str]:
        """
        List all registered surrogates.
        
        Returns:
            Dict mapping names to surrogate types
        """
        result = {}
        for name, surrogate in self._surrogates.items():
            if isinstance(surrogate, SurrogateInterface):
                result[name] = f"{surrogate.name} ({surrogate.surrogate_type})"
            else:
                result[name] = str(surrogate)
        return result
    
    def create_dummy(self) -> SurrogateInterface:
        """Create a dummy surrogate (convenience method)."""
        return DummySurrogate()
    
    def load_bc_surrogate(self, 
                         checkpoint_path: Path,
                         is_active: bool = False) -> SurrogateInterface:
        """
        Load a BC surrogate from checkpoint (convenience method).
        
        Args:
            checkpoint_path: Path to checkpoint
            is_active: Whether to enable active learning
            
        Returns:
            BCSurrogateWrapper instance
        """
        if is_active:
            # Get all components for active learning
            predict_fn, update_fn, net, params, opt_state = create_bc_surrogate(
                checkpoint_path, 
                allow_updates=True,
                return_components=True
            )
            # Create wrapper with all components
            wrapped = BCSurrogateWrapper(
                predict_fn, update_fn, 
                name="BC_Active", 
                is_active=True,
                net=net,
                params=params,
                opt_state=opt_state
            )
        else:
            # Static version - just get functions
            predict_fn, update_fn = create_bc_surrogate(checkpoint_path, allow_updates=False)
            wrapped = wrap_bc_surrogate(predict_fn, update_fn, is_active)
        
        wrapped._checkpoint_path = checkpoint_path
        return wrapped


# Global registry instance
_global_registry = SurrogateRegistry()


def get_registry() -> SurrogateRegistry:
    """Get the global surrogate registry."""
    return _global_registry


def register_surrogate(name: str, surrogate: Union[SurrogateInterface, Callable, str, Path]) -> None:
    """Register a surrogate in the global registry."""
    _global_registry.register(name, surrogate)


def get_surrogate(name: str) -> Optional[SurrogateInterface]:
    """Get a surrogate from the global registry."""
    return _global_registry.get(name)