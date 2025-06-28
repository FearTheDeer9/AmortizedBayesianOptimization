#!/usr/bin/env python3
"""
Mechanism Descriptors for Serialization

This module provides serializable descriptions of mechanism functions that can
be used to recreate the original mechanisms after pickle/JSON serialization.

The descriptors store the "quote" of the mechanism (its configuration) rather
than the actual function, enabling perfect reconstruction while being pickle-safe.
"""

from dataclasses import dataclass
from typing import Dict, List, Union, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Type aliases for clarity
MechanismFunction = Callable[[Dict[str, float], Any], float]  # (parent_values, noise_key) -> value


@dataclass
class LinearMechanismDescriptor:
    """
    Serializable descriptor for linear mechanisms.
    
    Stores all information needed to recreate a linear mechanism:
    Y = intercept + sum(coefficient_i * parent_i) + noise
    """
    parents: List[str]
    coefficients: Dict[str, float]
    intercept: float
    noise_scale: float
    mechanism_type: str = "linear"
    
    def __post_init__(self):
        """Validate descriptor consistency."""
        if set(self.parents) != set(self.coefficients.keys()):
            raise ValueError(
                f"Parents {self.parents} don't match coefficient keys {list(self.coefficients.keys())}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mechanism_type": self.mechanism_type,
            "parents": self.parents,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "noise_scale": self.noise_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearMechanismDescriptor":
        """Create descriptor from dictionary."""
        return cls(
            mechanism_type=data["mechanism_type"],
            parents=data["parents"],
            coefficients=data["coefficients"],
            intercept=data["intercept"],
            noise_scale=data["noise_scale"]
        )


@dataclass
class RootMechanismDescriptor:
    """
    Serializable descriptor for root mechanisms (variables with no parents).
    
    Stores all information needed to recreate a root mechanism:
    Y = mean + noise
    """
    mean: float
    noise_scale: float
    mechanism_type: str = "root"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mechanism_type": self.mechanism_type,
            "mean": self.mean,
            "noise_scale": self.noise_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RootMechanismDescriptor":
        """Create descriptor from dictionary."""
        return cls(
            mechanism_type=data["mechanism_type"],
            mean=data["mean"],
            noise_scale=data["noise_scale"]
        )


# Union type for all mechanism descriptors
MechanismDescriptor = Union[LinearMechanismDescriptor, RootMechanismDescriptor]


def descriptor_to_mechanism(descriptor: MechanismDescriptor) -> MechanismFunction:
    """
    Recreate a mechanism function from its descriptor.
    
    Args:
        descriptor: The mechanism descriptor to convert
        
    Returns:
        A mechanism function equivalent to the original
        
    Raises:
        ValueError: If descriptor type is not recognized
    """
    if isinstance(descriptor, LinearMechanismDescriptor):
        # Import here to avoid circular imports
        from .linear import create_linear_mechanism
        
        return create_linear_mechanism(
            parents=descriptor.parents,
            coefficients=descriptor.coefficients,
            intercept=descriptor.intercept,
            noise_scale=descriptor.noise_scale
        )
    
    elif isinstance(descriptor, RootMechanismDescriptor):
        # Import here to avoid circular imports
        from .linear import create_root_mechanism
        
        return create_root_mechanism(
            mean=descriptor.mean,
            noise_scale=descriptor.noise_scale
        )
    
    else:
        raise ValueError(f"Unknown mechanism descriptor type: {type(descriptor)}")


def mechanism_to_descriptor(
    mechanism_func: MechanismFunction,
    mechanism_type: str,
    **kwargs
) -> MechanismDescriptor:
    """
    Create a descriptor from mechanism parameters.
    
    Note: This function requires the original mechanism parameters since
    we cannot introspect the closure variables from the function alone.
    
    Args:
        mechanism_func: The mechanism function (currently unused but kept for API)
        mechanism_type: Type of mechanism ("linear" or "root")
        **kwargs: Mechanism parameters based on type
        
    Returns:
        Appropriate mechanism descriptor
        
    Raises:
        ValueError: If mechanism_type is not recognized or required parameters missing
    """
    if mechanism_type == "linear":
        required_params = {"parents", "coefficients", "intercept", "noise_scale"}
        if not required_params.issubset(kwargs.keys()):
            missing = required_params - kwargs.keys()
            raise ValueError(f"Missing required parameters for linear mechanism: {missing}")
        
        return LinearMechanismDescriptor(
            parents=kwargs["parents"],
            coefficients=kwargs["coefficients"],
            intercept=kwargs["intercept"],
            noise_scale=kwargs["noise_scale"]
        )
    
    elif mechanism_type == "root":
        required_params = {"mean", "noise_scale"}
        if not required_params.issubset(kwargs.keys()):
            missing = required_params - kwargs.keys()
            raise ValueError(f"Missing required parameters for root mechanism: {missing}")
        
        return RootMechanismDescriptor(
            mean=kwargs["mean"],
            noise_scale=kwargs["noise_scale"]
        )
    
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")


def descriptors_to_dict(descriptors: Dict[str, MechanismDescriptor]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a dictionary of mechanism descriptors to JSON-serializable format.
    
    Args:
        descriptors: Dictionary mapping variable names to descriptors
        
    Returns:
        JSON-serializable dictionary
    """
    return {var: desc.to_dict() for var, desc in descriptors.items()}


def descriptors_from_dict(data: Dict[str, Dict[str, Any]]) -> Dict[str, MechanismDescriptor]:
    """
    Recreate mechanism descriptors from JSON data.
    
    Args:
        data: JSON data with mechanism descriptors
        
    Returns:
        Dictionary mapping variable names to descriptors
    """
    descriptors = {}
    
    for var, desc_data in data.items():
        mechanism_type = desc_data["mechanism_type"]
        
        if mechanism_type == "linear":
            descriptors[var] = LinearMechanismDescriptor.from_dict(desc_data)
        elif mechanism_type == "root":
            descriptors[var] = RootMechanismDescriptor.from_dict(desc_data)
        else:
            raise ValueError(f"Unknown mechanism type for variable {var}: {mechanism_type}")
    
    return descriptors