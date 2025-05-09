"""
Utilities for the graph module.

This module provides utility functions for the graph module, such as
deprecation warnings and other common functionality.
"""
import warnings
import functools
from typing import Callable, TypeVar, Any, cast

F = TypeVar('F', bound=Callable[..., Any])


def deprecated(old_name: str, new_name: str) -> Callable[[F], F]:
    """
    Decorator to mark methods as deprecated.

    Args:
        old_name: The old method name that is deprecated
        new_name: The new method name that should be used instead

    Returns:
        A decorator function that wraps the deprecated method
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"'{old_name}' is deprecated and will be removed in a future version. "
                f"Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator 