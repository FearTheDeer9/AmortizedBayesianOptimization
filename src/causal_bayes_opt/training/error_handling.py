"""
Comprehensive error handling utilities for ACBO training.

This module provides pure functions for error handling, recovery strategies,
and robust training operations following functional programming principles.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import pyrsistent as pyr
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Warning, can continue
    MEDIUM = "medium"     # Error, can retry
    HIGH = "high"         # Critical, needs intervention
    FATAL = "fatal"       # Unrecoverable


class ErrorCategory(Enum):
    """Error categories for classification."""
    MEMORY = "memory"
    NUMERICAL = "numerical"  
    IO = "io"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrainingError:
    """Immutable error information."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: Dict[str, Any]
    traceback_str: str
    stage: str
    retry_count: int = 0
    recoverable: bool = True


@dataclass(frozen=True)
class ErrorRecoveryStrategy:
    """Immutable recovery strategy configuration."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    base_delay: float = 1.0
    memory_reduction_factor: float = 0.5
    enable_checkpoint_recovery: bool = True
    enable_parameter_reset: bool = False


@dataclass(frozen=True)
class ErrorContext:
    """Immutable error context for tracking."""
    errors: Tuple[TrainingError, ...] = ()
    total_errors: int = 0
    recovery_attempts: int = 0
    last_successful_checkpoint: Optional[str] = None
    current_stage: str = "unknown"


# Pure error classification functions
def classify_error(exception: Exception, context: Dict[str, Any]) -> ErrorCategory:
    """
    Classify error into appropriate category.
    
    Args:
        exception: The exception that occurred
        context: Additional context information
        
    Returns:
        ErrorCategory for the exception
    """
    error_msg = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    # Memory-related errors
    if any(keyword in error_msg for keyword in ['memory', 'oom', 'out of memory', 'cuda']):
        return ErrorCategory.MEMORY
    
    # Numerical errors
    if any(keyword in error_msg for keyword in ['nan', 'inf', 'overflow', 'underflow']):
        return ErrorCategory.NUMERICAL
    
    # Network errors (check before I/O to catch connection errors)
    if (any(keyword in error_type for keyword in ['connection', 'timeout']) or
        any(keyword in error_msg for keyword in ['network', 'connection', 'timeout', 'refused'])):
        return ErrorCategory.NETWORK
    
    # I/O errors - check both type and message
    if (any(keyword in error_type for keyword in ['io', 'file', 'permission', 'os']) or
        any(keyword in error_msg for keyword in ['file not found', 'permission denied', 'i/o'])):
        return ErrorCategory.IO
    
    # Configuration errors
    if any(keyword in error_msg for keyword in ['config', 'parameter', 'setting']):
        return ErrorCategory.CONFIGURATION
    
    # Data errors
    if any(keyword in error_msg for keyword in ['data', 'batch', 'shape', 'dimension']):
        return ErrorCategory.DATA
    
    # Model errors
    if any(keyword in error_msg for keyword in ['model', 'layer', 'weight']):
        return ErrorCategory.MODEL
    
    return ErrorCategory.UNKNOWN


def determine_error_severity(
    error_category: ErrorCategory,
    retry_count: int,
    context: Dict[str, Any]
) -> ErrorSeverity:
    """
    Determine error severity based on category and context.
    
    Args:
        error_category: Category of the error
        retry_count: Number of previous retry attempts
        context: Additional context information
        
    Returns:
        ErrorSeverity level
    """
    # Fatal errors after too many retries
    if retry_count >= 5:
        return ErrorSeverity.FATAL
    
    # Category-based severity
    severity_map = {
        ErrorCategory.MEMORY: ErrorSeverity.HIGH,
        ErrorCategory.NUMERICAL: ErrorSeverity.MEDIUM,
        ErrorCategory.IO: ErrorSeverity.MEDIUM,
        ErrorCategory.NETWORK: ErrorSeverity.LOW,
        ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
        ErrorCategory.DATA: ErrorSeverity.MEDIUM,
        ErrorCategory.MODEL: ErrorSeverity.MEDIUM,
        ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
    }
    
    base_severity = severity_map.get(error_category, ErrorSeverity.MEDIUM)
    
    # Escalate severity with retries
    if retry_count >= 3:
        if base_severity == ErrorSeverity.LOW:
            return ErrorSeverity.MEDIUM
        elif base_severity == ErrorSeverity.MEDIUM:
            return ErrorSeverity.HIGH
    
    return base_severity


def create_training_error(
    exception: Exception,
    stage: str,
    context: Dict[str, Any],
    retry_count: int = 0
) -> TrainingError:
    """
    Create TrainingError from exception with full context.
    
    Args:
        exception: The exception that occurred
        stage: Training stage where error occurred
        context: Additional context information
        retry_count: Number of previous retry attempts
        
    Returns:
        TrainingError with complete information
    """
    error_category = classify_error(exception, context)
    severity = determine_error_severity(error_category, retry_count, context)
    
    # Generate unique error ID
    error_id = f"{stage}_{error_category.value}_{int(time.time())}"
    
    # Determine if error is recoverable
    recoverable = (
        severity != ErrorSeverity.FATAL and
        retry_count < 5 and
        error_category != ErrorCategory.CONFIGURATION
    )
    
    return TrainingError(
        error_id=error_id,
        timestamp=time.time(),
        severity=severity,
        category=error_category,
        message=str(exception),
        context=context,
        traceback_str=traceback.format_exc(),
        stage=stage,
        retry_count=retry_count,
        recoverable=recoverable
    )


# Recovery strategy functions
def create_recovery_strategy(
    error: TrainingError,
    error_context: ErrorContext
) -> ErrorRecoveryStrategy:
    """
    Create appropriate recovery strategy based on error characteristics.
    
    Args:
        error: The training error to recover from
        error_context: Current error context
        
    Returns:
        ErrorRecoveryStrategy tailored to the error
    """
    # Base strategy
    strategy = ErrorRecoveryStrategy()
    
    # Customize based on error category
    if error.category == ErrorCategory.MEMORY:
        return ErrorRecoveryStrategy(
            max_retries=2,
            backoff_factor=1.5,
            base_delay=2.0,
            memory_reduction_factor=0.3,  # Aggressive memory reduction
            enable_checkpoint_recovery=True,
            enable_parameter_reset=False
        )
    
    elif error.category == ErrorCategory.NUMERICAL:
        return ErrorRecoveryStrategy(
            max_retries=3,
            backoff_factor=2.0,
            base_delay=0.5,
            memory_reduction_factor=1.0,  # No memory reduction
            enable_checkpoint_recovery=True,
            enable_parameter_reset=True  # Reset parameters on numerical issues
        )
    
    elif error.category == ErrorCategory.IO:
        return ErrorRecoveryStrategy(
            max_retries=5,
            backoff_factor=3.0,  # Longer backoff for I/O
            base_delay=1.0,
            memory_reduction_factor=1.0,
            enable_checkpoint_recovery=False,  # I/O issues might affect checkpoints
            enable_parameter_reset=False
        )
    
    elif error.category == ErrorCategory.NETWORK:
        return ErrorRecoveryStrategy(
            max_retries=10,  # Network issues are often transient
            backoff_factor=2.0,
            base_delay=5.0,  # Longer initial delay
            memory_reduction_factor=1.0,
            enable_checkpoint_recovery=False,
            enable_parameter_reset=False
        )
    
    # Default strategy for other categories
    return strategy


def calculate_backoff_delay(
    attempt: int,
    strategy: ErrorRecoveryStrategy
) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (0-indexed)
        strategy: Recovery strategy configuration
        
    Returns:
        Delay in seconds before next retry
    """
    delay = strategy.base_delay * (strategy.backoff_factor ** attempt)
    # Cap at reasonable maximum
    return min(delay, 300.0)  # Max 5 minutes


# Error handling decorators and utilities
def safe_training_step(
    step_fn: Callable,
    error_context: ErrorContext,
    stage: str,
    max_retries: int = 3
) -> Callable:
    """
    Wrap training step with comprehensive error handling.
    
    Args:
        step_fn: Training step function to wrap
        error_context: Current error context
        stage: Training stage name
        max_retries: Maximum retry attempts
        
    Returns:
        Safe training step function
    """
    def safe_step(*args, **kwargs):
        """Safe training step with error handling."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return step_fn(*args, **kwargs)
                
            except Exception as e:
                # Create error record
                context_info = {
                    'attempt': attempt,
                    'args_shapes': _extract_shape_info(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'stage': stage
                }
                
                error = create_training_error(e, stage, context_info, attempt)
                last_error = error
                
                # Log error
                logger.warning(f"Training step failed (attempt {attempt + 1}/{max_retries + 1}): {error.message}")
                
                # Check if we should retry
                if attempt < max_retries and error.recoverable:
                    strategy = create_recovery_strategy(error, error_context)
                    delay = calculate_backoff_delay(attempt, strategy)
                    
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    
                    # Apply recovery modifications if needed
                    if strategy.memory_reduction_factor < 1.0:
                        # Reduce batch sizes in kwargs
                        kwargs = _reduce_batch_sizes(kwargs, strategy.memory_reduction_factor)
                else:
                    break
        
        # All retries failed
        logger.error(f"Training step failed after {max_retries + 1} attempts")
        raise last_error
    
    return safe_step


def _extract_shape_info(args: Tuple) -> Dict[str, Any]:
    """Extract shape information from arguments for debugging."""
    shape_info = {}
    for i, arg in enumerate(args):
        if hasattr(arg, 'shape'):
            shape_info[f'arg_{i}_shape'] = arg.shape
        elif isinstance(arg, (list, tuple)):
            shape_info[f'arg_{i}_len'] = len(arg)
    return shape_info


def _reduce_batch_sizes(kwargs: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """Reduce batch sizes in kwargs for memory recovery."""
    new_kwargs = kwargs.copy()
    
    # Look for batch size parameters
    batch_keys = ['batch_size', 'per_device_batch_size', 'train_batch_size']
    for key in batch_keys:
        if key in new_kwargs and isinstance(new_kwargs[key], int):
            new_size = max(1, int(new_kwargs[key] * factor))
            new_kwargs[key] = new_size
            logger.info(f"Reduced {key} from {kwargs[key]} to {new_size}")
    
    return new_kwargs


# Error context management
def add_error_to_context(
    context: ErrorContext,
    error: TrainingError
) -> ErrorContext:
    """Add error to error context (pure function)."""
    return ErrorContext(
        errors=context.errors + (error,),
        total_errors=context.total_errors + 1,
        recovery_attempts=context.recovery_attempts,
        last_successful_checkpoint=context.last_successful_checkpoint,
        current_stage=error.stage
    )


def should_abort_training(error_context: ErrorContext) -> bool:
    """Determine if training should be aborted based on error history."""
    # Abort if too many errors
    if error_context.total_errors > 20:
        return True
    
    # Abort if too many high-severity errors
    high_severity_errors = sum(
        1 for error in error_context.errors
        if error.severity in (ErrorSeverity.HIGH, ErrorSeverity.FATAL)
    )
    if high_severity_errors > 5:
        return True
    
    # Abort if recent error rate is too high
    recent_errors = [
        error for error in error_context.errors
        if time.time() - error.timestamp < 300  # Last 5 minutes
    ]
    if len(recent_errors) > 10:
        return True
    
    return False


__all__ = [
    'ErrorSeverity',
    'ErrorCategory', 
    'TrainingError',
    'ErrorRecoveryStrategy',
    'ErrorContext',
    'classify_error',
    'determine_error_severity',
    'create_training_error',
    'create_recovery_strategy',
    'calculate_backoff_delay',
    'safe_training_step',
    'add_error_to_context',
    'should_abort_training'
]