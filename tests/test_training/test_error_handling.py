"""
Comprehensive tests for error handling infrastructure.

This module tests error classification, recovery strategies, and safe training
operations using property-based testing with Hypothesis.
"""

import pytest
import time
import traceback
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
import jax.numpy as jnp
import jax.random as random

from causal_bayes_opt.training.error_handling import (
    ErrorSeverity, ErrorCategory, TrainingError, ErrorRecoveryStrategy,
    ErrorContext, classify_error, determine_error_severity,
    create_training_error, create_recovery_strategy, calculate_backoff_delay,
    safe_training_step, add_error_to_context, should_abort_training
)


class TestErrorClassification:
    """Test error classification functionality."""
    
    def test_memory_error_classification(self):
        """Test classification of memory-related errors."""
        memory_errors = [
            RuntimeError("CUDA out of memory"),
            Exception("OOM when allocating tensor"),
            MemoryError("Cannot allocate memory"),
            RuntimeError("memory limit exceeded")
        ]
        
        for error in memory_errors:
            category = classify_error(error, {})
            assert category == ErrorCategory.MEMORY
    
    def test_numerical_error_classification(self):
        """Test classification of numerical errors."""
        numerical_errors = [
            ValueError("Input contains NaN"),
            RuntimeError("Detected inf values"),
            OverflowError("Math overflow"),
            ArithmeticError("Numerical underflow")
        ]
        
        for error in numerical_errors:
            category = classify_error(error, {})
            assert category == ErrorCategory.NUMERICAL
    
    def test_io_error_classification(self):
        """Test classification of I/O errors."""
        io_errors = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            IOError("I/O operation failed")
        ]
        
        for error in io_errors:
            category = classify_error(error, {})
            assert category == ErrorCategory.IO
    
    def test_network_error_classification(self):
        """Test classification of network errors."""
        network_errors = [
            ConnectionError("Network connection failed"),
            TimeoutError("Request timeout"),
            Exception("Connection refused")
        ]
        
        for error in network_errors:
            category = classify_error(error, {})
            assert category == ErrorCategory.NETWORK
    
    def test_unknown_error_classification(self):
        """Test classification of unknown errors."""
        unknown_error = ValueError("Some random error")
        category = classify_error(unknown_error, {})
        assert category == ErrorCategory.UNKNOWN
    
    @given(
        error_message=st.text(min_size=1, max_size=100),
        context_keys=st.lists(st.text(min_size=1, max_size=20), max_size=5)
    )
    @settings(max_examples=20)
    def test_classify_error_properties(self, error_message, context_keys):
        """Property-based test for error classification."""
        error = Exception(error_message)
        context = {key: f"value_{i}" for i, key in enumerate(context_keys)}
        
        category = classify_error(error, context)
        
        # Should always return a valid category
        assert isinstance(category, ErrorCategory)
        assert category in list(ErrorCategory)


class TestErrorSeverity:
    """Test error severity determination."""
    
    @given(
        retry_count=st.integers(min_value=0, max_value=10),
        category=st.sampled_from(list(ErrorCategory))
    )
    @settings(max_examples=30)
    def test_severity_escalation_with_retries(self, retry_count, category):
        """Test that severity escalates with retry count."""
        severity = determine_error_severity(category, retry_count, {})
        
        # Fatal after too many retries
        if retry_count >= 5:
            assert severity == ErrorSeverity.FATAL
        
        # Should be a valid severity
        assert isinstance(severity, ErrorSeverity)
        assert severity in list(ErrorSeverity)
    
    def test_category_based_severity(self):
        """Test severity assignment based on error category."""
        # Memory errors should be high severity
        severity = determine_error_severity(ErrorCategory.MEMORY, 0, {})
        assert severity == ErrorSeverity.HIGH
        
        # Configuration errors should be high severity
        severity = determine_error_severity(ErrorCategory.CONFIGURATION, 0, {})
        assert severity == ErrorSeverity.HIGH
        
        # Network errors should be low severity initially
        severity = determine_error_severity(ErrorCategory.NETWORK, 0, {})
        assert severity == ErrorSeverity.LOW
        
        # But escalate with retries
        severity = determine_error_severity(ErrorCategory.NETWORK, 3, {})
        assert severity == ErrorSeverity.MEDIUM
    
    def test_severity_escalation(self):
        """Test severity escalation with increasing retry count."""
        # Start with low severity error
        initial_severity = determine_error_severity(ErrorCategory.NETWORK, 0, {})
        escalated_severity = determine_error_severity(ErrorCategory.NETWORK, 3, {})
        
        # Should escalate (or at least not decrease)
        severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.FATAL]
        initial_idx = severities.index(initial_severity)
        escalated_idx = severities.index(escalated_severity)
        
        assert escalated_idx >= initial_idx


class TestTrainingError:
    """Test TrainingError creation and properties."""
    
    def test_create_training_error_basic(self):
        """Test basic TrainingError creation."""
        exception = ValueError("Test error")
        stage = "test_stage"
        context = {"key": "value"}
        
        error = create_training_error(exception, stage, context)
        
        assert isinstance(error, TrainingError)
        assert error.message == "Test error"
        assert error.stage == stage
        assert error.context == context
        assert error.retry_count == 0
        assert error.error_id.startswith(f"{stage}_")
        assert isinstance(error.timestamp, float)
        assert isinstance(error.severity, ErrorSeverity)
        assert isinstance(error.category, ErrorCategory)
        assert isinstance(error.recoverable, bool)
    
    @given(
        retry_count=st.integers(min_value=0, max_value=10),
        stage=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd', 'Pc')))
    )
    @settings(max_examples=20)
    def test_create_training_error_properties(self, retry_count, stage):
        """Property-based test for TrainingError creation."""
        exception = RuntimeError("Test error")
        context = {"retry_count": retry_count}
        
        error = create_training_error(exception, stage, context, retry_count)
        
        # Properties that should always hold
        assert error.retry_count == retry_count
        assert error.stage == stage
        assert error.timestamp > 0
        
        # Error should become non-recoverable after many retries
        if retry_count >= 5:
            assert error.recoverable is False or error.severity == ErrorSeverity.FATAL
    
    def test_training_error_immutability(self):
        """Test that TrainingError is immutable."""
        error = create_training_error(ValueError("test"), "stage", {})
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            error.message = "new message"
        
        with pytest.raises(AttributeError):
            error.retry_count = 5


class TestRecoveryStrategy:
    """Test error recovery strategy creation."""
    
    def test_memory_error_recovery_strategy(self):
        """Test recovery strategy for memory errors."""
        error = TrainingError(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            message="OOM",
            context={},
            traceback_str="",
            stage="test",
            retry_count=0,
            recoverable=True
        )
        context = ErrorContext()
        
        strategy = create_recovery_strategy(error, context)
        
        assert isinstance(strategy, ErrorRecoveryStrategy)
        assert strategy.max_retries <= 3  # Conservative for memory errors
        assert strategy.memory_reduction_factor < 1.0  # Should reduce memory usage
        assert strategy.enable_checkpoint_recovery is True
    
    def test_numerical_error_recovery_strategy(self):
        """Test recovery strategy for numerical errors."""
        error = TrainingError(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NUMERICAL,
            message="NaN detected",
            context={},
            traceback_str="",
            stage="test",
            retry_count=0,
            recoverable=True
        )
        context = ErrorContext()
        
        strategy = create_recovery_strategy(error, context)
        
        assert strategy.enable_parameter_reset is True  # Reset params for numerical issues
        assert strategy.memory_reduction_factor == 1.0  # No memory reduction needed
    
    def test_network_error_recovery_strategy(self):
        """Test recovery strategy for network errors."""
        error = TrainingError(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.NETWORK,
            message="Connection failed",
            context={},
            traceback_str="",
            stage="test",
            retry_count=0,
            recoverable=True
        )
        context = ErrorContext()
        
        strategy = create_recovery_strategy(error, context)
        
        assert strategy.max_retries >= 5  # More retries for transient network issues
        assert strategy.base_delay >= 1.0  # Longer delays for network
        assert strategy.enable_checkpoint_recovery is False  # Network issues don't affect checkpoints
    
    @given(
        category=st.sampled_from(list(ErrorCategory)),
        retry_count=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=20)
    def test_recovery_strategy_properties(self, category, retry_count):
        """Property-based test for recovery strategy creation."""
        error = TrainingError(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=category,
            message="Test error",
            context={},
            traceback_str="",
            stage="test",
            retry_count=retry_count,
            recoverable=True
        )
        context = ErrorContext()
        
        strategy = create_recovery_strategy(error, context)
        
        # All strategies should have reasonable bounds
        assert strategy.max_retries >= 1
        assert strategy.max_retries <= 20
        assert strategy.backoff_factor >= 1.0
        assert strategy.backoff_factor <= 5.0
        assert strategy.base_delay >= 0.1
        assert strategy.base_delay <= 10.0
        assert 0.1 <= strategy.memory_reduction_factor <= 1.0


class TestBackoffCalculation:
    """Test exponential backoff delay calculation."""
    
    @given(
        attempt=st.integers(min_value=0, max_value=10),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        backoff_factor=st.floats(min_value=1.0, max_value=5.0)
    )
    @settings(max_examples=30)
    def test_backoff_delay_properties(self, attempt, base_delay, backoff_factor):
        """Property-based test for backoff delay calculation."""
        strategy = ErrorRecoveryStrategy(
            base_delay=base_delay,
            backoff_factor=backoff_factor
        )
        
        delay = calculate_backoff_delay(attempt, strategy)
        
        # Should always return a positive delay
        assert delay > 0
        
        # Should be capped at reasonable maximum
        assert delay <= 300.0  # 5 minutes max
        
        # Should increase with attempt number
        if attempt > 0:
            prev_delay = calculate_backoff_delay(attempt - 1, strategy)
            assert delay >= prev_delay
    
    def test_backoff_delay_exponential_growth(self):
        """Test exponential growth of backoff delay."""
        strategy = ErrorRecoveryStrategy(base_delay=1.0, backoff_factor=2.0)
        
        delays = [calculate_backoff_delay(i, strategy) for i in range(5)]
        
        # Should roughly double each time (until capped)
        for i in range(1, len(delays)):
            if delays[i] < 300.0:  # Not capped
                assert delays[i] >= delays[i-1] * 1.5  # Allow some tolerance


class TestSafeTrainingStep:
    """Test safe training step wrapper."""
    
    def test_safe_training_step_success(self):
        """Test safe training step with successful execution."""
        def successful_step(x):
            return x * 2
        
        error_context = ErrorContext()
        safe_step = safe_training_step(successful_step, error_context, "test_stage")
        
        result = safe_step(5)
        assert result == 10
    
    def test_safe_training_step_single_failure_then_success(self):
        """Test safe training step with one failure then success."""
        call_count = 0
        
        def flaky_step(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return x * 2
        
        error_context = ErrorContext()
        safe_step = safe_training_step(flaky_step, error_context, "test_stage", max_retries=2)
        
        # Should eventually succeed
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = safe_step(5)
        
        assert result == 10
        assert call_count == 2
    
    def test_safe_training_step_permanent_failure(self):
        """Test safe training step with permanent failure."""
        def failing_step(x):
            raise ValueError("Permanent failure")
        
        error_context = ErrorContext()
        safe_step = safe_training_step(failing_step, error_context, "test_stage", max_retries=2)
        
        # Should eventually raise the error
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(Exception):  # Will raise the TrainingError
                safe_step(5)
    
    def test_safe_training_step_memory_error_recovery(self):
        """Test safe training step with memory error and batch size reduction."""
        call_count = 0
        
        def memory_failing_step(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory")
            # Should succeed with reduced batch size
            assert kwargs.get('batch_size', 8) < 8  # Should be reduced
            return "success"
        
        error_context = ErrorContext()
        safe_step = safe_training_step(memory_failing_step, error_context, "test_stage", max_retries=2)
        
        with patch('time.sleep'):
            result = safe_step(batch_size=8)
        
        assert result == "success"
        assert call_count == 2


class TestErrorContext:
    """Test error context management."""
    
    def test_empty_error_context(self):
        """Test empty error context creation."""
        context = ErrorContext()
        
        assert context.errors == ()
        assert context.total_errors == 0
        assert context.recovery_attempts == 0
        assert context.last_successful_checkpoint is None
        assert context.current_stage == "unknown"
    
    def test_add_error_to_context(self):
        """Test adding errors to context."""
        context = ErrorContext()
        error = create_training_error(ValueError("test"), "stage1", {})
        
        new_context = add_error_to_context(context, error)
        
        # Original context should be unchanged (immutable)
        assert context.total_errors == 0
        assert len(context.errors) == 0
        
        # New context should have the error
        assert new_context.total_errors == 1
        assert len(new_context.errors) == 1
        assert new_context.errors[0] == error
        assert new_context.current_stage == "stage1"
    
    def test_error_context_immutability(self):
        """Test that ErrorContext is immutable."""
        context = ErrorContext(total_errors=5)
        
        with pytest.raises(AttributeError):
            context.total_errors = 10
    
    @given(
        num_errors=st.integers(min_value=0, max_value=25),
        high_severity_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=20)
    def test_should_abort_training_properties(self, num_errors, high_severity_count):
        """Property-based test for training abortion decision."""
        assume(high_severity_count <= num_errors)
        
        # Create context with specified number of errors
        context = ErrorContext(total_errors=num_errors)
        
        # Add high severity errors
        errors = []
        for i in range(high_severity_count):
            error = TrainingError(
                error_id=f"error_{i}",
                timestamp=time.time(),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.MEMORY,
                message="High severity error",
                context={},
                traceback_str="",
                stage="test",
                retry_count=0,
                recoverable=True
            )
            errors.append(error)
        
        # Add low severity errors for remaining count
        for i in range(high_severity_count, num_errors):
            error = TrainingError(
                error_id=f"error_{i}",
                timestamp=time.time(),
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.NETWORK,
                message="Low severity error",
                context={},
                traceback_str="",
                stage="test",
                retry_count=0,
                recoverable=True
            )
            errors.append(error)
        
        context_with_errors = ErrorContext(
            errors=tuple(errors),
            total_errors=num_errors,
            recovery_attempts=0,
            last_successful_checkpoint=None,
            current_stage="test"
        )
        
        should_abort = should_abort_training(context_with_errors)
        
        # Should abort if too many total errors
        if num_errors > 20:
            assert should_abort is True
        
        # Should abort if too many high severity errors
        if high_severity_count > 5:
            assert should_abort is True
        
        # Should not abort for reasonable error counts
        if num_errors <= 5 and high_severity_count <= 2:
            assert should_abort is False


class TestErrorHandlingIntegration:
    """Integration tests for error handling components."""
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow from error to recovery."""
        # Simulate a training function that fails then succeeds
        attempts = 0
        
        def training_function(batch_size=16):
            nonlocal attempts
            attempts += 1
            
            if attempts == 1:
                raise RuntimeError("CUDA out of memory")
            elif attempts == 2:
                raise ValueError("NaN detected")
            else:
                return {"loss": 0.5, "batch_size": batch_size}
        
        # Start with empty error context
        error_context = ErrorContext()
        
        # Wrap with safe training step
        safe_training_fn = safe_training_step(
            training_function, 
            error_context, 
            "training",
            max_retries=3
        )
        
        # Should eventually succeed with recovery
        with patch('time.sleep'):  # Speed up test
            result = safe_training_fn(batch_size=16)
        
        assert result["loss"] == 0.5
        assert attempts == 3  # Should have tried 3 times
        # Batch size might be reduced due to memory error recovery
    
    def test_error_context_accumulation(self):
        """Test accumulation of errors in context over time."""
        context = ErrorContext()
        
        # Add multiple errors over time
        error1 = create_training_error(RuntimeError("Error 1"), "stage1", {})
        context = add_error_to_context(context, error1)
        
        error2 = create_training_error(ValueError("Error 2"), "stage2", {})
        context = add_error_to_context(context, error2)
        
        error3 = create_training_error(MemoryError("Error 3"), "stage3", {})
        context = add_error_to_context(context, error3)
        
        # Should accumulate correctly
        assert context.total_errors == 3
        assert len(context.errors) == 3
        assert context.current_stage == "stage3"  # Latest stage
        
        # Should maintain error order
        assert context.errors[0].message == "Error 1"
        assert context.errors[1].message == "Error 2"
        assert context.errors[2].message == "Error 3"


class TestErrorHandlingEdgeCases:
    """Test edge cases and error conditions in error handling."""
    
    def test_error_with_none_message(self):
        """Test handling of errors with None message."""
        error = Exception(None)
        training_error = create_training_error(error, "test", {})
        
        # Should handle gracefully
        assert isinstance(training_error.message, str)
    
    def test_error_classification_with_empty_context(self):
        """Test error classification with empty context."""
        error = ValueError("test error")
        category = classify_error(error, {})
        
        assert isinstance(category, ErrorCategory)
    
    def test_recovery_strategy_with_extreme_values(self):
        """Test recovery strategy creation with extreme error values."""
        error = TrainingError(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.FATAL,
            category=ErrorCategory.UNKNOWN,
            message="Extreme error",
            context={},
            traceback_str="",
            stage="test",
            retry_count=100,  # Very high retry count
            recoverable=False
        )
        context = ErrorContext(total_errors=1000)  # Many errors
        
        strategy = create_recovery_strategy(error, context)
        
        # Should still return a valid strategy
        assert isinstance(strategy, ErrorRecoveryStrategy)
        assert strategy.max_retries >= 0
    
    def test_safe_training_step_with_non_exception_error(self):
        """Test safe training step with non-Exception error (like KeyboardInterrupt)."""
        def failing_step(x):
            raise KeyboardInterrupt("User interrupted")
        
        error_context = ErrorContext()
        safe_step = safe_training_step(failing_step, error_context, "test_stage")
        
        # Should re-raise KeyboardInterrupt without wrapping
        with pytest.raises(KeyboardInterrupt):
            safe_step(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])