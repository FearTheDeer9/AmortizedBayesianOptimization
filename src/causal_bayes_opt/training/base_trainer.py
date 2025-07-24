#!/usr/bin/env python3
"""
Base Trainer for Behavioral Cloning

Abstract base trainer following Rich Hickey's principles:
- Single responsibility: Each trainer does ONE thing well
- Pure functions with immutable state
- Composition over inheritance
- Explicit over implicit

This provides the common interface and shared functionality for all BC trainers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, NamedTuple, Callable
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import optax
import pyrsistent as pyr
from pyrsistent import PRecord, field

logger = logging.getLogger(__name__)


# Immutable training state
@dataclass(frozen=True)
class TrainingMetrics:
    """Immutable training metrics"""
    loss: float
    accuracy: float
    learning_rate: float
    epoch: int
    step: int
    elapsed_time: float


@dataclass(frozen=True)
class ValidationMetrics:
    """Immutable validation metrics"""
    loss: float
    accuracy: float
    improvement: float  # Improvement over previous validation
    best_loss: float
    epochs_without_improvement: int


@dataclass(frozen=True)
class TrainingState:
    """Immutable training state"""
    model_params: Dict[str, Any]
    optimizer_state: Any  # Keep as JAX structure
    epoch: int
    step: int
    best_loss: float = float('inf')
    epochs_without_improvement: int = 0
    training_metrics: Tuple[TrainingMetrics, ...] = ()
    validation_metrics: Tuple[ValidationMetrics, ...] = ()
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    def replace(self, **kwargs):
        """Create new state with updated fields."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)
    
    def append_training_metric(self, metric: TrainingMetrics):
        """Create new state with appended training metric."""
        new_metrics = self.training_metrics + (metric,)
        return self.replace(training_metrics=new_metrics)
    
    def append_validation_metric(self, metric: ValidationMetrics):
        """Create new state with appended validation metric."""
        new_metrics = self.validation_metrics + (metric,)
        return self.replace(validation_metrics=new_metrics)


class TrainingConfig(NamedTuple):
    """Training configuration"""
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    min_epochs: int = 5
    early_stopping_patience: int = 10
    validation_frequency: int = 5
    gradient_clip_norm: Optional[float] = 1.0
    weight_decay: float = 1e-4
    use_jax_compilation: bool = True
    random_seed: int = 42


class BaseBCTrainer(ABC):
    """
    Abstract base class for behavioral cloning trainers.
    
    Following functional principles:
    - Immutable state transitions
    - Pure functions where possible
    - Explicit error handling
    - Single responsibility
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._rng_key = jax.random.PRNGKey(config.random_seed)
        self._optimizer = self._create_optimizer()
        self._training_start_time = None
        
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with gradient clipping and weight decay."""
        transforms = []
        
        if self.config.gradient_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.config.gradient_clip_norm))
        
        transforms.append(optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        ))
        
        return optax.chain(*transforms)
    
    @abstractmethod
    def _initialize_model_params(self, rng_key: jax.Array, sample_input: Any) -> pyr.PMap:
        """Initialize model parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _forward_pass(self, params: pyr.PMap, inputs: Any, rng_key: jax.Array) -> Any:
        """Forward pass through the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_loss_function(self) -> Callable[[Any, Dict[str, Any], Any, jax.Array], Tuple[float, Dict[str, float]]]:
        """Return pure loss computation function that can be JAX-compiled."""
        pass
    
    @abstractmethod
    def _get_model_apply_function(self) -> Callable:
        """Return pure model apply function that can be JAX-compiled."""
        pass
    
    @abstractmethod
    def _compute_accuracy(self, predictions: Any, targets: Any) -> float:
        """Compute accuracy metric. Must be implemented by subclasses."""
        pass
    
    def initialize_training_state(self, sample_input: Any) -> TrainingState:
        """
        Initialize training state with model parameters and optimizer state.
        
        Args:
            sample_input: Sample input to determine model parameter shapes
            
        Returns:
            Initial TrainingState
        """
        self._rng_key, init_key = jax.random.split(self._rng_key)
        
        # Initialize model parameters
        model_params_pyr = self._initialize_model_params(init_key, sample_input)
        model_params_dict = dict(model_params_pyr)
        
        # Initialize optimizer state
        optimizer_state = self._optimizer.init(model_params_dict)
        
        return TrainingState(
            model_params=model_params_dict,
            optimizer_state=optimizer_state,
            epoch=0,
            step=0,
            metadata={
                'model_type': self.__class__.__name__,
                'config': self.config._asdict(),
                'initialized_at': time.time()
            }
        )
    
    def _create_train_step_fn(self):
        """Create JAX-compiled training step function."""
        
        # Get pure functions from subclass - these don't reference self
        compute_loss_fn = self._get_loss_function()
        model_apply_fn = self._get_model_apply_function()
        
        # Capture config values needed inside JAX function
        learning_rate = self.config.learning_rate
        training_start_time = self._training_start_time
        
        def train_step(state: TrainingState, batch: Any, rng_key: jax.Array) -> Tuple[TrainingState, TrainingMetrics]:
            """Single training step - pure function."""
            
            def loss_fn(params):
                # Now calling pure function with no self references
                loss, metrics = compute_loss_fn(model_apply_fn, params, batch, rng_key)
                return loss, metrics
            
            # Compute gradients
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model_params)
            
            # Apply gradients
            updates, new_optimizer_state = self._optimizer.update(grads, state.optimizer_state, state.model_params)
            new_model_params = optax.apply_updates(state.model_params, updates)
            
            # Create training metrics
            training_metrics = TrainingMetrics(
                loss=float(loss),
                accuracy=metrics.get('accuracy', 0.0),
                learning_rate=learning_rate,
                epoch=state.epoch,
                step=state.step + 1,
                elapsed_time=time.time() - training_start_time if training_start_time else 0.0
            )
            
            # Create new state (immutable update)
            new_state = state.replace(
                model_params=new_model_params,
                optimizer_state=new_optimizer_state,
                step=state.step + 1
            ).append_training_metric(training_metrics)
            
            return new_state, training_metrics
        
        return jax.jit(train_step) if self.config.use_jax_compilation else train_step
    
    def _create_validation_step_fn(self):
        """Create JAX-compiled validation step function."""
        
        # Get pure functions from subclass - these don't reference self
        compute_loss_fn = self._get_loss_function()
        model_apply_fn = self._get_model_apply_function()
        
        def validation_step(state: TrainingState, batch: Any, rng_key: jax.Array) -> Dict[str, float]:
            """Single validation step - pure function."""
            loss, metrics = compute_loss_fn(model_apply_fn, state.model_params, batch, rng_key)
            return {'loss': float(loss), **metrics}
        
        return jax.jit(validation_step) if self.config.use_jax_compilation else validation_step
    
    def train_epoch(
        self, 
        state: TrainingState, 
        train_batches: List[Any], 
        val_batches: Optional[List[Any]] = None
    ) -> TrainingState:
        """
        Train for one epoch with functional state updates.
        
        Args:
            state: Current training state
            train_batches: List of training batches
            val_batches: Optional validation batches
            
        Returns:
            Updated training state
        """
        if self._training_start_time is None:
            self._training_start_time = time.time()
        
        # Create compiled training step
        train_step = self._create_train_step_fn()
        
        # Process all training batches
        current_state = state
        epoch_metrics = []
        
        for batch in train_batches:
            self._rng_key, step_key = jax.random.split(self._rng_key)
            current_state, metrics = train_step(current_state, batch, step_key)
            epoch_metrics.append(metrics)
        
        # Update epoch count
        current_state = current_state.replace(epoch=state.epoch + 1)
        
        # Run validation if provided
        if val_batches is not None and (current_state.epoch % self.config.validation_frequency == 0):
            current_state = self._run_validation(current_state, val_batches)
        
        # Log epoch summary
        if epoch_metrics:
            avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
            avg_accuracy = sum(m.accuracy for m in epoch_metrics) / len(epoch_metrics)
            logger.info(f"Epoch {current_state.epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        return current_state
    
    def _run_validation(self, state: TrainingState, val_batches: List[Any]) -> TrainingState:
        """Run validation and update state with validation metrics."""
        validation_step = self._create_validation_step_fn()
        
        val_metrics = []
        for batch in val_batches:
            self._rng_key, val_key = jax.random.split(self._rng_key)
            batch_metrics = validation_step(state, batch, val_key)
            val_metrics.append(batch_metrics)
        
        # Aggregate validation metrics
        avg_val_loss = sum(m['loss'] for m in val_metrics) / len(val_metrics)
        avg_val_accuracy = sum(m.get('accuracy', 0.0) for m in val_metrics) / len(val_metrics)
        
        # Check for improvement
        improvement = state.best_loss - avg_val_loss
        new_best_loss = min(state.best_loss, avg_val_loss)
        epochs_without_improvement = 0 if improvement > 0 else state.epochs_without_improvement + 1
        
        validation_metrics = ValidationMetrics(
            loss=avg_val_loss,
            accuracy=avg_val_accuracy,
            improvement=improvement,
            best_loss=new_best_loss,
            epochs_without_improvement=epochs_without_improvement
        )
        
        # Update state with validation results
        new_state = state.replace(
            best_loss=new_best_loss,
            epochs_without_improvement=epochs_without_improvement
        ).append_validation_metric(validation_metrics)
        
        logger.info(f"Validation: Loss={avg_val_loss:.4f}, Accuracy={avg_val_accuracy:.4f}, "
                   f"Improvement={improvement:.4f}, No improvement for {epochs_without_improvement} epochs")
        
        return new_state
    
    def should_stop_early(self, state: TrainingState) -> bool:
        """Check if training should stop early based on validation metrics."""
        if state.epoch < self.config.min_epochs:
            return False
        
        if state.epochs_without_improvement >= self.config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {state.epochs_without_improvement} epochs without improvement")
            return True
        
        return False
    
    def fit(
        self, 
        train_data: List[Any], 
        val_data: Optional[List[Any]] = None,
        sample_input: Optional[Any] = None
    ) -> TrainingState:
        """
        Complete training loop with early stopping.
        
        Args:
            train_data: Training data batches
            val_data: Validation data batches
            sample_input: Sample input for parameter initialization
            
        Returns:
            Final training state
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        
        # Initialize training state
        if sample_input is None:
            sample_input = train_data[0]
        
        state = self.initialize_training_state(sample_input)
        logger.info(f"Starting training with {len(train_data)} training batches, "
                   f"{len(val_data) if val_data else 0} validation batches")
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            state = self.train_epoch(state, train_data, val_data)
            
            # Check early stopping
            if self.should_stop_early(state):
                break
        
        logger.info(f"Training completed after {state.epoch} epochs. Best loss: {state.best_loss:.4f}")
        return state
    
    def predict(self, state: TrainingState, inputs: Any) -> Any:
        """
        Make predictions using trained model.
        
        Args:
            state: Trained model state
            inputs: Input data for prediction
            
        Returns:
            Model predictions
        """
        self._rng_key, pred_key = jax.random.split(self._rng_key)
        return self._forward_pass(pyr.pmap(state.model_params), inputs, pred_key)
    
    def get_training_summary(self, state: TrainingState) -> Dict[str, Any]:
        """Get summary of training progress."""
        training_metrics = list(state.training_metrics)
        validation_metrics = list(state.validation_metrics)
        
        summary = {
            'total_epochs': state.epoch,
            'total_steps': state.step,
            'best_loss': state.best_loss,
            'final_training_loss': training_metrics[-1].loss if training_metrics else None,
            'final_validation_loss': validation_metrics[-1].loss if validation_metrics else None,
            'final_training_accuracy': training_metrics[-1].accuracy if training_metrics else None,
            'final_validation_accuracy': validation_metrics[-1].accuracy if validation_metrics else None,
            'epochs_without_improvement': state.epochs_without_improvement,
            'config': dict(state.metadata['config']),
            'training_time': training_metrics[-1].elapsed_time if training_metrics else 0.0
        }
        
        return summary