"""
Model-Agnostic Meta-Learning (MAML) implementation for fast adaptation.

This module implements the MAML algorithm as described in the paper:
"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
by Finn et al. 2017.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterator
import copy
import logging
from collections import OrderedDict
import torch.nn.functional as F

try:
    import torch_geometric
    from torch_geometric.data import Data as GraphData, Batch as GraphBatch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False

from causal_meta.graph.causal_graph import CausalGraph

logger = logging.getLogger(__name__)


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML) implementation.

    This class implements the MAML algorithm for fast adaptation. It supports
    arbitrary model architectures, custom learning rates, and configurable
    adaptation steps for both inner and outer optimization loops.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 1,
        first_order: bool = False,
        device: Union[str, torch.device] = "cpu",
        inner_loss_fn: Optional[Callable] = None,
        outer_loss_fn: Optional[Callable] = None,
        outer_optimizer: Optional[Any] = None,
        per_param_lr: bool = False,
    ):
        """
        Initialize the MAML algorithm.

        Args:
            model: Base model architecture (will be meta-learned)
            inner_lr: Learning rate for the inner loop (task-specific adaptation)
            outer_lr: Learning rate for the outer loop (meta-parameters)
            num_inner_steps: Number of gradient steps for inner loop adaptation
            first_order: Whether to use first-order approximation (ignore second derivative)
            device: Device to use for computation (cpu or cuda)
            inner_loss_fn: Loss function for inner loop optimization (default: model's loss if available, else MSE)
            outer_loss_fn: Loss function for outer loop optimization (default: same as inner_loss_fn)
            outer_optimizer: Optimizer for the outer loop (default: Adam with outer_lr)
            per_param_lr: Whether to use parameter-specific learning rates for inner loop
        """
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.device = torch.device(device)
        self.per_param_lr = per_param_lr

        # Set default loss functions if not provided
        self.inner_loss_fn = inner_loss_fn or self._get_default_loss()
        self.outer_loss_fn = outer_loss_fn or self.inner_loss_fn

        # Set up parameter-specific learning rates if needed
        if per_param_lr:
            self._init_per_param_lr()

        # Set default optimizer if not provided
        self.outer_optimizer = outer_optimizer or optim.Adam(
            self.model.parameters(), lr=self.outer_lr
        )

        # Initialize metadata for tracking
        self.meta_iterations = 0
        self.task_adaptations = 0
        self.meta_losses = []

    def _init_per_param_lr(self):
        """Initialize per-parameter learning rates for the inner loop."""
        self.param_lrs = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_lrs[name] = nn.Parameter(
                    torch.ones_like(param) * self.inner_lr)

        # Add these to parameters for the outer loop to optimize
        self.lr_params = nn.ParameterDict(self.param_lrs)

        # Recreate the optimizer with the learning rate parameters
        all_params = list(self.model.parameters()) + \
            list(self.lr_params.values())
        self.outer_optimizer = optim.Adam(all_params, lr=self.outer_lr)

    def _get_default_loss(self) -> Callable:
        """Get a default loss function if none is provided."""
        # Try to use the model's loss function if available
        if hasattr(self.model, 'loss') and callable(self.model.loss):
            return self.model.loss

        # Otherwise use mean squared error as a sensible default
        return nn.MSELoss()

    def _process_batch(self, batch: Any) -> Tuple[Any, Any]:
        """
        Process a batch of data to prepare for forward pass.

        This method handles different types of data, including tensors and 
        PyTorch Geometric graph data.

        Args:
            batch: A batch of data (tensors, graphs, etc.)

        Returns:
            Tuple of (inputs, targets) ready to be used in the forward pass
        """
        # Handle different types of data
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            # Simple case: tuple of (inputs, targets)
            inputs, targets = batch
            return inputs.to(self.device), targets.to(self.device)

        elif HAS_PYGEOMETRIC and isinstance(batch, (GraphData, GraphBatch)):
            # Graph data: use x as inputs, y as targets
            inputs = batch.to(self.device)
            if hasattr(batch, 'y'):
                targets = batch.y.to(self.device)
            else:
                # If no targets, use the same data (e.g., for reconstruction tasks)
                targets = batch.to(self.device)
            return inputs, targets

        else:
            # Default case: assume the batch is the input, no targets
            # (target can be derived from input, e.g., in self-supervised learning)
            return batch.to(self.device), None

    def _compute_loss(self, model: nn.Module, inputs: Any, targets: Any) -> torch.Tensor:
        """
        Compute the loss for a batch of data.

        Args:
            model: The model to use for forward pass
            inputs: Input data
            targets: Target data (can be None for self-supervised tasks)

        Returns:
            Loss tensor
        """
        # For graph data
        if HAS_PYGEOMETRIC and isinstance(inputs, (GraphData, GraphBatch)):
            outputs = model(inputs)

            # Handle different model output and target combinations
            if targets is None:
                # Self-supervised case: model should define its own loss
                if hasattr(model, 'loss') and callable(model.loss):
                    return model.loss(outputs, inputs)
                else:
                    raise ValueError("For self-supervised learning with graph data, "
                                     "model must have a 'loss' method.")
            else:
                return self.inner_loss_fn(outputs, targets)

        # Standard tensor case
        else:
            outputs = model(inputs)
            return self.inner_loss_fn(outputs, targets)

    def adapt(
        self,
        batch: Any,
        num_steps: Optional[int] = None,
        create_graph: Optional[bool] = None,
        allow_unused: bool = True,
        return_task_params: bool = False,
    ) -> Union[nn.Module, Tuple[nn.Module, OrderedDict]]:
        """
        Adapt the model to a new task using the support set.

        Performs inner loop optimization to adapt the model parameters
        to a specific task.

        Args:
            batch: A batch of data (can be tensors, graphs, etc.)
            num_steps: Number of adaptation steps (overrides self.num_inner_steps if provided)
            create_graph: Whether to create a computation graph for meta-learning
                          (overrides self.first_order if provided)
            allow_unused: Whether to allow unused parameters in gradient computation
            return_task_params: Whether to return the adapted parameters separately

        Returns:
            If return_task_params is False:
                Adapted model for the specific task
            If return_task_params is True:
                Tuple of (adapted model, adapted parameters)
        """
        # Initialize parameters for task adaptation
        steps = num_steps if num_steps is not None else self.num_inner_steps
        create_graph = not self.first_order if create_graph is None else create_graph

        # Create a copy of the model for this task
        task_model = copy.deepcopy(self.model)
        task_model.train()

        # Process the batch
        inputs, targets = self._process_batch(batch)

        # Store parameters that will be updated in the task model
        task_params = OrderedDict(
            (name, param.clone())
            for (name, param) in task_model.named_parameters()
            if param.requires_grad
        )

        # Perform adaptation steps
        for step in range(steps):
            # Update model parameters with current task_params
            self._set_params(task_model, task_params)

            # Forward pass
            loss = self._compute_loss(task_model, inputs, targets)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                task_params.values(),
                create_graph=create_graph,
                allow_unused=allow_unused,
                retain_graph=True
            )

            # Update parameters
            for (name, param), grad in zip(task_params.items(), grads):
                if grad is not None:
                    if self.per_param_lr:
                        # Use parameter-specific learning rate
                        lr = self.param_lrs[name]
                        task_params[name] = param - lr * grad
                    else:
                        # Use global learning rate
                        task_params[name] = param - self.inner_lr * grad

        # Update model with adapted parameters
        self._set_params(task_model, task_params)

        # Count adaptation
        self.task_adaptations += 1

        if return_task_params:
            return task_model, task_params
        return task_model

    def _set_params(self, model: nn.Module, params: OrderedDict):
        """
        Set model parameters from the given parameter dictionary.

        Args:
            model: Model to update
            params: Dictionary of parameters
        """
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name].data)

    def _detach_params(self, params: OrderedDict) -> OrderedDict:
        """
        Detach parameters from the computation graph.

        Args:
            params: Dictionary of parameters

        Returns:
            Dictionary of detached parameters
        """
        return OrderedDict(
            (name, param.detach())
            for (name, param) in params.items()
        )

    def inner_loop(
        self,
        support_batch: Any,
        query_batch: Optional[Any] = None,
        num_steps: Optional[int] = None,
        create_graph: Optional[bool] = None,
        return_adapted_model: bool = False
    ) -> Union[Tuple[torch.Tensor, nn.Module], torch.Tensor, nn.Module]:
        """
        Perform the inner loop optimization for a single task.

        Args:
            support_batch: Batch of support data for adaptation
            query_batch: Batch of query data for evaluation (optional)
            num_steps: Number of adaptation steps
            create_graph: Whether to create a computation graph
            return_adapted_model: Whether to return the adapted model along with loss

        Returns:
            If query_batch is provided and return_adapted_model is False:
                Loss on the query set after adaptation
            If query_batch is provided and return_adapted_model is True:
                Tuple of (query loss, adapted model)
            If query_batch is not provided:
                Adapted model
        """
        # Adapt model to this task
        adapted_model = self.adapt(
            support_batch,
            num_steps=num_steps,
            create_graph=create_graph
        )

        # If query data is provided, evaluate on query set
        if query_batch is not None:
            query_inputs, query_targets = self._process_batch(query_batch)

            adapted_model.eval()
            with torch.no_grad():
                query_loss = self._compute_loss(
                    adapted_model, query_inputs, query_targets)

            if return_adapted_model:
                return query_loss, adapted_model
            return query_loss

        # Otherwise, just return the adapted model
        return adapted_model

    def clone_model_with_updates(self, updates: OrderedDict) -> nn.Module:
        """
        Clone the model and apply parameter updates.

        Args:
            updates: Dictionary of parameter updates to apply

        Returns:
            Updated model
        """
        clone = copy.deepcopy(self.model)
        with torch.no_grad():
            for (name, param) in clone.named_parameters():
                if name in updates:
                    param.copy_(updates[name])
        return clone.to(self.device)

    def adapt_with_optim(
        self,
        batch: Any,
        num_steps: Optional[int] = None,
        inner_optimizer: Optional[Any] = None,
        inner_scheduler: Optional[Any] = None,
    ) -> nn.Module:
        """
        Adapt the model using an optimizer instead of manual gradient updates.

        This provides an alternative to the standard `adapt` method, using
        PyTorch optimizers for inner loop updates. This can be helpful for
        more complex optimization strategies.

        Args:
            batch: A batch of data
            num_steps: Number of adaptation steps
            inner_optimizer: Optimizer to use (default: SGD with self.inner_lr)
            inner_scheduler: Learning rate scheduler (optional)

        Returns:
            Adapted model
        """
        # Initialize parameters
        steps = num_steps if num_steps is not None else self.num_inner_steps

        # Create a copy of the model for this task
        task_model = copy.deepcopy(self.model)
        task_model.train()
        task_model.to(self.device)

        # Create optimizer if not provided
        if inner_optimizer is None:
            inner_optimizer = optim.SGD(
                task_model.parameters(), lr=self.inner_lr)

        # Process batch
        inputs, targets = self._process_batch(batch)

        # Perform adaptation steps
        for step in range(steps):
            # Zero gradients
            inner_optimizer.zero_grad()

            # Forward pass and loss computation
            loss = self._compute_loss(task_model, inputs, targets)

            # Backward pass
            loss.backward()

            # Update parameters
            inner_optimizer.step()

            # Step scheduler if provided
            if inner_scheduler is not None:
                inner_scheduler.step()

        # Count adaptation
        self.task_adaptations += 1

        return task_model

    def outer_loop(
        self,
        tasks: List[Tuple[Any, Any]],
        meta_batch_size: Optional[int] = None,
        grad_clip: Optional[float] = None,
        accumulate_grad: bool = False,
        optimizer_step: bool = True,
        track_higher_grads: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Perform the outer loop optimization across multiple tasks.

        Args:
            tasks: List of (support_batch, query_batch) tuples for each task
            meta_batch_size: Number of tasks to use in each meta-update (all if None)
            grad_clip: Value for gradient clipping (None = no clipping)
            accumulate_grad: Whether to accumulate gradients without resetting
            optimizer_step: Whether to perform an optimizer step after computing gradients
            track_higher_grads: Whether to track higher-order gradients 
                               (None = use not self.first_order)

        Returns:
            Dictionary with meta-loss statistics
        """
        if meta_batch_size is None:
            meta_batch_size = len(tasks)

        # Use first-order approximation setting if track_higher_grads is not specified
        if track_higher_grads is None:
            track_higher_grads = not self.first_order

        # Sample a batch of tasks if necessary
        meta_batch_indices = torch.randperm(len(tasks))[:meta_batch_size]
        meta_batch_tasks = [tasks[i] for i in meta_batch_indices]

        # Zero the gradients before meta-update (unless accumulating)
        if not accumulate_grad:
            self.outer_optimizer.zero_grad()

        # Accumulate statistics
        meta_losses = []
        task_metrics = {}

        # Process each task
        for task_idx, (support_batch, query_batch) in enumerate(meta_batch_tasks):
            # Adapt model to task (inner loop)
            adapted_model = self.adapt(
                support_batch,
                create_graph=track_higher_grads
            )

            # Process query batch
            query_inputs, query_targets = self._process_batch(query_batch)

            # Evaluate on query set
            adapted_model.eval()  # Set to eval mode for consistent behavior
            query_outputs = adapted_model(query_inputs)

            # Handle different output types
            if isinstance(query_outputs, tuple):
                # Some models might return multiple outputs
                primary_outputs = query_outputs[0]
                query_loss = self.outer_loss_fn(primary_outputs, query_targets)

                # Store additional outputs if provided
                if len(query_outputs) > 1 and hasattr(query_outputs, '_fields'):
                    # Handle namedtuple outputs
                    for field_name in query_outputs._fields[1:]:
                        field_value = getattr(query_outputs, field_name)
                        if isinstance(field_value, torch.Tensor) and field_value.numel() == 1:
                            field_name = f"{field_name}"
                            if field_name not in task_metrics:
                                task_metrics[field_name] = []
                            task_metrics[field_name].append(field_value.item())
            else:
                # Standard single output
                query_loss = self.outer_loss_fn(query_outputs, query_targets)

            # Scale the loss if necessary (for gradient accumulation)
            if accumulate_grad and meta_batch_size > 0:
                query_loss = query_loss / meta_batch_size

            # Accumulate meta-gradient
            query_loss.backward()
            meta_losses.append(query_loss.item())

            # Log progress for long batches
            if (task_idx + 1) % 10 == 0 and task_idx < len(meta_batch_tasks) - 1:
                logger.debug(
                    f"Processed {task_idx + 1}/{len(meta_batch_tasks)} tasks in outer loop")

        # Clip gradients if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Also clip learning rate parameters if using per-param learning rates
            if self.per_param_lr:
                torch.nn.utils.clip_grad_norm_(
                    self.lr_params.parameters(), grad_clip)

        # Update meta-parameters if requested
        if optimizer_step:
            self.outer_optimizer.step()

        # Calculate statistics
        avg_meta_loss = sum(meta_losses) / \
            len(meta_losses) if meta_losses else 0.0

        # Prepare result metrics
        result_metrics = {
            'meta_loss': avg_meta_loss
        }

        # Add any additional metrics
        for metric_name, values in task_metrics.items():
            if values:
                result_metrics[metric_name] = sum(values) / len(values)

        # Update metadata
        self.meta_losses.append(avg_meta_loss)
        self.meta_iterations += 1

        return result_metrics

    def meta_train(
        self,
        task_generator: Callable[[], List[Tuple[Any, Any]]],
        num_iterations: int,
        meta_batch_size: Optional[int] = None,
        log_interval: int = 10,
        eval_interval: Optional[int] = None,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
        grad_clip: Optional[float] = None,
        optimizer_scheduler: Optional[Any] = None,
        accumulation_steps: int = 1,
        callbacks: Optional[List[Callable]] = None,
        early_stopping: bool = False,
        patience: int = 10,
        early_stopping_metric: str = 'meta_loss',
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = False,
        task_sampling_strategy: str = 'random',
        task_sampling_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Perform meta-training for a specified number of iterations.

        Args:
            task_generator: Function that returns a list of tasks for each iteration
            num_iterations: Number of meta-training iterations
            meta_batch_size: Number of tasks to use in each meta-update
            log_interval: Interval for logging progress
            eval_interval: Interval for evaluation (if None, no evaluation)
            eval_fn: Function to evaluate the meta-learned model
            grad_clip: Value for gradient clipping (None = no clipping)
            optimizer_scheduler: Learning rate scheduler for the outer loop optimizer
            accumulation_steps: Number of steps to accumulate gradients over
            callbacks: List of callback functions to call after each iteration
            early_stopping: Whether to enable early stopping
            patience: Number of iterations to wait for improvement before early stopping
            early_stopping_metric: Metric to monitor for early stopping
            checkpoint_dir: Directory to save model checkpoints
            save_best_only: Whether to save only the best model
            task_sampling_strategy: Strategy for sampling tasks ('random', 'weighted', 'curriculum')
            task_sampling_params: Parameters for the task sampling strategy
            verbose: Whether to print verbose output

        Returns:
            Dictionary with training metrics (meta_losses and eval_metrics if provided)
        """
        # Initialize results dictionary
        results = {
            'meta_losses': []
        }

        # Setup callbacks
        callbacks = callbacks or []

        # Setup early stopping
        # For minimizing (assumes loss-like metric)
        best_metric_value = float('inf')
        minimize_metric = True  # Default assumption for early stopping metric
        patience_counter = 0
        best_model_state = None

        # Ensure checkpoint directory exists if specified
        if checkpoint_dir is not None:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpoint directory: {checkpoint_dir}")

        # Setup task sampling parameters
        task_sampling_params = task_sampling_params or {}

        # Initialize metric history for curriculum learning if needed
        task_metrics_history = {}
        if task_sampling_strategy == 'curriculum':
            logger.info("Using curriculum learning strategy for task sampling")
        elif task_sampling_strategy == 'weighted':
            logger.info("Using weighted sampling strategy for task sampling")

        # Set up progress tracking
        from tqdm import tqdm
        progress_bar = tqdm(range(num_iterations), disable=not verbose)

        # Training loop
        for iteration in range(num_iterations):
            # Generate tasks for this iteration based on sampling strategy
            tasks = task_generator()

            # Apply task sampling strategy
            if task_sampling_strategy != 'random' and len(tasks) > meta_batch_size:
                if task_sampling_strategy == 'weighted' and 'task_weights' in task_sampling_params:
                    # Sample tasks based on provided weights
                    weights = task_sampling_params['task_weights']
                    task_indices = torch.multinomial(
                        torch.tensor(weights[:len(tasks)]),
                        meta_batch_size,
                        replacement=False
                    ).tolist()
                    tasks = [tasks[i] for i in task_indices]
                elif task_sampling_strategy == 'curriculum' and task_metrics_history:
                    # Sort tasks by difficulty and select progressively harder ones
                    difficulty_ratio = min(
                        0.9, (iteration / num_iterations) + 0.1)
                    task_indices = sorted(range(len(tasks)),
                                          key=lambda i: task_metrics_history.get(i, 0.0))
                    cutoff = int(len(task_indices) * difficulty_ratio)
                    selected_indices = task_indices[:cutoff]
                    # Randomly sample from the selected indices
                    import random
                    sampled_indices = random.sample(
                        selected_indices,
                        min(meta_batch_size, len(selected_indices))
                    )
                    tasks = [tasks[i] for i in sampled_indices]

            # Handle gradient accumulation
            if accumulation_steps > 1:
                # Compute effective batch size
                effective_batch_size = meta_batch_size // accumulation_steps if meta_batch_size else len(
                    tasks) // accumulation_steps
                optimizer_step = False  # Don't step until we've accumulated all gradients

                # Process tasks in chunks
                all_metrics = []
                for acc_step in range(accumulation_steps):
                    # Get task subset for this accumulation step
                    start_idx = acc_step * effective_batch_size
                    end_idx = min((acc_step + 1) *
                                  effective_batch_size, len(tasks))
                    acc_tasks = tasks[start_idx:end_idx]

                    if not acc_tasks:
                        continue

                    # Determine if this is the last accumulation step
                    is_last_step = (acc_step == accumulation_steps - 1)

                    # Perform outer loop update
                    step_metrics = self.outer_loop(
                        acc_tasks,
                        meta_batch_size=effective_batch_size,
                        grad_clip=None,  # Clip after all accumulation steps
                        accumulate_grad=True,
                        optimizer_step=False
                    )
                    all_metrics.append(step_metrics)

                # After accumulation, clip gradients and take optimizer step
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip)
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()

                # Compute average metrics across accumulation steps
                meta_loss = sum(m['meta_loss']
                                for m in all_metrics) / len(all_metrics)
                results['meta_losses'].append(meta_loss)
            else:
                # Standard single-step optimization
                metrics = self.outer_loop(
                    tasks,
                    meta_batch_size=meta_batch_size,
                    grad_clip=grad_clip
                )
                meta_loss = metrics['meta_loss']
                results['meta_losses'].append(meta_loss)

                # Store additional metrics if available
                for k, v in metrics.items():
                    if k != 'meta_loss':
                        if k not in results:
                            results[k] = []
                        results[k].append(v)

                # Update task metrics history for curriculum learning
                if task_sampling_strategy == 'curriculum' and meta_batch_size:
                    for i, (support, query) in enumerate(tasks[:meta_batch_size]):
                        # Use a simple hash of the support set as a task identifier
                        task_id = i
                        task_metrics_history[task_id] = meta_loss

            # Step the learning rate scheduler if provided
            if optimizer_scheduler is not None:
                if hasattr(optimizer_scheduler, 'get_last_lr'):
                    current_lr = optimizer_scheduler.get_last_lr()[0]
                else:
                    # For schedulers that don't have get_last_lr
                    param_groups = self.outer_optimizer.param_groups
                    current_lr = param_groups[0]['lr'] if param_groups else self.outer_lr
                optimizer_scheduler.step()

            # Update progress bar
            if verbose:
                progress_dict = {"meta_loss": f"{meta_loss:.5f}"}
                if optimizer_scheduler is not None:
                    progress_dict["lr"] = f"{current_lr:.6f}"
                progress_bar.set_postfix(progress_dict)
                progress_bar.update(1)

            # Logging
            if (iteration + 1) % log_interval == 0:
                lr_str = ''
                if optimizer_scheduler is not None:
                    lr_str = f", LR: {current_lr:.6f}"

                logger.info(
                    f"Meta iteration {iteration + 1}/{num_iterations}, Meta loss: {meta_loss:.5f}{lr_str}")

            # Evaluation
            eval_metrics = {}
            if eval_interval is not None and eval_fn is not None and (iteration + 1) % eval_interval == 0:
                eval_metrics = eval_fn(self.model)

                if verbose:
                    logger.info(f"Evaluation metrics: {eval_metrics}")

                # Store evaluation metrics
                for key, value in eval_metrics.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)

                # Check if this metric should be minimized or maximized
                if early_stopping_metric in eval_metrics:
                    metric_value = eval_metrics[early_stopping_metric]
                    # Detect if this is a metric to be maximized (e.g., accuracy)
                    if early_stopping_metric.lower() in ['accuracy', 'f1', 'auroc', 'auprc', 'precision', 'recall']:
                        minimize_metric = False
                        is_better = metric_value > best_metric_value if not minimize_metric else metric_value < best_metric_value
                    else:
                        minimize_metric = True
                        is_better = metric_value < best_metric_value if minimize_metric else metric_value > best_metric_value

                    # Update best metric value and save model if better
                    if is_better:
                        best_metric_value = metric_value
                        patience_counter = 0
                        best_model_state = copy.deepcopy(
                            self.model.state_dict())

                        # Save best model if directory is specified
                        if checkpoint_dir is not None and save_best_only:
                            self.save(f"{checkpoint_dir}/best_model.pt")
                            logger.info(
                                f"Saved best model with {early_stopping_metric}: {best_metric_value:.5f}")
                    else:
                        patience_counter += 1
                        logger.info(
                            f"Early stopping patience: {patience_counter}/{patience}")

                    # Check for early stopping
                    if early_stopping and patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered after {iteration + 1} iterations")

                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            logger.info(
                                f"Restored best model with {early_stopping_metric}: {best_metric_value:.5f}")

                        break

            # Save checkpoint if directory is specified
            if checkpoint_dir is not None and not save_best_only and (iteration + 1) % eval_interval == 0:
                self.save(f"{checkpoint_dir}/checkpoint_{iteration + 1}.pt")

                # Save latest model
                self.save(f"{checkpoint_dir}/latest_model.pt")

            # Call callbacks
            for callback in callbacks:
                callback(iteration, self.model,
                         meta_loss, results, eval_metrics)

        # Close progress bar
        progress_bar.close()

        # Restore best model if needed
        if early_stopping and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Training complete. Restored best model with {early_stopping_metric}: {best_metric_value:.5f}")

        return results

    def distributed_outer_loop(
        self,
        tasks: List[Tuple[Any, Any]],
        rank: int,
        world_size: int,
        meta_batch_size: Optional[int] = None,
        grad_clip: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Perform the outer loop optimization in a distributed training setting.

        This method distributes tasks across multiple processes and handles
        gradient synchronization for distributed training.

        Args:
            tasks: List of (support_batch, query_batch) tuples for each task
            rank: Rank of this process in the distributed group
            world_size: Total number of processes
            meta_batch_size: Number of tasks per process to use in each meta-update
            grad_clip: Value for gradient clipping (None = no clipping)

        Returns:
            Dictionary with meta-loss statistics
        """
        # Ensure torch.distributed is available
        if not torch.distributed.is_available():
            raise RuntimeError(
                "torch.distributed is not available. Cannot use distributed training.")

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. Call torch.distributed.init_process_group() first.")

        # Compute local batch size
        if meta_batch_size is None:
            # Divide tasks evenly across processes
            meta_batch_size = len(tasks) // world_size
            if rank < len(tasks) % world_size:
                meta_batch_size += 1

        # Get the local tasks for this process
        start_idx = rank * meta_batch_size
        end_idx = min(start_idx + meta_batch_size, len(tasks))
        local_tasks = tasks[start_idx:end_idx]

        # Zero gradients
        self.outer_optimizer.zero_grad()

        # Compute meta-gradients on local tasks
        meta_losses = []

        for support_batch, query_batch in local_tasks:
            # Adapt model to task
            adapted_model = self.adapt(
                support_batch, create_graph=not self.first_order)

            # Process query batch
            query_inputs, query_targets = self._process_batch(query_batch)

            # Evaluate on query set
            query_logits = adapted_model(query_inputs)
            if isinstance(query_logits, tuple):
                query_logits = query_logits[0]

            query_loss = self.outer_loss_fn(query_logits, query_targets)

            # Backward pass
            query_loss.backward()
            meta_losses.append(query_loss.item())

        # All-reduce gradients across processes (average)
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad.div_(world_size)

        # Clip gradients if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # Update parameters
        self.outer_optimizer.step()

        # All-reduce loss statistics
        if meta_losses:
            avg_meta_loss = sum(meta_losses) / len(meta_losses)
            meta_loss_tensor = torch.tensor(
                [avg_meta_loss], device=self.device)
            torch.distributed.all_reduce(
                meta_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            meta_loss_tensor.div_(world_size)
            avg_meta_loss = meta_loss_tensor.item()
        else:
            avg_meta_loss = 0.0

        # Update metadata
        self.meta_losses.append(avg_meta_loss)
        self.meta_iterations += 1

        return {'meta_loss': avg_meta_loss}

    def meta_test(
        self,
        test_tasks: List[Tuple[Any, Any]],
        additional_adaptation_steps: int = 0,
        compute_confidence_intervals: bool = False,
        compute_per_task_metrics: bool = False,
        return_adapted_models: bool = False,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        verbose: bool = True,
        adaptation_evaluation_steps: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the meta-learned model on a set of test tasks.

        Args:
            test_tasks: List of (support_batch, query_batch) tuples for test tasks
            additional_adaptation_steps: Additional inner loop steps to perform during testing
                                       (beyond self.num_inner_steps)
            compute_confidence_intervals: Whether to compute confidence intervals for metrics
            compute_per_task_metrics: Whether to return metrics for each individual task
            return_adapted_models: Whether to return the adapted models for each task
            custom_metrics: Dictionary of custom metric functions to apply
            verbose: Whether to print verbose output
            adaptation_evaluation_steps: If provided, evaluate at these specific step counts 
                                        to track adaptation progress

        Returns:
            Dictionary with evaluation metrics, and optionally confidence intervals,
            per-task metrics, and adapted models
        """
        if verbose:
            from tqdm import tqdm
            test_iter = tqdm(test_tasks, desc="Meta-testing tasks")
        else:
            test_iter = test_tasks

        # Initialize results containers
        query_losses = []
        all_metrics = {}
        per_task_metrics = {} if compute_per_task_metrics else None
        adapted_models = {} if return_adapted_models else None

        # Initialize custom metrics
        custom_metrics = custom_metrics or {}
        for metric_name in custom_metrics:
            all_metrics[metric_name] = []

        # Set up adaptation step evaluation
        adaptation_step_metrics = {}
        if adaptation_evaluation_steps:
            for step in adaptation_evaluation_steps:
                adaptation_step_metrics[step] = {'query_loss': []}
                for metric_name in custom_metrics:
                    adaptation_step_metrics[step][metric_name] = []

        # Process each test task
        for task_idx, (support_batch, query_batch) in enumerate(test_iter):
            # Dictionary to store this task's metrics
            if compute_per_task_metrics:
                per_task_metrics[task_idx] = {}

            # Track adaptation progress if requested
            if adaptation_evaluation_steps:
                # Initialize a model for this task
                task_model = copy.deepcopy(self.model)
                task_model.to(self.device)

                # Process batches
                support_inputs, support_targets = self._process_batch(
                    support_batch)
                query_inputs, query_targets = self._process_batch(query_batch)

                # Evaluate at each adaptation step
                for step in adaptation_evaluation_steps:
                    # Create a clean copy for this evaluation
                    step_model = copy.deepcopy(self.model)
                    step_model.to(self.device)

                    # Adapt for exactly this many steps
                    adapted_model = self.adapt(
                        support_batch,
                        num_steps=step
                    )

                    # Evaluate
                    adapted_model.eval()
                    with torch.no_grad():
                        query_outputs = adapted_model(query_inputs)
                        step_loss = self.outer_loss_fn(
                            query_outputs, query_targets)
                        adaptation_step_metrics[step]['query_loss'].append(
                            step_loss.item())

                        # Compute custom metrics
                        for metric_name, metric_fn in custom_metrics.items():
                            try:
                                metric_value = metric_fn(
                                    query_outputs, query_targets)
                                adaptation_step_metrics[step][metric_name].append(
                                    metric_value)
                            except Exception as e:
                                logger.warning(
                                    f"Error computing metric {metric_name}: {e}")

            # Adapt model to task, potentially with more steps
            num_steps = self.num_inner_steps + additional_adaptation_steps

            if return_adapted_models or compute_per_task_metrics:
                # Need the model and loss separately
                query_loss, adapted_model = self.inner_loop(
                    support_batch,
                    query_batch,
                    num_steps=num_steps,
                    return_adapted_model=True
                )
                query_losses.append(query_loss.item())

                if compute_per_task_metrics:
                    per_task_metrics[task_idx]['query_loss'] = query_loss.item()

                if return_adapted_models:
                    adapted_models[task_idx] = adapted_model

                # Compute custom metrics
                if custom_metrics:
                    query_inputs, query_targets = self._process_batch(
                        query_batch)
                    adapted_model.eval()
                    with torch.no_grad():
                        query_outputs = adapted_model(query_inputs)

                        for metric_name, metric_fn in custom_metrics.items():
                            try:
                                metric_value = metric_fn(
                                    query_outputs, query_targets)

                                if metric_name not in all_metrics:
                                    all_metrics[metric_name] = []
                                all_metrics[metric_name].append(metric_value)

                                if compute_per_task_metrics:
                                    per_task_metrics[task_idx][metric_name] = metric_value
                            except Exception as e:
                                logger.warning(
                                    f"Error computing metric {metric_name}: {e}")

            else:
                # Just need the loss
                query_loss = self.inner_loop(
                    support_batch,
                    query_batch,
                    num_steps=num_steps
                )
                query_losses.append(query_loss.item())

        # Calculate average metrics
        avg_query_loss = sum(query_losses) / \
            len(query_losses) if query_losses else 0.0

        # Prepare results dictionary
        results = {
            'avg_query_loss': avg_query_loss
        }

        # Add custom metrics
        for metric_name, values in all_metrics.items():
            if values:
                results[f'avg_{metric_name}'] = sum(values) / len(values)

        # Add confidence intervals if requested
        if compute_confidence_intervals:
            import numpy as np
            from scipy import stats

            # Confidence interval for query loss
            loss_array = np.array(query_losses)
            n = len(loss_array)
            if n > 1:
                se = stats.sem(loss_array)
                ci_95 = se * stats.t.ppf((1 + 0.95) / 2, n-1)
                results['query_loss_95ci'] = ci_95
                results['query_loss_std'] = loss_array.std()

            # Confidence intervals for custom metrics
            for metric_name, values in all_metrics.items():
                if len(values) > 1:
                    try:
                        metric_array = np.array(values)
                        se = stats.sem(metric_array)
                        ci_95 = se * \
                            stats.t.ppf((1 + 0.95) / 2, len(metric_array)-1)
                        results[f'{metric_name}_95ci'] = ci_95
                        results[f'{metric_name}_std'] = metric_array.std()
                    except Exception as e:
                        logger.warning(
                            f"Error computing CI for {metric_name}: {e}")

        # Add per-task metrics if requested
        if compute_per_task_metrics:
            results['per_task_metrics'] = per_task_metrics

        # Add adapted models if requested
        if return_adapted_models:
            results['adapted_models'] = adapted_models

        # Add adaptation progress metrics if requested
        if adaptation_evaluation_steps:
            adaptation_progress = {}
            for step, metrics in adaptation_step_metrics.items():
                step_results = {}
                for metric_name, values in metrics.items():
                    if values:
                        step_results[f'avg_{metric_name}'] = sum(
                            values) / len(values)
                adaptation_progress[step] = step_results
            results['adaptation_progress'] = adaptation_progress

        return results

    def visualize_adaptation(
        self,
        task: Tuple[Any, Any],
        max_steps: int = 10,
        step_interval: int = 1,
        metric_fn: Optional[Callable] = None,
        metric_name: str = 'accuracy',
        plot_loss: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        plot_baseline: bool = True
    ) -> 'matplotlib.figure.Figure':
        """
        Visualize the adaptation process for a single task.

        Args:
            task: Tuple of (support_batch, query_batch)
            max_steps: Maximum number of adaptation steps to evaluate
            step_interval: Interval between step evaluations
            metric_fn: Function to compute a custom metric (e.g., accuracy)
            metric_name: Name of the custom metric for plotting
            plot_loss: Whether to plot the loss curve
            figsize: Figure size for the plot
            plot_baseline: Whether to plot baseline (no adaptation) performance

        Returns:
            Matplotlib figure with the adaptation visualization
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.error(
                "Matplotlib is required for visualization. Install with 'pip install matplotlib'")
            return None

        # Unpack task
        support_batch, query_batch = task

        # Process query batch for evaluation
        query_inputs, query_targets = self._process_batch(query_batch)

        # Initialize containers
        steps = list(range(0, max_steps + 1, step_interval))
        losses = []
        metrics = []

        # Evaluate at each step
        for step in steps:
            # Create a copy of the model
            task_model = copy.deepcopy(self.model)
            task_model.to(self.device)

            if step > 0:
                # Adapt the model for this many steps
                task_model = self.adapt(
                    support_batch,
                    num_steps=step
                )

            # Evaluate on query set
            task_model.eval()
            with torch.no_grad():
                query_outputs = task_model(query_inputs)
                step_loss = self.outer_loss_fn(query_outputs, query_targets)
                losses.append(step_loss.item())

                # Compute custom metric if provided
                if metric_fn is not None:
                    try:
                        metric_value = metric_fn(query_outputs, query_targets)
                        metrics.append(metric_value)
                    except Exception as e:
                        logger.warning(f"Error computing metric: {e}")
                        metrics.append(np.nan)

        # Create figure
        fig, axes = plt.subplots(
            1, 2 if plot_loss and metric_fn else 1, figsize=figsize)

        if plot_loss and metric_fn:
            ax1, ax2 = axes
        else:
            ax1 = axes if not isinstance(axes, np.ndarray) else axes[0]

        # Plot loss
        if plot_loss:
            ax1.plot(steps, losses, 'o-', color='blue', label='Query Loss')

            if plot_baseline:
                # Add horizontal line for baseline (no adaptation)
                ax1.axhline(y=losses[0], color='red',
                            linestyle='--', label='No Adaptation')

            ax1.set_xlabel('Adaptation Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Adaptation Loss Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Plot custom metric
        if metric_fn is not None:
            ax = ax2 if plot_loss else ax1
            ax.plot(steps, metrics, 'o-', color='green',
                    label=metric_name.title())

            if plot_baseline:
                # Add horizontal line for baseline (no adaptation)
                ax.axhline(y=metrics[0], color='red',
                           linestyle='--', label='No Adaptation')

            ax.set_xlabel('Adaptation Steps')
            ax.set_ylabel(metric_name.title())
            ax.set_title(f'Adaptation {metric_name.title()} Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        return fig

    def integrate_with_gnn(
        self,
        gnn_encoder: nn.Module,
        gnn_decoder: nn.Module,
        graph_batcher=None,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Integrate MAML with GNN encoder and decoder components.

        Creates a composite model that can be used for meta-learning on graph tasks,
        combining the GNN components with MAML optimization.

        Args:
            gnn_encoder: The GNN encoder model
            gnn_decoder: The GNN decoder model
            graph_batcher: Optional graph batching utility for handling graph data
            device: Device to move the model to

        Returns:
            Integrated model that can be used with MAML
        """
        # Import GraphBatcher if not provided
        if graph_batcher is None and HAS_PYGEOMETRIC:
            try:
                from causal_meta.inference.models.graph_utils import GraphBatcher
                graph_batcher = GraphBatcher(pad_to_max=True)
            except ImportError:
                logger.warning(
                    "Could not import GraphBatcher from causal_meta.inference.models.graph_utils")

        # Create a composite model
        class GNNCompositeModel(nn.Module):
            def __init__(self, encoder, decoder, device=None):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.device = device or torch.device('cpu')
                self.to(self.device)

            def forward(self, batch):
                # Handle batched graph data
                if HAS_PYGEOMETRIC and isinstance(batch, (GraphData, GraphBatch)):
                    # Encode the graph to latent representation
                    z = self.encoder(batch)

                    # Decode latent representation back to graph
                    if hasattr(batch, 'num_nodes'):
                        num_nodes = batch.num_nodes
                    else:
                        num_nodes = batch.x.size(0)

                    reconstructed = self.decoder(z, num_nodes)

                    return reconstructed
                else:
                    # Handle non-graph data
                    return self.decoder(self.encoder(batch))

            def loss(self, outputs, targets):
                """
                Compute reconstruction loss for graph data.

                Args:
                    outputs: Model outputs from forward pass
                    targets: Target graph data

                Returns:
                    Reconstruction loss
                """
                # Check if we're dealing with graph data
                if HAS_PYGEOMETRIC and isinstance(targets, (GraphData, GraphBatch)):
                    # Extract edge indices and node features from outputs and targets
                    if isinstance(outputs, tuple):
                        # Handle case where decoder returns multiple outputs
                        edge_logits = outputs[0]
                        if len(outputs) > 1:
                            node_features = outputs[1]
                        else:
                            node_features = None
                    else:
                        # Handle case where decoder returns single output
                        edge_logits = outputs
                        node_features = None

                    # Compute edge prediction loss
                    target_adj = torch_geometric.utils.to_dense_adj(
                        targets.edge_index,
                        max_num_nodes=targets.x.size(0)
                    ).squeeze(0)

                    # Binary cross entropy loss for edge prediction
                    edge_loss = nn.BCEWithLogitsLoss()(edge_logits, target_adj)

                    # Optionally compute node feature reconstruction loss
                    if node_features is not None and hasattr(targets, 'x'):
                        node_loss = nn.MSELoss()(node_features, targets.x)
                        # Combine losses (can adjust weights as needed)
                        return edge_loss + 0.1 * node_loss
                    else:
                        return edge_loss
                else:
                    # Default to MSE loss for non-graph data
                    return nn.MSELoss()(outputs, targets)

        # Create the composite model
        composite_model = GNNCompositeModel(
            gnn_encoder, gnn_decoder, device or self.device)

        # Attach the graph batcher if available
        if graph_batcher is not None:
            composite_model.graph_batcher = graph_batcher

        return composite_model

    def save(self, path: str):
        """
        Save the MAML model and metadata.

        Args:
            path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.outer_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'num_inner_steps': self.num_inner_steps,
            'first_order': self.first_order,
            'meta_iterations': self.meta_iterations,
            'task_adaptations': self.task_adaptations,
            'meta_losses': self.meta_losses,
            'per_param_lr': self.per_param_lr,
        }

        # Save param-specific learning rates if used
        if self.per_param_lr:
            save_dict['param_lrs'] = {
                name: param.data for name, param in self.param_lrs.items()
            }

        torch.save(save_dict, path)
        logger.info(f"MAML model saved to {path}")

    def load(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Load the MAML model and metadata.

        Args:
            path: Path to the saved model
            map_location: Location to map tensors to (for loading on different devices)
        """
        checkpoint = torch.load(path, map_location=map_location)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.outer_optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

        # Load hyperparameters
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.outer_lr = checkpoint.get('outer_lr', self.outer_lr)
        self.num_inner_steps = checkpoint.get(
            'num_inner_steps', self.num_inner_steps)
        self.first_order = checkpoint.get('first_order', self.first_order)
        self.per_param_lr = checkpoint.get('per_param_lr', self.per_param_lr)

        # Load param-specific learning rates if used
        if self.per_param_lr and 'param_lrs' in checkpoint:
            for name, value in checkpoint['param_lrs'].items():
                if name in self.param_lrs:
                    self.param_lrs[name].data.copy_(value)

        # Load metadata
        self.meta_iterations = checkpoint.get('meta_iterations', 0)
        self.task_adaptations = checkpoint.get('task_adaptations', 0)
        self.meta_losses = checkpoint.get('meta_losses', [])

        logger.info(f"MAML model loaded from {path}")


class SimpleMAML(nn.Module):
    """
    Simplified MAML implementation for demos.
    
    This class provides a basic implementation of Model-Agnostic Meta-Learning
    to adapt neural networks for causal discovery using interventional data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 3
    ):
        """Initialize the MAML model."""
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def clone_model(self) -> nn.Module:
        """Create a clone of the model for adaptation."""
        clone = copy.deepcopy(self.model)
        return clone
    
    def prepare_graph_inputs(self, x: torch.Tensor, graph: Optional[CausalGraph] = None):
        """
        Prepare graph structured inputs for AmortizedCausalDiscovery models.
        
        Args:
            x: Input tensor [batch_size, seq_length, num_nodes]
            graph: Optional graph structure
            
        Returns:
            Dictionary with node_features, edge_index, and batch tensors
        """
        batch_size, seq_length, num_nodes = x.shape
        device = x.device
        
        # Create node features (mean of sequence for simplicity)
        node_features = x.mean(dim=1).reshape(batch_size * num_nodes, 1)
        
        # Create edge index from fully connected graph if no graph provided
        if graph is None:
            # Create a fully connected graph for each batch
            edge_index_list = []
            for b in range(batch_size):
                # Offset for this batch
                offset = b * num_nodes
                
                # Create edges for all possible connections in this batch
                source_nodes = []
                target_nodes = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:  # No self-loops
                            source_nodes.append(i + offset)
                            target_nodes.append(j + offset)
                
                batch_edge_index = torch.tensor([source_nodes, target_nodes], 
                                                dtype=torch.long, device=device)
                edge_index_list.append(batch_edge_index)
            
            # Concatenate all batch edge indices
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            # Extract edge information from the provided graph
            adj_matrix = torch.tensor(graph.get_adjacency_matrix(), 
                                     dtype=torch.float32, device=device)
            
            # Convert adjacency matrix to edge index
            edge_index_list = []
            for b in range(batch_size):
                # Offset for this batch
                offset = b * num_nodes
                
                # Find edges in the adjacency matrix
                edges = torch.nonzero(adj_matrix, as_tuple=True)
                sources, targets = edges[0], edges[1]
                
                # Add batch offset
                sources = sources + offset
                targets = targets + offset
                
                # Stack to create edge index
                batch_edge_index = torch.stack([sources, targets], dim=0)
                edge_index_list.append(batch_edge_index)
            
            # Concatenate all batch edge indices
            if edge_index_list:
                edge_index = torch.cat(edge_index_list, dim=1)
            else:
                # If no edges, create empty edge index
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Create batch tensor (one batch index per node)
        batch = torch.repeat_interleave(torch.arange(batch_size, device=device), 
                                        repeats=num_nodes)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'batch': batch
        }
    
    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        graph: Optional[CausalGraph] = None,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the model using support data.
        
        Args:
            support_data: Tuple of (inputs, targets) for adaptation
            graph: Causal graph structure (optional)
            num_steps: Number of adaptation steps (overrides default if provided)
            
        Returns:
            Adapted model
        """
        # Initialize parameters for adaptation
        steps = num_steps if num_steps is not None else self.num_inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = self.clone_model()
        adapted_model.train()
        
        # Get the inputs and targets from support data
        inputs, targets = support_data
        
        # Create an optimizer for the inner loop
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Prepare graph inputs for AmortizedCausalDiscovery
        if hasattr(adapted_model, 'graph_encoder') and inputs.dim() >= 3:
            graph_inputs = self.prepare_graph_inputs(inputs, graph)
        
        # Adaptation steps
        for step in range(steps):
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass 
            if hasattr(adapted_model, 'graph_encoder') and inputs.dim() >= 3:
                # For AmortizedCausalDiscovery models with the comprehensive interface
                try:
                    # Try with the full interface
                    outputs = adapted_model(
                        x=inputs,
                        node_features=graph_inputs['node_features'],
                        edge_index=graph_inputs['edge_index'],
                        batch=graph_inputs['batch']
                    )
                    
                    # Handle different output types
                    if isinstance(outputs, dict) and 'adjacency' in outputs:
                        predictions = outputs['adjacency']
                    elif isinstance(outputs, torch.Tensor):
                        predictions = outputs
                    else:
                        # Just use graph encoder output directly
                        predictions = adapted_model.graph_encoder(inputs)
                    
                    # Ensure predictions and targets have the same shape
                    if predictions.shape != targets.shape:
                        # If predictions is [batch, nodes, nodes] and targets is [nodes, nodes]
                        if predictions.dim() > targets.dim():
                            targets = targets.unsqueeze(0).expand_as(predictions)
                        # If targets is [batch, nodes, nodes] and predictions is [nodes, nodes]
                        elif targets.dim() > predictions.dim():
                            predictions = predictions.unsqueeze(0).expand_as(targets)
                    
                    # Basic MSE loss
                    loss = F.mse_loss(predictions, targets)
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    # Fallback to using just the graph encoder
                    predictions = adapted_model.graph_encoder(inputs)
                    
                    # Ensure predictions and targets have the same shape
                    if predictions.shape != targets.shape:
                        # If predictions is [batch, nodes, nodes] and targets is [nodes, nodes]
                        if predictions.dim() > targets.dim():
                            targets = targets.unsqueeze(0).expand_as(predictions)
                        # If targets is [batch, nodes, nodes] and predictions is [nodes, nodes]
                        elif targets.dim() > predictions.dim():
                            predictions = predictions.unsqueeze(0).expand_as(targets)
                    
                    loss = F.mse_loss(predictions, targets)
            else:
                # For simpler models
                outputs = adapted_model(inputs)
                loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        # Return the adapted model
        return adapted_model


class MAMLForCausalDiscovery(nn.Module):
    """
    MAML implementation for causal discovery.
    
    This class integrates the MAML algorithm with AmortizedCausalDiscovery for
    fast adaptation to new causal structures with interventional data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 3,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize the MAML model for causal discovery."""
        super().__init__()
        
        self.model = model
        self.device = torch.device(device)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Initialize MAML algorithm
        self.maml = SimpleMAML(
            model=model,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps
        )
        
    def adapt(
        self,
        graph: CausalGraph,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the model to a new causal graph using interventional data.
        
        Args:
            graph: Causal graph structure
            support_data: Tuple of (inputs, targets) for adaptation
            num_steps: Number of adaptation steps (overrides default if provided)
            
        Returns:
            Adapted model for the given causal graph
        """
        # Use the MAML algorithm to adapt the model
        adapted_model = self.maml.adapt(
            support_data=support_data,
            graph=graph,
            num_steps=num_steps or self.num_inner_steps
        )
        
        return adapted_model
    
    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[CausalGraph] = None,
        adapted_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            graph: Optional causal graph (if adaptation is needed)
            adapted_model: Optional pre-adapted model (if already adapted)
            
        Returns:
            Model predictions
        """
        if adapted_model is not None:
            # Use the pre-adapted model
            return adapted_model(x)
        else:
            # Use the base model without adaptation
            return self.model(x)
