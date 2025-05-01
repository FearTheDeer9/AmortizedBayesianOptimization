import logging
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np # Ensure numpy is imported
import networkx as nx # Added networkx import for type hint
import pandas as pd # Added for type hinting
import torch # Added for checkpointing
import os # Added for checkpointing paths
import copy # Added for deepcopy in checkpointing
from causal_meta.graph.causal_graph import CausalGraph # Corrected path
# Placeholder imports - replace with actuals when available
from ..environments.scm import StructuralCausalModel
# from ..environments.interventions import Intervention # Maybe not needed directly here
# from ..graph.causal_graph import CausalGraph # Already imported above
from .task_representation import TaskRepresentation
from .maml import MAML # Assuming MAML framework exists
# Import the actual handler
# from ..graph.causal_graph_handler import CausalGraphHandler # Removed - Handler not found
# from causal_meta.utils.causal_effect_estimator import estimate_causal_effect_via_backdoor # Example estimator - Module not found
# from causal_meta.utils.visualization import plot_graph # Example visualizer - Module not found

# Remove dummy types if real imports are available
# StructuralCausalModel = Any # Use real SCM type if available
# Intervention = Any # Define if needed
# CausalGraph = Any # Use real import
# TaskRepresentation = Any # Use real import
# MAML = Any # Use real import

# BoTorch / GPyTorch Imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, AcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
# --- Add required GPyTorch imports for Subtask 4.1 ---
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
# --- End imports for Subtask 4.1 ---


class MetaCBO:
    """Core Meta-CBO implementation combining meta-learning and causal Bayesian optimization."""

    def __init__(
        self,
        task_representation_model: TaskRepresentation,
        maml_framework: MAML, # Integrate MAML properly now
        # causal_graph_handler: CausalGraphHandler, # Removed - Handler not found
        # maml_config: Dict[str, Any], # MAML config might be handled by MAML framework itself
        logger: Optional[logging.Logger] = None,
        random_seed: Optional[int] = None,
        # Add other necessary parameters based on PRD/API spec
        meta_lr: float = 0.001, # Example - needed for meta_train below
        # Add params for GP configuration
        gp_kernel_nu: float = 2.5,
        gp_noise_prior_scale: float = 1.0, # Example prior scale for noise
    ):
        """
        Initializes the MetaCBO framework.

        Args:
            task_representation_model: Model for embedding tasks.
            maml_framework: The MAML framework instance to use.
            # causal_graph_handler: Handler for causal graph operations. # Removed
            logger: Optional logger instance.
            random_seed: Optional random seed for reproducibility.
            meta_lr: Example meta learning rate.
            gp_kernel_nu: Smoothness parameter for the Matern kernel.
            gp_noise_prior_scale: Scale for the noise prior in GaussianLikelihood.
        """
        self.task_representation_model = task_representation_model
        self.maml_framework = maml_framework
        # self.causal_graph_handler = causal_graph_handler # Removed
        # self.maml_config = maml_config # Store config, MAML might use it internally

        if logger is None:
            # Get default logger if none provided
            self.logger = logging.getLogger(__name__)

            # Check if our specific handler is already present
            handler_exists = any(
                isinstance(h, logging.StreamHandler) and
                isinstance(h.formatter, logging.Formatter) and
                h.formatter._fmt == '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                for h in self.logger.handlers
            )

            # Add our default handler only if it doesn't exist
            if not handler_exists:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            # Always set default level to INFO if no logger was passed
            self.logger.setLevel(logging.INFO)
            # Prevent propagation to avoid duplicate messages if root logger is configured
            self.logger.propagate = False
        else:
            # Use the provided logger
            self.logger = logger

        # Store meta_lr if needed for MAML call
        self.meta_lr = meta_lr

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            # Set other seeds (e.g., torch)
            torch.manual_seed(self.random_seed) # Add torch seed
            self.logger.info(f"Set random seed to {self.random_seed}")

        # Internal state attributes (add as needed)
        self._meta_parameters = maml_framework.meta_parameters # Get initial params from MAML
        self._is_meta_trained = False
        self._meta_training_history: List[Dict[str, Any]] = [] # For tracking progress
        self._best_meta_params: Optional[Dict[str, Any]] = None
        self._best_meta_performance: float = -float('inf')
        # Add placeholder for storing task-specific graph info if needed
        self._task_graphs: Dict[Any, CausalGraph] = {} # Store CausalGraph instances

        # --- GP Model Configuration (Subtask 4.1) ---
        self.gp_kernel_nu = gp_kernel_nu
        self.gp_noise_prior_scale = gp_noise_prior_scale
        self.likelihood = GaussianLikelihood(
            noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05), # Example prior
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        # Factory function to create kernel based on input dimension
        self.kernel_factory = lambda dim: ScaleKernel(
            MaternKernel(nu=self.gp_kernel_nu, ard_num_dims=dim)
        )
        # --- Dataset and Model Storage (Subtask 4.1) ---
        self.datasets: Dict[Any, Dict[str, torch.Tensor]] = {} # task_id -> {'train_x': tensor, 'train_y': tensor}
        self.models: Dict[Any, Tuple[SingleTaskGP, GaussianLikelihood]] = {} # task_id -> (model, likelihood)
        # Node mapping cache (populated when encoding interventions)
        self._node_to_idx: Optional[Dict[str, int]] = None
        self._idx_to_node: Optional[Dict[int, str]] = None
        self._current_task_nodes: Optional[List[str]] = None # Track nodes for current task context


    def _validate_and_store_graph(self, task: StructuralCausalModel) -> Optional[CausalGraph]:
        """Validates the causal graph of a task and stores it."""
        # Use a proper task identifier if available on the SCM object
        task_id = getattr(task, 'task_id', id(task)) # Example identifier
        if task_id in self._task_graphs:
            self.logger.debug(f"Returning cached graph for task {task_id}")
            return self._task_graphs[task_id] # Return cached graph

        graph = None
        try:
            # --- Attempt to get graph directly from task --- #
            # Prioritize .graph attribute used in tests
            if hasattr(task, 'graph') and isinstance(task.graph, CausalGraph):
                self.logger.debug(f"Found .graph attribute of type CausalGraph for task {task_id}")
                graph = task.graph
            elif hasattr(task, 'get_causal_graph') and callable(task.get_causal_graph):
                self.logger.debug(f"Calling get_causal_graph() for task {task_id}")
                graph = task.get_causal_graph()
            elif isinstance(task, CausalGraph):
                self.logger.debug(f"Task object itself is a CausalGraph for task {task_id}")
                graph = task # If the task object itself is the graph
            else:
                 # Fallback/Placeholder: Try creating from adjacency if possible?
                 self.logger.warning(f"Cannot directly extract CausalGraph from task {task_id} via .graph or get_causal_graph(). Trying adjacency matrix.")
                 if hasattr(task, 'adjacency_matrix'):
                     adj = task.adjacency_matrix()
                     # Assuming CausalGraph has a class method for this
                     if hasattr(CausalGraph, 'from_adjacency_matrix') and callable(CausalGraph.from_adjacency_matrix):
                         graph = CausalGraph.from_adjacency_matrix(adj)
                         self.logger.info(f"Inferred CausalGraph from adjacency matrix for task {task_id}.")
                     else:
                         self.logger.error(f"Task {task_id} has adjacency_matrix but CausalGraph lacks from_adjacency_matrix method.")
                 else:
                      self.logger.error(f"Cannot obtain CausalGraph from task {task_id}. No suitable method or attribute found.")
                      return None # Explicitly return None if no graph found
            # --- End Graph Retrieval --- #

            # --- Validate the obtained graph --- #
            if graph is None:
                 self.logger.error(f"Graph retrieval failed for task {task_id}. Cannot validate.")
                 return None

            if not isinstance(graph, CausalGraph):
                 self.logger.error(f"Retrieved object for task {task_id} is not a CausalGraph instance (Type: {type(graph)}). Validation failed.")
                 return None

            # Perform validation checks (e.g., acyclicity)
            if not graph.is_dag(): # Assuming CausalGraph inherits or implements is_dag
                 self.logger.warning(f"Graph for task {task_id} is not a DAG. Invalid causal structure.")
                 return None

            # --- Store and return --- #
            self.logger.debug(f"Successfully validated graph for task {task_id}")
            self._task_graphs[task_id] = graph
            return graph

        except Exception as e:
            self.logger.exception(f"Unexpected exception during graph validation for task {task_id}: {e}") # Use .exception for traceback
            return None

    def meta_train(self, task_family: List[StructuralCausalModel], **kwargs) -> None:
        """
        Performs meta-training across a family of tasks.

        Args:
            task_family: A list of SCM tasks for meta-training.
            **kwargs: Additional training parameters (e.g., epochs, batch_size,
                      patience for early stopping, checkpoint_dir).
        """
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 16)
        patience = kwargs.get('patience', 10)
        checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger.info(f"Starting meta-training on {len(task_family)} tasks for {epochs} epochs...")
        self.logger.info(f"Batch size: {batch_size}, Patience: {patience}, Checkpoint Dir: {checkpoint_dir}")

        # --- Early Stopping Setup ---
        best_meta_loss = float('inf')
        patience_counter = 0
        self._best_meta_params = copy.deepcopy(self._meta_parameters) # Initialize best params

        for epoch in range(epochs):
            self.logger.info(f"Starting Meta Epoch {epoch+1}/{epochs}")
            # --- Shuffle tasks for batching --- #
            np.random.shuffle(task_family) # Shuffle task list in-place
            epoch_meta_losses = []
            task_embeddings = {} # Initialize the dictionary here

            # --- Mini-batch Loop --- #
            for i in range(0, len(task_family), batch_size):
                current_batch = task_family[i:i + batch_size]
                self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(task_family) + batch_size - 1)//batch_size} (size: {len(current_batch)}) ")

                # --- Task Representation & Graph Handling per batch --- #
                maml_input_batch = []
                for task in current_batch:
                    task_id = getattr(task, 'id', task) # Simple example

                    # Validate and store graph first
                    graph = self._validate_and_store_graph(task)
                    if graph is None:
                        self.logger.warning(f"Skipping task {task_id} due to invalid graph.")
                        continue

                    # Try embedding
                    try:
                        # Task representation might need the graph or data
                        # Adapt embedding logic as needed
                        task_embedding = self.task_representation_model.embed_task(task, graph=graph)
                        task_embeddings[task_id] = task_embedding
                        self.logger.debug(f"Embedded task {task_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to embed task {task_id}: {e}")
                        continue # Skip task if embedding fails

                    # Prepare data for MAML
                    support_data = getattr(task, 'support_data', None) # Example
                    query_data = getattr(task, 'query_data', None)     # Example
                    if support_data is not None and query_data is not None:
                         maml_input_batch.append({
                             'task': task,
                             'graph': graph, # Include graph info
                             'embedding': task_embeddings[task_id],
                             'support': support_data,
                             'query': query_data
                         })
                    else:
                         self.logger.warning(f"Skipping task {task_id} in meta-train due to missing data.")

                if not maml_input_batch:
                    self.logger.error("No valid tasks with data found for meta-training.")
                    continue

                # Call MAML framework's outer loop update
                # The exact signature depends on the MAML implementation
                try:
                    # Pass necessary args like batch, epochs, etc.
                    # MAML might handle batching internally based on config
                    # Track metrics from MAML update
                    meta_update_metrics = self.maml_framework.outer_loop_update(
                        task_batch=maml_input_batch, # Ensure batch has necessary info (support/query data, embeddings)
                        epochs=kwargs.get('epochs', 1), # Example kwarg
                        # Add other MAML-specific parameters as needed
                        meta_lr=self.meta_lr, # Pass meta learning rate
                        # **self.maml_config.get('outer_loop_args', {}) # Pass other config args if needed
                    )
                    self._meta_parameters = self.maml_framework.meta_parameters # Update meta parameters
                    self._is_meta_trained = True

                    # --- Progress Tracking & Checkpointing ---
                    epoch_metrics = {
                        'epoch': kwargs.get('current_epoch', 0), # Assuming epoch info is passed
                        **meta_update_metrics # Include metrics from MAML update (e.g., meta-loss)
                    }
                    self._meta_training_history.append(epoch_metrics)
                    self.logger.info(f"Meta Epoch {epoch_metrics['epoch']} Metrics: {epoch_metrics}")

                    # Simple checkpointing based on meta-loss (assuming lower is better)
                    current_performance = -epoch_metrics.get('meta_loss', float('inf')) # Example metric
                    if current_performance > self._best_meta_performance:
                        self._best_meta_performance = current_performance
                        self._best_meta_params = self._meta_parameters # Store best params
                        self.logger.info(f"New best meta-performance: {current_performance}. Checkpointing parameters.")
                        # Add actual checkpoint saving logic here (e.g., save to file)

                    # Decide if partial results should be kept or state reset
                    # self._is_meta_trained = False # Reset state on error?

                except Exception as e:
                    self.logger.error(f"Error during MAML outer loop update: {e}")
            # --- Epoch End: Calculate Average Loss & Checkpointing/Early Stopping --- #
            if not epoch_meta_losses:
                self.logger.warning(f"No batches were successfully processed in epoch {epoch+1}. Skipping metric update and early stopping check.")
                continue

            avg_epoch_meta_loss = np.mean([m.get('meta_loss', np.inf) for m in epoch_meta_losses])
            self.logger.info(f"Meta Epoch {epoch+1} completed. Average Meta Loss: {avg_epoch_meta_loss:.4f}")

            # Update history (use the average loss for the epoch)
            epoch_summary_metrics = {'epoch': epoch + 1, 'avg_meta_loss': avg_epoch_meta_loss}
            # Optionally add other aggregated metrics from epoch_meta_losses if needed
            self._meta_training_history.append(epoch_summary_metrics)

            # Checkpointing based on average epoch meta-loss
            if avg_epoch_meta_loss < best_meta_loss:
                best_meta_loss = avg_epoch_meta_loss
                self._best_meta_params = copy.deepcopy(self._meta_parameters) # Store best params
                patience_counter = 0 # Reset patience
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.logger.info(
                    f"New best meta-loss: {best_meta_loss:.4f}. "
                    f"Resetting patience and saving checkpoint to {checkpoint_path}"
                )
                self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                self.logger.info(f"Meta loss did not improve. Patience: {patience_counter}/{patience}")

            # Early Stopping Check
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after epoch {epoch+1}. No improvement in meta loss for {patience} epochs.")
                break # Exit meta-training loop

        self.logger.info(f"Meta-training finished after {epoch+1} epochs.")
        if self._best_meta_params:
             self.logger.info(f"Best meta-loss achieved: {best_meta_loss:.4f}")
             # Optionally load the best parameters back into the main parameters
             # self._meta_parameters = self._best_meta_params
        else:
             self.logger.warning("No best parameters recorded during meta-training.")


    def adapt(self, new_task: StructuralCausalModel, adaptation_data: Any, **kwargs) -> Any:
        """
        Adapts the meta-learned model to a new task.

        Args:
            new_task: The new SCM task to adapt to.
            adaptation_data: Data specific to the new task for adaptation.
            **kwargs: Additional adaptation parameters.

        Returns:
            The adapted model parameters or the adapted model object.
        """
        self.logger.info(f"Adapting to new task...")

        # --- Validate Graph for New Task ---
        graph = self._validate_and_store_graph(new_task)
        if graph is None:
            raise ValueError("Cannot adapt: Invalid causal graph for the new task.")

        # --- Get Task Embedding ---
        try:
            # Adapt embedding logic if it needs the graph
            task_embedding = self.task_representation_model.embed_task(new_task, graph=graph)
        except Exception as e:
            self.logger.error(f"Failed to embed new task for adaptation: {e}")
            raise ValueError("Cannot adapt: Failed to embed the new task.") from e

        # --- Call MAML Inner Loop ---
        if not self._is_meta_trained:
            self.logger.warning("Adapting from non-meta-trained parameters.")
            initial_params = self._meta_parameters # Or MAML default initial params
        else:
             initial_params = self._best_meta_params if self._best_meta_params else self._meta_parameters

        try:
            # The exact signature depends on the MAML implementation
            # Pass necessary info: data, embedding, initial params, inner_lr, steps, etc.
            adapted_params = self.maml_framework.inner_loop_update(
                task_data=adaptation_data,
                task_embedding=task_embedding, # Pass embedding
                initial_params=initial_params,
                # inner_lr=self.inner_lr, # Get from config or kwargs
                # num_steps=self.n_inner_steps, # Get from config or kwargs
                **kwargs # Pass other MAML inner loop args
            )
            self.logger.info("Adaptation completed successfully.")
            # Store adaptation history if needed
            # self._adaptation_history.append({'task_id': ..., 'adapted_params': adapted_params})
            return adapted_params
        except Exception as e:
            self.logger.error(f"Error during MAML inner loop adaptation: {e}")
            raise RuntimeError("Adaptation failed.") from e


    def evaluate(self, task: StructuralCausalModel, test_data: Any, adapted_model: Optional[Any] = None, **kwargs) -> Dict[str, float]:
        """
        Evaluates the performance of the model (meta-learned or adapted) on a task.

        Args:
            task: The SCM task for evaluation.
            test_data: Data used for evaluation.
            adapted_model: Optional adapted model; if None, uses meta-learned parameters.
            **kwargs: Additional evaluation parameters.

        Returns:
            A dictionary of evaluation metrics.
        """
        self.logger.info(f"Evaluating model on task...")
        task_id = getattr(task, 'id', task)
        graph = self._task_graphs.get(task_id)
        if graph is None:
             graph = self._validate_and_store_graph(task)
             if graph is None:
                 self.logger.error("Cannot evaluate task with invalid graph.")
                 return {} # Return empty metrics

        # Implement evaluation logic using either meta-parameters or adapted_model
        # Evaluation might involve causal effect estimation using the graph handler
        # Example:
        # effect = self.causal_graph_handler.estimate_effect(graph, adapted_model, ...)
        self.logger.warning("Evaluation logic not implemented.")
        raise NotImplementedError("Evaluation is not yet implemented.")
        # metrics = {"metric1": 0.0, "metric2": 0.0}
        # self.logger.info(f"Evaluation metrics: {metrics}")
        # return metrics


    def optimize_interventions(
        self,
        task: StructuralCausalModel,
        budget: int,
        target_variable: str,
        objective_metric: str = 'expected_improvement',
        initial_data: Optional[pd.DataFrame] = None, # Specify DataFrame
        adapted_model: Optional[Any] = None,
        **kwargs
    ) -> List[Tuple[str, Any]]: # Intervention is (node_name, value)
        """
        Optimizes interventions for a given task using Bayesian optimization,
        guided by the causal graph and potentially meta-learned priors.

        Args:
            task: The SCM representing the task for intervention optimization.
            budget: The maximum number of interventions allowed.
            target_variable: The variable whose outcome we want to optimize.
            objective_metric: Acquisition function or optimization metric.
            initial_data: Initial observational or interventional data.
            adapted_model: Task-adapted model parameters or model object.
            **kwargs: Additional optimization parameters (e.g., BO settings).

        Returns:
            A list of recommended interventions (node, value pairs).
        """
        self.logger.info(f"Starting intervention optimization for target '{target_variable}' with budget {budget}...")

        # --- Validate Graph for the Task ---
        graph = self._validate_and_store_graph(task)
        if graph is None:
            raise ValueError("Cannot optimize interventions: Invalid causal graph for the task.")

        # --- Setup Bayesian Optimization ---
        # 1. Define Search Space
        potential_intervention_nodes = self._get_valid_interventions(graph, target_variable)
        if not potential_intervention_nodes:
             self.logger.warning("No valid intervention nodes found based on graph structure.")
             return []

        # Convert simple search space (assuming continuous [0,1] for now) to BoTorch format
        # Requires mapping node names to dimension indices
        node_list = sorted(list(potential_intervention_nodes))
        node_to_dim = {node: i for i, node in enumerate(node_list)}
        dim_to_node = {i: node for node, i in node_to_dim.items()}
        search_dim = len(node_list)
        # BoTorch bounds are (2, d) tensor: [[lower_bounds], [upper_bounds]]
        bounds = torch.tensor([[0.0] * search_dim, [1.0] * search_dim], dtype=torch.double)
        self.logger.debug(f"Intervention search space dimension: {search_dim}, Bounds: {bounds.tolist()}")
        self.logger.debug(f"Node to dimension map: {node_to_dim}")

        # 2. Prepare Initial Data for GP
        # Need to convert initial_data (DataFrame) into train_X (interventions) and train_Y (outcomes)
        # This requires knowing the format of initial_data (how interventions are stored)
        # Placeholder:
        train_X = torch.empty((0, search_dim), dtype=torch.double)
        train_Y = torch.empty((0, 1), dtype=torch.double)
        if initial_data is not None and not initial_data.empty:
            self.logger.info(f"Processing {len(initial_data)} initial data points...")
            # TODO: Implement logic to extract intervention vectors (train_X) and target outcomes (train_Y)
            # from the initial_data DataFrame based on node_to_dim mapping.
            # Example structure needed in initial_data: columns for each node in node_list + target_variable
            try:
                # Assume columns exist for intervention nodes and target
                intervention_cols = [col for col in initial_data.columns if col in node_to_dim]
                if not intervention_cols or target_variable not in initial_data.columns:
                    self.logger.error("Initial data missing required intervention node columns or target variable column.")
                else:
                    # Simple case: assume rows represent interventions where node columns have the value
                    # Need a more robust way to identify which node was intervened on per row if not all nodes are set
                    train_X_list = []
                    for _, row in initial_data.iterrows():
                        intervention_vec = torch.full((search_dim,), float('nan'), dtype=torch.double) # Start with NaN
                        # Find which node was intervened on (this logic is likely too simple)
                        intervened_node = None
                        intervened_value = float('nan')
                        for node in node_list:
                            if node in row and pd.notna(row[node]): # Check if node value exists
                                intervened_node = node
                                intervened_value = row[node]
                                break # Assume one intervention per row for now
                        if intervened_node is not None:
                            intervention_vec[node_to_dim[intervened_node]] = intervened_value
                            # Need to handle non-intervened dimensions (e.g., fill with default/observed?)
                            # For now, maybe filter only rows with full intervention specs?

                        # THIS PART NEEDS REFINEMENT BASED ON HOW DATA IS LOGGED

                    train_X = torch.tensor(initial_data[node_list].values, dtype=torch.double)
                    train_Y = torch.tensor(initial_data[[target_variable]].values, dtype=torch.double)
                    # Filter out rows with NaNs in train_Y (or train_X) after assignment
                    nan_mask_y = torch.isnan(train_Y).any(dim=1)
                    nan_mask_x = torch.isnan(train_X).any(dim=1)
                    valid_mask = ~nan_mask_y & ~nan_mask_x # Consider NaNs in both X and Y

                    if (~valid_mask).any():
                         self.logger.warning(f"Removing {(~valid_mask).sum()} rows with NaN values from initial data.")
                         train_X = train_X[valid_mask]
                         train_Y = train_Y[valid_mask]

                    self.logger.warning("Initial data processing logic is a placeholder and needs refinement.")

            except KeyError as e:
                self.logger.error(f"Initial data processing failed. Missing column: {e}")
            except Exception as e:
                self.logger.exception(f"Unexpected error processing initial data: {e}")

        # 3. Initialize GP Model
        if len(train_Y) > 0:
             gp_model = SingleTaskGP(train_X, train_Y)
             mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
             fit_gpytorch_model(mll)
             self.logger.info(f"Initialized GP model with {len(train_Y)} initial points.")
        else:
             gp_model = None
             self.logger.warning("No valid initial data points. GP model not initialized. Starting with random exploration or prior.")
             # Handle the case with no initial data (e.g., use prior mean, random sampling for first steps)

        # --- Optimization Loop ---
        recommended_interventions: List[Tuple[str, Any]] = []
        observed_interventions: List[Dict[str, Any]] = [] # Initialize list to store history
        for i in range(budget):
            self.logger.info(f"Intervention optimization step {i+1}/{budget}")

            # The acquisition function should incorporate causal estimates
            next_intervention_point = self._select_next_intervention(
                acquisition_function_name=objective_metric, # Pass name
                gp_model=gp_model, # Pass current GP
                best_f=train_Y.max().item() if len(train_Y) > 0 else -float('inf'),
                bounds=bounds,
                graph=graph,
                target_variable=target_variable,
                current_data=pd.concat([initial_data, pd.DataFrame(recommended_interventions)]) if recommended_interventions else initial_data,
                **kwargs
            )

            if next_intervention_point is None:
                self.logger.warning("Could not select a next intervention. Stopping optimization.")
                break

            # Convert BoTorch point back to (node, value) - ASSUMES SINGLE POINT OPTIMIZATION
            # This logic needs refinement if optimize_acqf suggests multiple points or complex interventions
            intervention_dim_index = torch.argmax(torch.abs(next_intervention_point - 0.5)).item() # Heuristic: find dim furthest from 0.5?
            next_intervention_node = dim_to_node[intervention_dim_index]
            next_intervention_value = next_intervention_point[0, intervention_dim_index].item()
            intervention_tuple = (next_intervention_node, next_intervention_value)
            self.logger.info(f"Selected intervention: do({intervention_tuple[0]}={intervention_tuple[1]:.4f}) based on point {next_intervention_point.tolist()}")
            recommended_interventions.append(intervention_tuple)

            # 5. Apply intervention and observe outcome (simulate or query environment)
            try:
                # outcome = task.perform_intervention(intervention_tuple[0], intervention_tuple[1], target_variable)
                # Placeholder - Requires SCM/Environment interaction
                outcome = np.random.normal(loc=next_intervention_value, scale=0.1) # Example outcome based on value
                self.logger.info(f"Observed outcome for {target_variable}: {outcome:.4f}")
            except Exception as e:
                 self.logger.exception(f"Failed to apply intervention or observe outcome: {e}")
                 # Decide how to handle failure (e.g., stop, skip update)
                 break

            # 6. Update data and GP model
            new_X = next_intervention_point
            new_Y = torch.tensor([[outcome]], dtype=torch.double)

            # Append new data
            train_X = torch.cat([train_X, new_X], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)
            observed_interventions.append({'intervention': intervention_tuple, 'outcome': outcome, **dict(zip(node_list, new_X.tolist()[0]))})

            # Retrain GP model
            if gp_model is None:
                 gp_model = SingleTaskGP(train_X, train_Y)
            else:
                 # Use state dict method for efficiency if supported by GP library
                 gp_model = gp_model.condition_on_observations(X=new_X, Y=new_Y)

            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            try:
                fit_gpytorch_model(mll)
                self.logger.debug("GP model retrained successfully.")
            except Exception as e:
                 self.logger.error(f"Failed to retrain GP model: {e}")
                 # Handle retraining failure (e.g., revert to previous model state?)

        self.logger.info(f"Intervention optimization finished. Recommended interventions: {recommended_interventions}")
        # TODO: Actually implement BO logic above
        if not recommended_interventions and budget > 0:
             # Raise error if budget > 0 but loop finished early without recommendation (likely due to missing BO impl.)
             raise NotImplementedError("Bayesian Optimization logic for intervention selection is not implemented.")
        return recommended_interventions


    def _get_valid_interventions(self, graph: CausalGraph, target_variable: Optional[str] = None) -> Set[Any]:
        """
        Identifies valid intervention nodes based on the causal graph.

        Args:
            graph: The CausalGraph object for the task.
            target_variable: The final outcome variable (optional, to exclude).

        Returns:
            A set of node identifiers that are valid targets for intervention.
        """
        if graph is None or not isinstance(graph, CausalGraph):
             self.logger.error("_get_valid_interventions called with invalid graph.")
             return set()

        all_nodes = set(graph.get_nodes())
        valid_nodes = set()

        for node in all_nodes:
            # Exclude the target variable itself
            if node == target_variable:
                continue

            # Add other potential constraints based on strategy:
            # - e.g., Exclude nodes with no causal path to target?
            # if not graph.has_directed_path(node, target_variable):
            #     continue
            # - e.g., Exclude descendants of the target?
            # if target_variable is not None and graph.is_descendant(node, target_variable):
            #    continue

            valid_nodes.add(node)

        self.logger.debug(f"Identified valid intervention nodes: {valid_nodes}")
        return valid_nodes


    def _estimate_causal_effect(
        self,
        graph: CausalGraph,
        intervention_node: Any,
        intervention_value: Any,
        outcome_variable: str,
        current_data: pd.DataFrame
    ) -> float:
        """
        Estimates the causal effect of an intervention. Placeholder implementation.

        Args:
            graph: The CausalGraph object.
            intervention_node: The node being intervened upon.
            intervention_value: The value set for the intervention.
            outcome_variable: The target variable to measure the effect on.
            current_data: The currently available data.

        Returns:
            An estimated causal effect (e.g., change in outcome mean).
            Returns 0.0 if estimation fails or is not possible.
        """
        self.logger.debug(f"Estimating effect of do({intervention_node}={intervention_value}) on {outcome_variable}")
        if graph is None or not isinstance(graph, CausalGraph):
             self.logger.error("_estimate_causal_effect called with invalid graph.")
             return 0.0
        if current_data is None or current_data.empty:
             self.logger.warning("Cannot estimate causal effect without data.")
             return 0.0

        try:
            # Example: Use backdoor adjustment if possible
            # We now use the graph object directly
            adjustment_set = graph.get_backdoor_adjustment_set(intervention_node, outcome_variable)
            if adjustment_set is not None:
                 # effect = estimate_causal_effect_via_backdoor( # Commented out - Module not found
                 #     data=current_data,
                 #     treatment=intervention_node,
                 #     outcome=outcome_variable,
                 #     adjustment_set=adjustment_set,
                 #     treatment_value=intervention_value # Need to handle how value is used
                 # )
                 # self.logger.debug(f"Estimated effect using backdoor adjustment: {effect}")
                 # return effect if effect is not None else 0.0
                 self.logger.warning("Backdoor estimation logic requires causal_effect_estimator module.")
                 raise NotImplementedError("Causal effect estimation via backdoor not implemented yet.")
            else:
                 self.logger.warning(f"No valid backdoor adjustment set found for {intervention_node} -> {outcome_variable}")
                 # TODO: Implement other estimation strategies (frontdoor, IV, etc.)
                 return 0.0 # Placeholder: return neutral effect if cannot estimate

        except Exception as e:
            self.logger.error(f"Error during causal effect estimation: {e}")
            return 0.0


    def _select_next_intervention(
        self,
        acquisition_function_name: str,
        gp_model: Optional[SingleTaskGP],
        best_f: float,
        bounds: torch.Tensor,
        graph: CausalGraph,
        target_variable: str,
        current_data: Optional[pd.DataFrame],
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Selects the next intervention point using BoTorch.

        Args:
            acquisition_function_name: Name of the acquisition function (e.g., 'EI', 'CausalEI').
            gp_model: The trained Gaussian Process model.
            best_f: The best observed outcome so far.
            bounds: The search bounds for the intervention space (d-dimensional tensor).
            graph: The causal graph for estimating effects.
            target_variable: The target variable.
            current_data: Current data for causal effect estimation.
            **kwargs: Additional parameters (e.g., q for batch size).

        Returns:
            A (1 x d) tensor representing the next intervention point, or None.
        """
        self.logger.debug(f"Selecting next intervention point using {acquisition_function_name}")

        if gp_model is None:
            # Handle case with no model (e.g., initial random sampling)
            self.logger.warning("GP model not available. Performing random exploration.")
            # Return a random point within bounds
            # Assuming bounds is (2, d)
            next_point = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(bounds.shape[1], dtype=bounds.dtype)
            return next_point.unsqueeze(0) # Return as (1 x d) tensor

        # --- Define Acquisition Function --- #
        if acquisition_function_name.lower() == 'ei':
            acqf = ExpectedImprovement(model=gp_model, best_f=best_f)
        elif acquisition_function_name.lower() == 'causalei':
            # Instantiate the custom Causal EI function
            acqf = CausalExpectedImprovement(
                model=gp_model,
                graph=graph,
                causal_estimator=self._estimate_causal_effect, # Pass estimator method
                current_data=current_data,
                target_variable=target_variable,
                best_f=best_f,
                maximize=kwargs.get('maximize', True) # Assuming maximization
            )
        # TODO: Add other acquisition functions (UCB, CausalUCB, etc.)
        else:
            self.logger.error(f"Unsupported acquisition function: {acquisition_function_name}")
            # Fallback to standard EI?
            acqf = ExpectedImprovement(model=gp_model, best_f=best_f)

        # --- Optimize Acquisition Function --- #
        try:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=kwargs.get('q', 1), # Number of points to select (batch size)
                num_restarts=kwargs.get('num_restarts', 5),
                raw_samples=kwargs.get('raw_samples', 20), # Used for initialization heuristic
                options=kwargs.get('acqf_options', {"batch_limit": 5, "maxiter": 200}),
            )
            # Return the best candidate point found
            next_point = candidates.detach()
            self.logger.debug(f"Optimizer selected candidate point: {next_point.tolist()}")
            return next_point
        except Exception as e:
            self.logger.exception(f"Error optimizing acquisition function: {e}")
            return None # Indicate failure


# --- Custom Causal Acquisition Function --- #
# Needs to be defined outside the class or imported
class CausalExpectedImprovement(AcquisitionFunction):
    """ Custom EI acquisition function incorporating causal effects. """
    def __init__(self,
                 model: SingleTaskGP,
                 graph: CausalGraph,
                 causal_estimator: callable,
                 current_data: Optional[pd.DataFrame],
                 target_variable: str,
                 best_f: float,
                 maximize: bool = True):
        super().__init__(model)
        self.graph = graph
        self.causal_estimator = causal_estimator # Method like MetaCBO._estimate_causal_effect
        self.current_data = current_data
        self.target_variable = target_variable
        self.best_f = best_f
        self.maximize = maximize
        self.logger = logging.getLogger(__name__) # Get logger

    # Define _required_forward_args if needed by BoTorch internals
    # @property
    # def _required_forward_args(self) -> Set[str]:
    #     return {"X"}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Evaluate Causal EI on the candidate points X. """
        # X is (batch_shape) x q x d tensor
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        d = X.shape[-1]

        # 1. Standard EI calculation
        mean, sigma = self.model.posterior(X).mean_var
        sigma = sigma.sqrt().clamp_min(1e-9)
        u = (mean - self.best_f) / sigma if self.maximize else (self.best_f - mean) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        standard_ei = sigma * (updf + u * ucdf)

        # 2. Causal Adjustment Factor
        causal_factors = torch.ones_like(standard_ei) # Default to 1 (no adjustment)

        # Reshape X for iterating: typically (q, d) if batch_shape is empty
        X_reshaped = X.reshape(-1, d)

        # Estimate causal effect for each point in the batch
        # This is slow if batch size (q) is large! Requires vectorization if possible.
        estimated_effects = []
        for i in range(X_reshaped.shape[0]):
            point = X_reshaped[i]
            # Convert point back to (node, value) - Requires dim_to_node mapping
            # This part assumes single intervention per point - needs refinement
            # For now, just pass the raw point and let estimator handle it (or raise error)
            # We need a robust way to map a d-dim point back to a specific intervention
            intervention_node = None # Placeholder
            intervention_value = None # Placeholder
            try:
                # Need to map the point tensor to intervention details
                # This mapping logic is complex and depends on how search space is defined
                # Skipping the actual estimation call for now as it needs the mapping
                # effect = self.causal_estimator(
                #     graph=self.graph,
                #     intervention_node=intervention_node,
                #     intervention_value=intervention_value,
                #     outcome_variable=self.target_variable,
                #     current_data=self.current_data
                # )
                effect = 0.0 # Placeholder - assume no effect
            except NotImplementedError:
                self.logger.warning("Causal effect estimation is not implemented. Skipping causal adjustment.", exc_info=False)
                effect = 0.0 # Default to no effect if estimation not implemented
            except Exception as e:
                self.logger.error(f"Error estimating causal effect for point {point.tolist()}: {e}", exc_info=False)
                effect = 0.0 # Default to no effect on error
            estimated_effects.append(effect)

        # Convert effects to a tensor and reshape to match standard_ei
        effects_tensor = torch.tensor(estimated_effects, device=X.device, dtype=X.dtype).reshape(standard_ei.shape)

        # Define causal adjustment factor (example: scale by exp(effect))
        # Needs careful design based on desired causal influence
        # Simple example: boost EI if effect is positive (assuming maximization)
        causal_factors = torch.exp(effects_tensor * 0.1) # Small scaling factor

        # 3. Combine standard EI and causal factor
        causal_ei = standard_ei * causal_factors
        self.logger.debug(f"Calculated CausalEI: Min={causal_ei.min():.3f}, Max={causal_ei.max():.3f}")
        return causal_ei


# --- End Causal Acquisition Function --- #


    # --- Helper methods for Task 19.4 ---
    def get_meta_training_history(self) -> List[Dict[str, Any]]:
        """Returns the history of metrics recorded during meta-training."""
        return self._meta_training_history

    def plot_meta_training_progress(self, metric_key: str = 'meta_loss'):
        """Plots a specific metric from the meta-training history."""
        # Requires matplotlib or similar plotting library
        try:
            import matplotlib.pyplot as plt
            epochs = [m.get('epoch', i) for i, m in enumerate(self._meta_training_history)]
            values = [m.get(metric_key) for m in self._meta_training_history]
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                self.logger.warning(f"No valid data found for metric '{metric_key}' to plot.")
                return

            plt.figure()
            plt.plot(epochs, values, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel(metric_key.replace('_', ' ').title())
            plt.title(f"Meta-Training Progress ({metric_key})")
            plt.grid(True)
            plt.show()
        except ImportError:
            self.logger.error("matplotlib is required for plotting. Please install it.")
        except Exception as e:
            self.logger.error(f"Error plotting meta-training progress: {e}")

    def save_checkpoint(self, filepath: str):
        """Saves the current meta-model state (parameters, history)."""
        try:
            state = {
                'meta_parameters': self._meta_parameters,
                'is_meta_trained': self._is_meta_trained,
                'meta_training_history': self._meta_training_history,
                'best_meta_params': self._best_meta_params,
                'best_meta_performance': self._best_meta_performance,
                'random_seed': self.random_seed,
                # Save state dicts if available
                'maml_framework_state': self.maml_framework.state_dict()
                    if hasattr(self.maml_framework, 'state_dict') else None,
                'task_representation_model_state': self.task_representation_model.state_dict()
                    if hasattr(self.task_representation_model, 'state_dict') else None,
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(state, filepath)
            self.logger.info(f"MetaCBO checkpoint saved successfully to {filepath}")
        except Exception as e:
            self.logger.exception(f"Failed to save checkpoint to {filepath}: {e}")

    def load_checkpoint(self, filepath: str):
        """Loads the MetaCBO state from a checkpoint file."""
        self.logger.info(f"Loading checkpoint from {filepath}")
        if not os.path.exists(filepath):
            self.logger.error(f"Checkpoint file not found: {filepath}")
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        try:
            state = torch.load(filepath)

            self._meta_parameters = state.get('meta_parameters')
            self._is_meta_trained = state.get('is_meta_trained', False)
            self._meta_training_history = state.get('meta_training_history', [])
            self._best_meta_params = state.get('best_meta_params')
            self._best_meta_performance = state.get('best_meta_performance', -float('inf'))
            loaded_seed = state.get('random_seed')
            # Optionally restore seed or just log mismatch
            if self.random_seed is not None and self.random_seed != loaded_seed:
                 self.logger.warning(f"Loaded checkpoint has different random seed ({loaded_seed}) than current instance ({self.random_seed}).")
            # self.random_seed = loaded_seed # Uncomment to restore seed

            # Load state dicts if available and methods exist
            maml_state = state.get('maml_framework_state')
            if maml_state and hasattr(self.maml_framework, 'load_state_dict'):
                self.maml_framework.load_state_dict(maml_state)
                self.logger.info("Loaded MAML framework state.")

            task_repr_state = state.get('task_representation_model_state')
            if task_repr_state and hasattr(self.task_representation_model, 'load_state_dict'):
                self.task_representation_model.load_state_dict(task_repr_state)
                self.logger.info("Loaded Task Representation model state.")

            self.logger.info(f"MetaCBO checkpoint loaded successfully from {filepath}")

            # Important: Update _meta_parameters from loaded MAML state if applicable
            if hasattr(self.maml_framework, 'meta_parameters'):
                 self._meta_parameters = self.maml_framework.meta_parameters

        except Exception as e:
            self.logger.exception(f"Failed to load checkpoint from {filepath}: {e}")
            raise

    def evaluate_adaptation_performance(self, task: StructuralCausalModel, validation_data: Any, adapted_model: Any, **kwargs) -> Dict[str, float]:
        """Evaluates the performance of the adapted model on validation data."""
        # This should be similar to the main evaluate method but specifically for post-adaptation assessment
        # self.logger.warning("evaluate_adaptation_performance is not fully implemented.")
        # Placeholder - delegate to main evaluate or implement separately
        # return self.evaluate(task, validation_data, adapted_model, **kwargs)
        self.logger.info(f"Evaluating adaptation performance for task...") # Add task id if possible
        if adapted_model is None:
             self.logger.error("Cannot evaluate adaptation performance without an adapted model.")
             return {}

        # Reuse the main evaluate method, passing the adapted model explicitly
        try:
            # Pass a flag or use kwargs to indicate this is an adaptation evaluation if needed by evaluate()
            metrics = self.evaluate(task, validation_data, adapted_model=adapted_model, **kwargs)
            self.logger.info(f"Adaptation performance metrics: {metrics}")
            return metrics
        except NotImplementedError:
            self.logger.error("Cannot evaluate adaptation performance because the main evaluate method is not implemented.")
            return {}
        except Exception as e:
            self.logger.exception(f"Error during adaptation performance evaluation: {e}")
            return {}

    def visualize_task_graph(self, task_id: Any, show: bool = True, save_path: Optional[str] = None):
        """
        Visualizes the causal graph for a specific task.

        Args:
            task_id: Identifier of the task whose graph to visualize.
            show: Whether to display the plot.
            save_path: Optional path to save the visualization.
        """
        if task_id not in self._task_graphs:
             self.logger.warning(f"No graph found or validated for task {task_id}. Cannot visualize.")
             # Maybe try validating on the fly?
             # task = self._get_task_by_id(task_id) # Need a way to get task object
             # if task:
             #     graph = self._validate_and_store_graph(task)
             # else:
             #     return
             return

        graph = self._task_graphs[task_id]
        self.logger.info(f"Visualizing graph for task {task_id}")
        try:
            # Assuming plot_graph is available from utils or handler
            # plot_graph(graph, show=show, save_path=save_path, title=f"Causal Graph for Task {task_id}") # Commented out - Module not found
            self.logger.warning("Graph visualization requires visualization module.")
            raise NotImplementedError("Graph visualization not implemented yet.")
        except Exception as e:
            self.logger.error(f"Error during graph visualization for task {task_id}: {e}")

    # --- End Helper methods for Task 19.4 ---

    # --- Data Transformation Methods (Subtask 4.1) ---
    def _update_node_mappings(self, nodes: List[str]):
        """Updates the node-to-index mappings if they don't exist or nodes changed."""
        if self._node_to_idx is None or self._current_task_nodes != nodes:
            self._current_task_nodes = nodes
            self._node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            self._idx_to_node = {idx: node for node, idx in self._node_to_idx.items()}
            self.logger.debug(f"Updated node mappings for nodes: {nodes}")

    def _encode_intervention(self, intervention_dict: Dict[str, Any], nodes: List[str]) -> torch.Tensor:
        """
        Convert intervention dictionary to tensor representation using current node order.

        Args:
            intervention_dict: Dict mapping node names to intervention values.
                               Assumes single node intervention for now.
            nodes: The ordered list of nodes for the current task context.

        Returns:
            torch.Tensor: Shape (1, 2) where row is [node_idx, value].
                          Needs adaptation for multi-node interventions or different encoding.
        """
        self._update_node_mappings(nodes)

        if not intervention_dict:
            raise ValueError("Intervention dictionary cannot be empty for encoding.")
        if len(intervention_dict) > 1:
            # TODO: Extend encoding for multi-node interventions if needed.
            # Could concatenate multiple [idx, val] pairs or use a different format.
            self.logger.warning("Encoding only supports single-node interventions currently. Using the first item.")

        node, value = next(iter(intervention_dict.items()))

        if node not in self._node_to_idx:
            raise ValueError(f"Node '{node}' not found in current task nodes: {self._current_task_nodes}")
        node_idx = self._node_to_idx[node]

        # Simple encoding: [node_index, value]
        # Note: GP kernel needs to handle this mixed representation.
        intervention_tensor = torch.tensor([[float(node_idx), float(value)]], dtype=torch.float32)
        self.logger.debug(f"Encoded {intervention_dict} to tensor: {intervention_tensor}")
        return intervention_tensor

    def _decode_intervention(self, intervention_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Convert tensor representation back to intervention dictionary.

        Args:
            intervention_tensor: torch.Tensor of shape (1, 2) with [node_idx, value].

        Returns:
            Dict: Mapping node name to intervention value.
        """
        if self._idx_to_node is None:
            raise RuntimeError("Node mappings not initialized. Call _encode_intervention or _update_node_mappings first.")
        if intervention_tensor.shape != (1, 2):
            raise ValueError(f"Expected intervention tensor shape (1, 2), got {intervention_tensor.shape}")

        node_idx_float, value_float = intervention_tensor.squeeze().tolist()
        node_idx = int(node_idx_float)

        if node_idx not in self._idx_to_node:
            raise ValueError(f"Node index {node_idx} not found in current mappings.")
        node_name = self._idx_to_node[node_idx]

        decoded_dict = {node_name: value_float}
        self.logger.debug(f"Decoded tensor {intervention_tensor} to dict: {decoded_dict}")
        return decoded_dict

    # --- GP Model Fitting and Update Methods (Subtask 4.1) ---
    def _initialize_model(self, train_x: torch.Tensor, train_y: torch.Tensor) -> Tuple[SingleTaskGP, GaussianLikelihood]:
        """
        Initialize and fit a GP model with the given data.

        Args:
            train_x: torch.Tensor of shape (n_samples, n_features).
            train_y: torch.Tensor of shape (n_samples) or (n_samples, 1).

        Returns:
            tuple: (fitted_model, likelihood). Returns the *class* likelihood instance.
        """
        if train_x.numel() == 0 or train_y.numel() == 0:
            raise ValueError("Training data cannot be empty for model initialization.")
        if train_x.shape[0] != train_y.shape[0]:
             raise ValueError(f"Shape mismatch: train_x ({train_x.shape}) and train_y ({train_y.shape})")

        # Ensure inputs are properly shaped (n_samples, n_features) and (n_samples, 1)
        if train_x.dim() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_y.dim() == 1:
            train_y = train_y.unsqueeze(-1) # BoTorch models expect (n, 1) output dim

        # Initialize model using the class likelihood instance
        # Ensure kernel dimension matches input features
        input_dim = train_x.shape[-1]
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            likelihood=self.likelihood, # Use the class instance
            covar_module=self.kernel_factory(input_dim)
        )
        # Initialize MLL using the same likelihood instance
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit model hyperparameters
        try:
            self.logger.debug(f"Fitting GP model with {train_x.shape[0]} points...")
            fit_gpytorch_model(mll)
            self.logger.debug("GP model fitting complete.")
        except Exception as e:
            self.logger.error(f"GP model fitting failed: {e}")
            # Decide how to handle fitting failure (e.g., use priors, raise error)
            # For now, log and potentially proceed with priors
            # raise # Uncomment to propagate error

        return model, self.likelihood # Return model and the *class* likelihood

    def _update_model(self, task_id: Any, new_x: Optional[torch.Tensor] = None, new_y: Optional[torch.Tensor] = None) -> Optional[Tuple[SingleTaskGP, GaussianLikelihood]]:
        """
        Update the dataset for a task and refit the GP model.

        Args:
            task_id: Identifier for the task.
            new_x: New intervention tensor(s) to add (shape [m, d]). Optional.
            new_y: New outcome tensor(s) to add (shape [m]). Optional.

        Returns:
            tuple: Updated (model, likelihood) or (None, None) if no data.
        """
        # Get existing data or initialize empty tensors with correct dtype
        current_data = self.datasets.setdefault(task_id, {
            'train_x': torch.tensor([], dtype=torch.float32),
            'train_y': torch.tensor([], dtype=torch.float32)
        })
        train_x = current_data['train_x']
        train_y = current_data['train_y']

        # Add new data if provided
        if new_x is not None and new_y is not None:
            # Ensure consistent shapes and types
            if new_x.dim() == 1: new_x = new_x.unsqueeze(0) # Shape [1, d]
            if new_y.dim() == 0: new_y = new_y.unsqueeze(0) # Shape [1]
            new_y = new_y.unsqueeze(-1) # Ensure shape [m, 1] for SingleTaskGP

            if train_x.numel() > 0 and new_x.shape[-1] != train_x.shape[-1]:
                 raise ValueError(f"Feature dimension mismatch: existing {train_x.shape[-1]}, new {new_x.shape[-1]}")

            # Concatenate new data
            train_x = torch.cat([train_x, new_x.to(dtype=torch.float32)], dim=0)
            train_y = torch.cat([train_y, new_y.to(dtype=torch.float32)], dim=0)
            self.logger.debug(f"Added {new_x.shape[0]} points to dataset for task {task_id}. New size: {train_x.shape[0]}")

            # Update stored dataset
            self.datasets[task_id]['train_x'] = train_x
            self.datasets[task_id]['train_y'] = train_y

        # Skip model fitting if no data exists *at all*
        if train_x.numel() == 0:
            self.logger.warning(f"No training data for task {task_id}, cannot fit model.")
            self.models.pop(task_id, None) # Remove any potentially stale model
            return None, None

        # Re-initialize and fit model with the *entire* updated dataset
        try:
            model, likelihood = self._initialize_model(train_x, train_y)
            # Store the updated model
            self.models[task_id] = (model, likelihood)
            self.logger.debug(f"Updated and refit model for task {task_id}.")
            return model, likelihood
        except Exception as e:
            self.logger.error(f"Failed to update model for task {task_id}: {e}")
            # Keep old model if refitting fails? Or remove? Let's remove for now.
            self.models.pop(task_id, None)
            return None, None

    # --- Public Data Handling Methods (Subtask 4.1) ---
    def add_data_point(self, task_id: Any, intervention: Dict[str, Any], outcome: float, nodes: List[str]):
        """
        Add a single intervention-outcome pair and update the corresponding GP model.

        Args:
            task_id: Identifier for the task (e.g., index in TaskFamily).
            intervention: Dictionary mapping node name to intervention value.
            outcome: Observed outcome value (e.g., causal effect).
            nodes: List of node names for the specific task graph (needed for encoding).
        """
        try:
            x_tensor = self._encode_intervention(intervention, nodes) # Shape [1, 2]
            y_tensor = torch.tensor([[outcome]], dtype=torch.float32) # Shape [1, 1]

            # Update dataset and refit model
            self._update_model(task_id, new_x=x_tensor, new_y=y_tensor)
            self.logger.info(f"Added data point for task {task_id}: {intervention} -> {outcome:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to add data point for task {task_id}: {e}")

    def get_task_data(self, task_id: Any) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve the dataset for a specific task.

        Args:
            task_id: Identifier for the task.

        Returns:
            dict: {'train_x': tensor, 'train_y': tensor} or None if task not found.
        """
        return self.datasets.get(task_id)

    def get_task_model(self, task_id: Any) -> Optional[Tuple[SingleTaskGP, GaussianLikelihood]]:
        """
        Retrieve the fitted model and likelihood for a specific task.

        Args:
            task_id: Identifier for the task.

        Returns:
            tuple: (model, likelihood) or None if no model is fitted for the task.
        """
        return self.models.get(task_id)

    # --- Existing Methods ---
    def _validate_and_store_graph(self, task: StructuralCausalModel) -> Optional[CausalGraph]:
        """Validates the causal graph of a task and stores it."""
        # Use a proper task identifier if available on the SCM object
        task_id = getattr(task, 'task_id', id(task)) # Example identifier
        if task_id in self._task_graphs:
            self.logger.debug(f"Returning cached graph for task {task_id}")
            return self._task_graphs[task_id] # Return cached graph

        graph = None
        try:
            # --- Attempt to get graph directly from task --- #
            # Prioritize .graph attribute used in tests
            if hasattr(task, 'graph') and isinstance(task.graph, CausalGraph):
                self.logger.debug(f"Found .graph attribute of type CausalGraph for task {task_id}")
                graph = task.graph
            elif hasattr(task, 'get_causal_graph') and callable(task.get_causal_graph):
                self.logger.debug(f"Calling get_causal_graph() for task {task_id}")
                graph = task.get_causal_graph()
            elif isinstance(task, CausalGraph):
                self.logger.debug(f"Task object itself is a CausalGraph for task {task_id}")
                graph = task # If the task object itself is the graph
            else:
                 # Fallback/Placeholder: Try creating from adjacency if possible?
                 self.logger.warning(f"Cannot directly extract CausalGraph from task {task_id} via .graph or get_causal_graph(). Trying adjacency matrix.")
                 if hasattr(task, 'adjacency_matrix'):
                     adj = task.adjacency_matrix()
                     # Assuming CausalGraph has a class method for this
                     if hasattr(CausalGraph, 'from_adjacency_matrix') and callable(CausalGraph.from_adjacency_matrix):
                         graph = CausalGraph.from_adjacency_matrix(adj)
                         self.logger.info(f"Inferred CausalGraph from adjacency matrix for task {task_id}.")
                     else:
                         self.logger.error(f"Task {task_id} has adjacency_matrix but CausalGraph lacks from_adjacency_matrix method.")
                 else:
                      self.logger.error(f"Cannot obtain CausalGraph from task {task_id}. No suitable method or attribute found.")
                      return None # Explicitly return None if no graph found
            # --- End Graph Retrieval --- #

            # --- Validate the obtained graph --- #
            if graph is None:
                 self.logger.error(f"Graph retrieval failed for task {task_id}. Cannot validate.")
                 return None

            if not isinstance(graph, CausalGraph):
                 self.logger.error(f"Retrieved object for task {task_id} is not a CausalGraph instance (Type: {type(graph)}). Validation failed.")
                 return None

            # Perform validation checks (e.g., acyclicity)
            if not graph.is_dag(): # Assuming CausalGraph inherits or implements is_dag
                 self.logger.warning(f"Graph for task {task_id} is not a DAG. Invalid causal structure.")
                 return None

            # --- Store and return --- #
            self.logger.debug(f"Successfully validated graph for task {task_id}")
            self._task_graphs[task_id] = graph
            return graph

        except Exception as e:
            self.logger.exception(f"Unexpected exception during graph validation for task {task_id}: {e}") # Use .exception for traceback
            return None

    # --- End Data Transformation Methods (Subtask 4.1) ---

    # --- GP Model Fitting and Update Methods (Subtask 4.1) ---
    def _initialize_model(self, train_x: torch.Tensor, train_y: torch.Tensor) -> Tuple[SingleTaskGP, GaussianLikelihood]:
        """
        Initialize and fit a GP model with the given data.

        Args:
            train_x: torch.Tensor of shape (n_samples, n_features).
            train_y: torch.Tensor of shape (n_samples) or (n_samples, 1).

        Returns:
            tuple: (fitted_model, likelihood). Returns the *class* likelihood instance.
        """
        if train_x.numel() == 0 or train_y.numel() == 0:
            raise ValueError("Training data cannot be empty for model initialization.")
        if train_x.shape[0] != train_y.shape[0]:
             raise ValueError(f"Shape mismatch: train_x ({train_x.shape}) and train_y ({train_y.shape})")

        # Ensure inputs are properly shaped (n_samples, n_features) and (n_samples, 1)
        if train_x.dim() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_y.dim() == 1:
            train_y = train_y.unsqueeze(-1) # BoTorch models expect (n, 1) output dim

        # Initialize model using the class likelihood instance
        # Ensure kernel dimension matches input features
        input_dim = train_x.shape[-1]
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            likelihood=self.likelihood, # Use the class instance
            covar_module=self.kernel_factory(input_dim)
        )
        # Initialize MLL using the same likelihood instance
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit model hyperparameters
        try:
            self.logger.debug(f"Fitting GP model with {train_x.shape[0]} points...")
            fit_gpytorch_model(mll)
            self.logger.debug("GP model fitting complete.")
        except Exception as e:
            self.logger.error(f"GP model fitting failed: {e}")
            # Decide how to handle fitting failure (e.g., use priors, raise error)
            # For now, log and potentially proceed with priors
            # raise # Uncomment to propagate error

        return model, self.likelihood # Return model and the *class* likelihood

    def _update_model(self, task_id: Any, new_x: Optional[torch.Tensor] = None, new_y: Optional[torch.Tensor] = None) -> Optional[Tuple[SingleTaskGP, GaussianLikelihood]]:
        """
        Update the dataset for a task and refit the GP model.

        Args:
            task_id: Identifier for the task.
            new_x: New intervention tensor(s) to add (shape [m, d]). Optional.
            new_y: New outcome tensor(s) to add (shape [m]). Optional.

        Returns:
            tuple: Updated (model, likelihood) or (None, None) if no data.
        """
        # Get existing data or initialize empty tensors with correct dtype
        current_data = self.datasets.setdefault(task_id, {
            'train_x': torch.tensor([], dtype=torch.float32),
            'train_y': torch.tensor([], dtype=torch.float32)
        })
        train_x = current_data['train_x']
        train_y = current_data['train_y']

        # Add new data if provided
        if new_x is not None and new_y is not None:
            # Ensure consistent shapes and types
            if new_x.dim() == 1: new_x = new_x.unsqueeze(0) # Shape [1, d]
            if new_y.dim() == 0: new_y = new_y.unsqueeze(0) # Shape [1]
            new_y = new_y.unsqueeze(-1) # Ensure shape [m, 1] for SingleTaskGP

            if train_x.numel() > 0 and new_x.shape[-1] != train_x.shape[-1]:
                 raise ValueError(f"Feature dimension mismatch: existing {train_x.shape[-1]}, new {new_x.shape[-1]}")

            # Concatenate new data
            train_x = torch.cat([train_x, new_x.to(dtype=torch.float32)], dim=0)
            train_y = torch.cat([train_y, new_y.to(dtype=torch.float32)], dim=0)
            self.logger.debug(f"Added {new_x.shape[0]} points to dataset for task {task_id}. New size: {train_x.shape[0]}")

            # Update stored dataset
            self.datasets[task_id]['train_x'] = train_x
            self.datasets[task_id]['train_y'] = train_y

        # Skip model fitting if no data exists *at all*
        if train_x.numel() == 0:
            self.logger.warning(f"No training data for task {task_id}, cannot fit model.")
            self.models.pop(task_id, None) # Remove any potentially stale model
            return None, None

        # Re-initialize and fit model with the *entire* updated dataset
        try:
            model, likelihood = self._initialize_model(train_x, train_y)
            # Store the updated model
            self.models[task_id] = (model, likelihood)
            self.logger.debug(f"Updated and refit model for task {task_id}.")
            return model, likelihood
        except Exception as e:
            self.logger.error(f"Failed to update model for task {task_id}: {e}")
            # Keep old model if refitting fails? Or remove? Let's remove for now.
            self.models.pop(task_id, None)
            return None, None

    # --- Public Data Handling Methods (Subtask 4.1) ---
    def add_data_point(self, task_id: Any, intervention: Dict[str, Any], outcome: float, nodes: List[str]):
        """
        Add a single intervention-outcome pair and update the corresponding GP model.

        Args:
            task_id: Identifier for the task (e.g., index in TaskFamily).
            intervention: Dictionary mapping node name to intervention value.
            outcome: Observed outcome value (e.g., causal effect).
            nodes: List of node names for the specific task graph (needed for encoding).
        """
        try:
            x_tensor = self._encode_intervention(intervention, nodes) # Shape [1, 2]
            y_tensor = torch.tensor([[outcome]], dtype=torch.float32) # Shape [1, 1]

            # Update dataset and refit model
            self._update_model(task_id, new_x=x_tensor, new_y=y_tensor)
            self.logger.info(f"Added data point for task {task_id}: {intervention} -> {outcome:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to add data point for task {task_id}: {e}")

    def get_task_data(self, task_id: Any) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve the dataset for a specific task.

        Args:
            task_id: Identifier for the task.

        Returns:
            dict: {'train_x': tensor, 'train_y': tensor} or None if task not found.
        """
        return self.datasets.get(task_id)

    def get_task_model(self, task_id: Any) -> Optional[Tuple[SingleTaskGP, GaussianLikelihood]]:
        """
        Retrieve the fitted model and likelihood for a specific task.

        Args:
            task_id: Identifier for the task.

        Returns:
            tuple: (model, likelihood) or None if no model is fitted for the task.
        """
        return self.models.get(task_id)

    # --- End GP Model Fitting and Update Methods (Subtask 4.1) ---

    # --- End Helper methods for Task 19.4 --- 