"""
Progressive intervention loop for causal graph structure learning.

This module provides components for iteratively selecting interventions
to improve causal graph structure learning.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

from causal_meta.structure_learning.config import ProgressiveInterventionConfig
from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.structure_learning.graph_structure_acquisition import GraphStructureAcquisition
from causal_meta.structure_learning.training import (
    SimpleGraphLearnerTrainer, evaluate_graph, calculate_structural_hamming_distance
)
from causal_meta.structure_learning.data_processing import (
    normalize_data, convert_to_tensor
)
from causal_meta.structure_learning.data_utils import (
    generate_observational_data, generate_interventional_data, create_intervention_mask
)
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_edge_probability_distribution,
    plot_threshold_sensitivity,
    compare_intervention_strategies
)


# Re-export components
from causal_meta.structure_learning.config import ProgressiveInterventionConfig
from causal_meta.structure_learning.graph_structure_acquisition import GraphStructureAcquisition


class ProgressiveInterventionLoop:
    """
    Progressive intervention loop for causal graph structure learning.
    
    This class implements an iterative process for improving causal graph
    structure learning through strategically selected interventions. The
    process involves training a model on observational data, selecting
    informative interventions, collecting new data, and updating the model.
    
    Args:
        config: Configuration for the experiment
        scm: Structural causal model for generating data
        obs_data: Initial observational data
        true_adj_matrix: True adjacency matrix (for evaluation)
        device: Device to use for computation
    """
    
    def __init__(
        self,
        config: ProgressiveInterventionConfig,
        scm: Any,  # Changed from LinearSCM to Any
        obs_data: np.ndarray,
        true_adj_matrix: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the progressive intervention loop."""
        self.config = config
        self.scm = scm
        self.obs_data = obs_data
        self.true_adj_matrix = true_adj_matrix
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state
        self.iteration = 0
        self.all_data = obs_data.copy()
        self.int_data = None
        self.int_mask = None
        self.model = None
        self.trainer = None
        self.acquisition = None
        self.results = []
        
        # Initialize acquisition strategy
        self.acquisition = GraphStructureAcquisition(
            strategy_type=config.acquisition_strategy,
            intervention_values=config.int_values
        )
        
        # Initialize output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = config.experiment_name or "progressive_intervention"
        self.output_dir = os.path.join(config.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Normalize data
        self.obs_data_norm, self.scaler = normalize_data(obs_data)
        self.all_data_norm = self.obs_data_norm.copy()
        
        # Convert to tensors
        self.obs_tensor = convert_to_tensor(self.obs_data_norm)
        self.all_tensor = self.obs_tensor.clone()  # Initially same as obs_tensor
    
    def initialize_model(self) -> SimpleGraphLearner:
        """
        Initialize the graph learning model.
        
        Returns:
            Initialized SimpleGraphLearner model
        """
        # Create model with anti-bias features
        model = SimpleGraphLearner(
            input_dim=self.config.num_nodes,
            hidden_dim=self.config.hidden_dim,
            num_layers=getattr(self.config, 'num_layers', 2),
            dropout=getattr(self.config, 'dropout', 0.1),
            sparsity_weight=0.01,  # Reduce this from the default 0.1
            acyclicity_weight=getattr(self.config, 'acyclicity_weight', 1.0),
            pos_weight=getattr(self.config, 'pos_weight', 5.0),
            consistency_weight=getattr(self.config, 'consistency_weight', 0.1),
            edge_prob_bias=0.7,  # Increase this to encourage edge prediction
            expected_density=getattr(self.config, 'expected_density', 0.3),
            density_weight=getattr(self.config, 'density_weight', 0.1),
            intervention_weight=50.0
        )
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def initialize_trainer(self, model: SimpleGraphLearner) -> SimpleGraphLearnerTrainer:
        """
        Initialize the model trainer.
        
        Args:
            model: Graph learning model
            
        Returns:
            SimpleGraphLearnerTrainer instance
        """
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=model,
            lr=getattr(self.config, 'learning_rate', 0.001),  # Default to 0.001 if not present
            weight_decay=getattr(self.config, 'weight_decay', 0.0),  # Default to 0.0 if not present
            device=self.device
        )
        
        return trainer
    
    def train_model(
        self,
        data: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model without supervision from true adjacency matrix.
        """
        # Create model if not exists
        if self.model is None:
            self.model = self.initialize_model()
        
        # Create trainer if not exists
        if self.trainer is None:
            self.trainer = self.initialize_trainer(self.model)
        
        # Get training parameters
        batch_size = getattr(self.config, 'batch_size', 32)
        epochs = getattr(self.config, 'epochs', 100)
        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)
        
        # Train the model without true adjacency matrix
        train_history = self.trainer.train(
            train_data=data,
            val_data=None,
            train_intervention_mask=intervention_mask,
            val_intervention_mask=None,
            true_adj=None,  # No more supervision!
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            normalize=False,
            verbose=True,
            checkpoint_dir=checkpoint_path
        )
        
        return train_history 
        

    def evaluate_model(self, data, intervention_mask=None):
        """Evaluate with different thresholds and debugging."""
        metrics = {}
        
        # Normal evaluation with different thresholds
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            threshold_metrics = self.trainer.evaluate(
                data=data,
                intervention_mask=intervention_mask,
                true_adj=self.true_adj_matrix,
                pre_intervention_data=self.pre_int_tensor if hasattr(self, 'pre_int_tensor') else None,
                post_intervention_data=self.post_int_tensor if hasattr(self, 'post_int_tensor') else None,
                threshold=threshold
            )
            # Add threshold to metric names
            metrics.update({f"{k}_t{threshold}": v for k, v in threshold_metrics.items()})
        
        # Add debug evaluation
        debug_metrics = self.trainer.evaluate_with_debug(
            data=data,
            intervention_mask=intervention_mask,
            true_adj=self.true_adj_matrix,
            pre_intervention_data=self.pre_int_tensor if hasattr(self, 'pre_int_tensor') else None,
            post_intervention_data=self.post_int_tensor if hasattr(self, 'post_int_tensor') else None,
            threshold=0.3
        )
        
        # Add debug metrics
        metrics.update(debug_metrics)
        
        return metrics

    def select_intervention(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Select the next intervention using the acquisition strategy.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dictionary specifying the selected intervention
        """
        # Ensure model exists
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Select intervention
        intervention = self.acquisition.select_intervention(
            model=self.model,
            data=data,
            budget=self.config.int_budget
        )
        
        return intervention
    
    def perform_intervention(
        self,
        intervention: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the specified intervention and generate data.
        
        Args:
            intervention: Dictionary specifying the intervention
            
        Returns:
            Tuple of (intervention data, intervention mask)
        """
        # Extract intervention details
        target_node = intervention['target_node']
        value = intervention['value']
        
        # Get the actual variable names from the SCM
        scm_variables = self.scm.get_variable_names()
        
        # Map the integer target_node to the corresponding SCM variable name
        if isinstance(target_node, int) and target_node < len(scm_variables):
            # Note: In a real SCM, the variable name might not simply be f"x{target_node}"
            # We need to find the actual mapping
            
            # Get the mapping of SCM variable names to their column indices in data
            if hasattr(self, 'column_mapping'):
                # Use existing mapping if already computed
                node_name = self.column_mapping.get(target_node)
            else:
                # Create a mapping of integer indices to variable names
                # This assumes the first data we received has columns in the same order as scm_variables
                self.column_mapping = {}
                for i, name in enumerate(scm_variables):
                    self.column_mapping[i] = name
                
                node_name = self.column_mapping.get(target_node)
            
            if node_name is None:
                # Fallback to direct name matching
                if f"x{target_node}" in scm_variables:
                    node_name = f"x{target_node}"
                else:
                    # As a last resort, just use the scm_variables at the same index
                    # This might not be correct if the order doesn't match
                    node_name = scm_variables[target_node] if target_node < len(scm_variables) else str(target_node)
        else:
            # If target_node is already a string or not a valid index, use it directly
            node_name = str(target_node)
        
        # Debug info
        print(f"Performing intervention on node {target_node} (name: {node_name}) with value {value}")
        print(f"Available variables: {scm_variables}")
        
        # Generate interventional data
        pre_int_data = generate_observational_data(
            scm=self.scm,
            n_samples=self.config.num_int_samples,
            as_tensor=False
        )
        
        # Generate post-intervention data
        post_int_data = generate_interventional_data(
            scm=self.scm,
            node=node_name,
            value=value,
            n_samples=self.config.num_int_samples,
            as_tensor=False
        )
        
        # Create intervention mask
        int_mask = np.zeros((self.config.num_int_samples, self.config.num_nodes))
        int_mask[:, target_node] = 1
        
        return pre_int_data, post_int_data, int_mask
    
    def update_data(
        self,
        pre_int_data: Union[pd.DataFrame, np.ndarray],
        post_int_data: Union[pd.DataFrame, np.ndarray],
        int_mask: np.ndarray
    ) -> None:
        """
        Update the dataset with new interventional data.
        
        Args:
            pre_int_data: Pre-intervention data
            post_int_data: Post-intervention data
            int_mask: Intervention mask
        """
        # Ensure pre_int_data and post_int_data are DataFrames with proper column names
        if not isinstance(pre_int_data, pd.DataFrame):
            # If it's a numpy array, convert to DataFrame with proper column names
            if hasattr(self.scm, 'get_variable_names'):
                var_names = self.scm.get_variable_names()
            elif isinstance(self.obs_data, pd.DataFrame):
                var_names = list(self.obs_data.columns)
            else:
                var_names = [f'x{i}' for i in range(pre_int_data.shape[1])]
            
            pre_int_data = pd.DataFrame(pre_int_data, columns=var_names)
            post_int_data = pd.DataFrame(post_int_data, columns=var_names)
        
        # Print debug info
        print(f"Pre-intervention data columns: {pre_int_data.columns}")
        print(f"Post-intervention data columns: {post_int_data.columns}")
        print(f"Observational data columns: {self.obs_data.columns if isinstance(self.obs_data, pd.DataFrame) else 'not a DataFrame'}")
        
        
        # Ensure consistent column order (add this before normalization)
        if isinstance(self.obs_data, pd.DataFrame) and isinstance(post_int_data, pd.DataFrame):
            # Get the expected column order from observational data
            expected_columns = list(self.obs_data.columns)
        
             # Reorder the columns in pre_int_data and post_int_data
            pre_int_data = pre_int_data[expected_columns]
            post_int_data = post_int_data[expected_columns]
            
            print(f"Reordered columns to match: {expected_columns}")
            
        # Apply normalization
        pre_int_data_norm = normalize_data(pre_int_data, self.scaler)
        post_int_data_norm = normalize_data(post_int_data, self.scaler)
        
        # Initialize new intervention data structures if this is the first intervention
        if not hasattr(self, 'pre_int_data'):
            self.pre_int_data = pre_int_data
            self.pre_int_data_norm = pre_int_data_norm
            self.post_int_data = post_int_data
            self.post_int_data_norm = post_int_data_norm
            self.int_mask = int_mask
            
            # Initialize the original int_data attributes for compatibility
            self.int_data = post_int_data
            self.int_data_norm = post_int_data_norm
        else:
            # Append to existing data
            self.pre_int_data = pd.concat([self.pre_int_data, pre_int_data], ignore_index=True)
            self.pre_int_data_norm = pd.concat([self.pre_int_data_norm, pre_int_data_norm], ignore_index=True)
            self.post_int_data = pd.concat([self.post_int_data, post_int_data], ignore_index=True)
            self.post_int_data_norm = pd.concat([self.post_int_data_norm, post_int_data_norm], ignore_index=True)
            
            # Update int_mask
            self.int_mask = np.concatenate([self.int_mask, int_mask], axis=0)
            
            # Update original int_data attributes for compatibility
            self.int_data = pd.concat([self.int_data, post_int_data], ignore_index=True)
            self.int_data_norm = pd.concat([self.int_data_norm, post_int_data_norm], ignore_index=True)
        
        # Update tensors
        self.pre_int_tensor = torch.tensor(self.pre_int_data_norm.values, dtype=torch.float32)
        self.post_int_tensor = torch.tensor(self.post_int_data_norm.values, dtype=torch.float32)
        self.int_tensor = torch.tensor(self.int_data_norm.values, dtype=torch.float32)
        self.int_mask_tensor = torch.tensor(self.int_mask, dtype=torch.float32)
        
        # Update all data
        if isinstance(self.obs_data, pd.DataFrame):
            self.all_data = pd.concat([self.obs_data, self.post_int_data], ignore_index=True)
        else:
            obs_data_df = pd.DataFrame(self.obs_data, columns=post_int_data.columns)
            self.all_data = pd.concat([obs_data_df, self.post_int_data], ignore_index=True)
        
        # Similar handling for normalized data
        if isinstance(self.obs_data_norm, pd.DataFrame):
            self.all_data_norm = pd.concat([self.obs_data_norm, self.post_int_data_norm], ignore_index=True)
        else:
            obs_data_norm_df = pd.DataFrame(self.obs_data_norm, columns=post_int_data_norm.columns)
            self.all_data_norm = pd.concat([obs_data_norm_df, self.post_int_data_norm], ignore_index=True)
        
        # Update all_tensor
        self.all_tensor = torch.tensor(self.all_data_norm.values, dtype=torch.float32)    

    def run_iteration(self) -> Dict[str, Any]:
        """
        Run a single iteration of the intervention loop with intervention-based learning.
        """
        # Increment iteration counter
        self.iteration += 1
        
        # 1. Train the model on current data
        checkpoint_path = os.path.join(self.output_dir, f"model_iter_{self.iteration}.pt")
        
        # If we have interventional data, include it in training
        if hasattr(self, 'pre_int_tensor') and hasattr(self, 'post_int_tensor'):
            print(f"\nTraining with {len(self.pre_int_tensor)} intervention pairs")
            print(f"Learning causal structure through intervention outcomes")
            
            train_history = self.trainer.train(
                train_data=self.all_tensor,
                val_data=None,
                train_intervention_mask=self.int_mask_tensor,
                val_intervention_mask=None,
                true_adj=None,  # No more supervised learning!
                pre_intervention_data=self.pre_int_tensor,
                post_intervention_data=self.post_int_tensor,
                batch_size=getattr(self.config, 'batch_size', 32),
                epochs=getattr(self.config, 'epochs', 100),
                early_stopping_patience=getattr(self.config, 'early_stopping_patience', 10),
                normalize=False,  # Data is already normalized
                verbose=True,
                checkpoint_dir=checkpoint_path if self.config.save_checkpoints else None
            )
        else:
            # First iteration - train on observational data only with regularization
            print("\nNo intervention data yet - using regularization only")
            train_history = self.train_model(
                data=self.obs_tensor,
                checkpoint_path=checkpoint_path if self.config.save_checkpoints else None
            )
        
        # 2. Evaluate the model (using true_adj only for evaluation)
        metrics = self.evaluate_model(self.all_tensor)

        # Add our special forced evaluation for debugging
        if hasattr(self.trainer, 'evaluate_with_forced_edges'):
            print("Running evaluation with forced edges...")
            forced_metrics = self.trainer.evaluate_with_forced_edges(
                data=self.all_tensor,
                intervention_mask=None,
                true_adj=self.true_adj_matrix,
                threshold=0.3
            )
            # Add these metrics with a prefix
            metrics.update({f"forced_{k}": v for k, v in forced_metrics.items()})
        
        
        # 3. Select an intervention
        intervention = self.select_intervention(self.all_tensor)
        
        # 4. Perform the intervention and generate data
        pre_int_data, post_int_data, int_mask = self.perform_intervention(intervention)
        
        # 5. Update the dataset
        self.update_data(pre_int_data, post_int_data, int_mask)
        
        # Store results
        result = {
            'iteration': self.iteration,
            'model': self.model,
            'metrics': metrics,
            'intervention': intervention,
            'train_history': train_history
        }
        
        self.results.append(result)
        
        return result
    
    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run the full experiment with multiple iterations.
        
        This method runs the initial evaluation on observational data,
        followed by multiple iterations of intervention loop.
        
        Returns:
            List of dictionaries with results from each iteration
        """
        print(f"Starting progressive intervention experiment with {self.config.num_iterations} iterations")
        
        # Initialize model
        self.model = self.initialize_model()
        self.trainer = self.initialize_trainer(self.model)
        
        # Initial training on observational data
        print("Initial training on observational data...")
        checkpoint_path = os.path.join(self.output_dir, "model_initial.pt")
        train_history = self.train_model(
            data=self.obs_tensor,
            checkpoint_path=checkpoint_path if self.config.save_checkpoints else None
        )
        
        # Initial evaluation
        metrics = self.evaluate_model(self.obs_tensor)
        
        # Store initial results
        initial_result = {
            'iteration': 0,
            'model': self.model,
            'metrics': metrics,
            'train_history': train_history
        }
        self.results = [initial_result]
        
        # Run iterations
        for i in range(self.config.num_iterations):
            print(f"\nIteration {i+1}/{self.config.num_iterations}")
            result = self.run_iteration()
            
            # Print metrics
            print(f"Iteration {i+1} metrics:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value:.4f}")
        
        # Save final results
        self.save_results()
        
        return self.results
    
    def plot_graph_comparison(self) -> None:
        """
        Plot comparison between true and learned graphs.
        
        This method creates a visualization comparing the true graph
        with the learned graph from the final iteration.
        """
        if self.model is None or self.true_adj_matrix is None:
            return
        
        # Get the final learned graph
        with torch.no_grad():
            edge_probs = self.model(self.all_tensor)
            learned_adj = self.model.threshold_edge_probabilities(edge_probs, threshold=self.config.threshold).detach().cpu().numpy()
        
        # Create the basic plot (existing functionality)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot true graph
        ax = axes[0]
        ax.imshow(self.true_adj_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('True Graph')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        
        # Add grid
        for i in range(self.config.num_nodes + 1):
            ax.axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        
        # Plot learned graph
        ax = axes[1]
        ax.imshow(learned_adj, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Learned Graph (Iteration {self.iteration})')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        
        # Add grid
        for i in range(self.config.num_nodes + 1):
            ax.axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        
        # Add metrics as text
        shd = calculate_structural_hamming_distance(learned_adj, self.true_adj_matrix)
        metrics = evaluate_graph(learned_adj, self.true_adj_matrix)
        
        metrics_text = f"SHD: {shd}\n"
        metrics_text += f"Accuracy: {metrics['accuracy']:.4f}\n"
        if 'precision' in metrics:
            metrics_text += f"Precision: {metrics['precision']:.4f}\n"
        if 'recall' in metrics:
            metrics_text += f"Recall: {metrics['recall']:.4f}\n"
        if 'f1' in metrics:
            metrics_text += f"F1: {metrics['f1']:.4f}"
        
        fig.text(0.5, 0.01, metrics_text, ha='center', va='center', fontsize=10)
        
        # Save the plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(self.output_dir, "graph_comparison.png"))
        plt.close()
        
        # Create enhanced visualization with edge probabilities
        edge_probs_np = edge_probs.detach().cpu().numpy()
        plot_edge_probabilities(
            true_adj=self.true_adj_matrix,
            edge_probs=edge_probs_np,
            thresholded_adj=learned_adj,
            threshold=self.config.threshold,
            save_path=os.path.join(self.output_dir, "edge_probability_comparison.png")
        )
        
        # Plot edge probability histogram
        plot_edge_probability_histogram(
            edge_probs=edge_probs_np,
            threshold=self.config.threshold,
            save_path=os.path.join(self.output_dir, "edge_probability_histogram.png")
        )
        
        # Plot edge probability distribution
        plot_edge_probability_distribution(
            true_adj=self.true_adj_matrix,
            edge_probs=edge_probs_np,
            save_path=os.path.join(self.output_dir, "edge_probability_distribution.png")
        )
    
    def analyze_threshold_sensitivity(self) -> None:
        """
        Analyze sensitivity of graph recovery to different thresholds.
        
        This method creates visualizations showing how different thresholds
        affect the graph recovery metrics.
        """
        if self.model is None or self.true_adj_matrix is None:
            return
        
        # Get edge probabilities
        with torch.no_grad():
            edge_probs = self.model(self.all_tensor)
            edge_probs_np = edge_probs.detach().cpu().numpy()
        
        # Create threshold sensitivity plot
        plot_threshold_sensitivity(
            true_adj=self.true_adj_matrix,
            edge_probs=edge_probs_np,
            save_path=os.path.join(self.output_dir, "threshold_sensitivity.png")
        )
    
    def save_results(self) -> None:
        """
        Save the experiment results.
        
        This method saves metrics, graph visualizations, and plots
        summarizing the experiment results.
        """
        print(f"Saving results to {self.output_dir}")
        
        # Save metrics
        metrics_df = pd.DataFrame([
            {
                'iteration': result['iteration'],
                **result['metrics']
            }
            for result in self.results
        ])
        metrics_df.to_csv(os.path.join(self.output_dir, "metrics.csv"), index=False)
        
        # Plot metrics over iterations
        self.plot_metrics()
        
        # Save final graph visualization
        if self.model is not None and self.true_adj_matrix is not None:
            self.plot_graph_comparison()
            
            # Add threshold sensitivity analysis
            self.analyze_threshold_sensitivity()
    
    def plot_metrics(self) -> None:
        """
        Plot metrics over iterations.
        
        This method creates plots showing how different metrics
        change over the course of the experiment.
        """
        metrics_to_plot = self.config.evaluation_metrics
        
        if not metrics_to_plot or len(self.results) < 2:
            return
        
        iterations = [result['iteration'] for result in self.results]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics_to_plot:
            if all(metric in result['metrics'] for result in self.results):
                values = [result['metrics'][metric] for result in self.results]
                ax.plot(iterations, values, marker='o', label=metric)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics Progression Over Iterations')
        ax.grid(True)
        ax.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_progression.png"))
        plt.close()
    
    def compare_intervention_strategies(
        self,
        random_results: List[Dict],
        strategic_results: List[Dict],
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare random and strategic intervention strategies.
        
        Args:
            random_results: Results from random intervention strategy
            strategic_results: Results from strategic intervention strategy
            save_path: Path to save the comparison plot
        """
        if not random_results or not strategic_results:
            return
        
        # Extract iterations
        max_iter = max(
            max(result['iteration'] for result in random_results),
            max(result['iteration'] for result in strategic_results)
        )
        iterations = list(range(max_iter + 1))
        
        # Extract metrics from both strategies
        random_metrics = {
            'accuracy': [0.0] * (max_iter + 1),
            'precision': [0.0] * (max_iter + 1),
            'recall': [0.0] * (max_iter + 1),
            'f1': [0.0] * (max_iter + 1),
            'shd': [0] * (max_iter + 1)
        }
        
        strategic_metrics = {
            'accuracy': [0.0] * (max_iter + 1),
            'precision': [0.0] * (max_iter + 1),
            'recall': [0.0] * (max_iter + 1),
            'f1': [0.0] * (max_iter + 1),
            'shd': [0] * (max_iter + 1)
        }
        
        # Fill in random metrics
        for result in random_results:
            iter_idx = result['iteration']
            for metric, value in result['metrics'].items():
                if metric in random_metrics:
                    random_metrics[metric][iter_idx] = value
        
        # Fill in strategic metrics
        for result in strategic_results:
            iter_idx = result['iteration']
            for metric, value in result['metrics'].items():
                if metric in strategic_metrics:
                    strategic_metrics[metric][iter_idx] = value
        
        # Create comparison plot
        if not save_path:
            save_path = os.path.join(self.output_dir, "intervention_strategy_comparison.png")
        
        compare_intervention_strategies(
            iterations=iterations,
            random_metrics=random_metrics,
            strategic_metrics=strategic_metrics,
            save_path=save_path
        ) 