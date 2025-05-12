"""
Benchmark Suite for Causal Discovery and Intervention Optimization.

This module provides benchmark tools for evaluating various causal discovery and
intervention optimization algorithms, including both neural and traditional approaches.
The benchmarks focus on:
1. Structural accuracy (graph recovery)
2. Intervention effectiveness
3. Computational efficiency
4. Scalability

The module integrates with the Component Registry to ensure proper reuse of existing
components and consistent interfaces across the codebase.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import os
import psutil  # Add psutil for memory profiling
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import json
from datetime import datetime
import signal

from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.meta_learning.graph_inference_utils import GraphMetrics, compute_shd, compute_precision_recall
from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
from causal_meta.meta_learning.amortized_cbo import AmortizedCBO

# Configure logging
logger = logging.getLogger(__name__)


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.
    
    This class defines the interface for all benchmark implementations and provides
    common utility methods for evaluating and reporting results.
    
    Args:
        name: Name of the benchmark
        output_dir: Directory to save benchmark results
        seed: Random seed for reproducibility
        device: Device to run benchmark on
    """
    
    def __init__(
        self,
        name: str,
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize the benchmark."""
        self.name = name
        self.output_dir = output_dir
        self.seed = seed
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # Create output directory
        self.benchmark_dir = os.path.join(output_dir, name)
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        # Initialize storage for models, baselines, and datasets
        self.models = {}
        self.baselines = {}
        self.datasets = {}
        self.results = None
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the benchmark, generating datasets and preparing evaluation.
        
        This method should be implemented by subclasses to generate the necessary
        data for benchmarking.
        """
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark and return results.
        
        This method should be implemented by subclasses to evaluate all models
        and baselines and return a structured results dictionary.
        
        Returns:
            Dictionary containing benchmark results
        """
        pass
    
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the benchmark.
        
        Args:
            name: Name of the model
            model: Model instance to evaluate
        """
        self.models[name] = model
    
    def add_baseline(self, name: str, baseline: Any) -> None:
        """
        Add a baseline method to the benchmark.
        
        Args:
            name: Name of the baseline
            baseline: Baseline method instance
        """
        self.baselines[name] = baseline
    
    def add_dataset(self, name: str, dataset: Any) -> None:
        """
        Add a dataset to the benchmark.
        
        Args:
            name: Name of the dataset
            dataset: Dataset to use for evaluation
        """
        self.datasets[name] = dataset
    
    def evaluate_structure_recovery(
        self,
        true_graph: CausalGraph,
        pred_graph: CausalGraph
    ) -> Dict[str, float]:
        """
        Evaluate graph structure recovery performance.
        
        Args:
            true_graph: Ground truth causal graph
            pred_graph: Predicted causal graph
            
        Returns:
            Dictionary of metrics including:
            - shd: Structural Hamming Distance
            - precision: Precision of edge recovery
            - recall: Recall of edge recovery
            - f1: F1 score
        """
        # Get adjacency matrices
        true_adj = true_graph.get_adjacency_matrix()
        pred_adj = pred_graph.get_adjacency_matrix()
        
        # Ensure same node ordering
        if len(true_graph.get_nodes()) != len(pred_graph.get_nodes()):
            raise ValueError("True and predicted graphs must have the same number of nodes")
        
        # Calculate metrics
        true_edges = set()
        pred_edges = set()
        
        for i in range(true_adj.shape[0]):
            for j in range(true_adj.shape[1]):
                if true_adj[i, j] > 0:
                    true_edges.add((i, j))
                if pred_adj[i, j] > 0:
                    pred_edges.add((i, j))
        
        # True positives, false positives, false negatives
        tp = len(true_edges.intersection(pred_edges))
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # SHD (Structural Hamming Distance)
        shd = fp + fn
        
        return {
            "shd": float(shd),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def evaluate_intervention_prediction(
        self,
        scm: StructuralCausalModel,
        model: Any,
        interventions: Dict[str, Any],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate intervention prediction performance.
        
        Args:
            scm: Ground truth structural causal model
            model: Model to evaluate
            interventions: Dictionary of interventions to evaluate
            num_samples: Number of samples to use for evaluation
            
        Returns:
            Dictionary of metrics including:
            - mse: Mean squared error
            - mae: Mean absolute error
            - r2: R-squared
        """
        # Generate true interventional data
        true_int_data = scm.sample_interventional_data(interventions, sample_size=num_samples)
        
        # Get predictions from model
        try:
            pred_int_data = None
            
            if hasattr(model, "predict_intervention_outcomes"):
                pred_int_data = model.predict_intervention_outcomes(
                    scm.sample_data(sample_size=num_samples),
                    interventions=interventions
                )
            elif hasattr(model, "sample_interventional_data"):
                pred_int_data = model.sample_interventional_data(
                    interventions=interventions,
                    sample_size=num_samples
                )
            else:
                logger.warning(f"Model doesn't have method to predict interventions")
                return {
                    "mse": float('nan'),
                    "mae": float('nan'),
                    "r2": float('nan'),
                    "error": "Model doesn't support intervention prediction"
                }
            
            # Convert to DataFrame if needed
            if not isinstance(pred_int_data, pd.DataFrame):
                # Assume it's a numpy array or tensor
                if isinstance(pred_int_data, torch.Tensor):
                    pred_int_data = pred_int_data.detach().cpu().numpy()
                
                # Ensure shape compatibility
                if pred_int_data.shape != true_int_data.values.shape:
                    # Try to reshape if dimensions are compatible
                    if pred_int_data.size == true_int_data.size:
                        pred_int_data = pred_int_data.reshape(true_int_data.values.shape)
                    else:
                        logger.warning(f"Shape mismatch: pred_shape={pred_int_data.shape}, true_shape={true_int_data.shape}")
                        return {
                            "mse": float('nan'),
                            "mae": float('nan'),
                            "r2": float('nan'),
                            "error": f"Shape mismatch: {pred_int_data.shape} vs {true_int_data.shape}"
                        }
                
                # Create DataFrame with same columns as true data
                pred_int_data = pd.DataFrame(pred_int_data, columns=true_int_data.columns)
            
            # Ensure columns match
            if not all(col in pred_int_data.columns for col in true_int_data.columns):
                missing_cols = [col for col in true_int_data.columns if col not in pred_int_data.columns]
                logger.warning(f"Missing columns in predictions: {missing_cols}")
                
                # Use only common columns
                common_cols = [col for col in true_int_data.columns if col in pred_int_data.columns]
                if not common_cols:
                    return {
                        "mse": float('nan'),
                        "mae": float('nan'),
                        "r2": float('nan'),
                        "error": "No common columns between predictions and ground truth"
                    }
                
                true_int_data = true_int_data[common_cols]
                pred_int_data = pred_int_data[common_cols]
            
            # Calculate metrics
            mse = ((true_int_data - pred_int_data) ** 2).mean().mean()
            mae = (true_int_data - pred_int_data).abs().mean().mean()
            
            # Calculate R-squared
            ss_tot = ((true_int_data - true_int_data.mean()) ** 2).sum().sum()
            ss_res = ((true_int_data - pred_int_data) ** 2).sum().sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            }
        except Exception as e:
            logger.error(f"Error evaluating intervention prediction: {str(e)}")
            return {
                "mse": float('nan'),
                "mae": float('nan'),
                "r2": float('nan'),
                "error": str(e)
            }
    
    def time_performance(
        self,
        callable_fn: Callable,
        repeat: int = 3,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Measure the performance of a callable in terms of execution time.
        
        Args:
            callable_fn: Function to time
            repeat: Number of times to repeat the measurement
            *args: Positional arguments to pass to callable_fn
            **kwargs: Keyword arguments to pass to callable_fn
            
        Returns:
            Dictionary with timing statistics and result:
            - mean: Mean execution time in seconds
            - std: Standard deviation of execution time
            - min: Minimum execution time
            - max: Maximum execution time
            - result: Result of the last function call
        """
        times = []
        result = None
        
        for _ in range(repeat):
            start_time = time.time()
            try:
                result = callable_fn(*args, **kwargs)
            except Exception as e:
                print(f"Error during timing: {str(e)}")
                continue
            end_time = time.time()
            times.append(end_time - start_time)
        
        if not times:
            return {
                "mean": float('nan'),
                "std": float('nan'),
                "min": float('nan'),
                "max": float('nan'),
                "error": "All timing attempts failed"
            }
        
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "result": result
        }
    
    def plot_results(
        self,
        metrics: Dict[str, Any],
        title: str = "Benchmark Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot benchmark results.
        
        Args:
            metrics: Dictionary of metrics to plot
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract model names and metrics
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.8 / len(metric_names)
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json") -> str:
        """
        Save benchmark results to a file.
        
        Args:
            results: Dictionary of results to save
            filename: Name of the file to save to
            
        Returns:
            Path to the saved file
        """
        # Process results to ensure JSON serializable
        def process_for_json(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return obj.tolist()
            elif isinstance(obj, (tuple, list)):
                return [process_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: process_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, 'to_dict'):
                return process_for_json(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)
        
        # Process results
        processed_results = process_for_json(results)
        
        # Add timestamp
        processed_results['timestamp'] = datetime.now().isoformat()
        
        # Save to file
        output_path = os.path.join(self.benchmark_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        return output_path
    
    def load_results(self, filename: str = "results.json") -> Dict[str, Any]:
        """
        Load benchmark results from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Dictionary of results
        """
        input_path = os.path.join(self.benchmark_dir, filename)
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        return results

class CausalDiscoveryBenchmark(Benchmark):
    """
    Benchmark for evaluating causal discovery methods.
    
    This benchmark evaluates how well different methods can recover the 
    structure of a causal graph from observational data.
    """
    
    def __init__(
        self,
        name: str = "causal_discovery_benchmark",
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        num_nodes: int = 10,
        num_graphs: int = 20,
        num_samples: int = 1000,
        graph_type: str = "random",
        edge_prob: float = 0.3
    ):
        """Initialize the causal discovery benchmark."""
        super().__init__(name, output_dir, seed, device)
        
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.num_samples = num_samples
        self.graph_type = graph_type
        self.edge_prob = edge_prob
        
        # Initialize graph factory
        self.graph_factory = GraphFactory()
        
        # Initialize metric trackers
        self.structure_metrics = {}
        self.runtime_metrics = {}
    
    def setup(self) -> None:
        """
        Set up the benchmark by generating synthetic graphs and data.
        """
        logger.info(f"Setting up benchmark: {self.name}")
        
        # Generate synthetic graphs and SCMs
        self.graphs = []
        self.scms = []
        self.datasets = {}
        
        for i in range(self.num_graphs):
            # Generate a random graph with named nodes
            graph = self.graph_factory.create_graph(
                graph_type=self.graph_type,
                num_nodes=self.num_nodes,
                edge_probability=self.edge_prob,
                seed=self.seed + i if self.seed is not None else None
            )
            
            # Ensure nodes are named x0, x1, etc. if not already
            node_mapping = {}
            for j, node in enumerate(graph.get_nodes()):
                if not isinstance(node, str) or not node.startswith('x'):
                    node_name = f"x{j}"
                    node_mapping[node] = node_name
            
            # If we need to rename nodes
            if node_mapping:
                new_graph = CausalGraph()
                # Add nodes
                for j in range(self.num_nodes):
                    new_graph.add_node(f"x{j}")
                
                # Add edges (ensuring lower index -> higher index for acyclicity)
                for u in graph.get_nodes():
                    for v in graph.get_children(u):
                        u_name = node_mapping.get(u, u)
                        v_name = node_mapping.get(v, v)
                        
                        # Extract indices to ensure acyclicity
                        if u_name.startswith('x') and v_name.startswith('x'):
                            u_idx = int(u_name[1:])
                            v_idx = int(v_name[1:])
                            
                            # Only add edge if u_idx < v_idx to ensure acyclicity
                            if u_idx < v_idx:
                                new_graph.add_edge(u_name, v_name)
                else:
                    # If no cycles are detected, just use as is
                    if not new_graph.has_cycle():
                        graph = new_graph
                    else:
                        # Create a simple chain as fallback
                        fallback_graph = CausalGraph()
                        for j in range(self.num_nodes):
                            fallback_graph.add_node(f"x{j}")
                        
                        # Add edges in a chain: x0->x1->x2->...
                        for j in range(self.num_nodes - 1):
                            fallback_graph.add_edge(f"x{j}", f"x{j+1}")
                        
                        graph = fallback_graph
            else:
                # Ensure the existing graph is acyclic
                if graph.has_cycle():
                    # Create a simple chain as fallback
                    fallback_graph = CausalGraph()
                    for j in range(self.num_nodes):
                        fallback_graph.add_node(f"x{j}")
                    
                    # Add edges in a chain: x0->x1->x2->...
                    for j in range(self.num_nodes - 1):
                        fallback_graph.add_edge(f"x{j}", f"x{j+1}")
                    
                    graph = fallback_graph
            
            self.graphs.append(graph)
            
            # Create an SCM from the graph
            scm = StructuralCausalModel(
                causal_graph=graph,
                random_state=self.seed + i if self.seed is not None else None
            )
            
            # Add variables to the SCM
            for node in graph.get_nodes():
                scm.add_variable(node)
            
            # Define linear Gaussian equations for all nodes
            for node in graph.get_nodes():
                parents = graph.get_parents(node)
                if parents:
                    # Create random coefficients for parents
                    coeffs = {parent: np.random.uniform(-1, 1) for parent in parents}
                    scm.define_linear_gaussian_equation(
                        node, coeffs, intercept=0, noise_std=0.1
                    )
                else:
                    # Exogenous nodes have no parents
                    scm.define_linear_gaussian_equation(
                        node, {}, intercept=0, noise_std=1.0
                    )
            
            self.scms.append(scm)
            
            # Generate observational data
            obs_data = scm.sample_data(sample_size=self.num_samples)
            
            # Generate some interventional data - intervene on 20% of nodes
            num_interventions = max(1, int(0.2 * self.num_nodes))
            intervention_indices = np.random.choice(
                self.num_nodes, 
                size=num_interventions, 
                replace=False
            )
            
            interventions = {}
            intervention_data = []
            
            for idx in intervention_indices:
                # Create a do-intervention
                target_name = f"x{idx}"
                intervention_value = np.random.normal(0, 1)
                interventions[target_name] = intervention_value
                
                # Sample from interventional distribution
                int_data = scm.sample_interventional_data(
                    interventions={target_name: intervention_value},
                    sample_size=self.num_samples // num_interventions
                )
                
                # Add intervention indicator
                int_data[f"do_{target_name}"] = 1
                intervention_data.append(int_data)
            
            # Combine all intervention datasets
            if intervention_data:
                int_data_combined = pd.concat(intervention_data, ignore_index=True)
                
                # Store datasets
                self.datasets[f"graph_{i}"] = {
                    "graph": graph,
                    "scm": scm,
                    "observational_data": obs_data,
                    "interventional_data": int_data_combined,
                    "interventions": interventions
                }
            else:
                # If no interventions (shouldn't happen with our setup)
                self.datasets[f"graph_{i}"] = {
                    "graph": graph,
                    "scm": scm,
                    "observational_data": obs_data,
                    "interventional_data": None,
                    "interventions": {}
                }
        
        logger.info(f"Setup complete: generated {len(self.graphs)} graphs with {self.num_samples} samples each")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark on all registered models and baselines.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running benchmark: {self.name}")
        
        # Initialize results storage
        self.results = {
            "benchmark_config": {
                "num_nodes": self.num_nodes,
                "num_graphs": self.num_graphs,
                "num_samples": self.num_samples,
                "graph_type": self.graph_type,
                "edge_prob": self.edge_prob
            },
            "models": {},
            "baselines": {}
        }
        
        # Run benchmark on each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = self._evaluate_method(model_name, model)
            self.results["models"][model_name] = model_results
        
        # Run benchmark on each baseline
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            baseline_results = self._evaluate_method(baseline_name, baseline)
            self.results["baselines"][baseline_name] = baseline_results
        
        # Aggregate results
        self._aggregate_results()
        
        # Save results
        results_path = self.save_results(self.results)
        logger.info(f"Results saved to: {results_path}")
        
        return self.results
    
    def _evaluate_method(self, method_name: str, method: Any) -> Dict[str, Any]:
        """
        Evaluate a single causal discovery method on all datasets.
        
        Args:
            method_name: Name of the method
            method: Method instance to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        method_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            # Extract dataset components
            graph = dataset["graph"]
            scm = dataset["scm"]
            obs_data = dataset["observational_data"]
            int_data = dataset["interventional_data"]
            
            # Track dataset results
            dataset_results = {}
            
            # Measure structure learning performance
            try:
                # Time the structure learning process
                timing_results = self.time_performance(
                    self._run_structure_learning,
                    3,  # repeat 3 times
                    method=method,
                    obs_data=obs_data,
                    int_data=int_data
                )
                
                # Extract learned graph
                pred_graph = timing_results["result"]
                
                # Calculate structural metrics
                if pred_graph is not None:
                    struct_metrics = self.evaluate_structure_recovery(graph, pred_graph)
                    dataset_results["structure_metrics"] = struct_metrics
                else:
                    dataset_results["structure_metrics"] = {"error": "Failed to learn graph"}
                
                # Add timing results
                dataset_results["timing"] = {
                    "min_time": timing_results["min"],
                    "max_time": timing_results["max"],
                    "avg_time": timing_results["mean"]
                }
                
                # Evaluate intervention prediction if applicable
                if hasattr(method, "predict_intervention_outcomes") or hasattr(method, "sample_interventional_data"):
                    # Choose a random intervention
                    target_idx = np.random.randint(0, self.num_nodes)
                    target_var = f"x{target_idx}"
                    intervention_value = 1.0  # Simple intervention value
                    
                    # Evaluate intervention prediction
                    int_metrics = self.evaluate_intervention_prediction(
                        scm=scm,
                        model=method,
                        interventions={target_var: intervention_value},
                        num_samples=100
                    )
                    
                    dataset_results["intervention_metrics"] = int_metrics
            
            except Exception as e:
                logger.error(f"Error evaluating {method_name} on {dataset_name}: {str(e)}")
                dataset_results["error"] = str(e)
            
            # Store results for this dataset
            method_results[dataset_name] = dataset_results
        
        return method_results
    
    def _run_structure_learning(self, method: Any, obs_data: pd.DataFrame, int_data: Optional[pd.DataFrame] = None) -> Optional[CausalGraph]:
        """
        Run structure learning using the provided method.
        
        This is a helper method that handles different method interfaces.
        
        Args:
            method: Structure learning method to evaluate
            obs_data: Observational data
            int_data: Interventional data (if available)
            
        Returns:
            Learned causal graph, or None if learning failed
        """
        try:
            # Check for different method interfaces
            if hasattr(method, "learn_graph"):
                # Standard interface
                if int_data is not None and "int_data" in method.learn_graph.__code__.co_varnames:
                    # Method accepts interventional data
                    return method.learn_graph(obs_data=obs_data, int_data=int_data)
                else:
                    # Method only accepts observational data
                    return method.learn_graph(obs_data=obs_data)
            
            elif hasattr(method, "fit") and hasattr(method, "predict_graph"):
                # Fit-predict interface
                method.fit(obs_data)
                return method.predict_graph()
            
            elif isinstance(method, AmortizedCausalDiscovery):
                # Handle AmortizedCausalDiscovery interface
                # Convert data to tensor format expected by the model
                X = torch.tensor(obs_data.values, dtype=torch.float32)
                X = X.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    graph_probs = method.infer_graph(X)
                
                # Convert to CausalGraph
                adj_matrix = (graph_probs > 0.5).cpu().numpy()
                
                # Create a new graph with named nodes
                graph = CausalGraph()
                
                # Add nodes with proper names
                node_names = obs_data.columns if len(obs_data.columns) == adj_matrix.shape[0] else [f"x{i}" for i in range(adj_matrix.shape[0])]
                for node_name in node_names:
                    graph.add_node(node_name)
                
                # Add edges based on adjacency matrix
                for i in range(adj_matrix.shape[0]):
                    for j in range(adj_matrix.shape[1]):
                        if adj_matrix[i, j] > 0:
                            graph.add_edge(node_names[i], node_names[j])
                
                return graph
            
            elif hasattr(method, "__call__"):
                # Callable interface
                result = method(obs_data)
                if isinstance(result, CausalGraph):
                    return result
                elif isinstance(result, np.ndarray) or isinstance(result, torch.Tensor):
                    # Assume adjacency matrix
                    if isinstance(result, torch.Tensor):
                        result = result.cpu().numpy()
                    
                    # Create a new graph with named nodes
                    graph = CausalGraph()
                    
                    # Add nodes with proper names
                    node_names = obs_data.columns if len(obs_data.columns) == result.shape[0] else [f"x{i}" for i in range(result.shape[0])]
                    for node_name in node_names:
                        graph.add_node(node_name)
                    
                    # Add edges based on adjacency matrix
                    for i in range(result.shape[0]):
                        for j in range(result.shape[1]):
                            if result[i, j] > 0:
                                graph.add_edge(node_names[i], node_names[j])
                    
                    return graph
            
            # If we can't determine the interface
            logger.warning(f"Unknown method interface, couldn't run structure learning")
            return None
            
        except Exception as e:
            logger.error(f"Error in structure learning: {str(e)}")
            return None
    
    def _aggregate_results(self) -> None:
        """
        Aggregate results across all datasets and methods.
        """
        # Initialize aggregated results
        self.results["aggregated"] = {
            "models": {},
            "baselines": {}
        }
        
        # Aggregate model results
        for model_name, model_results in self.results["models"].items():
            # Initialize metrics trackers
            shd_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            runtime_values = []
            
            # Collect metrics across datasets
            for dataset_name, dataset_results in model_results.items():
                if "structure_metrics" in dataset_results:
                    metrics = dataset_results["structure_metrics"]
                    if "shd" in metrics:
                        shd_values.append(metrics["shd"])
                    if "precision" in metrics:
                        precision_values.append(metrics["precision"])
                    if "recall" in metrics:
                        recall_values.append(metrics["recall"])
                    if "f1" in metrics:
                        f1_values.append(metrics["f1"])
                
                if "timing" in dataset_results:
                    runtime_values.append(dataset_results["timing"]["avg_time"])
            
            # Compute aggregate statistics
            self.results["aggregated"]["models"][model_name] = {
                "shd": {
                    "mean": np.mean(shd_values) if shd_values else None,
                    "std": np.std(shd_values) if shd_values else None,
                    "min": np.min(shd_values) if shd_values else None,
                    "max": np.max(shd_values) if shd_values else None
                },
                "precision": {
                    "mean": np.mean(precision_values) if precision_values else None,
                    "std": np.std(precision_values) if precision_values else None
                },
                "recall": {
                    "mean": np.mean(recall_values) if recall_values else None,
                    "std": np.std(recall_values) if recall_values else None
                },
                "f1": {
                    "mean": np.mean(f1_values) if f1_values else None,
                    "std": np.std(f1_values) if f1_values else None
                },
                "runtime": {
                    "mean": np.mean(runtime_values) if runtime_values else None,
                    "std": np.std(runtime_values) if runtime_values else None
                }
            }
        
        # Do the same for baselines
        for baseline_name, baseline_results in self.results["baselines"].items():
            # Similar aggregation as above
            # (code omitted for brevity, same pattern as model aggregation)
            # Initialize metrics trackers
            shd_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            runtime_values = []
            
            # Collect metrics across datasets
            for dataset_name, dataset_results in baseline_results.items():
                if "structure_metrics" in dataset_results:
                    metrics = dataset_results["structure_metrics"]
                    if "shd" in metrics:
                        shd_values.append(metrics["shd"])
                    if "precision" in metrics:
                        precision_values.append(metrics["precision"])
                    if "recall" in metrics:
                        recall_values.append(metrics["recall"])
                    if "f1" in metrics:
                        f1_values.append(metrics["f1"])
                
                if "timing" in dataset_results:
                    runtime_values.append(dataset_results["timing"]["avg_time"])
            
            # Compute aggregate statistics
            self.results["aggregated"]["baselines"][baseline_name] = {
                "shd": {
                    "mean": np.mean(shd_values) if shd_values else None,
                    "std": np.std(shd_values) if shd_values else None,
                    "min": np.min(shd_values) if shd_values else None,
                    "max": np.max(shd_values) if shd_values else None
                },
                "precision": {
                    "mean": np.mean(precision_values) if precision_values else None,
                    "std": np.std(precision_values) if precision_values else None
                },
                "recall": {
                    "mean": np.mean(recall_values) if recall_values else None,
                    "std": np.std(recall_values) if recall_values else None
                },
                "f1": {
                    "mean": np.mean(f1_values) if f1_values else None,
                    "std": np.std(f1_values) if f1_values else None
                },
                "runtime": {
                    "mean": np.mean(runtime_values) if runtime_values else None,
                    "std": np.std(runtime_values) if runtime_values else None
                }
            }
    
    def plot_results(
        self,
        metrics: List[str] = ["shd", "f1", "runtime"],
        title: str = "Causal Discovery Benchmark Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot benchmark results for visual comparison.
        
        Args:
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, "results") or not self.results:
            logger.warning("No results to plot. Run the benchmark first.")
            return None
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Get all method names
        all_methods = list(self.results.get("aggregated", {}).get("models", {}).keys())
        all_methods.extend(list(self.results.get("aggregated", {}).get("baselines", {}).keys()))
        all_methods = sorted(set(all_methods))  # Deduplicate and sort
        
        # Define colors for models and baselines
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods)))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot models
            x_positions = []
            x_labels = []
            current_x = 0
            
            for j, method in enumerate(all_methods):
                # Check if method exists in models
                model_data = self.results.get("aggregated", {}).get("models", {}).get(method, {})
                baseline_data = self.results.get("aggregated", {}).get("baselines", {}).get(method, {})
                
                # Plot model data if available
                if model_data and metric in model_data and model_data[metric]["mean"] is not None:
                    mean = model_data[metric]["mean"]
                    std = model_data[metric]["std"] if "std" in model_data[metric] else 0
                    
                    ax.bar(current_x, mean, yerr=std, color=colors[j], alpha=0.7, 
                           label=f"Model: {method}")
                    x_positions.append(current_x)
                    x_labels.append(method)
                    current_x += 1
                
                # Plot baseline data if available
                elif baseline_data and metric in baseline_data and baseline_data[metric]["mean"] is not None:
                    mean = baseline_data[metric]["mean"]
                    std = baseline_data[metric]["std"] if "std" in baseline_data[metric] else 0
                    
                    ax.bar(current_x, mean, yerr=std, color=colors[j], hatch='//', alpha=0.7,
                           label=f"Baseline: {method}")
                    x_positions.append(current_x)
                    x_labels.append(method)
                    current_x += 1
            
            # Set labels and title
            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel(metric.upper())
            
            # Add a grid for better readability
            ax.grid(True, linestyle="--", alpha=0.7)
            
            # Show legend
            ax.legend()
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure if path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class CBOBenchmark(Benchmark):
    """
    Benchmark for evaluating causal Bayesian optimization methods.
    
    This benchmark evaluates how well different methods can optimize
    an objective function through carefully selected interventions
    on a causal system.
    """
    
    def __init__(
        self,
        name: str = "cbo_benchmark",
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        num_nodes: int = 10,
        num_graphs: int = 10,
        num_samples: int = 1000,
        graph_type: str = "random",
        edge_prob: float = 0.3,
        target_node: Optional[str] = None,
        objective_type: str = "maximize",
        num_interventions: int = 5,
        intervention_budget: int = 10
    ):
        """Initialize the CBO benchmark."""
        super().__init__(name, output_dir, seed, device)
        
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.num_samples = num_samples
        self.graph_type = graph_type
        self.edge_prob = edge_prob
        self.target_node = target_node
        self.objective_type = objective_type
        self.num_interventions = num_interventions
        self.intervention_budget = intervention_budget
        
        # Initialize graph factory
        self.graph_factory = GraphFactory()
        
        # Initialize metric trackers
        self.intervention_metrics = {}
        self.runtime_metrics = {}
    
    def setup(self) -> None:
        """
        Set up the benchmark by generating optimization problems.
        """
        logger.info(f"Setting up benchmark: {self.name}")
        
        # Generate synthetic graphs and SCMs for optimization
        self.graphs = []
        self.scms = []
        self.optimization_problems = {}
        
        for i in range(self.num_graphs):
            # Generate a random graph with named nodes
            graph = self.graph_factory.create_graph(
                graph_type=self.graph_type,
                num_nodes=self.num_nodes,
                edge_probability=self.edge_prob,
                seed=self.seed + i if self.seed is not None else None
            )
            
            # Ensure nodes are named x0, x1, etc. if not already
            node_mapping = {}
            for j, node in enumerate(graph.get_nodes()):
                if not isinstance(node, str) or not node.startswith('x'):
                    node_name = f"x{j}"
                    node_mapping[node] = node_name
            
            # If we need to rename nodes
            if node_mapping:
                new_graph = CausalGraph()
                # Add nodes
                for j in range(self.num_nodes):
                    new_graph.add_node(f"x{j}")
                
                # Add edges (ensuring lower index -> higher index for acyclicity)
                for u in graph.get_nodes():
                    for v in graph.get_children(u):
                        u_name = node_mapping.get(u, u)
                        v_name = node_mapping.get(v, v)
                        
                        # Extract indices to ensure acyclicity
                        if u_name.startswith('x') and v_name.startswith('x'):
                            u_idx = int(u_name[1:])
                            v_idx = int(v_name[1:])
                            
                            # Only add edge if u_idx < v_idx to ensure acyclicity
                            if u_idx < v_idx:
                                new_graph.add_edge(u_name, v_name)
                else:
                    # If no cycles are detected, just use as is
                    if not new_graph.has_cycle():
                        graph = new_graph
                    else:
                        # Create a simple chain as fallback
                        fallback_graph = CausalGraph()
                        for j in range(self.num_nodes):
                            fallback_graph.add_node(f"x{j}")
                        
                        # Add edges in a chain: x0->x1->x2->...
                        for j in range(self.num_nodes - 1):
                            fallback_graph.add_edge(f"x{j}", f"x{j+1}")
                        
                        graph = fallback_graph
            else:
                # Ensure the existing graph is acyclic
                if graph.has_cycle():
                    # Create a simple chain as fallback
                    fallback_graph = CausalGraph()
                    for j in range(self.num_nodes):
                        fallback_graph.add_node(f"x{j}")
                    
                    # Add edges in a chain: x0->x1->x2->...
                    for j in range(self.num_nodes - 1):
                        fallback_graph.add_edge(f"x{j}", f"x{j+1}")
                    
                    graph = fallback_graph
            
            self.graphs.append(graph)
            
            # Create an SCM from the graph
            scm = StructuralCausalModel(
                causal_graph=graph,
                random_state=self.seed + i if self.seed is not None else None
            )
            
            # Add variables to the SCM
            for node in graph.get_nodes():
                scm.add_variable(node)
            
            # Define linear Gaussian equations for all nodes
            for node in graph.get_nodes():
                parents = graph.get_parents(node)
                if parents:
                    # Create random coefficients for parents
                    coeffs = {parent: np.random.uniform(-1, 1) for parent in parents}
                    scm.define_linear_gaussian_equation(
                        node, coeffs, intercept=0, noise_std=0.1
                    )
                else:
                    # Exogenous nodes have no parents
                    scm.define_linear_gaussian_equation(
                        node, {}, intercept=0, noise_std=1.0
                    )
            
            self.scms.append(scm)
            
            # Generate observational data
            obs_data = scm.sample_data(sample_size=self.num_samples)
            
            # Choose a target node if not specified
            if self.target_node is None:
                # Choose a random leaf node (with no children) as target if possible
                leaf_nodes = []
                for node in graph.get_nodes():
                    if len(graph.get_children(node)) == 0:
                        leaf_nodes.append(node)
                
                if leaf_nodes:
                    target_node = np.random.choice(leaf_nodes)
                else:
                    # Fallback to a random node
                    nodes = list(graph.get_nodes())
                    target_node = np.random.choice(nodes)
            else:
                target_node = self.target_node
            
            # Choose potential intervention targets (nodes that affect the target)
            # Exclude the target node itself
            potential_targets = []
            
            # Find ancestors (direct and indirect causes) of the target
            ancestors = graph.get_ancestors(target_node)
            
            # Add ancestors as potential intervention targets
            for ancestor in ancestors:
                if ancestor != target_node:  # Skip target itself
                    potential_targets.append(ancestor)
            
            # If not enough ancestors, add some random nodes
            if len(potential_targets) < self.num_interventions:
                other_nodes = [node for node in graph.get_nodes() 
                              if node != target_node and node not in potential_targets]
                
                # Randomly select additional nodes if needed
                if other_nodes:
                    num_additional = min(self.num_interventions - len(potential_targets), len(other_nodes))
                    additional_targets = np.random.choice(other_nodes, size=num_additional, replace=False)
                    potential_targets.extend(additional_targets)
            
            # Define intervention ranges for each target
            intervention_ranges = {}
            for target in potential_targets:
                # Simple intervention range: (-2, 2)
                intervention_ranges[target] = (-2.0, 2.0)
            
            # Define objective function
            def objective_fn(interventions: Dict[str, float]) -> float:
                """
                Objective function for optimization.
                
                Args:
                    interventions: Dictionary mapping node names to intervention values
                    
                Returns:
                    Value of the target node under interventions
                """
                # Sample data under interventions
                intervention_data = scm.sample_interventional_data(
                    interventions=interventions,
                    sample_size=100
                )
                # Return mean value of target node
                return intervention_data[target_node].mean()
            
            # Store optimization problem
            self.optimization_problems[f"problem_{i}"] = {
                "graph": graph,
                "scm": scm,
                "observational_data": obs_data,
                "target_node": target_node,
                "potential_targets": potential_targets,
                "intervention_ranges": intervention_ranges,
                "objective_fn": objective_fn,
                "objective_type": self.objective_type
            }
        
        logger.info(f"Setup complete: defined {len(self.optimization_problems)} optimization problems")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark on all registered models and baselines.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running benchmark: {self.name}")
        
        # Initialize results storage
        self.results = {
            "benchmark_config": {
                "num_nodes": self.num_nodes,
                "num_graphs": self.num_graphs,
                "num_samples": self.num_samples,
                "graph_type": self.graph_type,
                "edge_prob": self.edge_prob,
                "objective_type": self.objective_type,
                "num_interventions": self.num_interventions,
                "intervention_budget": self.intervention_budget
            },
            "models": {},
            "baselines": {}
        }
        
        # Run benchmark on each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = self._evaluate_method(model_name, model)
            self.results["models"][model_name] = model_results
        
        # Run benchmark on each baseline
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            baseline_results = self._evaluate_method(baseline_name, baseline)
            self.results["baselines"][baseline_name] = baseline_results
        
        # Aggregate results
        self._aggregate_results()
        
        # Save results
        results_path = self.save_results(self.results)
        logger.info(f"Results saved to: {results_path}")
        
        return self.results
    
    def _evaluate_method(self, method_name: str, method: Any) -> Dict[str, Any]:
        """
        Evaluate a single CBO method on all optimization problems.
        
        Args:
            method_name: Name of the method
            method: Method instance to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        method_results = {}
        
        for problem_name, problem in self.optimization_problems.items():
            # Extract problem components
            graph = problem["graph"]
            scm = problem["scm"]
            obs_data = problem["observational_data"]
            target_node = problem["target_node"]
            potential_targets = problem["potential_targets"]
            intervention_ranges = problem["intervention_ranges"]
            objective_fn = problem["objective_fn"]
            
            # Track problem results
            problem_results = {}
            
            try:
                # Time the optimization process
                timing_results = self.time_performance(
                    self._run_optimization,
                    3,  # repeat 3 times
                    method=method,
                    graph=graph,
                    scm=scm,
                    obs_data=obs_data,
                    target_node=target_node,
                    potential_targets=potential_targets,
                    intervention_ranges=intervention_ranges,
                    objective_fn=objective_fn
                )
                
                # Extract optimization results
                opt_results = timing_results["result"]
                
                if opt_results:
                    # Calculate metrics
                    problem_results["best_intervention"] = opt_results["best_intervention"]
                    problem_results["best_value"] = opt_results["best_value"]
                    problem_results["num_evaluations"] = opt_results["num_evaluations"]
                    problem_results["intervention_sequence"] = opt_results["intervention_sequence"]
                    problem_results["value_sequence"] = opt_results["value_sequence"]
                    
                    # Compare to optimal value if available (here we don't have it)
                    # In a real implementation, you could compute or estimate optimal values
                    
                    # Compare to random baseline
                    random_values = []
                    for _ in range(10):  # Evaluate 10 random interventions
                        random_intervention = {}
                        for target in np.random.choice(potential_targets, size=min(3, len(potential_targets)), replace=False):
                            min_val, max_val = intervention_ranges[target]
                            random_intervention[target] = np.random.uniform(min_val, max_val)
                        
                        random_value = objective_fn(random_intervention)
                        random_values.append(random_value)
                    
                    random_best = max(random_values)
                    problem_results["random_baseline_value"] = random_best
                    problem_results["improvement_over_random"] = (problem_results["best_value"] - random_best) / abs(random_best) if random_best != 0 else 0
                else:
                    problem_results["error"] = "Failed to run optimization"
                
                # Add timing results
                problem_results["timing"] = {
                    "min_time": timing_results["min"],
                    "max_time": timing_results["max"],
                    "avg_time": timing_results["mean"]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {method_name} on {problem_name}: {str(e)}")
                problem_results["error"] = str(e)
            
            # Store results for this problem
            method_results[problem_name] = problem_results
        
        return method_results
    
    def _run_optimization(
        self, 
        method: Any,
        graph: CausalGraph,
        scm: StructuralCausalModel,
        obs_data: pd.DataFrame,
        target_node: str,
        potential_targets: List[str],
        intervention_ranges: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, float]], float]
    ) -> Dict[str, Any]:
        """
        Run optimization using the provided method.
        
        This is a helper method that handles different method interfaces.
        
        Args:
            method: Optimization method to evaluate
            graph: Causal graph
            scm: Structural causal model
            obs_data: Observational data
            target_node: Target node for optimization
            potential_targets: List of potential intervention targets
            intervention_ranges: Dictionary mapping intervention targets to (min, max) ranges
            objective_fn: Objective function to optimize
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Track optimization process
            best_intervention = None
            best_value = float('-inf') if self.objective_type == "maximize" else float('inf')
            num_evaluations = 0
            intervention_sequence = []
            value_sequence = []
            
            # Check for different method interfaces
            if isinstance(method, AmortizedCBO):
                # Handle AmortizedCBO interface
                
                # Convert data to tensor format expected by the model
                X = torch.tensor(obs_data.values, dtype=torch.float32)
                X = X.to(self.device)
                
                # Get feature names
                feature_names = obs_data.columns.tolist()
                
                # Configure optimization problem
                method.configure_optimization(
                    X=X,
                    target_node=target_node,
                    feature_names=feature_names,
                    intervention_targets=potential_targets,
                    intervention_ranges=intervention_ranges,
                    objective_fn=objective_fn,
                    maximize=self.objective_type == "maximize"
                )
                
                # Run optimization
                for i in range(self.intervention_budget):
                    # Get next intervention to try
                    next_intervention = method.suggest_intervention()
                    
                    # Evaluate intervention
                    value = objective_fn(next_intervention)
                    num_evaluations += 1
                    
                    # Update model with observation
                    method.update(next_intervention, value)
                    
                    # Track values
                    intervention_sequence.append(next_intervention)
                    value_sequence.append(value)
                    
                    # Update best found
                    if self.objective_type == "maximize" and value > best_value:
                        best_value = value
                        best_intervention = next_intervention
                    elif self.objective_type == "minimize" and value < best_value:
                        best_value = value
                        best_intervention = next_intervention
                
                # Get final best suggestion
                final_intervention = method.get_best_intervention()
                final_value = objective_fn(final_intervention)
                num_evaluations += 1
                
                # Update best if final suggestion is better
                if self.objective_type == "maximize" and final_value > best_value:
                    best_value = final_value
                    best_intervention = final_intervention
                elif self.objective_type == "minimize" and final_value < best_value:
                    best_value = final_value
                    best_intervention = final_intervention
            
            elif hasattr(method, "optimize"):
                # Standard optimization interface
                result = method.optimize(
                    scm=scm,
                    graph=graph,
                    obs_data=obs_data,
                    target_node=target_node,
                    potential_targets=potential_targets,
                    intervention_ranges=intervention_ranges,
                    objective_fn=objective_fn,
                    num_iterations=self.intervention_budget,
                    maximize=self.objective_type == "maximize"
                )
                
                # Extract results
                if isinstance(result, dict):
                    best_intervention = result.get("best_intervention", None)
                    best_value = result.get("best_value", None)
                    num_evaluations = result.get("num_evaluations", None)
                    intervention_sequence = result.get("intervention_sequence", [])
                    value_sequence = result.get("value_sequence", [])
                else:
                    # Simple interface just returning best intervention
                    best_intervention = result
                    best_value = objective_fn(best_intervention)
                    num_evaluations = 1
            
            elif hasattr(method, "__call__"):
                # Callable interface
                result = method(
                    scm=scm,
                    graph=graph,
                    obs_data=obs_data,
                    target_node=target_node,
                    potential_targets=potential_targets,
                    intervention_ranges=intervention_ranges,
                    objective_fn=objective_fn,
                    num_iterations=self.intervention_budget,
                    maximize=self.objective_type == "maximize"
                )
                
                # Extract results
                if isinstance(result, dict):
                    best_intervention = result.get("best_intervention", None)
                    best_value = result.get("best_value", None)
                    num_evaluations = result.get("num_evaluations", None)
                    intervention_sequence = result.get("intervention_sequence", [])
                    value_sequence = result.get("value_sequence", [])
                else:
                    # Simple interface just returning best intervention
                    best_intervention = result
                    best_value = objective_fn(best_intervention)
                    num_evaluations = 1
            
            else:
                # If we can't determine the interface
                logger.warning(f"Unknown method interface, couldn't run optimization")
                return None
            
            # Return optimization results
            return {
                "best_intervention": best_intervention,
                "best_value": best_value,
                "num_evaluations": num_evaluations,
                "intervention_sequence": intervention_sequence,
                "value_sequence": value_sequence
            }
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return None
    
    def _aggregate_results(self) -> None:
        """
        Aggregate results across all optimization problems and methods.
        """
        # Initialize aggregated results
        self.results["aggregated"] = {
            "models": {},
            "baselines": {}
        }
        
        # Aggregate model results
        for model_name, model_results in self.results["models"].items():
            # Initialize metrics trackers
            best_values = []
            improvements = []
            runtime_values = []
            num_evals = []
            
            # Collect metrics across problems
            for problem_name, problem_results in model_results.items():
                if "best_value" in problem_results:
                    best_values.append(problem_results["best_value"])
                
                if "improvement_over_random" in problem_results:
                    improvements.append(problem_results["improvement_over_random"])
                
                if "timing" in problem_results:
                    runtime_values.append(problem_results["timing"]["avg_time"])
                
                if "num_evaluations" in problem_results:
                    num_evals.append(problem_results["num_evaluations"])
            
            # Compute aggregate statistics
            self.results["aggregated"]["models"][model_name] = {
                "best_value": {
                    "mean": np.mean(best_values) if best_values else None,
                    "std": np.std(best_values) if best_values else None,
                    "min": np.min(best_values) if best_values else None,
                    "max": np.max(best_values) if best_values else None
                },
                "improvement_over_random": {
                    "mean": np.mean(improvements) if improvements else None,
                    "std": np.std(improvements) if improvements else None
                },
                "runtime": {
                    "mean": np.mean(runtime_values) if runtime_values else None,
                    "std": np.std(runtime_values) if runtime_values else None
                },
                "num_evaluations": {
                    "mean": np.mean(num_evals) if num_evals else None,
                    "std": np.std(num_evals) if num_evals else None
                }
            }
        
        # Do the same for baselines
        for baseline_name, baseline_results in self.results["baselines"].items():
            # Initialize metrics trackers
            best_values = []
            improvements = []
            runtime_values = []
            num_evals = []
            
            # Collect metrics across problems
            for problem_name, problem_results in baseline_results.items():
                if "best_value" in problem_results:
                    best_values.append(problem_results["best_value"])
                
                if "improvement_over_random" in problem_results:
                    improvements.append(problem_results["improvement_over_random"])
                
                if "timing" in problem_results:
                    runtime_values.append(problem_results["timing"]["avg_time"])
                
                if "num_evaluations" in problem_results:
                    num_evals.append(problem_results["num_evaluations"])
            
            # Compute aggregate statistics
            self.results["aggregated"]["baselines"][baseline_name] = {
                "best_value": {
                    "mean": np.mean(best_values) if best_values else None,
                    "std": np.std(best_values) if best_values else None,
                    "min": np.min(best_values) if best_values else None,
                    "max": np.max(best_values) if best_values else None
                },
                "improvement_over_random": {
                    "mean": np.mean(improvements) if improvements else None,
                    "std": np.std(improvements) if improvements else None
                },
                "runtime": {
                    "mean": np.mean(runtime_values) if runtime_values else None,
                    "std": np.std(runtime_values) if runtime_values else None
                },
                "num_evaluations": {
                    "mean": np.mean(num_evals) if num_evals else None,
                    "std": np.std(num_evals) if num_evals else None
                }
            }
    
    def plot_results(
        self,
        metrics: List[str] = ["best_value", "improvement_over_random", "runtime"],
        title: str = "CBO Benchmark Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot benchmark results for visual comparison.
        
        Args:
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, "results") or not self.results:
            logger.warning("No results to plot. Run the benchmark first.")
            return None
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Get all method names
        all_methods = list(self.results.get("aggregated", {}).get("models", {}).keys())
        all_methods.extend(list(self.results.get("aggregated", {}).get("baselines", {}).keys()))
        all_methods = sorted(set(all_methods))  # Deduplicate and sort
        
        # Define colors for models and baselines
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods)))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot models
            x_positions = []
            x_labels = []
            current_x = 0
            
            for j, method in enumerate(all_methods):
                # Check if method exists in models
                model_data = self.results.get("aggregated", {}).get("models", {}).get(method, {})
                baseline_data = self.results.get("aggregated", {}).get("baselines", {}).get(method, {})
                
                # Plot model data if available
                if model_data and metric in model_data and model_data[metric]["mean"] is not None:
                    mean = model_data[metric]["mean"]
                    std = model_data[metric]["std"] if "std" in model_data[metric] else 0
                    
                    ax.bar(current_x, mean, yerr=std, color=colors[j], alpha=0.7, 
                           label=f"Model: {method}")
                    x_positions.append(current_x)
                    x_labels.append(method)
                    current_x += 1
                
                # Plot baseline data if available
                elif baseline_data and metric in baseline_data and baseline_data[metric]["mean"] is not None:
                    mean = baseline_data[metric]["mean"]
                    std = baseline_data[metric]["std"] if "std" in baseline_data[metric] else 0
                    
                    ax.bar(current_x, mean, yerr=std, color=colors[j], hatch='//', alpha=0.7,
                           label=f"Baseline: {method}")
                    x_positions.append(current_x)
                    x_labels.append(method)
                    current_x += 1
            
            # Set labels and title
            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel(metric.upper())
            
            # Add a grid for better readability
            ax.grid(True, linestyle="--", alpha=0.7)
            
            # Show legend
            ax.legend()
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure if path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class ScalabilityBenchmark(Benchmark):
    """
    Benchmark for evaluating scalability of causal discovery and intervention
    optimization methods with respect to graph size.
    
    This benchmark automatically runs tests on graphs of increasing size and
    tracks memory usage, runtime, and other performance metrics to analyze
    how methods scale with problem complexity.
    """
    
    def __init__(
        self,
        name: str = "scalability_benchmark",
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        min_nodes: int = 5,
        max_nodes: int = 50,
        step_size: int = 5,
        num_graphs_per_size: int = 3,
        num_samples: int = 1000,
        graph_type: str = "erdos_renyi",
        edge_density: float = None,
        measure_mode: str = "both",
        track_gpu_memory: bool = True,
    ):
        """
        Initialize the scalability benchmark.
        
        Args:
            name: Name of the benchmark
            output_dir: Directory to store benchmark results
            seed: Random seed for reproducibility
            device: Device to run the benchmark on
            min_nodes: Minimum number of nodes in test graphs
            max_nodes: Maximum number of nodes in test graphs
            step_size: Increment between graph sizes
            num_graphs_per_size: Number of test graphs to generate for each size
            num_samples: Number of samples for each test graph
            graph_type: Type of graphs to generate ("erdos_renyi", "scale_free", etc.)
            edge_density: Fixed edge density; if None, uses 2/n for avg degree ~2
            measure_mode: What to measure ("discovery", "cbo", or "both")
            track_gpu_memory: Whether to track GPU memory usage
        """
        super().__init__(name=name, output_dir=output_dir, seed=seed, device=device)
        
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.step_size = step_size
        self.num_graphs_per_size = num_graphs_per_size
        self.num_samples = num_samples
        self.graph_type = graph_type
        self.edge_density = edge_density
        self.measure_mode = measure_mode
        self.track_gpu_memory = track_gpu_memory
        
        # Generate the list of node sizes to test
        self.node_sizes = list(range(min_nodes, max_nodes + 1, step_size))
        
        # Initialize scalability results storage
        self.scalability_results = {}
        
    def setup(self) -> None:
        """
        Set up the benchmark by preparing test harnesses for different graph sizes.
        """
        logger.info(f"Setting up scalability benchmark: {self.name}")
        
        # Create subdirectories for different graph sizes
        self.size_dirs = {}
        for size in self.node_sizes:
            size_dir = os.path.join(self.benchmark_dir, f"size_{size}")
            os.makedirs(size_dir, exist_ok=True)
            self.size_dirs[size] = size_dir
        
        # Initialize test graphs and problems
        self.test_graphs = {}
        self.test_problems = {}
        
        for size in self.node_sizes:
            logger.info(f"Generating test cases for graph size {size}")
            
            # Generate graphs of this size
            graphs = []
            problems = []
            
            for i in range(self.num_graphs_per_size):
                # Generate random graph
                if self.graph_type == "erdos_renyi":
                    from causal_meta.graph.generators.random_graphs import RandomGraphGenerator
                    raw_graph = RandomGraphGenerator.erdos_renyi(
                        num_nodes=size,
                        edge_probability=2.0 / size if self.edge_density is None else self.edge_density,
                        ensure_dag=True,  # Ensure the graph is a DAG
                        seed=self.seed + i if self.seed is not None else None
                    )
                elif self.graph_type == "scale_free":
                    from causal_meta.graph.generators.scale_free import ScaleFreeNetworkGenerator
                    raw_graph = ScaleFreeNetworkGenerator.barabasi_albert(
                        num_nodes=size,
                        m=2,  # Default parameter for new edges
                        seed=self.seed + i if self.seed is not None else None
                    )
                    # Scale-free networks can have cycles, so we need to ensure it's a DAG
                    if raw_graph.has_cycle():
                        # Create a simple chain as fallback
                        raw_graph = CausalGraph()
                        for j in range(size):
                            raw_graph.add_node(j)
                        
                        # Add edges in a chain: 0->1->2->...
                        for j in range(size - 1):
                            raw_graph.add_edge(j, j+1)
                else:
                    raise ValueError(f"Unsupported graph type: {self.graph_type}")
                
                # Create a new graph with string node names
                graph = CausalGraph()
                
                # Add nodes with string names
                node_map = {}
                for j, node in enumerate(raw_graph.get_nodes()):
                    node_name = f"x{j}"
                    graph.add_node(node_name)
                    node_map[node] = node_name
                
                # Add edges with mapped node names
                for node in raw_graph.get_nodes():
                    for child in raw_graph.get_children(node):
                        graph.add_edge(node_map[node], node_map[child])
                
                # Generate SCM based on the graph
                from causal_meta.environments.scm import StructuralCausalModel
                scm = StructuralCausalModel(
                    causal_graph=graph,
                    random_state=self.seed + i + 100 if self.seed is not None else None
                )
                
                # Add variables to the SCM
                for node in graph.get_nodes():
                    scm.add_variable(node)
                
                # Define linear Gaussian equations for all nodes
                for node in graph.get_nodes():
                    parents = graph.get_parents(node)
                    if parents:
                        # Create random coefficients for parents
                        coeffs = {parent: np.random.uniform(-1, 1) for parent in parents}
                        scm.define_linear_gaussian_equation(
                            node, coeffs, intercept=0, noise_std=0.1
                        )
                    else:
                        # Exogenous nodes have no parents
                        scm.define_linear_gaussian_equation(
                            node, {}, intercept=0, noise_std=1.0
                        )
                
                # Generate observational data
                obs_data = scm.sample_data(sample_size=self.num_samples)
                
                # Create a test problem
                problem = {
                    "graph": graph,
                    "scm": scm,
                    "observational_data": obs_data
                }
                
                # If measuring CBO, add optimization problem components
                if self.measure_mode in ["cbo", "both"]:
                    # Randomly select a target node
                    import random
                    rng = random.Random(self.seed + i + 200 if self.seed is not None else None)
                    
                    # Only select target nodes that have parents
                    potential_targets = []
                    for node_name in graph.get_nodes():
                        if len(graph.get_parents(node_name)) > 0:
                            potential_targets.append(node_name)
                    
                    if potential_targets:
                        target_node = rng.choice(potential_targets)
                        
                        # Get potential intervention targets (parents of target)
                        intervention_targets = graph.get_parents(target_node)
                        
                        # Define intervention ranges
                        intervention_ranges = {
                            node: (-2.0, 2.0) for node in intervention_targets
                        }
                        
                        # Create objective function (maximize output of target node)
                        def objective_fn(scm, interventions):
                            # Sample with interventions
                            intervention_data = scm.sample_interventional_data(
                                interventions=interventions,
                                sample_size=100
                            )
                            # Return mean value of target node
                            return intervention_data[target_node].mean()
                        
                        # Add optimization problem components
                        problem.update({
                            "target_node": target_node,
                            "potential_targets": intervention_targets,
                            "intervention_ranges": intervention_ranges,
                            "objective_fn": objective_fn
                        })
                
                # Add to problems list
                problems.append(problem)
                
                # Save problem data for reuse
                problem_file = os.path.join(self.size_dirs[size], f"problem_{i}.npz")
                np.savez(
                    problem_file,
                    graph=graph.get_adjacency_matrix(),
                    observational_data=obs_data
                )
            
            # Store test cases for this size
            self.test_problems[size] = problems
        
        logger.info(f"Scalability benchmark setup complete with {sum(len(p) for p in self.test_problems.values())} test problems")
    
    def memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage metrics
        """
        # Get system memory usage
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        result = {
            "cpu_memory_mb": cpu_memory
        }
        
        # Add GPU memory if requested and available
        if self.track_gpu_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            result.update({
                "gpu_memory_mb": gpu_memory,
                "gpu_memory_reserved_mb": gpu_memory_reserved
            })
        
        return result
    
    def measure_memory_usage(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """
        Measure the memory usage and execution time of a function.
        
        Args:
            func: Function to measure
            **kwargs: Arguments to pass to the function
            
        Returns:
            Dictionary with measured metrics and function result
        """
        import time
        
        # Record initial memory usage
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        initial_gpu_memory = 0
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
        # Record peak memory during execution
        peak_cpu_memory = initial_cpu_memory
        peak_gpu_memory = initial_gpu_memory
        
        def update_peak_memory():
            nonlocal peak_cpu_memory, peak_gpu_memory
            current_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            peak_cpu_memory = max(peak_cpu_memory, current_cpu_memory)
            
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
                current_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
        
        # Setup memory monitoring thread
        stop_monitoring = False
        
        def memory_monitor():
            while not stop_monitoring:
                update_peak_memory()
                time.sleep(0.1)  # Check every 100ms
        
        # Start monitoring thread
        import threading
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute function and measure time
        start_time = time.time()
        result = None
        exception = None
        
        try:
            result = func(**kwargs)
        except Exception as e:
            exception = str(e)
            logger = logging.getLogger(__name__)
            logger.error(f"Error executing function: {e}")
        
        execution_time = time.time() - start_time
        
        # Stop monitoring thread
        stop_monitoring = True
        monitor_thread.join(timeout=1.0)
        
        # Final memory check
        update_peak_memory()
        
        # Calculate memory usage
        cpu_memory_used = peak_cpu_memory - initial_cpu_memory
        gpu_memory_used = peak_gpu_memory - initial_gpu_memory if self.device != "cpu" and torch.cuda.is_available() else 0
        
        metrics = {
            "execution_time": execution_time,
            "peak_cpu_memory_mb": peak_cpu_memory,
            "cpu_memory_used_mb": cpu_memory_used,
        }
        
        if self.device != "cpu" and torch.cuda.is_available():
            metrics.update({
                "peak_gpu_memory_mb": peak_gpu_memory,
                "gpu_memory_used_mb": gpu_memory_used,
            })
        
        return {
            "metrics": metrics,
            "result": result,
            "exception": exception
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the scalability benchmark on all registered models and baselines.
        
        For each model and each graph size, the benchmark measures runtime,
        memory usage, and accuracy metrics.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Running scalability benchmark: {self.name}")
        
        # Initialize results storage
        self.results = {
            "benchmark_config": {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "step_size": self.step_size,
                "num_graphs_per_size": self.num_graphs_per_size,
                "num_samples": self.num_samples,
                "graph_type": self.graph_type,
                "edge_density": self.edge_density,
                "measure_mode": self.measure_mode
            },
            "models": {},
            "baselines": {},
            "scalability_metrics": {}
        }
        
        # Run benchmark on each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model scalability: {model_name}")
            model_results = self.evaluate_method_scalability(model_name, model, is_baseline=False)
            self.results["models"][model_name] = model_results
        
        # Run benchmark on each baseline
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline scalability: {baseline_name}")
            baseline_results = self.evaluate_method_scalability(baseline_name, baseline, is_baseline=True)
            self.results["baselines"][baseline_name] = baseline_results
        
        # Analyze scaling behavior
        self.analyze_scaling()
        
        # Save results
        results_path = self.save_results(self.results)
        logger.info(f"Results saved to: {results_path}")
        
        return self.results
    
    def evaluate_method_scalability(self, method_name: str, method: Any, is_baseline: bool = False) -> Dict[str, Any]:
        """
        Evaluate the scalability of a single method across different graph sizes.
        
        Args:
            method_name: Name of the method
            method: Method instance to evaluate
            is_baseline: Whether this is a baseline method
            
        Returns:
            Dictionary containing evaluation results for each graph size
        """
        logger = logging.getLogger(__name__)
        import numpy as np
        import time
        
        # Initialize results for this method
        method_results = {}
        
        # Store raw measurements for trend analysis
        measurements = {
            "graph_sizes": [],
            "runtime": [],
            "cpu_memory": [],
            "gpu_memory": [],
            "accuracy": []
        }
        
        # Evaluate on each graph size
        for size in self.node_sizes:
            logger.info(f"Testing {method_name} on graph size {size}")
            
            # Track results for this size
            size_results = {
                "problems": [],
                "avg_runtime": None,
                "avg_cpu_memory": None,
                "avg_gpu_memory": None,
                "avg_accuracy": None,
                "failure_rate": 0.0
            }
            
            # Variables to track statistics
            runtimes = []
            cpu_memories = []
            gpu_memories = []
            accuracies = []
            failures = 0
            
            # Time limit for each problem (increases with graph size)
            timeout = max(30, size * 2)  # seconds
            
            # Run tests on all problems of this size
            for i, problem in enumerate(self.test_problems[size]):
                logger.info(f"Running problem {i+1}/{len(self.test_problems[size])} for size {size}")
                
                problem_result = {
                    "size": size,
                    "problem_index": i,
                    "success": False,
                    "timeout": False
                }
                
                # Measure discovery or optimization based on measure_mode
                if self.measure_mode in ["discovery", "both"]:
                    # Check if method has learn_graph method
                    if hasattr(method, 'learn_graph'):
                        try:
                            # Set timeout for this evaluation
                            with_timeout = lambda **kwargs: self._run_with_timeout(
                                method.learn_graph, timeout=timeout, **kwargs
                            )
                            
                            # Measure causal discovery
                            measurement = self.measure_memory_usage(
                                with_timeout, data=problem["observational_data"]
                            )
                            
                            if measurement["exception"] is not None:
                                if "timeout" in measurement["exception"].lower():
                                    problem_result["timeout"] = True
                                failures += 1
                            else:
                                # Record metrics
                                problem_result.update(measurement["metrics"])
                                
                                # Calculate accuracy if true graph is available
                                if "graph" in problem:
                                    true_adj = problem["graph"].adjacency_matrix()
                                    pred_adj = measurement["result"]
                                    
                                    # Structural Hamming Distance
                                    from causal_meta.graph.metrics import structural_hamming_distance
                                    shd = structural_hamming_distance(true_adj, pred_adj)
                                    
                                    problem_result["shd"] = shd
                                    problem_result["accuracy"] = 1.0 - shd / (size * (size - 1))  # Normalize
                                    
                                    if "accuracy" in problem_result:
                                        accuracies.append(problem_result["accuracy"])
                                
                                problem_result["success"] = True
                                
                                # Add metrics to aggregated statistics
                                runtimes.append(problem_result["execution_time"])
                                cpu_memories.append(problem_result["cpu_memory_used_mb"])
                                
                                if "gpu_memory_used_mb" in problem_result:
                                    gpu_memories.append(problem_result["gpu_memory_used_mb"])
                        except Exception as e:
                            logger.error(f"Error evaluating {method_name} on size {size}, problem {i}: {e}")
                            failures += 1
                
                # If measuring CBO, evaluate optimization performance
                if self.measure_mode in ["cbo", "both"] and "target_node" in problem:
                    # Check if method has optimize method
                    if hasattr(method, 'optimize'):
                        try:
                            # Set timeout for this evaluation
                            with_timeout = lambda **kwargs: self._run_with_timeout(
                                method.optimize, timeout=timeout, **kwargs
                            )
                            
                            # Measure optimization
                            measurement = self.measure_memory_usage(
                                with_timeout, 
                                graph=problem["graph"],
                                scm=problem["scm"],
                                obs_data=problem["observational_data"],
                                target_node=problem["target_node"],
                                potential_targets=problem["potential_targets"],
                                intervention_ranges=problem["intervention_ranges"],
                                objective_fn=problem["objective_fn"]
                            )
                            
                            if measurement["exception"] is not None:
                                if "timeout" in measurement["exception"].lower():
                                    problem_result["timeout"] = True
                                failures += 1
                            else:
                                # Record metrics
                                cbo_metrics = measurement["metrics"]
                                
                                # If this is a combined run, aggregate metrics
                                if "execution_time" in problem_result:
                                    problem_result["execution_time"] += cbo_metrics["execution_time"]
                                    problem_result["cpu_memory_used_mb"] = max(
                                        problem_result["cpu_memory_used_mb"],
                                        cbo_metrics["cpu_memory_used_mb"]
                                    )
                                    
                                    if "gpu_memory_used_mb" in problem_result and "gpu_memory_used_mb" in cbo_metrics:
                                        problem_result["gpu_memory_used_mb"] = max(
                                            problem_result["gpu_memory_used_mb"],
                                            cbo_metrics["gpu_memory_used_mb"]
                                        )
                                else:
                                    # First metrics for this problem
                                    problem_result.update(cbo_metrics)
                                    
                                    # Add metrics to aggregated statistics
                                    runtimes.append(problem_result["execution_time"])
                                    cpu_memories.append(problem_result["cpu_memory_used_mb"])
                                    
                                    if "gpu_memory_used_mb" in problem_result:
                                        gpu_memories.append(problem_result["gpu_memory_used_mb"])
                                
                                problem_result["success"] = True
                        except Exception as e:
                            logger.error(f"Error evaluating {method_name} on size {size}, problem {i}: {e}")
                            failures += 1
                
                # Add result for this problem
                size_results["problems"].append(problem_result)
            
            # Calculate aggregated metrics for this size
            total_problems = len(self.test_problems[size])
            size_results["failure_rate"] = failures / total_problems if total_problems > 0 else 1.0
            
            if runtimes:
                size_results["avg_runtime"] = float(np.mean(runtimes))
                size_results["std_runtime"] = float(np.std(runtimes))
            
            if cpu_memories:
                size_results["avg_cpu_memory"] = float(np.mean(cpu_memories))
                size_results["std_cpu_memory"] = float(np.std(cpu_memories))
            
            if gpu_memories:
                size_results["avg_gpu_memory"] = float(np.mean(gpu_memories))
                size_results["std_gpu_memory"] = float(np.std(gpu_memories))
            
            if accuracies:
                size_results["avg_accuracy"] = float(np.mean(accuracies))
                size_results["std_accuracy"] = float(np.std(accuracies))
            
            # Save results for this size
            method_results[f"size_{size}"] = size_results
            
            # Store measurements for trend analysis
            measurements["graph_sizes"].append(size)
            measurements["runtime"].append(size_results.get("avg_runtime"))
            measurements["cpu_memory"].append(size_results.get("avg_cpu_memory"))
            if size_results.get("avg_gpu_memory") is not None:
                measurements["gpu_memory"].append(size_results.get("avg_gpu_memory"))
            measurements["accuracy"].append(size_results.get("avg_accuracy"))
        
        # Store raw measurements for later analysis
        method_category = "baselines" if is_baseline else "models"
        self.scalability_results[f"{method_category}.{method_name}"] = measurements
        
        return method_results
    
    def _run_with_timeout(self, func, timeout=30, **kwargs):
        """Run a function with a timeout."""
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {timeout} seconds")
        
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(**kwargs)
        finally:
            # Cancel the timeout
            signal.alarm(0)
        
        return result
    
    def analyze_scaling(self) -> Dict[str, Any]:
        """
        Analyze the scaling behavior of all methods based on runtime and memory measurements.
        
        This function fits scaling models (e.g., polynomial, exponential) to the measured data
        and identifies the computational complexity class for each method.
        
        Returns:
            Dictionary containing scaling analysis results
        """
        import numpy as np
        
        scaling_analysis = {}
        
        for method_key, measurements in self.scalability_results.items():
            graph_sizes = np.array(measurements["graph_sizes"])
            
            # Skip if not enough data points (need at least 3 for reliable fitting)
            if len(graph_sizes) < 3:
                continue
                
            method_analysis = {}
            
            # Analyze runtime scaling
            if all(x is not None for x in measurements["runtime"]):
                runtimes = np.array(measurements["runtime"])
                
                # Fit polynomial models of different degrees
                poly_fits = {}
                for degree in [1, 2, 3]:  # Linear, quadratic, cubic
                    poly = np.polyfit(graph_sizes, runtimes, degree)
                    poly_fn = np.poly1d(poly)
                    residuals = runtimes - poly_fn(graph_sizes)
                    mse = np.mean(residuals ** 2)
                    poly_fits[degree] = {
                        "coefficients": poly.tolist(),
                        "mse": float(mse)
                    }
                
                # Find best polynomial fit based on MSE
                best_poly_degree = min(poly_fits.keys(), key=lambda d: poly_fits[d]["mse"])
                
                # Try exponential fit (y = a * exp(b * x))
                try:
                    # log(y) = log(a) + b*x
                    # Only use positive values for log
                    valid_indices = runtimes > 0
                    if np.sum(valid_indices) >= 3:
                        log_runtimes = np.log(runtimes[valid_indices])
                        exp_fit = np.polyfit(graph_sizes[valid_indices], log_runtimes, 1)
                        a = np.exp(exp_fit[1])
                        b = exp_fit[0]
                        exp_fn = lambda x: a * np.exp(b * x)
                        exp_residuals = runtimes[valid_indices] - exp_fn(graph_sizes[valid_indices])
                        exp_mse = np.mean(exp_residuals ** 2)
                        
                        exp_fit_result = {
                            "a": float(a),
                            "b": float(b),
                            "mse": float(exp_mse)
                        }
                    else:
                        exp_fit_result = {"error": "Not enough positive values for exponential fit"}
                except Exception as e:
                    exp_fit_result = {"error": str(e)}
                
                # Determine scaling class based on best fit
                if "error" not in exp_fit_result and exp_fit_result["mse"] < poly_fits[best_poly_degree]["mse"]:
                    scaling_class = "exponential"
                    best_fit = exp_fit_result
                else:
                    if best_poly_degree == 1:
                        scaling_class = "linear"
                    elif best_poly_degree == 2:
                        scaling_class = "quadratic"
                    elif best_poly_degree == 3:
                        scaling_class = "cubic"
                    best_fit = poly_fits[best_poly_degree]
                
                method_analysis["runtime_scaling"] = {
                    "scaling_class": scaling_class,
                    "best_fit": best_fit,
                    "polynomial_fits": poly_fits,
                    "exponential_fit": exp_fit_result
                }
            
            # Analyze memory scaling (similar approach as runtime)
            if all(x is not None for x in measurements["cpu_memory"]):
                memory_usage = np.array(measurements["cpu_memory"])
                
                # Fit polynomial models
                poly_fits = {}
                for degree in [1, 2, 3]:  # Linear, quadratic, cubic
                    poly = np.polyfit(graph_sizes, memory_usage, degree)
                    poly_fn = np.poly1d(poly)
                    residuals = memory_usage - poly_fn(graph_sizes)
                    mse = np.mean(residuals ** 2)
                    poly_fits[degree] = {
                        "coefficients": poly.tolist(),
                        "mse": float(mse)
                    }
                
                # Find best polynomial fit
                best_poly_degree = min(poly_fits.keys(), key=lambda d: poly_fits[d]["mse"])
                
                # Try exponential fit
                try:
                    valid_indices = memory_usage > 0
                    if np.sum(valid_indices) >= 3:
                        log_memory = np.log(memory_usage[valid_indices])
                        exp_fit = np.polyfit(graph_sizes[valid_indices], log_memory, 1)
                        a = np.exp(exp_fit[1])
                        b = exp_fit[0]
                        exp_fn = lambda x: a * np.exp(b * x)
                        exp_residuals = memory_usage[valid_indices] - exp_fn(graph_sizes[valid_indices])
                        exp_mse = np.mean(exp_residuals ** 2)
                        
                        exp_fit_result = {
                            "a": float(a),
                            "b": float(b),
                            "mse": float(exp_mse)
                        }
                    else:
                        exp_fit_result = {"error": "Not enough positive values for exponential fit"}
                except Exception as e:
                    exp_fit_result = {"error": str(e)}
                
                # Determine scaling class
                if "error" not in exp_fit_result and exp_fit_result["mse"] < poly_fits[best_poly_degree]["mse"]:
                    scaling_class = "exponential"
                    best_fit = exp_fit_result
                else:
                    if best_poly_degree == 1:
                        scaling_class = "linear"
                    elif best_poly_degree == 2:
                        scaling_class = "quadratic"
                    elif best_poly_degree == 3:
                        scaling_class = "cubic"
                    best_fit = poly_fits[best_poly_degree]
                
                method_analysis["memory_scaling"] = {
                    "scaling_class": scaling_class,
                    "best_fit": best_fit,
                    "polynomial_fits": poly_fits,
                    "exponential_fit": exp_fit_result
                }
            
            scaling_analysis[method_key] = method_analysis
        
        self.scaling_analysis = scaling_analysis
        return scaling_analysis
    
    def plot_scaling_curves(self, metric: str = "runtime", log_scale: bool = True, 
                           save_path: Optional[str] = None) -> None:
        """
        Plot scaling curves for all methods.
        
        Args:
            metric: Metric to plot ("runtime", "cpu_memory", "gpu_memory", "accuracy")
            log_scale: Whether to use log scale for both axes
            save_path: Path to save the plot, if None, the plot is shown
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.scalability_results)))
        
        for i, (method_key, measurements) in enumerate(self.scalability_results.items()):
            graph_sizes = measurements["graph_sizes"]
            metric_values = measurements[metric]
            
            # Skip methods with missing data
            if not all(v is not None for v in metric_values):
                continue
            
            # Get method name for display
            if "." in method_key:
                category, method_name = method_key.split(".", 1)
            else:
                method_name = method_key
            
            # Plot raw data points
            plt.scatter(graph_sizes, metric_values, color=colors[i], label=f"{method_name} (data)")
            
            # Plot fitted curve if we have enough data points
            if len(graph_sizes) >= 3 and method_key in self.scaling_analysis:
                analysis = self.scaling_analysis[method_key]
                if f"{metric}_scaling" in analysis:
                    scaling_info = analysis[f"{metric}_scaling"]
                    
                    # Create smooth curve for plotting
                    x_smooth = np.linspace(min(graph_sizes), max(graph_sizes), 100)
                    
                    if scaling_info["scaling_class"] == "exponential":
                        a = scaling_info["best_fit"]["a"]
                        b = scaling_info["best_fit"]["b"]
                        y_smooth = a * np.exp(b * x_smooth)
                        curve_label = f"{method_name} (exp fit)"
                    else:
                        # Polynomial fit
                        degree = {"linear": 1, "quadratic": 2, "cubic": 3}[scaling_info["scaling_class"]]
                        coeffs = scaling_info["best_fit"]["coefficients"]
                        poly = np.poly1d(coeffs)
                        y_smooth = poly(x_smooth)
                        curve_label = f"{method_name} ({scaling_info['scaling_class']} fit)"
                    
                    plt.plot(x_smooth, y_smooth, color=colors[i], linestyle="--", label=curve_label)
        
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")
        
        # Add labels and title
        metric_labels = {
            "runtime": "Runtime (seconds)",
            "cpu_memory": "CPU Memory (MB)",
            "gpu_memory": "GPU Memory (MB)",
            "accuracy": "Accuracy"
        }
        
        plt.xlabel("Graph Size (number of nodes)")
        plt.ylabel(metric_labels.get(metric, metric))
        plt.title(f"Scaling of {metric_labels.get(metric, metric)} with Graph Size")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def generate_scaling_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report on the scaling behavior of all methods.
        
        Args:
            output_path: Path to save the report as a JSON file, if None, only returns the report
            
        Returns:
            Dictionary containing the scaling report
        """
        # Ensure scaling analysis has been performed
        if not hasattr(self, "scaling_analysis"):
            self.analyze_scaling()
        
        # Prepare report
        from datetime import datetime
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_config": {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "step_size": self.step_size,
                "num_graphs_per_size": self.num_graphs_per_size,
                "graph_type": self.graph_type,
                "measure_mode": self.measure_mode
            },
            "system_info": {
                "cpu_info": psutil.cpu_count(logical=False),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "device": str(self.device)
            },
            "methods": {},
            "scaling_recommendations": {}
        }
        
        # Add method-specific results
        for method_key, analysis in self.scaling_analysis.items():
            if "." in method_key:
                category, method_name = method_key.split(".", 1)
            else:
                method_name = method_key
                category = "unknown"
            
            method_report = {
                "category": category,
                "scaling": {}
            }
            
            # Add scaling results for each metric
            for metric in ["runtime", "memory"]:
                scaling_key = f"{metric}_scaling"
                if scaling_key in analysis:
                    method_report["scaling"][metric] = {
                        "class": analysis[scaling_key]["scaling_class"],
                        "fit_details": analysis[scaling_key]["best_fit"]
                    }
            
            report["methods"][method_name] = method_report
        
        # Generate recommendations based on scaling behavior
        recommendations = []
        
        for method_name, method_report in report["methods"].items():
            if "runtime" in method_report["scaling"]:
                runtime_class = method_report["scaling"]["runtime"]["class"]
                
                if runtime_class == "exponential":
                    recommendations.append({
                        "method": method_name,
                        "issue": "Exponential scaling",
                        "recommendation": f"The {method_name} method shows exponential scaling with graph size. " +
                                        "Not recommended for graphs larger than " +
                                        f"{max(self.node_sizes)} nodes without significant optimization."
                    })
                elif runtime_class == "cubic":
                    recommendations.append({
                        "method": method_name,
                        "issue": "Cubic scaling",
                        "recommendation": f"The {method_name} method shows cubic scaling with graph size. " +
                                        "May become prohibitively slow for large graphs (100+ nodes)."
                    })
                elif runtime_class == "quadratic":
                    max_recommended = max(self.node_sizes) * 2
                    recommendations.append({
                        "method": method_name,
                        "issue": "Quadratic scaling",
                        "recommendation": f"The {method_name} method shows quadratic scaling. " +
                                        f"Should be usable for graphs up to approximately {max_recommended} nodes."
                    })
                elif runtime_class == "linear":
                    recommendations.append({
                        "method": method_name,
                        "issue": "Linear scaling",
                        "recommendation": f"The {method_name} method shows linear scaling with graph size. " +
                                        "Excellent scalability for larger graphs."
                    })
        
        report["scaling_recommendations"] = recommendations
        
        # Save report if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        
        return report