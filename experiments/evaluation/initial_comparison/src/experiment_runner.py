"""
Core experiment runner for comparing causal discovery methods.

This module contains the main logic for running experiments comparing
different policy-surrogate combinations on various SCM sizes.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import pandas as pd
from dataclasses import dataclass

# Add paths for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.core.metric_collector import MetricCollector
from experiments.evaluation.initial_comparison.src.graph_metrics import (
    evaluate_graph_discovery,
    compute_parent_accuracy_per_variable
)

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    scm_sizes: List[int]
    n_scms_per_size: int
    n_observational_samples: int
    n_interventions: int
    structure_types: List[str]
    edge_density: float
    seed: int
    metrics_to_track: List[str]


@dataclass
class MethodConfig:
    """Configuration for a method."""
    name: str
    policy_type: str  # 'random', 'oracle', 'checkpoint', 'untrained'
    policy_checkpoint: Optional[Path] = None
    surrogate_checkpoint: Optional[Path] = None
    use_surrogate: bool = True


class ExperimentRunner:
    """Runs comparison experiments between different methods."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.rng_key = random.PRNGKey(config.seed)
        self.scm_factory = VariableSCMFactory(seed=config.seed)
        self.results = []
        self.trained_checkpoint_path = None  # Track trained model for architecture matching
        
        logger.info(f"Initialized ExperimentRunner with config: {config}")
    
    def generate_scms(self) -> List[Tuple[int, Dict]]:
        """
        Generate SCMs for all sizes.
        
        Returns:
            List of (size, scm) tuples
        """
        scms = []
        
        for size in self.config.scm_sizes:
            for i in range(self.config.n_scms_per_size):
                # Rotate through structure types
                structure_idx = i % len(self.config.structure_types)
                structure_type = self.config.structure_types[structure_idx]
                
                # Generate SCM
                scm = self.scm_factory.create_variable_scm(
                    num_variables=size,
                    structure_type=structure_type,
                    edge_density=self.config.edge_density
                )
                
                scms.append((size, scm))
                
        logger.info(f"Generated {len(scms)} SCMs across sizes {self.config.scm_sizes}")
        return scms
    
    def load_method(self, method_config: MethodConfig) -> Tuple[Callable, Optional[Any]]:
        """
        Load a method (policy and optionally surrogate).
        
        Args:
            method_config: Method configuration
            
        Returns:
            Tuple of (acquisition_function, surrogate_model)
        """
        # Load policy
        if method_config.policy_type == 'random':
            acquisition_fn = ModelLoader.load_baseline('random', seed=self.config.seed)
        elif method_config.policy_type == 'oracle':
            # Oracle will be created per-SCM
            acquisition_fn = None
        elif method_config.policy_type == 'untrained':
            # Create untrained policy matching trained architecture if available
            acquisition_fn = ModelLoader.create_untrained_policy(
                seed=self.config.seed,
                checkpoint_reference=self.trained_checkpoint_path
            )
        elif method_config.policy_type == 'checkpoint':
            if method_config.policy_checkpoint:
                try:
                    acquisition_fn = ModelLoader.load_policy(
                        method_config.policy_checkpoint,
                        seed=self.config.seed
                    )
                    # Store checkpoint path for architecture matching
                    self.trained_checkpoint_path = method_config.policy_checkpoint
                except Exception as e:
                    logger.warning(f"Failed to load policy checkpoint: {e}")
                    logger.info("Using untrained policy instead")
                    acquisition_fn = ModelLoader.create_untrained_policy(
                        seed=self.config.seed,
                        checkpoint_reference=method_config.policy_checkpoint
                    )
            else:
                acquisition_fn = ModelLoader.create_untrained_policy(
                    seed=self.config.seed,
                    checkpoint_reference=self.trained_checkpoint_path
                )
        else:
            raise ValueError(f"Unknown policy type: {method_config.policy_type}")
        
        # Load surrogate if needed
        surrogate_model = None
        if method_config.use_surrogate and method_config.surrogate_checkpoint:
            try:
                surrogate_params, surrogate_arch = ModelLoader.load_surrogate(
                    method_config.surrogate_checkpoint
                )
                surrogate_model = (surrogate_params, surrogate_arch)
            except Exception as e:
                logger.warning(f"Failed to load surrogate: {e}")
        
        return acquisition_fn, surrogate_model
    
    def run_single_experiment(self,
                            scm: Dict,
                            method_config: MethodConfig,
                            acquisition_fn: Callable,
                            surrogate_model: Optional[Any]) -> Dict[str, Any]:
        """
        Run a single experiment (one method on one SCM).
        
        Args:
            scm: SCM to use
            method_config: Method configuration
            acquisition_fn: Acquisition function
            surrogate_model: Optional surrogate model
            
        Returns:
            Dictionary with experiment results
        """
        # Setup
        variables = get_variables(scm)  # Returns FrozenSet[str]
        variables = list(variables)     # Convert to list for indexing compatibility
        target_var = get_target(scm)
        mapper = VariableMapper(variables, target_variable=target_var)
        true_parents = {var: get_parents(scm, var) for var in variables}
        
        # Create oracle acquisition if needed
        if method_config.policy_type == 'oracle':
            acquisition_fn = ModelLoader.load_baseline('oracle', scm=scm)
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        obs_samples = sample_from_linear_scm(
            scm, 
            n_samples=self.config.n_observational_samples,
            seed=int(self.rng_key[0]) % 1000000
        )
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Track metrics
        trajectory = []
        graph_predictions = []
        
        # Run interventions
        for step in range(self.config.n_interventions):
            # Convert buffer to tensor
            tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
            
            # Get surrogate predictions if available
            posterior = None
            if surrogate_model is not None:
                # TODO: Actually run surrogate inference here
                # For now, create dummy predictions
                posterior = {
                    'parent_probs': {var: np.random.random(len(variables)) 
                                   for var in variables}
                }
            
            # Get intervention from policy
            self.rng_key, action_key = random.split(self.rng_key)
            intervention_dict = acquisition_fn(tensor, posterior, target_var, variables)
            
            # Apply intervention - handle frozenset properly
            if isinstance(intervention_dict['targets'], frozenset):
                selected_var = list(intervention_dict['targets'])[0]
            else:
                selected_var = intervention_dict['targets'][0] if isinstance(intervention_dict['targets'], list) else intervention_dict['targets']
            intervention_value = intervention_dict['values'][selected_var]
            
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            # Sample post-intervention
            self.rng_key, sample_key = random.split(self.rng_key)
            post_samples = sample_with_intervention(
                scm, intervention, 1, seed=int(sample_key[0])
            )
            
            if post_samples:
                post_sample = post_samples[0]
                buffer.add_intervention(intervention, post_sample)
                
                # Record trajectory
                # Access values from PMap structure
                target_value = post_sample['values'][target_var]
                    
                trajectory.append({
                    'step': step,
                    'selected_var': selected_var,
                    'intervention_value': float(intervention_value),
                    'target_value': float(target_value),
                    'is_parent': selected_var in true_parents[target_var]
                })
                
                # Record graph predictions if we have a surrogate
                if posterior and 'parent_probs' in posterior:
                    graph_predictions.append(posterior['parent_probs'])
        
        # Compute final metrics
        final_target = trajectory[-1]['target_value'] if trajectory else 0.0
        best_target = min(t['target_value'] for t in trajectory) if trajectory else 0.0
        
        # Compute graph metrics if we have predictions
        graph_metrics = {}
        if graph_predictions:
            # Use last prediction
            last_pred = graph_predictions[-1]
            graph_metrics = evaluate_graph_discovery(
                true_parents, last_pred, variables, threshold=0.5
            )
        
        return {
            'method': method_config.name,
            'trajectory': trajectory,
            'final_target': final_target,
            'best_target': best_target,
            'graph_metrics': graph_metrics,
            'n_interventions': len(trajectory)
        }
    
    def run_all_experiments(self, methods: List[MethodConfig]) -> pd.DataFrame:
        """
        Run all experiments for all methods and SCMs.
        
        Args:
            methods: List of method configurations
            
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        # Pre-scan for checkpoint methods to get architecture reference
        for method_config in methods:
            if method_config.policy_type == 'checkpoint' and method_config.policy_checkpoint:
                if method_config.policy_checkpoint.exists():
                    self.trained_checkpoint_path = method_config.policy_checkpoint
                    logger.info(f"Found trained checkpoint for architecture matching: {method_config.policy_checkpoint}")
                    break
        
        # Generate SCMs
        scms = self.generate_scms()
        total_experiments = len(scms) * len(methods)
        
        logger.info(f"Running {total_experiments} experiments...")
        
        for method_config in methods:
            logger.info(f"Loading method: {method_config.name}")
            acquisition_fn, surrogate_model = self.load_method(method_config)
            
            for scm_idx, (size, scm) in enumerate(scms):
                logger.debug(f"Running {method_config.name} on SCM {scm_idx+1}/{len(scms)} (size={size})")
                
                try:
                    result = self.run_single_experiment(
                        scm, method_config, acquisition_fn, surrogate_model
                    )
                    
                    # Add metadata
                    result['scm_size'] = size
                    result['scm_idx'] = scm_idx
                    result['scm_type'] = scm.get('metadata', {}).get('structure', 'unknown')
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed experiment: {method_config.name} on SCM {scm_idx}: {e}")
                    # Record failure
                    all_results.append({
                        'method': method_config.name,
                        'scm_size': size,
                        'scm_idx': scm_idx,
                        'error': str(e),
                        'final_target': np.nan,
                        'best_target': np.nan
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        logger.info(f"Completed {len(all_results)} experiments")
        
        return df
    
    def aggregate_results(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Aggregate results by method and SCM size.
        
        Args:
            df: Raw results DataFrame
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregated = {}
        
        # Filter out failed experiments
        valid_df = df[~df['final_target'].isna()].copy()
        
        # Aggregate by method and size
        metrics_to_agg = ['final_target', 'best_target']
        
        # Add graph metrics if available
        if 'graph_metrics' in valid_df.columns:
            # Extract graph metrics into separate columns
            for idx, row in valid_df.iterrows():
                if isinstance(row['graph_metrics'], dict):
                    for key, value in row['graph_metrics'].items():
                        valid_df.loc[idx, f'graph_{key}'] = value
            
            # Add graph metrics to aggregation
            graph_metric_cols = [col for col in valid_df.columns if col.startswith('graph_') and col != 'graph_metrics']
            metrics_to_agg.extend(graph_metric_cols)
        
        # Group by method and size
        aggregated['by_method_and_size'] = valid_df.groupby(['method', 'scm_size'])[metrics_to_agg].agg(['mean', 'std'])
        
        # Group by method only
        aggregated['by_method'] = valid_df.groupby('method')[metrics_to_agg].agg(['mean', 'std'])
        
        # Group by size only  
        aggregated['by_size'] = valid_df.groupby('scm_size')[metrics_to_agg].agg(['mean', 'std'])
        
        return aggregated