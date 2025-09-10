#!/usr/bin/env python3
"""
Model Evaluation Suite.

Comprehensive evaluation of trained policy and surrogate models.
Computes metrics for structure learning (F1), intervention quality (regret),
and causal discovery (parent selection rate).

Example usage:
    # Evaluate single checkpoint pair
    python full_evaluation.py \\
        --policy-path policy.pkl \\
        --surrogate-path surrogate.pkl \\
        --num-episodes 30
    
    # Evaluate with baselines for comparison
    python full_evaluation.py \\
        --policy-path policy.pkl \\
        --surrogate-path surrogate.pkl \\
        --baselines
    
    # Evaluate training progression across checkpoints
    python full_evaluation.py \\
        --checkpoint-dir checkpoints/ \\
        --surrogate-path surrogate.pkl
    
    # Generate plots of results
    python full_evaluation.py \\
        --policy-path policy.pkl \\
        --surrogate-path surrogate.pkl \\
        --plot

Output:
    - JSON file with detailed metrics per episode
    - Summary statistics (mean, std) across episodes
    - Optional: Comparison with random/oracle baselines
    - Optional: Trajectory plots showing learning progression
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values

# Import statistical features calculation if available
try:
    # Import from training script where it's defined
    sys.path.append(str(project_root / 'experiments' / 'policy-only-training'))
    from train_grpo_ground_truth_rotation import calculate_statistical_features
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("[Warning] Statistical features calculation not available")

# Import baselines - use absolute import when running as script
try:
    from .baselines import create_baseline
except ImportError:
    from baselines import create_baseline


class MetricsTracker:
    """Track and compute evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episodes = []
        self.current_episode = None
    
    def start_episode(self, scm_info: Dict[str, Any]):
        """Start tracking a new episode."""
        self.current_episode = {
            'scm_info': scm_info,
            'interventions': [],
            'parent_probabilities': [],
            'target_values': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'parent_selections': [],
            'cumulative_regret': [],
            'simple_regret': [],
            'best_target_so_far': float('inf')
        }
    
    def add_intervention(self, intervention_info: Dict[str, Any]):
        """Add intervention results to current episode."""
        if self.current_episode is None:
            raise ValueError("No episode started")
        
        self.current_episode['interventions'].append(intervention_info['intervention'])
        self.current_episode['parent_probabilities'].append(intervention_info['parent_probs'])
        self.current_episode['target_values'].append(intervention_info['target_value'])
        self.current_episode['parent_selections'].append(intervention_info['is_parent'])
        
        # Update best target (skip if inf for data-only evaluation)
        if intervention_info['target_value'] != float('inf'):
            if intervention_info['target_value'] < self.current_episode['best_target_so_far']:
                self.current_episode['best_target_so_far'] = intervention_info['target_value']
        
        # Calculate structure learning metrics
        true_parents = intervention_info['true_parents']
        predicted_parents = {var for var, prob in intervention_info['parent_probs'].items() 
                           if prob > 0.5}
        
        tp = len(true_parents & predicted_parents)
        fp = len(predicted_parents - true_parents)
        fn = len(true_parents - predicted_parents)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        self.current_episode['f1_scores'].append(f1)
        self.current_episode['precision_scores'].append(precision)
        self.current_episode['recall_scores'].append(recall)
        
        # Calculate regret metrics (skip if inf for data-only evaluation)
        if intervention_info['target_value'] != float('inf'):
            optimal_value = intervention_info.get('optimal_value', 0.0)
            current_regret = intervention_info['target_value'] - optimal_value
            
            if len(self.current_episode['cumulative_regret']) == 0:
                self.current_episode['cumulative_regret'].append(current_regret)
            else:
                self.current_episode['cumulative_regret'].append(
                    self.current_episode['cumulative_regret'][-1] + current_regret
                )
            
            self.current_episode['simple_regret'].append(
                self.current_episode['best_target_so_far'] - optimal_value
            )
        else:
            # For data-only evaluation, skip regret calculations
            self.current_episode['cumulative_regret'].append(0.0)
            self.current_episode['simple_regret'].append(0.0)
    
    def end_episode(self):
        """Finalize current episode and add to episodes list."""
        if self.current_episode is None:
            raise ValueError("No episode to end")
        
        # Calculate episode summary statistics
        # Handle data-only evaluation (no interventions performed)
        valid_targets = [t for t in self.current_episode['target_values'] if t != float('inf')]
        valid_selections = [s for i, s in enumerate(self.current_episode['parent_selections']) 
                          if self.current_episode['interventions'][i] != (None, None)]
        
        self.current_episode['summary'] = {
            'final_f1': self.current_episode['f1_scores'][-1] if self.current_episode['f1_scores'] else 0.0,
            'best_f1': max(self.current_episode['f1_scores']) if self.current_episode['f1_scores'] else 0.0,
            'mean_f1': np.mean(self.current_episode['f1_scores']) if self.current_episode['f1_scores'] else 0.0,
            'parent_selection_rate': np.mean(valid_selections) if valid_selections else 0.0,
            'final_target': valid_targets[-1] if valid_targets else float('inf'),
            'best_target': min(valid_targets) if valid_targets else float('inf'),
            'target_improvement': (valid_targets[0] - valid_targets[-1]) if len(valid_targets) > 1 else 0.0,
            'final_cumulative_regret': self.current_episode['cumulative_regret'][-1] if self.current_episode['cumulative_regret'] else 0.0,
            'final_simple_regret': self.current_episode['simple_regret'][-1] if self.current_episode['simple_regret'] else 0.0
        }
        
        self.episodes.append(self.current_episode)
        self.current_episode = None
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all episodes."""
        if not self.episodes:
            return {}
        
        aggregate = defaultdict(list)
        
        for episode in self.episodes:
            for key, value in episode['summary'].items():
                aggregate[key].append(value)
        
        # Calculate mean and std for each metric
        result = {}
        for key, values in aggregate.items():
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_std'] = float(np.std(values))
            result[f'{key}_min'] = float(np.min(values))
            result[f'{key}_max'] = float(np.max(values))
        
        return result


def load_models(policy_path: Path, surrogate_path: Path) -> Tuple[Any, Any, Any, Any, Dict]:
    """Load both policy and surrogate models with metadata."""
    
    # Load surrogate
    surrogate_checkpoint = load_checkpoint(surrogate_path)
    surrogate_params = surrogate_checkpoint['params']
    surrogate_architecture = surrogate_checkpoint.get('architecture', {})
    
    # Create surrogate network
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=surrogate_architecture.get('hidden_dim', 128),
            num_heads=surrogate_architecture.get('num_heads', 8),
            num_layers=surrogate_architecture.get('num_layers', 8),
            key_size=surrogate_architecture.get('key_size', 32),
            dropout=surrogate_architecture.get('dropout', 0.1) if is_training else 0.0,
            use_temperature_scaling=True,
            temperature_init=0.0
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    
    # Load policy
    policy_checkpoint = load_checkpoint(policy_path)
    policy_params = policy_checkpoint['params']
    policy_architecture = policy_checkpoint.get('architecture', {})
    
    # Detect policy type and create appropriate network
    architecture_type = policy_architecture.get('architecture_type', 'quantile')
    
    # Check if using statistical features
    using_stats = architecture_type == 'quantile_with_stats' or 'quantile_with_stats' in str(policy_architecture)
    
    # Create policy network
    if using_stats:
        from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy_with_stats
        stats_weight = policy_architecture.get('stats_weight', 0.1)
        print(f"[Evaluation] Using quantile_with_stats policy with stats_weight={stats_weight}")
        policy_fn = create_quantile_policy_with_stats(
            hidden_dim=policy_architecture.get('hidden_dim', 256),
            stats_weight=stats_weight
        )
    else:
        from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy
        print(f"[Evaluation] Using standard quantile policy")
        policy_fn = create_quantile_policy(
            hidden_dim=policy_architecture.get('hidden_dim', 256)
        )
    
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Extract training metadata
    metadata = {
        'policy_iteration': policy_checkpoint.get('iteration', -1),
        'policy_architecture': policy_architecture,
        'surrogate_architecture': surrogate_architecture,
        'policy_path': str(policy_path),
        'surrogate_path': str(surrogate_path),
        'using_statistical_features': using_stats,
        'stats_weight': policy_architecture.get('stats_weight', 0.1) if using_stats else None
    }
    
    return policy_net, policy_params, surrogate_net, surrogate_params, metadata


def evaluate_episode(
    policy_net, policy_params,
    surrogate_net, surrogate_params,
    scm, metrics_tracker: MetricsTracker,
    num_interventions: int = 30,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    max_history_size: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
    baseline: Optional[Any] = None,
    use_oracle_surrogate: bool = False,
    use_statistical_features: bool = False
) -> Dict[str, Any]:
    """
    Evaluate one episode with comprehensive metrics tracking.
    
    Args:
        initial_observations: Number of observational samples to start with
        initial_interventions: Number of random interventional samples to start with
        num_interventions: Number of policy-driven interventions to perform
        max_history_size: Maximum number of samples for surrogate (None = use all)
    """
    
    # Get SCM information
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    variable_ranges = scm.get('metadata', {}).get('variable_ranges', {})
    
    # Extract parent coefficients from metadata
    all_coefficients = scm.get('metadata', {}).get('coefficients', {})
    parent_coefficients = {}
    for edge_str, coeff in all_coefficients.items():
        # Parse edge string (could be tuple string representation)
        if isinstance(edge_str, str) and ',' in edge_str:
            # Handle string representation of tuple like "('X1', 'X2')"
            edge_str = edge_str.strip('()')
            parts = [p.strip().strip("'").strip('"') for p in edge_str.split(',')]
            if len(parts) == 2:
                from_var, to_var = parts
                if to_var == target_var and from_var in true_parents:
                    parent_coefficients[from_var] = coeff
        elif isinstance(edge_str, tuple) and len(edge_str) == 2:
            from_var, to_var = edge_str
            if to_var == target_var and from_var in true_parents:
                parent_coefficients[from_var] = coeff
    
    # Log detailed SCM information for debugging
    import logging
    logger = logging.getLogger(__name__)
    structure_type = scm.get('metadata', {}).get('structure_type', 'unknown')
    logger.info(f"\n=== Episode Debug Info ===")
    logger.info(f"Structure: {structure_type} with {len(variables)} variables")
    logger.info(f"Variables: {variables}")
    logger.info(f"Target: {target_var}")
    logger.info(f"True parents of target: {true_parents}")
    logger.info(f"Number of parents: {len(true_parents)}")
    
    # Log coefficient information
    if parent_coefficients:
        logger.info(f"Parent coefficients: {parent_coefficients}")
        coeff_magnitudes = [abs(c) for c in parent_coefficients.values()]
        logger.info(f"Coefficient magnitudes - Min: {min(coeff_magnitudes):.3f}, "
                   f"Max: {max(coeff_magnitudes):.3f}, "
                   f"Mean: {np.mean(coeff_magnitudes):.3f}")
    
    # Validate that target is not a root node
    if len(true_parents) == 0:
        logger.error(f"ERROR: Target {target_var} has NO PARENTS (root node)!")
        logger.error(f"This should not happen - targets must have at least one parent")
        logger.error(f"SCM metadata: {scm.get('metadata', {})}")
    
    # Calculate optimal value (for regret calculation)
    # This would require knowledge of the true SCM - simplified here
    optimal_value = -5.0  # Placeholder - in reality would need to solve the SCM
    
    # Start episode tracking
    coeff_stats = {}
    if parent_coefficients:
        coeff_magnitudes = [abs(c) for c in parent_coefficients.values()]
        coeff_stats = {
            'min_coefficient': min(coeff_magnitudes),
            'max_coefficient': max(coeff_magnitudes),
            'mean_coefficient': float(np.mean(coeff_magnitudes)),
            'parent_coefficients': parent_coefficients
        }
    
    metrics_tracker.start_episode({
        'num_variables': len(variables),
        'target': target_var,
        'true_parents': list(true_parents),
        'structure_type': scm.get('metadata', {}).get('structure_type', 'unknown'),
        **coeff_stats
    })
    
    if verbose:
        print(f"\n📊 Episode: {len(variables)} vars, target={target_var}, parents={true_parents}")
        if parent_coefficients:
            coeff_magnitudes = [abs(c) for c in parent_coefficients.values()]
            print(f"   Parent coefficients - Min: {min(coeff_magnitudes):.3f}, "
                  f"Max: {max(coeff_magnitudes):.3f}, Mean: {np.mean(coeff_magnitudes):.3f}")
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    
    # Debug: Print data configuration
    
    # Add observational samples
    rng_key, sample_key = random.split(rng_key)
    samples = sample_from_linear_scm(scm, n_samples=initial_observations, seed=int(sample_key[0]))
    for sample in samples:
        buffer.add_observation(sample)
    
    
    # Add initial random interventions if specified
    if initial_interventions > 0:
        # Get non-target variables for intervention
        non_target_vars = [v for v in variables if v != target_var]
        
        for i in range(initial_interventions):
            # Randomly select a variable to intervene on
            rng_key, var_key = random.split(rng_key)
            selected_var_idx = random.choice(var_key, len(non_target_vars))
            selected_var = non_target_vars[int(selected_var_idx)]
            
            # Sample intervention value from variable's range
            var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
            rng_key, value_key = random.split(rng_key)
            # Use uniform distribution across the range
            intervened_value = float(random.uniform(value_key, 
                                                   minval=var_range[0], 
                                                   maxval=var_range[1]))
            
            # Apply intervention and sample
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervened_value}
            )
            
            samples = sample_with_intervention(
                scm, intervention, n_samples=1,
                seed=seed + i + 5000  # Different seed offset for initial interventions
            )
            
            # Add to buffer
            for sample in samples:
                buffer.add_intervention({selected_var: intervened_value}, sample)
    
    
    # Create surrogate wrapper
    if use_oracle_surrogate:
        # Use oracle surrogate with perfect parent predictions
        from baselines import OracleSurrogateBaseline
        oracle_surrogate = OracleSurrogateBaseline(scm)
        surrogate_fn = oracle_surrogate
    else:
        # Use learned surrogate
        def surrogate_fn(tensor_3ch, target_var_name, variable_list):
            """Wrapper for surrogate predictions."""
            target_idx = list(variable_list).index(target_var_name)
            rng_key_surrogate = random.PRNGKey(42)
            predictions = surrogate_net.apply(surrogate_params, rng_key_surrogate, tensor_3ch, target_idx, False)
            parent_probs = predictions.get('parent_probabilities', jnp.full(len(variable_list), 0.5))
            return {'parent_probs': parent_probs}
    
    # Special case: data-only evaluation (no interventions)
    if num_interventions == 0:
        # Get final buffer state and make predictions
        # Use all available data (no artificial limit)
        tensor_4ch, mapper, _ = buffer_to_four_channel_tensor(
            buffer, target_var, 
            surrogate_fn=surrogate_fn,
            max_history_size=max_history_size,  # Use provided limit or all data
            standardize=True
        )
        
        # Extract parent probabilities from the 4th channel
        current_parent_probs = {}
        for i, var in enumerate(mapper.variables):
            if var != target_var:
                prob = float(tensor_4ch[-1, i, 3])
                current_parent_probs[var] = prob
        
        # Calculate F1 score for structure learning
        predicted_parents = {var for var, prob in current_parent_probs.items() if prob > 0.5}
        tp = len(true_parents & predicted_parents)
        fp = len(predicted_parents - true_parents)
        fn = len(true_parents - predicted_parents)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Add single "evaluation" entry to tracker
        metrics_tracker.add_intervention({
            'intervention': (None, None),  # No intervention
            'parent_probs': current_parent_probs,
            'target_value': float('inf'),  # No target update  
            'is_parent': False,  # Not applicable
            'true_parents': true_parents,
            'optimal_value': optimal_value
        })
        
        if verbose:
            print(f"  Data-only eval: F1={f1:.3f}, Predicted parents: {predicted_parents}, True: {true_parents}")
    
    # Run interventions
    for intervention_idx in range(num_interventions):
        # Convert buffer to 4-channel tensor
        # Use all available data (no artificial limit)
        tensor_4ch, mapper, _ = buffer_to_four_channel_tensor(
            buffer, target_var, 
            surrogate_fn=surrogate_fn,
            max_history_size=max_history_size,  # Use provided limit or all data
            standardize=True
        )
        
        # Get current parent probability predictions
        current_parent_probs = {}
        for i, var in enumerate(mapper.variables):
            if var != target_var:
                prob = float(tensor_4ch[-1, i, 3])
                current_parent_probs[var] = prob
        
        # Select intervention
        if baseline is not None:
            # Use baseline for intervention selection
            selected_var, intervened_value = baseline.select_intervention(
                buffer, target_var, variables, variable_ranges,
                parent_probs=current_parent_probs
            )
        else:
            # Use policy for intervention selection
            # Get target index
            target_idx = mapper.get_index(target_var)
            
            # Calculate statistical features if needed
            if use_statistical_features and STATS_AVAILABLE:
                statistical_features = calculate_statistical_features(tensor_4ch, target_idx, mapper)
                # Call policy with statistical features
                policy_output = policy_net.apply(policy_params, tensor_4ch, target_idx, statistical_features)
            else:
                # Call policy without statistical features
                policy_output = policy_net.apply(policy_params, tensor_4ch, target_idx)
            
            # Decode quantile output
            quantile_scores = policy_output['quantile_scores']
            flat_scores = quantile_scores.flatten()
            
            # Sample from distribution
            rng_key, sample_key = random.split(rng_key)
            probs = jax.nn.softmax(flat_scores)
            selected_flat_idx = random.choice(sample_key, len(flat_scores), p=probs)
            
            # Map back to variable and quantile
            selected_var_idx = selected_flat_idx // 3
            selected_quantile_idx = selected_flat_idx % 3
            selected_var = mapper.variables[int(selected_var_idx)]
            
            # Map quantile to value
            quantile_values = {0: 0.25, 1: 0.50, 2: 0.75}
            percentile = quantile_values[int(selected_quantile_idx)]
            var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
            intervened_value = var_range[0] + (var_range[1] - var_range[0]) * percentile
            
            # Add exploration noise
            rng_key, noise_key = random.split(rng_key)
            noise = random.normal(noise_key) * 0.1
            intervened_value = float(jnp.clip(intervened_value + noise, var_range[0], var_range[1]))
        
        # Check if parent
        is_parent = selected_var in true_parents
        
        # Apply intervention
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: intervened_value}
        )
        
        samples = sample_with_intervention(
            scm, intervention, n_samples=1, 
            seed=seed + intervention_idx + 1000
        )
        
        # Get target value after intervention
        for sample in samples:
            target_value = float(get_values(sample)[target_var])
            buffer.add_intervention({selected_var: intervened_value}, sample)
        
        # Track metrics
        metrics_tracker.add_intervention({
            'intervention': (selected_var, intervened_value),
            'parent_probs': current_parent_probs,
            'target_value': target_value,
            'is_parent': is_parent,
            'true_parents': true_parents,
            'optimal_value': optimal_value
        })
        
        if verbose and (intervention_idx + 1) % 10 == 0:
            current_f1 = metrics_tracker.current_episode['f1_scores'][-1]
            print(f"  Intervention {intervention_idx+1}: F1={current_f1:.3f}, "
                  f"target={target_value:.2f}, parent_rate="
                  f"{np.mean(metrics_tracker.current_episode['parent_selections']):.2%}")
    
    # End episode
    metrics_tracker.end_episode()
    
    return metrics_tracker.episodes[-1]['summary']


def evaluate_checkpoint_pair(
    policy_path: Path,
    surrogate_path: Path,
    num_episodes: int = 10,
    num_interventions: int = 30,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    max_history_size: Optional[int] = None,
    min_parent_coefficient: Optional[float] = None,
    structure_types: List[str] = ['fork', 'chain', 'scale_free'],
    num_vars_list: List[int] = [5, 8],
    seed: int = 42,
    verbose: bool = True,
    include_baselines: bool = False,
    use_oracle_surrogate: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a policy-surrogate checkpoint pair across multiple SCMs.
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating Checkpoint Pair")
    print(f"Policy: {policy_path.name}")
    if use_oracle_surrogate:
        print(f"Surrogate: ORACLE (perfect parent predictions)")
    else:
        print(f"Surrogate: {surrogate_path.name}")
    print(f"{'='*70}")
    
    # Load models
    policy_net, policy_params, surrogate_net, surrogate_params, metadata = load_models(
        policy_path, surrogate_path
    )
    
    # Initialize metrics trackers
    metrics_tracker = MetricsTracker()
    random_tracker = MetricsTracker() if include_baselines else None
    oracle_tracker = MetricsTracker() if include_baselines else None
    
    # Create SCM factory with optional coefficient filtering
    factory = VariableSCMFactory(
        seed=seed,
        noise_scale=0.5,
        coefficient_range=(-3.0, 3.0),
        min_parent_coefficient=min_parent_coefficient,
        vary_intervention_ranges=True,
        use_output_bounds=True
    )
    
    # Evaluate across different SCM configurations
    episode_count = 0
    for structure_type in structure_types:
        for num_vars in num_vars_list:
            for episode_idx in range(num_episodes):
                episode_count += 1
                
                if verbose:
                    print(f"\n📊 Episode {episode_count}: {structure_type} with {num_vars} vars")
                
                # Create SCM
                scm = factory.create_variable_scm(
                    num_variables=num_vars,
                    structure_type=structure_type
                )
                
                # Evaluate episode with policy
                summary = evaluate_episode(
                    policy_net, policy_params,
                    surrogate_net, surrogate_params,
                    scm, metrics_tracker,
                    num_interventions=num_interventions,
                    initial_observations=initial_observations,
                    initial_interventions=initial_interventions,
                    max_history_size=max_history_size,
                    seed=seed + episode_count * 1000,
                    verbose=False,
                    use_oracle_surrogate=use_oracle_surrogate,
                    use_statistical_features=metadata.get('using_statistical_features', False)
                )
                
                if verbose:
                    print(f"  Policy: F1={summary['final_f1']:.3f}, "
                          f"Parent Rate={summary['parent_selection_rate']:.2%}, "
                          f"Target={summary['best_target']:.2f}")
                
                # Run baselines if requested
                if include_baselines:
                    # Random baseline - uses random intervention selection
                    # F1 score comes from surrogate predictions on the randomly collected data
                    random_baseline = create_baseline('random', seed=seed + episode_count * 2000)
                    random_summary = evaluate_episode(
                        policy_net, policy_params,
                        surrogate_net, surrogate_params,
                        scm, random_tracker,
                        num_interventions=num_interventions,
                        initial_observations=initial_observations,
                        initial_interventions=initial_interventions,
                        max_history_size=max_history_size,
                        seed=seed + episode_count * 1000,
                        verbose=False,
                        baseline=random_baseline,
                        use_oracle_surrogate=use_oracle_surrogate,
                        use_statistical_features=metadata.get('using_statistical_features', False)
                    )
                    
                    # Oracle baseline - uses perfect knowledge for intervention selection
                    # F1 score comes from surrogate predictions on the optimally collected data
                    oracle_baseline = create_baseline('oracle', scm=scm)
                    oracle_summary = evaluate_episode(
                        policy_net, policy_params,
                        surrogate_net, surrogate_params,
                        scm, oracle_tracker,
                        num_interventions=num_interventions,
                        initial_observations=initial_observations,
                        initial_interventions=initial_interventions,
                        max_history_size=max_history_size,
                        seed=seed + episode_count * 1000,
                        verbose=False,
                        baseline=oracle_baseline,
                        use_oracle_surrogate=use_oracle_surrogate,
                        use_statistical_features=metadata.get('using_statistical_features', False)
                    )
                    
                    if verbose:
                        print(f"  Random: F1={random_summary['final_f1']:.3f} (surrogate predictions), "
                              f"Parent Rate={random_summary['parent_selection_rate']:.2%} (actual selections)")
                        print(f"  Oracle: F1={oracle_summary['final_f1']:.3f} (surrogate predictions), "
                              f"Parent Rate={oracle_summary['parent_selection_rate']:.2%} (actual selections)")
    
    # Get aggregate metrics
    aggregate_metrics = metrics_tracker.get_aggregate_metrics()
    
    # Add metadata
    result = {
        'metadata': metadata,
        'evaluation_config': {
            'num_episodes': episode_count,
            'num_interventions': num_interventions,
            'structure_types': structure_types,
            'num_vars_list': num_vars_list
        },
        'aggregate_metrics': aggregate_metrics,
        'episodes': metrics_tracker.episodes
    }
    
    # Add baseline results if computed
    if include_baselines:
        result['baselines'] = {
            'random': {
                'aggregate_metrics': random_tracker.get_aggregate_metrics(),
                'episodes': random_tracker.episodes
            },
            'oracle': {
                'aggregate_metrics': oracle_tracker.get_aggregate_metrics(),
                'episodes': oracle_tracker.episodes
            }
        }
    
    return result


def evaluate_training_progression(
    checkpoint_dir: Path,
    surrogate_path: Path,
    checkpoint_iterations: Optional[List[int]] = None,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    max_history_size: Optional[int] = None,
    min_parent_coefficient: Optional[float] = None,
    **eval_kwargs
) -> pd.DataFrame:
    """
    Evaluate multiple checkpoints from training to understand performance progression.
    """
    
    results = []
    
    # Find all checkpoints
    if checkpoint_iterations is None:
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
        checkpoint_iterations = []
        for ckpt_file in checkpoint_files:
            try:
                iteration = int(ckpt_file.stem.split('_')[-1])
                checkpoint_iterations.append(iteration)
            except:
                continue
    
    print(f"\n📈 Evaluating training progression across {len(checkpoint_iterations)} checkpoints")
    
    for iteration in checkpoint_iterations:
        policy_path = checkpoint_dir / f"checkpoint_{iteration}.pkl"
        
        if not policy_path.exists():
            print(f"⚠️  Checkpoint not found: {policy_path}")
            continue
        
        print(f"\n🔄 Evaluating iteration {iteration}")
        
        # Evaluate this checkpoint
        eval_result = evaluate_checkpoint_pair(
            policy_path=policy_path,
            surrogate_path=surrogate_path,
            initial_observations=initial_observations,
            initial_interventions=initial_interventions,
            max_history_size=max_history_size,
            min_parent_coefficient=min_parent_coefficient,
            verbose=False,
            **eval_kwargs
        )
        
        # Extract key metrics
        metrics = eval_result['aggregate_metrics']
        row = {
            'iteration': iteration,
            'final_f1_mean': metrics.get('final_f1_mean', 0.0),
            'final_f1_std': metrics.get('final_f1_std', 0.0),
            'parent_selection_rate_mean': metrics.get('parent_selection_rate_mean', 0.0),
            'parent_selection_rate_std': metrics.get('parent_selection_rate_std', 0.0),
            'best_target_mean': metrics.get('best_target_mean', float('inf')),
            'best_target_std': metrics.get('best_target_std', 0.0),
            'final_cumulative_regret_mean': metrics.get('final_cumulative_regret_mean', 0.0),
            'final_simple_regret_mean': metrics.get('final_simple_regret_mean', 0.0)
        }
        results.append(row)
        
        # Print summary
        print(f"  F1: {row['final_f1_mean']:.3f} ± {row['final_f1_std']:.3f}")
        print(f"  Parent Rate: {row['parent_selection_rate_mean']:.2%} ± "
              f"{row['parent_selection_rate_std']:.2%}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('iteration')
    
    # Analyze progression
    print(f"\n📊 Training Progression Analysis:")
    print(f"{'='*60}")
    
    if len(df) > 1:
        # Check for performance degradation
        early_f1 = df.iloc[0]['final_f1_mean']
        late_f1 = df.iloc[-1]['final_f1_mean']
        
        print(f"F1 Score:")
        print(f"  Early (iter {df.iloc[0]['iteration']}): {early_f1:.3f}")
        print(f"  Late (iter {df.iloc[-1]['iteration']}): {late_f1:.3f}")
        print(f"  Change: {late_f1 - early_f1:+.3f}")
        
        if late_f1 < early_f1 - 0.1:
            print("  ⚠️ SIGNIFICANT PERFORMANCE DEGRADATION DETECTED!")
        
        # Parent selection analysis
        early_parent = df.iloc[0]['parent_selection_rate_mean']
        late_parent = df.iloc[-1]['parent_selection_rate_mean']
        
        print(f"\nParent Selection Rate:")
        print(f"  Early: {early_parent:.2%}")
        print(f"  Late: {late_parent:.2%}")
        print(f"  Change: {late_parent - early_parent:+.2%}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation with metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single checkpoint
  %(prog)s --policy-path model.pkl --surrogate-path surrogate.pkl
  
  # Include baseline comparisons
  %(prog)s --policy-path model.pkl --surrogate-path surrogate.pkl --baselines
  
  # Evaluate checkpoint progression
  %(prog)s --checkpoint-dir checkpoints/ --surrogate-path surrogate.pkl
  
Metrics computed:
  - F1 Score: Structure learning accuracy
  - Parent Selection Rate: Percentage of interventions on true parents
  - Target Value: Optimization objective achieved
  - Regret: Difference from optimal intervention strategy
        """
    )
    
    # Model paths
    parser.add_argument('--policy-path', type=Path,
                       help='Path to single policy checkpoint')
    parser.add_argument('--checkpoint-dir', type=Path,
                       help='Path to directory with multiple checkpoints')
    parser.add_argument('--surrogate-path', type=Path, required=True,
                       help='Path to surrogate checkpoint')
    
    # Evaluation config
    parser.add_argument('--num-episodes', type=int, default=30,
                       help='Number of episodes per configuration (more episodes = more robust statistics)')
    parser.add_argument('--num-interventions', type=int, default=30,
                       help='Number of interventions per episode')
    parser.add_argument('--initial-observations', type=int, default=20,
                       help='Number of initial observational samples')
    parser.add_argument('--initial-interventions', type=int, default=10,
                       help='Number of initial random interventional samples')
    parser.add_argument('--max-history-size', type=int, default=None,
                       help='Maximum number of samples for surrogate (None = use all available data)')
    parser.add_argument('--min-parent-coefficient', type=float, default=None,
                       help='Minimum absolute coefficient value for parent edges (e.g., 0.5 to filter weak effects)')
    parser.add_argument('--structures', nargs='+', 
                       default=['fork', 'chain'],
                       choices=['fork', 'true_fork', 'chain', 'collider', 'mixed', 'random', 'scale_free', 'two_layer'],
                       help='SCM structure types to test (fork, true_fork, chain, collider, mixed, random, scale_free, two_layer)')
    parser.add_argument('--num-vars', nargs='+', type=int, default=[5, 8],
                       help='Number of variables to test')
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation_results'),
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--plot', action='store_true',
                       help='Generate trajectory plots')
    parser.add_argument('--baselines', action='store_true',
                       help='Include random and oracle baselines')
    parser.add_argument('--oracle-surrogate', action='store_true',
                       help='Use oracle surrogate with perfect parent predictions instead of learned surrogate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL EVALUATION WITH METRICS")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.checkpoint_dir:
        # Evaluate training progression
        print(f"\n📊 Evaluating training progression from: {args.checkpoint_dir}")
        
        df = evaluate_training_progression(
            checkpoint_dir=args.checkpoint_dir,
            surrogate_path=args.surrogate_path,
            num_episodes=args.num_episodes,
            num_interventions=args.num_interventions,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            max_history_size=args.max_history_size,
            min_parent_coefficient=args.min_parent_coefficient,
            structure_types=args.structures,
            num_vars_list=args.num_vars,
            seed=args.seed
        )
        
        # Save results
        csv_path = args.output_dir / f"progression_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # Display summary
        print(f"\n📋 Summary Table:")
        print(df[['iteration', 'final_f1_mean', 'parent_selection_rate_mean', 
                 'best_target_mean']].to_string(index=False))
        
        # Generate plots if requested
        if args.plot:
            from plotting_utils import plot_training_progression
            plot_training_progression(csv_path, args.output_dir, show_plots=True)
        
    elif args.policy_path:
        # Evaluate single checkpoint
        print(f"\n📊 Evaluating single checkpoint: {args.policy_path}")
        
        result = evaluate_checkpoint_pair(
            policy_path=args.policy_path,
            surrogate_path=args.surrogate_path,
            num_episodes=args.num_episodes,
            num_interventions=args.num_interventions,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            max_history_size=args.max_history_size,
            min_parent_coefficient=args.min_parent_coefficient,
            structure_types=args.structures,
            num_vars_list=args.num_vars,
            seed=args.seed,
            include_baselines=args.baselines,
            use_oracle_surrogate=args.oracle_surrogate
        )
        
        # Save results
        json_path = args.output_dir / f"evaluation_{timestamp}.json"
        
        # Convert to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, jnp.ndarray)):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        with open(json_path, 'w') as f:
            json.dump(convert_to_serializable(result), f, indent=2)
        print(f"\n💾 Results saved to: {json_path}")
        
        # Display summary with clear explanations
        metrics = result['aggregate_metrics']
        print(f"\n📋 Evaluation Summary:")
        print(f"\n  Policy (using trained GRPO model for intervention selection):")
        print(f"    F1 Score: {metrics.get('final_f1_mean', 0):.3f} ± "
              f"{metrics.get('final_f1_std', 0):.3f}  [Surrogate's structure prediction accuracy]")
        print(f"    Parent Selection: {metrics.get('parent_selection_rate_mean', 0):.2%} ± "
              f"{metrics.get('parent_selection_rate_std', 0):.2%}  [% of interventions on true parents]")
        print(f"    Best Target: {metrics.get('best_target_mean', 0):.2f} ± "
              f"{metrics.get('best_target_std', 0):.2f}  [Best target value achieved]")
        
        if args.baselines and 'baselines' in result:
            print(f"\n  📊 Baseline Comparisons:")
            print(f"  (Note: F1 scores measure surrogate's predictions on data collected by each method)")
            
            # Get typical SCM info from first episode
            if result['episodes'] and result['episodes'][0]:
                scm_info = result['episodes'][0].get('scm_info', {})
                num_parents = len(scm_info.get('true_parents', []))
                num_vars = scm_info.get('num_variables', 0)
                # Random selection from non-target variables
                # If there are P parents among (N-1) non-target variables
                # The probability of selecting a parent is P/(N-1)
                if num_vars > 1 and num_parents > 0:
                    expected_random_rate = num_parents / (num_vars - 1)
                elif num_parents == 0:
                    expected_random_rate = 0.0  # No parents to select
                else:
                    expected_random_rate = 0.33  # Default estimate
            else:
                expected_random_rate = 0.33  # Default estimate
            
            print(f"\n  Random Baseline (uniform random intervention selection):")
            random_metrics = result['baselines']['random']['aggregate_metrics']
            print(f"    F1 Score: {random_metrics.get('final_f1_mean', 0):.3f} ± "
                  f"{random_metrics.get('final_f1_std', 0):.3f}  [Surrogate accuracy on random data]")
            print(f"    Parent Selection: {random_metrics.get('parent_selection_rate_mean', 0):.2%}  "
                  f"[Expected: ~{expected_random_rate:.1%} for uniform random]")
            print(f"    Best Target: {random_metrics.get('best_target_mean', 0):.2f}")
            
            print(f"\n  Oracle Baseline (perfect knowledge of causal structure):")
            oracle_metrics = result['baselines']['oracle']['aggregate_metrics']
            print(f"    F1 Score: {oracle_metrics.get('final_f1_mean', 0):.3f} ± "
                  f"{oracle_metrics.get('final_f1_std', 0):.3f}  [Surrogate accuracy on optimal data]")
            print(f"    Parent Selection: {oracle_metrics.get('parent_selection_rate_mean', 0):.2%}  "
                  f"[Should be 100% - oracle always selects parents]")
            print(f"    Best Target: {oracle_metrics.get('best_target_mean', 0):.2f}  [Oracle's optimization result]")
        
        # Generate plots if requested
        if args.plot:
            from plotting_utils import plot_evaluation_trajectories
            plot_evaluation_trajectories(json_path, args.output_dir, show_plots=True)
    
    else:
        print("❌ Please provide either --policy-path or --checkpoint-dir")
        return 1
    
    print(f"\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())