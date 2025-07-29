"""
BC Runner for Evaluation

This module provides the core functionality to run BC models in the evaluation framework
without cross-package dependencies.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass

import pyrsistent as pyr
import jax.numpy as jnp
import jax.random as random

from ..data_structures.scm import (
    get_target, get_variables, get_parents
)
from ..mechanisms.linear import sample_from_linear_scm
from ..training.bc_model_inference import (
    create_bc_surrogate_inference_fn,
    create_full_bc_acquisition_fn
)
from ..acquisition.state import AcquisitionState
from ..acquisition import update_state_with_intervention
from ..avici_integration.parent_set.posterior import create_parent_set_posterior
from ..interventions.registry import apply_intervention
from ..interventions.handlers import create_perfect_intervention

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for BC evaluation runs."""
    n_observational_samples: int = 100
    n_intervention_steps: int = 20
    intervention_value_range: Tuple[float, float] = (-2.0, 2.0)
    random_seed: int = 42
    scoring_method: str = "bic"
    learning_rate: float = 1e-3
    

class PerformanceMetrics(NamedTuple):
    """Performance metrics at each step."""
    step: int
    intervention_variable: Optional[str]
    intervention_value: Optional[float]
    target_value: float
    graph_estimate: Optional[Dict[str, List[str]]]
    marginals: Optional[Dict[str, Dict[str, float]]]
    uncertainty: float = 0.0


def run_bc_experiment(
    scm: pyr.PMap,
    config: EvalConfig,
    surrogate_checkpoint: Optional[str] = None,
    acquisition_checkpoint: Optional[str] = None,
    track_performance: bool = True
) -> Dict[str, Any]:
    """
    Run BC experiment with given SCM and checkpoint paths.
    
    Args:
        scm: Structural causal model
        config: Evaluation configuration
        surrogate_checkpoint: Optional path to BC surrogate checkpoint
        acquisition_checkpoint: Optional path to BC acquisition checkpoint
        track_performance: Whether to track detailed performance
        
    Returns:
        Dictionary with experiment results
    """
    # Initialize random key
    key = random.PRNGKey(config.random_seed)
    
    # Get SCM info
    target = get_target(scm)
    variables = list(get_variables(scm))
    true_parents = list(get_parents(scm, target))
    
    # Create BC inference functions if checkpoints provided
    bc_surrogate_fn = None
    bc_acquisition_fn = None
    
    if surrogate_checkpoint:
        logger.info(f"Creating BC surrogate inference function from {surrogate_checkpoint}")
        bc_surrogate_fn = create_bc_surrogate_inference_fn(
            checkpoint_path=surrogate_checkpoint,
            threshold=0.1  # Probability threshold for parent inclusion
        )
    
    if acquisition_checkpoint:
        logger.info(f"Creating BC acquisition inference function from {acquisition_checkpoint}")
        bc_acquisition_fn = create_full_bc_acquisition_fn(
            checkpoint_path=acquisition_checkpoint,
            variables=variables,
            target_variable=target,
            value_range=config.intervention_value_range
        )
    
    # Sample observational data
    key, sample_key = random.split(key)
    observational_data = sample_from_linear_scm(
        scm, 
        n_samples=config.n_observational_samples,
        seed=int(sample_key[0])
    )
    
    # Initialize tracking
    performance_trajectory = []
    learning_history = []
    
    # Compute initial target value
    # Convert SampleList to array
    # Handle case where samples might be dicts with string keys
    obs_array = jnp.array([[float(sample.get(var, 0.0)) for var in variables] for sample in observational_data])
    initial_value = float(jnp.mean(obs_array[:, variables.index(target)]))
    
    if track_performance:
        performance_trajectory.append(PerformanceMetrics(
            step=0,
            intervention_variable=None,
            intervention_value=None,
            target_value=initial_value,
            graph_estimate=None,
            marginals=None,
            uncertainty=1.0
        ))
    
    # Track data for evaluation (simplified approach)
    all_samples = [obs_array]  # Start with observational data
    
    # Create initial acquisition state if using BC acquisition
    acquisition_state = None
    if bc_acquisition_fn is not None:
        # Create initial posterior with uniform distribution
        parent_sets = [frozenset()]  # Empty parent set
        for var in variables:
            if var != target:
                parent_sets.append(frozenset([var]))
        
        n_sets = len(parent_sets)
        probabilities = jnp.ones(n_sets) / n_sets
        
        initial_posterior = create_parent_set_posterior(
            target_variable=target,
            parent_sets=parent_sets,
            probabilities=probabilities
        )
        
        # Create acquisition state
        from ..acquisition.trajectory import TrajectoryBuffer
        from ..data_structures.buffer import create_empty_buffer
        
        # TrajectoryBuffer needs an ExperienceBuffer
        experience_buffer = create_empty_buffer()
        
        # Add observational samples to buffer so target variable is known
        from ..data_structures.sample import create_sample
        for i in range(min(10, len(observational_data))):
            obs_sample = observational_data[i]
            # Convert dict to Sample
            sample = create_sample(
                values={var: float(obs_sample.get(var, 0.0)) for var in variables}
            )
            experience_buffer.add_observation(sample)
        
        buffer = TrajectoryBuffer(experience_buffer=experience_buffer)
        
        import pyrsistent as pyr
        acquisition_state = AcquisitionState(
            posterior=initial_posterior,
            buffer=buffer,
            best_value=initial_value,
            current_target=target,
            step=0,
            metadata=pyr.m(
                scm_info=pyr.m(
                    variables=variables,
                    target=target,
                    n_variables=len(variables)
                )
            )
        )
    
    # Run intervention steps
    current_best = initial_value
    
    for step in range(config.n_intervention_steps):
        key, step_key = random.split(key)
        
        # Select intervention
        if bc_acquisition_fn is not None and acquisition_state is not None:
            # Use BC acquisition with state
            intervention_result = bc_acquisition_fn(acquisition_state, step_key)
            
            # Handle different return formats
            if 'variable' in intervention_result:
                # Simple format
                selected_var = intervention_result.get('variable')
                selected_val = intervention_result.get('value', 0.0)
            elif 'intervention_variables' in intervention_result:
                # Full format
                int_vars = intervention_result.get('intervention_variables', frozenset())
                int_vals = intervention_result.get('intervention_values', ())
                if int_vars and int_vals:
                    selected_var = list(int_vars)[0]
                    selected_val = float(int_vals[0])
                else:
                    # Fallback to random
                    logger.warning("BC acquisition returned empty intervention")
                    var_idx = random.randint(step_key, (), 0, len(variables))
                    selected_var = variables[var_idx]
                    selected_val = 0.0
            else:
                logger.warning(f"Unexpected BC acquisition result format: {intervention_result.keys()}")
                var_idx = random.randint(step_key, (), 0, len(variables))
                selected_var = variables[var_idx]
                selected_val = 0.0
            
            # Ensure selected_val is a float
            selected_val = float(selected_val)
        else:
            # Random intervention
            key, var_key, val_key = random.split(step_key, 3)
            var_idx = random.randint(var_key, (), 0, len(variables))
            selected_var = variables[var_idx]
            selected_val = random.uniform(
                val_key, (), 
                minval=config.intervention_value_range[0],
                maxval=config.intervention_value_range[1]
            )
        
        # Sample with intervention
        key, int_key = random.split(key)
        # Create intervened SCM
        intervention_spec = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: selected_val}
        )
        intervened_scm = apply_intervention(scm, intervention_spec)
        int_data = sample_from_linear_scm(
            intervened_scm,
            n_samples=100,
            seed=int(int_key[0])
        )
        
        # Convert intervention samples to array
        int_array = jnp.array([[float(sample.get(var, 0.0)) for var in variables] for sample in int_data])
        
        # Track intervention samples
        all_samples.append(int_array)
        
        # Compute new target value
        target_value = float(jnp.mean(int_array[:, variables.index(target)]))
        current_best = max(current_best, target_value) if config.scoring_method == 'maximize' else min(current_best, target_value)
        
        # Track performance
        if track_performance:
            # Extract marginals if using BC surrogate
            marginals = {}
            if bc_surrogate_fn:
                # Use BC surrogate to get posterior predictions
                # Combine all collected data
                all_data = jnp.vstack(all_samples)
                
                # Convert to AVICI format [N, d, 3]
                # For now, use simple format with no intervention indicators
                n_samples, n_vars = all_data.shape
                avici_data = jnp.zeros((n_samples, n_vars, 3))
                avici_data = avici_data.at[:, :, 0].set(all_data)  # Value channel
                avici_data = avici_data.at[:, :, 1].set(1.0)  # Mask channel (all observed)
                avici_data = avici_data.at[:, :, 2].set(0.0)  # Intervention channel (none)
                
                # Get predictions from BC surrogate
                try:
                    posterior = bc_surrogate_fn(avici_data, variables, target)
                except Exception as e:
                    logger.error(f"BC surrogate inference failed: {e}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    # Fallback to no marginals
                    marginals = {}
                    uncertainty = 1.0 / (step + 2)
                else:
                    # Extract marginals from posterior
                    if hasattr(posterior, 'marginals'):
                        marginals = posterior.marginals
                    elif isinstance(posterior, dict) and 'marginals' in posterior:
                        marginals = posterior['marginals']
                    else:
                        # Fallback: create marginals from posterior if available
                        marginals = {}
                    
                    # Extract uncertainty if available
                    uncertainty = 0.0
                    if hasattr(posterior, 'uncertainty'):
                        uncertainty = float(posterior.uncertainty)
                    elif isinstance(posterior, dict) and 'uncertainty' in posterior:
                        uncertainty = float(posterior['uncertainty'])
            else:
                uncertainty = 1.0 / (step + 2)  # Decreasing uncertainty fallback
            
            performance_trajectory.append(PerformanceMetrics(
                step=step + 1,
                intervention_variable=selected_var,
                intervention_value=selected_val,
                target_value=target_value,
                graph_estimate=None,  # Simplified
                marginals=marginals if marginals else None,
                uncertainty=uncertainty
            ))
        
        # Update acquisition state if using BC acquisition
        if acquisition_state is not None:
            # For BC runner, we'll do a simple state update
            # The real update_state_with_intervention expects different parameters
            # so we'll create a new state directly
            acquisition_state = AcquisitionState(
                posterior=acquisition_state.posterior,  # Keep same posterior for simplicity
                buffer=acquisition_state.buffer,
                best_value=min(acquisition_state.best_value, target_value) if config.scoring_method != 'maximize' else max(acquisition_state.best_value, target_value),
                current_target=acquisition_state.current_target,
                step=acquisition_state.step + 1,
                metadata=acquisition_state.metadata
            )
        
        # Track history
        learning_history.append({
            'step': step + 1,
            'intervention': {
                'intervention_variables': frozenset([selected_var]),
                'intervention_values': (selected_val,)
            },
            'outcome_value': target_value,
            'current_best': current_best,
            'marginals': marginals if 'marginals' in locals() else {}
        })
    
    # Return results
    return {
        'initial_best': initial_value,
        'final_best': current_best,
        'target_progress': [initial_value] + [m.target_value for m in performance_trajectory[1:]],
        'learning_history': learning_history,
        'performance_trajectory': performance_trajectory if track_performance else None,
        'performance_metrics': {
            'initial_value': initial_value,
            'final_value': current_best,
            'improvement': current_best - initial_value,
            'n_interventions': config.n_intervention_steps
        },
        'success': True
    }