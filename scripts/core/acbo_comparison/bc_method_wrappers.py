#!/usr/bin/env python3
"""
BC Method Wrappers for ACBO Comparison Framework

This module provides wrappers for BC-trained models to run them in the full ACBO
comparison framework with actual CBO experiments (not simulations).

Key Features:
1. Loads BC checkpoints and wraps them for ACBO
2. Runs actual CBO experiments using run_progressive_learning_demo_with_scm
3. Returns standardized metrics for comparison
4. Handles both surrogate and acquisition models

Design Principles (Rich Hickey Approved):
- Pure functions for method creation
- No side effects in experiment runners
- Clear separation of concerns
- Composable method interfaces
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pyrsistent as pyr
import jax.numpy as jnp
import jax.random as random
from omegaconf import DictConfig

from src.causal_bayes_opt.training.bc_model_loader import load_bc_model, validate_checkpoint
from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
from examples.demo_learning import DemoConfig
from .method_registry import ExperimentMethod
from src.causal_bayes_opt.evaluation.bc_performance_tracker import BCPerformanceTracker

logger = logging.getLogger(__name__)


def run_cbo_with_tracking(
    scm: pyr.PMap, 
    config: DemoConfig,
    pretrained_surrogate: Optional[Tuple[Any, Any, Any, Any, Any]] = None,
    pretrained_acquisition: Optional[Callable] = None,
    track_performance: bool = True
) -> Dict[str, Any]:
    """
    Run CBO experiment with performance tracking.
    
    This wraps run_progressive_learning_demo_with_scm to add:
    - SHD tracking at each step
    - F1 score tracking
    - Target value trajectory
    
    Args:
        scm: Structural causal model
        config: Demo configuration
        pretrained_surrogate: Optional BC surrogate tuple
        pretrained_acquisition: Optional BC acquisition function
        track_performance: Whether to track detailed metrics
        
    Returns:
        Results dict with added performance_trajectory
    """
    # Extract target variable
    from causal_bayes_opt.data_structures.scm import get_target
    target = get_target(scm)
    
    # Create performance tracker if requested
    tracker = BCPerformanceTracker(scm, target) if track_performance else None
    
    # Run the base experiment
    result = run_progressive_learning_demo_with_scm(
        scm, config,
        pretrained_surrogate=pretrained_surrogate,
        pretrained_acquisition=pretrained_acquisition
    )
    
    # Extract and track performance metrics
    if tracker and 'learning_history' in result:
        # Add initial observational step
        if 'initial_best' in result:
            tracker.record_step(
                intervention_variable=None,
                intervention_value=None,
                target_value=result['initial_best'],
                graph_estimate=None  # No estimate at start
            )
        
        # Track each intervention step
        for i, step_info in enumerate(result['learning_history']):
            # Extract intervention details
            intervention = step_info.get('intervention', {})
            intervention_vars = intervention.get('intervention_variables', frozenset())
            intervention_vals = intervention.get('intervention_values', ())
            
            intervention_var = list(intervention_vars)[0] if intervention_vars else None
            intervention_val = intervention_vals[0] if intervention_vals else None
            
            # Get target value from step or progress
            target_value = step_info.get('outcome_value', 
                                       result['target_progress'][i+1] if i+1 < len(result['target_progress']) else 0.0)
            
            # Extract graph estimate if available
            graph_estimate = None
            if 'marginals' in step_info:
                # Convert marginals to simple graph estimate
                # (In real implementation, would extract from posterior)
                marginals = step_info['marginals']
                graph_estimate = {}
                for var, parent_probs in marginals.items():
                    # Threshold at 0.5 for binary decision
                    if isinstance(parent_probs, dict):
                        parents = [p for p, prob in parent_probs.items() if prob > 0.5]
                    else:
                        parents = []
                    graph_estimate[var] = parents
            
            tracker.record_step(
                intervention_variable=intervention_var,
                intervention_value=intervention_val,
                target_value=target_value,
                graph_estimate=graph_estimate
            )
        
        # Add performance trajectory to results
        result['performance_trajectory'] = tracker.get_trajectory()
        result['performance_metrics'] = tracker.get_final_metrics()
    
    return result


def create_bc_surrogate_random_method(surrogate_checkpoint_path: str) -> ExperimentMethod:
    """
    Create BC surrogate with random acquisition method.
    
    Uses BC-trained surrogate model for structure learning but random
    intervention selection. This tests the value of the BC surrogate alone.
    
    Args:
        surrogate_checkpoint_path: Path to BC surrogate checkpoint
        
    Returns:
        ExperimentMethod configured for BC surrogate + random acquisition
    """
    def bc_surrogate_random(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with BC surrogate and random acquisition."""
        try:
            # Load BC trained surrogate
            logger.info(f"Loading BC surrogate from {surrogate_checkpoint_path}")
            
            # Validate checkpoint first
            validation = validate_checkpoint(surrogate_checkpoint_path)
            if not validation['valid']:
                raise ValueError(f"Invalid checkpoint: {validation.get('error', 'Unknown error')}")
            
            # Load the model - returns tuple for surrogate
            loaded_surrogate = load_bc_model(surrogate_checkpoint_path, 'surrogate')
            
            # Create ACBO config
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=0.0,  # No learning for BC surrogate (frozen)
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Import the real BC inference function
            from causal_bayes_opt.training.bc_model_inference import create_bc_surrogate_inference_fn
            
            # Create the real BC surrogate inference function
            bc_surrogate_fn = create_bc_surrogate_inference_fn(
                checkpoint_path=surrogate_checkpoint_path,
                threshold=0.1  # Probability threshold for parent set inclusion
            )
            
            # Wrap BC surrogate to match expected 4-argument signature
            def bc_surrogate_wrapper(avici_data, variables, target, current_params=None):
                # BC inference function expects 3 args, ignore current_params
                return bc_surrogate_fn(avici_data, variables, target)
            
            # loaded_surrogate is already a tuple: (init_fn, apply_fn, encoder_init, encoder_apply, params)
            init_fn, apply_fn, encoder_init, encoder_apply, params = loaded_surrogate
            
            # Create surrogate components matching expected format
            bc_surrogate_tuple = (
                bc_surrogate_wrapper,  # surrogate function with correct signature
                None,  # net (not needed for frozen model)
                params,  # params from loaded model
                None,  # opt_state (no optimization)
                lambda p, o, post, samples, vars, tgt: (p, o, (0.0, 0.0, 0.0, 0.0))  # no-op update
            )
            
            # Run actual CBO experiment with BC surrogate and tracking
            logger.info(f"Running CBO with BC surrogate on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config, 
                pretrained_surrogate=bc_surrogate_tuple,
                pretrained_acquisition=None,  # Use random acquisition
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'bc_surrogate_random'
            result['surrogate_checkpoint_used'] = surrogate_checkpoint_path
            result['used_bc_surrogate'] = True
            result['used_bc_acquisition'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"BC surrogate random method failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'bc_surrogate_random',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="BC Surrogate + Random",
        type="bc_surrogate_random",
        description="BC-trained surrogate with random acquisition policy",
        run_function=bc_surrogate_random,
        config={},
        requires_checkpoint=True,
        checkpoint_path=surrogate_checkpoint_path
    )


def create_bc_acquisition_learning_method(acquisition_checkpoint_path: str) -> ExperimentMethod:
    """
    Create learning surrogate with BC acquisition method.
    
    Uses learning surrogate model but BC-trained acquisition policy.
    This tests the value of the BC acquisition policy alone.
    
    Args:
        acquisition_checkpoint_path: Path to BC acquisition checkpoint
        
    Returns:
        ExperimentMethod configured for learning surrogate + BC acquisition
    """
    def bc_acquisition_learning(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with learning surrogate and BC acquisition."""
        try:
            # Load BC trained acquisition
            logger.info(f"Loading BC acquisition from {acquisition_checkpoint_path}")
            
            # Validate checkpoint first
            validation = validate_checkpoint(acquisition_checkpoint_path)
            if not validation['valid']:
                raise ValueError(f"Invalid checkpoint: {validation.get('error', 'Unknown error')}")
            
            # Load the model - returns callable for acquisition
            loaded_acquisition = load_bc_model(acquisition_checkpoint_path, 'acquisition')
            
            # Create ACBO config
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=1e-3,  # Enable learning for surrogate
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Import the real BC acquisition inference function
            from causal_bayes_opt.training.bc_model_inference import create_bc_acquisition_inference_fn
            
            # Get variables and target from SCM for acquisition setup
            from causal_bayes_opt.data_structures.scm import get_variables, get_target
            scm_variables = list(get_variables(scm))
            scm_target = get_target(scm)
            
            # Create the real BC acquisition inference function
            bc_acquisition_fn = create_bc_acquisition_inference_fn(
                checkpoint_path=acquisition_checkpoint_path,
                variables=scm_variables,
                target_variable=scm_target
            )
            
            # Create BC acquisition policy function
            def bc_acquisition_policy(variables: List[str], target: str, value_range: tuple):
                """BC-trained acquisition policy using real model."""
                # Return the inference function directly
                # (it already has the right signature: key -> intervention_decision)
                return bc_acquisition_fn
            
            # Get target variable (already imported above)
            target = scm_target
            variables = scm_variables
            
            # Create BC acquisition policy
            bc_acquisition_fn = bc_acquisition_policy(
                variables, target, acbo_config.intervention_value_range
            )
            
            # Run actual CBO experiment with BC acquisition and tracking
            logger.info(f"Running CBO with BC acquisition on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config,
                pretrained_surrogate=None,  # Use learning surrogate
                pretrained_acquisition=bc_acquisition_fn,
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'bc_acquisition_learning'
            result['acquisition_checkpoint_used'] = acquisition_checkpoint_path
            result['used_bc_surrogate'] = False
            result['used_bc_acquisition'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"BC acquisition learning method failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'bc_acquisition_learning',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="Learning + BC Acquisition",
        type="bc_acquisition_learning",
        description="Learning surrogate with BC-trained acquisition policy",
        run_function=bc_acquisition_learning,
        config={},
        requires_checkpoint=True,
        checkpoint_path=acquisition_checkpoint_path
    )


def create_bc_trained_both_method(
    surrogate_checkpoint_path: str,
    acquisition_checkpoint_path: str
) -> ExperimentMethod:
    """
    Create method with both BC-trained surrogate and acquisition.
    
    Uses both BC-trained models together. This tests the full BC pipeline.
    
    Args:
        surrogate_checkpoint_path: Path to BC surrogate checkpoint
        acquisition_checkpoint_path: Path to BC acquisition checkpoint
        
    Returns:
        ExperimentMethod configured for BC surrogate + BC acquisition
    """
    def bc_trained_both(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with both BC models."""
        try:
            # Load both BC models
            logger.info(f"Loading BC surrogate from {surrogate_checkpoint_path}")
            
            # Validate surrogate checkpoint
            validation = validate_checkpoint(surrogate_checkpoint_path)
            if not validation['valid']:
                raise ValueError(f"Invalid surrogate checkpoint: {validation.get('error', 'Unknown error')}")
            
            # Load surrogate - returns tuple
            loaded_surrogate = load_bc_model(surrogate_checkpoint_path, 'surrogate')
            
            logger.info(f"Loading BC acquisition from {acquisition_checkpoint_path}")
            
            # Validate acquisition checkpoint
            validation = validate_checkpoint(acquisition_checkpoint_path)
            if not validation['valid']:
                raise ValueError(f"Invalid acquisition checkpoint: {validation.get('error', 'Unknown error')}")
            
            # Load acquisition - returns callable
            loaded_acquisition = load_bc_model(acquisition_checkpoint_path, 'acquisition')
            
            # Create ACBO config
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=0.0,  # No learning for BC models (both frozen)
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Import the real BC inference functions
            from causal_bayes_opt.training.bc_model_inference import (
                create_bc_surrogate_inference_fn,
                create_bc_acquisition_inference_fn
            )
            
            # Create the real BC surrogate inference function
            bc_surrogate_fn = create_bc_surrogate_inference_fn(
                checkpoint_path=surrogate_checkpoint_path,
                threshold=0.1
            )
            
            # Wrap BC surrogate to match expected 4-argument signature
            def bc_surrogate_wrapper(avici_data, variables, target, current_params=None):
                # BC inference function expects 3 args, ignore current_params
                return bc_surrogate_fn(avici_data, variables, target)
            
            # loaded_surrogate is a tuple: (init_fn, apply_fn, encoder_init, encoder_apply, params)
            init_fn, apply_fn, encoder_init, encoder_apply, params = loaded_surrogate
            
            # Create surrogate components
            bc_surrogate_tuple = (
                bc_surrogate_wrapper,
                None,
                params,  # params from loaded model
                None,
                lambda p, o, post, samples, vars, tgt: (p, o, (0.0, 0.0, 0.0, 0.0))
            )
            
            # Get target and variables
            from causal_bayes_opt.data_structures.scm import get_target, get_variables
            target = get_target(scm)
            variables = list(get_variables(scm))
            
            # Create the real BC acquisition inference function
            bc_acquisition_fn = create_bc_acquisition_inference_fn(
                checkpoint_path=acquisition_checkpoint_path,
                variables=variables,
                target_variable=target
            )
            
            # Run actual CBO experiment with both BC models and tracking
            logger.info(f"Running CBO with both BC models on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config,
                pretrained_surrogate=bc_surrogate_tuple,
                pretrained_acquisition=bc_acquisition_fn,
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'bc_trained_both'
            result['surrogate_checkpoint_used'] = surrogate_checkpoint_path
            result['acquisition_checkpoint_used'] = acquisition_checkpoint_path
            result['used_bc_surrogate'] = True
            result['used_bc_acquisition'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"BC both trained method failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'bc_trained_both',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="BC Surrogate + BC Acquisition",
        type="bc_trained_both",
        description="Both surrogate and acquisition trained with behavioral cloning",
        run_function=bc_trained_both,
        config={},
        requires_checkpoint=True,
        checkpoint_path=None  # Both paths stored separately
    )