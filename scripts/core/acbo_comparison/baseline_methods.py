#!/usr/bin/env python3
"""
Baseline Methods for ACBO Comparison Framework

This module provides baseline methods (random and oracle) for meaningful
comparison with BC-trained models in the ACBO framework.

Key Baselines:
1. Random Baseline: No learning, random interventions (performance floor)
2. Oracle Baseline: Perfect causal knowledge (performance ceiling)

Design Principles (Rich Hickey Approved):
- Pure functions for baseline creation
- No side effects in experiment runners
- Clear performance bounds for comparison
- Composable method interfaces
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pyrsistent as pyr
import jax.numpy as jnp
import jax.random as random
from omegaconf import DictConfig

from causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
from examples.demo_learning import DemoConfig
from .method_registry import ExperimentMethod
from src.causal_bayes_opt.evaluation.bc_performance_tracker import BCPerformanceTracker
from .bc_method_wrappers import run_cbo_with_tracking

logger = logging.getLogger(__name__)


def create_random_baseline_method() -> ExperimentMethod:
    """
    Create pure random baseline method.
    
    This baseline uses:
    - No structure learning (untrained surrogate)
    - Random intervention selection
    
    Provides the performance floor for comparison.
    
    Returns:
        ExperimentMethod configured for random baseline
    """
    def random_baseline(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with no learning - pure random."""
        try:
            # Create ACBO config
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=0.0,  # No learning for baseline
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Create a non-learning surrogate that returns uniform posteriors
            def random_surrogate_fn(avici_data, variables, target, params=None):
                """Random baseline surrogate - returns uniform posteriors."""
                from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
                
                # For the target variable, return uniform posterior (no parents)
                parent_sets = [frozenset()]  # Empty parent set
                probabilities = jnp.array([1.0])
                
                return create_parent_set_posterior(
                    target_variable=target,
                    parent_sets=parent_sets,
                    probabilities=probabilities,
                    metadata={'type': 'random_baseline'}
                )
            
            # Create surrogate components for random baseline
            random_surrogate_tuple = (
                random_surrogate_fn,  # surrogate function
                None,  # net
                {},  # params (empty)
                None,  # opt_state
                lambda p, o, post, samples, vars, tgt: (p, o, (0.0, 0.0, 0.0, 0.0))  # no-op update
            )
            
            # Run CBO with random surrogate and random acquisition with tracking
            logger.info(f"Running random baseline on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config,
                pretrained_surrogate=random_surrogate_tuple,
                pretrained_acquisition=None,  # Random acquisition
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'random_baseline'
            result['is_baseline'] = True
            result['used_bc_surrogate'] = False
            result['used_bc_acquisition'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Random baseline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'random_baseline',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="Random Baseline",
        type="random_baseline",
        description="Pure random interventions with no learning",
        run_function=random_baseline,
        config={},
        requires_checkpoint=False
    )


def create_oracle_baseline_method() -> ExperimentMethod:
    """
    Create oracle baseline method.
    
    This baseline uses:
    - Perfect knowledge of causal structure
    - Optimal intervention selection (intervene on direct parents of target)
    
    Provides the performance ceiling for comparison.
    
    Returns:
        ExperimentMethod configured for oracle baseline
    """
    def oracle_baseline(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with perfect causal knowledge."""
        try:
            # Extract true causal structure
            target = get_target(scm)
            variables = list(get_variables(scm))
            true_parents = get_parents(scm, target)
            
            logger.info(f"Oracle knows target {target} has parents: {true_parents}")
            
            # Create oracle intervention policy
            def oracle_intervention_policy(variables: List[str], target: str, value_range: tuple):
                """Oracle policy that intervenes on true parents."""
                def policy(key):
                    # Always intervene on a parent of the target if available
                    if true_parents:
                        # Rotate through parents for diversity
                        step = int(key[0]) % len(true_parents)
                        selected_var = list(true_parents)[step]
                        
                        # Sample intervention value
                        value_key = random.fold_in(key, 1)
                        value = random.uniform(value_key, minval=value_range[0], maxval=value_range[1])
                        
                        return {
                            'intervention_variables': frozenset([selected_var]),
                            'intervention_values': (float(value),)
                        }
                    else:
                        # No parents - can't improve target through interventions
                        return {
                            'intervention_variables': frozenset(),
                            'intervention_values': tuple()
                        }
                
                return policy
            
            # Create perfect surrogate (knows true graph)
            def oracle_surrogate_fn(avici_data, variables, target, params=None):
                """Oracle surrogate that returns true causal structure."""
                from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
                
                # For the target variable, return posterior with true parents
                true_parent_set = frozenset(get_parents(scm, target))
                parent_sets = [true_parent_set]
                probabilities = jnp.array([1.0])
                
                return create_parent_set_posterior(
                    target_variable=target,
                    parent_sets=parent_sets,
                    probabilities=probabilities,
                    metadata={'type': 'oracle', 'true_parents': list(true_parent_set)}
                )
            
            # Create oracle surrogate components
            oracle_surrogate_tuple = (
                oracle_surrogate_fn,  # surrogate function
                None,  # net (no neural network needed)
                {},  # params (empty)
                None,  # opt_state (no optimizer)
                lambda p, o, post, samples, vars, tgt: (p, o, (0.0, 0.0, 0.0, 0.0))  # no-op update
            )
            
            # Create ACBO config
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=0.0,  # No learning needed
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Create oracle acquisition policy
            oracle_acquisition_fn = oracle_intervention_policy(variables, target, acbo_config.intervention_value_range)
            
            # Run CBO with oracle knowledge and tracking
            logger.info(f"Running oracle baseline on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config,
                pretrained_surrogate=oracle_surrogate_tuple,
                pretrained_acquisition=oracle_acquisition_fn,
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'oracle_baseline'
            result['is_baseline'] = True
            result['is_oracle'] = True
            result['used_bc_surrogate'] = False
            result['used_bc_acquisition'] = False
            result['true_parents'] = list(true_parents)
            
            return result
            
        except Exception as e:
            logger.error(f"Oracle baseline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'oracle_baseline',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="Oracle Baseline",
        type="oracle_baseline",
        description="Perfect knowledge of causal graph (performance ceiling)",
        run_function=oracle_baseline,
        config={},
        requires_checkpoint=False
    )


def create_learning_baseline_method() -> ExperimentMethod:
    """
    Create learning baseline method (optional).
    
    This baseline uses:
    - Active structure learning from interventions
    - Random intervention selection
    
    Tests pure active learning without any BC components.
    
    Returns:
        ExperimentMethod configured for learning baseline
    """
    def learning_baseline(scm: pyr.PMap, config: DictConfig, scm_idx: int, seed: int) -> Dict[str, Any]:
        """Run CBO with active learning but random interventions."""
        try:
            # Create ACBO config with learning enabled
            acbo_config = DemoConfig(
                n_observational_samples=config.experiment.target.n_observational_samples,
                n_intervention_steps=config.experiment.target.max_interventions,
                learning_rate=1e-3,  # Enable learning
                scoring_method='bic',
                intervention_value_range=(-2.0, 2.0),
                random_seed=seed
            )
            
            # Run CBO with learning surrogate but random interventions with tracking
            logger.info(f"Running learning baseline on SCM {scm_idx}")
            result = run_cbo_with_tracking(
                scm, acbo_config,
                pretrained_surrogate=None,  # Will create learning surrogate
                pretrained_acquisition=None,  # Random interventions
                track_performance=True
            )
            
            # Add method metadata
            result['method'] = 'learning_baseline'
            result['is_baseline'] = True
            result['used_bc_surrogate'] = False
            result['used_bc_acquisition'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Learning baseline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'learning_baseline',
                'final_target_value': 0.0,
                'target_improvement': 0.0
            }
    
    return ExperimentMethod(
        name="Learning Baseline",
        type="learning_baseline",
        description="Active learning with random interventions",
        run_function=learning_baseline,
        config={},
        requires_checkpoint=False
    )