#!/usr/bin/env python3
"""
PARENT_SCALE Algorithm Runner

Contains the main function for running the complete PARENT_SCALE Causal Bayesian 
Optimization algorithm with proper data generation and parameter space handling.
"""

from typing import List, Dict, Any, Optional
from copy import deepcopy
import sys

import pyrsistent as pyr

# Import our data structures
from causal_bayes_opt.data_structures.scm import get_target

# Add PARENT_SCALE to path
sys.path.insert(0, 'external/parent_scale')
sys.path.insert(0, 'external')
sys.path.insert(0, 'causal_bayes_opt_old')

# Import PARENT_SCALE components - handle gracefully if not available
try:
    from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
    from graphs.linear_collider_graph import LinearColliderGraph
    from graphs.data_setup import setup_observational_interventional
    from parent_scale.posterior_model.model import DoublyRobustModel
    PARENT_SCALE_AVAILABLE = True
except ImportError:
    # Create dummy classes when PARENT_SCALE not available
    class PARENT_SCALE:
        pass
    class LinearColliderGraph:
        pass
    def setup_observational_interventional(*args, **kwargs):
        raise ImportError("PARENT_SCALE not available")
    class DoublyRobustModel:
        pass
    PARENT_SCALE_AVAILABLE = False

# Import other components
from .data_conversion import (
    scm_to_graph_structure, samples_to_parent_scale_data, 
    parent_scale_results_to_posterior, generate_parent_scale_data_original
)


def run_parent_discovery(
    scm: pyr.PMap,
    samples: List[pyr.PMap],
    target_variable: str,
    num_bootstraps: int = 15,
    individual: bool = False
) -> Dict[str, Any]:
    """
    Complete parent discovery using PARENT_SCALE neural doubly robust method.
    
    Args:
        scm: Our SCM representation
        samples: List of our Sample objects
        target_variable: Variable to find parents for
        num_bootstraps: Number of bootstrap samples (use validated scaling)
        individual: Whether to use individual bootstrap method
        
    Returns:
        Parent discovery results in our format
    """
    # Convert SCM to PARENT_SCALE format
    graph = scm_to_graph_structure(scm)
    variables = graph.variables
    
    # Convert samples to PARENT_SCALE format
    data = samples_to_parent_scale_data(samples, variables)
    
    # Create and run PARENT_SCALE model
    model = DoublyRobustModel(
        graph=graph,
        topological_order=variables,
        target=target_variable,
        num_bootstraps=num_bootstraps,
        indivdual=individual  # Note: PARENT_SCALE has typo in parameter name
    )
        
    # Run inference
    estimate = model.run_method(data)
        
    # Convert results to our format
    posterior = parent_scale_results_to_posterior(
        model.prob_estimate, target_variable, variables
    )
        
    # Add metadata
    posterior['num_bootstraps'] = num_bootstraps
    posterior['num_samples'] = len(samples)
    posterior['method'] = 'neural_doubly_robust'
        
    return posterior


def run_full_parent_scale_algorithm(
    scm: pyr.PMap,
    samples: Optional[List[pyr.PMap]] = None,
    target_variable: str = None,
    T: int = 10,
    nonlinear: bool = True,
    causal_prior: bool = True,
    individual: bool = False,
    use_doubly_robust: bool = True,
    n_observational: int = 100,
    n_interventional: int = 2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the complete PARENT_SCALE Causal Bayesian Optimization algorithm.
    
    This uses PARENT_SCALE's exact original data generation process to ensure
    identical behavior with the original algorithm.
    
    Args:
        scm: Our SCM representation (used only for target variable)
        samples: Optional list of samples (ignored - use exact original data generation)
        target_variable: Variable to optimize (defaults to SCM target)
        T: Number of CBO iterations to run
        nonlinear: Whether to use nonlinear posterior model
        causal_prior: Whether to use causal priors
        individual: Whether to use individual bootstrap method
        use_doubly_robust: Whether to use doubly robust for initial probabilities
        n_observational: Number of observational samples to generate
        n_interventional: Number of interventional samples per exploration element
        seed: Random seed for reproducible data generation
        
    Returns:
        Complete expert trajectory with intervention decisions and reasoning
    """
    if not PARENT_SCALE_AVAILABLE:
        raise ImportError(
            "Full PARENT_SCALE algorithm not available. Please ensure external/parent_scale is properly set up."
        )
    
    print(f"Running PARENT_SCALE algorithm with EXACT original process...")
    
    # Use target from SCM if not specified
    if target_variable is None:
        target_variable = get_target(scm)
    
    # CRITICAL: Set random seed exactly like the original algorithm
    import numpy as np
    np.random.seed(seed)
    
    # CRITICAL: Use PARENT_SCALE's exact original data generation process
    # This ensures identical random number consumption and data structures
    
    # Create the original graph structure (exactly like original algorithm)
    graph = LinearColliderGraph(noiseless=False)
    
    # Generate data using PARENT_SCALE's exact original process
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type="Toy",
        n_obs=n_observational,
        n_int=n_interventional,
        noiseless=True,
        seed=seed,
        graph=graph,
        use_iscm=False
    )
    
    print(f"✓ Generated data using PARENT_SCALE's exact original process")
    print(f"  - D_O keys: {list(D_O.keys())}")
    print(f"  - D_I keys: {list(D_I.keys())}")
    print(f"  - Exploration set: {exploration_set}")
    
    # Use the original graph directly (no conversion needed)
    
    # Verify data completeness (should be complete with original generation)
    missing_exploration_elements = [es for es in exploration_set if es not in D_I]
    if missing_exploration_elements:
        raise ValueError(f"Missing D_I data for exploration elements: {missing_exploration_elements}. This should not happen with original data generation.")
    
    print(f"✓ Generated complete PARENT_SCALE data")
    print(f"  - {len(D_O[target_variable])} observational samples")
    print(f"  - {len(D_I)} intervention groups (all exploration elements covered)")
    print(f"  - Exploration set: {exploration_set}")
    
    # Initialize PARENT_SCALE algorithm with the original graph
    parent_scale = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        causal_prior=causal_prior,
        noiseless=True,  # For deterministic expert demonstrations
        cost_num=1,
        scale_data=True,
        individual=individual,
        use_doubly_robust=use_doubly_robust,
        use_iscm=False
    )
    
    # REMOVED CUSTOM DATA HANDLING: Use original PARENT_SCALE data flow exactly
    # No need for D_O_original workaround - the original algorithm handles this correctly
    
    # Set data and exploration set (exactly like original)
    parent_scale.set_values(D_O, D_I, exploration_set)
    
    print(f"✓ Initialized PARENT_SCALE algorithm")
    print(f"  - Target: {target_variable}")
    print(f"  - T={T} iterations")
    print(f"  - Nonlinear: {nonlinear}")
    print(f"  - Causal prior: {causal_prior}")
    
    # Run the complete algorithm
    try:
        print(f"  - About to call parent_scale.run_algorithm with T={T}")
        
        # Debug: Check the data structures before running
        print(f"  - D_O keys: {list(parent_scale.D_O.keys())}")
        print(f"  - D_I keys: {list(parent_scale.D_I.keys())}")
        print(f"  - Exploration set: {parent_scale.exploration_set}")
        
        # Verify that all exploration set elements have D_I data
        for es in parent_scale.exploration_set:
            es_tuple = tuple(es)
            if es_tuple not in parent_scale.D_I:
                print(f"  - WARNING: Missing D_I data for exploration element: {es_tuple}")
            else:
                print(f"  - D_I[{es_tuple}] keys: {list(parent_scale.D_I[es_tuple].keys())}")
        
        results = parent_scale.run_algorithm(
            T=T,
            show_graphics=False,
            file=None
        )
        print(f"  - parent_scale.run_algorithm completed successfully")
        print(f"  - Results type: {type(results)}")
        print(f"  - Results length: {len(results) if results and hasattr(results, '__len__') else 'N/A'}")
        
        # Validate results structure
        if not results:
            raise ValueError("parent_scale.run_algorithm returned None or empty results")
        
        if not hasattr(results, '__len__') or len(results) != 6:
            raise ValueError(f"parent_scale.run_algorithm returned unexpected format: {type(results)} with content {results}")
        
        (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
            average_uncertainty
        ) = results
        
        # Validate individual components
        if global_opt is None:
            raise ValueError("global_opt is None")
        if current_y is None:
            raise ValueError("current_y is None")
        if intervention_set is None:
            raise ValueError("intervention_set is None")
        
        print(f"✓ PARENT_SCALE algorithm completed {T} iterations")
        print(f"  - Final optimum: {global_opt[-1]:.4f}")
        print(f"  - Total interventions: {len(intervention_set)}")
        print(f"  - Final cost: {current_cost[-1]:.4f}")
        
        # Extract expert trajectory information
        trajectory = {
            'algorithm': 'PARENT_SCALE_CBO',
            'target_variable': target_variable,
            'iterations': T,
            'total_samples_used': n_observational + (n_interventional * len(exploration_set)),
            'status': 'completed',  # Add explicit success status
            
            # Complete trajectory data
            'intervention_sequence': intervention_set,
            'intervention_values': intervention_values,
            'target_outcomes': current_y,
            'global_optimum_trajectory': global_opt,
            'cost_trajectory': current_cost,
            'uncertainty_trajectory': average_uncertainty,
            
            # Final algorithm state
            'final_optimum': global_opt[-1],
            'final_cost': current_cost[-1],
            'total_interventions': len(intervention_set),
            
            # Algorithm configuration
            'config': {
                'nonlinear': nonlinear,
                'causal_prior': causal_prior,
                'individual': individual,
                'use_doubly_robust': use_doubly_robust
            },
            
            # Posterior information (if available)
            'final_posterior': getattr(parent_scale, 'prior_probabilities', {}),
            'final_graphs': list(getattr(parent_scale, 'graphs', {}).keys()),
            
            # Success metrics
            'convergence_rate': len([i for i, (prev, curr) in enumerate(zip(global_opt[:-1], global_opt[1:])) if curr < prev]) / T,
            'exploration_efficiency': len(set(intervention_set)) / len(intervention_set) if intervention_set else 0
        }
        
        return trajectory
        
    except Exception as e:
        import traceback
        print(f"❌ PARENT_SCALE algorithm failed: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        
        # Return partial results for debugging
        return {
            'algorithm': 'PARENT_SCALE_CBO',
            'target_variable': target_variable,
            'iterations': T,
            'total_samples_used': len(samples) if samples else n_observational + (n_interventional * len(exploration_set)),
            'error': str(e),
            'status': 'failed'
        }