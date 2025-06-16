#!/usr/bin/env python3
"""
PARENT_SCALE Algorithm Runner (Refactored)

Main API functions for running the complete PARENT_SCALE Causal Bayesian 
Optimization algorithm. This refactored version uses focused modules for
better maintainability and follows functional programming principles.
"""

from typing import List, Dict, Any, Optional
import sys
import time

import pyrsistent as pyr

# Import our data structures
from causal_bayes_opt.data_structures.scm import get_target

# Import refactored modules
from .data_generation import (
    check_parent_scale_availability,
    generate_parent_scale_data,
    validate_data_completeness,
    get_data_summary,
    create_graph_instance
)
from .trajectory_extraction import (
    validate_algorithm_results,
    extract_trajectory_components,
    create_expert_trajectory,
    create_failed_trajectory,
    convert_to_acbo_format
)
from .validation import (
    validate_trajectory_completeness,
    validate_algorithm_configuration
)

# Import PARENT_SCALE components - handle gracefully if not available
# Use the availability check from data_generation module
from .data_generation import check_parent_scale_availability

if check_parent_scale_availability():
    try:
        from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
        from parent_scale.posterior_model.model import DoublyRobustModel
    except ImportError:
        # Create dummy classes when specific components not available
        class PARENT_SCALE:
            pass
        class DoublyRobustModel:
            pass
else:
    # Create dummy classes when PARENT_SCALE not available
    class PARENT_SCALE:
        pass
    class DoublyRobustModel:
        pass

# Import legacy data conversion functions
from .data_conversion import (
    scm_to_graph_structure, 
    samples_to_parent_scale_data, 
    parent_scale_results_to_posterior
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
    
    This function provides legacy compatibility for the neural doubly robust
    approach using custom SCM conversion. Note: The main algorithm runner
    now uses PARENT_SCALE's native data generation for perfect fidelity.
    
    Args:
        scm: Our SCM representation
        samples: List of our Sample objects
        target_variable: Variable to find parents for
        num_bootstraps: Number of bootstrap samples (use validated scaling)
        individual: Whether to use individual bootstrap method
        
    Returns:
        Parent discovery results in our format
    """
    if not check_parent_scale_availability():
        raise ImportError(
            "PARENT_SCALE components not available. Please ensure external/parent_scale is properly set up."
        )
    
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
    identical behavior with the original algorithm (100% verified fidelity).
    
    Args:
        scm: Our SCM representation (used only for target variable extraction)
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
    # Check availability
    if not check_parent_scale_availability():
        raise ImportError(
            "Full PARENT_SCALE algorithm not available. Please ensure external/parent_scale is properly set up."
        )
    
    print(f"Running PARENT_SCALE algorithm with EXACT original process...")
    
    # Use target from SCM if not specified
    if target_variable is None:
        target_variable = get_target(scm)
    
    # Create algorithm configuration for validation and tracking
    algorithm_config = {
        'nonlinear': nonlinear,
        'causal_prior': causal_prior,
        'individual': individual,
        'use_doubly_robust': use_doubly_robust,
        'n_observational': n_observational,
        'n_interventional': n_interventional,
        'seed': seed
    }
    
    # Validate configuration
    config_valid, config_errors = validate_algorithm_configuration(algorithm_config)
    if not config_valid:
        error_msg = f"Invalid algorithm configuration: {config_errors}"
        print(f"❌ {error_msg}")
        return create_failed_trajectory(error_msg, target_variable, T, n_observational, algorithm_config)
    
    try:
        # Generate data using exact original process
        D_O, D_I, exploration_set = generate_parent_scale_data(
            n_observational=n_observational,
            n_interventional=n_interventional,
            seed=seed
        )
        
        # Validate data completeness
        validate_data_completeness(D_O, D_I, exploration_set, target_variable)
        
        # Log data summary
        data_summary = get_data_summary(D_O, D_I, exploration_set, target_variable)
        print(f"✓ Generated complete PARENT_SCALE data")
        print(f"  - {data_summary['observational_samples']} observational samples")
        print(f"  - {data_summary['intervention_groups']} intervention groups")
        print(f"  - Exploration set: {data_summary['exploration_set']}")
        
        # Create graph instance for algorithm
        graph = create_graph_instance()
        
        # Initialize PARENT_SCALE algorithm
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
        
        # Set data (using original PARENT_SCALE data flow exactly)
        parent_scale.set_values(D_O, D_I, exploration_set)
        
        print(f"✓ Initialized PARENT_SCALE algorithm")
        print(f"  - Target: {target_variable}")
        print(f"  - T={T} iterations")
        print(f"  - Configuration: {algorithm_config}")
        
        # Run the complete algorithm
        print(f"  - Running PARENT_SCALE optimization...")
        start_time = time.time()
        
        results = parent_scale.run_algorithm(
            T=T,
            show_graphics=False,
            file=None
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"  - Algorithm completed in {runtime:.2f}s")
        
        # Validate and extract results
        if not validate_algorithm_results(results):
            error_msg = f"Invalid algorithm results: {type(results)} with content {results}"
            print(f"❌ {error_msg}")
            return create_failed_trajectory(error_msg, target_variable, T, n_observational, algorithm_config)
        
        # Extract trajectory components
        components = extract_trajectory_components(results)
        
        print(f"✓ PARENT_SCALE algorithm completed {T} iterations")
        print(f"  - Final optimum: {components['global_opt'][-1]:.6f}")
        print(f"  - Total interventions: {len(components['intervention_set'])}")
        print(f"  - Final cost: {components['current_cost'][-1]:.6f}")
        
        # Create expert trajectory
        trajectory = create_expert_trajectory(
            components=components,
            target_variable=target_variable,
            T=T,
            n_observational=n_observational,
            n_interventional=n_interventional,
            algorithm_config=algorithm_config,
            parent_scale_algorithm=parent_scale
        )
        
        # Add runtime information
        trajectory['runtime'] = runtime
        
        # Validate trajectory completeness
        trajectory_valid, trajectory_errors = validate_trajectory_completeness(trajectory)
        if not trajectory_valid:
            print(f"⚠️  Trajectory validation warnings: {trajectory_errors}")
        
        print(f"✅ Expert trajectory extracted successfully")
        print(f"  - Status: {trajectory['status']}")
        print(f"  - Convergence rate: {trajectory['convergence_rate']:.1%}")
        print(f"  - Exploration efficiency: {trajectory['exploration_efficiency']:.1%}")
        
        return trajectory
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ PARENT_SCALE algorithm failed: {error_msg}")
        print(f"Full traceback:")
        traceback.print_exc()
        
        # Return structured failure information
        return create_failed_trajectory(
            error=error_msg,
            target_variable=target_variable,
            T=T,
            n_observational=n_observational,
            algorithm_config=algorithm_config
        )


def run_batch_expert_demonstrations(
    n_trajectories: int = 10,
    base_seed: int = 42,
    iterations_range: tuple = (5, 15),
    **algorithm_kwargs
) -> List[Dict[str, Any]]:
    """
    Collect a batch of expert demonstrations for training.
    
    Args:
        n_trajectories: Number of expert demonstrations to collect
        base_seed: Base random seed (each trajectory uses base_seed + i)
        iterations_range: Tuple of (min_iterations, max_iterations) for diversity
        **algorithm_kwargs: Additional arguments for run_full_parent_scale_algorithm
        
    Returns:
        List of expert demonstration trajectories
    """
    import random
    
    print(f"Collecting {n_trajectories} expert demonstrations...")
    
    demonstrations = []
    successful_count = 0
    
    for i in range(n_trajectories):
        # Vary parameters for diversity
        seed = base_seed + i
        iterations = random.randint(*iterations_range)
        
        print(f"\n--- Collecting trajectory {i+1}/{n_trajectories} (seed={seed}, T={iterations}) ---")
        
        # Collect demonstration
        trajectory = run_full_parent_scale_algorithm(
            scm=None,  # Use default LinearColliderGraph
            target_variable='Y',
            T=iterations,
            seed=seed,
            **algorithm_kwargs
        )
        
        demonstrations.append(trajectory)
        
        if trajectory.get('status') == 'completed':
            successful_count += 1
            print(f"✓ Trajectory {i+1} completed successfully")
        else:
            print(f"❌ Trajectory {i+1} failed: {trajectory.get('error', 'Unknown error')}")
    
    success_rate = successful_count / n_trajectories
    print(f"\n✅ Batch collection complete:")
    print(f"  - Total trajectories: {n_trajectories}")
    print(f"  - Successful: {successful_count}")
    print(f"  - Success rate: {success_rate:.1%}")
    
    return demonstrations


def convert_trajectory_to_acbo_format(trajectory: Dict[str, Any]) -> pyr.PMap:
    """
    Convert expert demonstration trajectory to immutable ACBO format.
    
    Args:
        trajectory: Expert demonstration trajectory dictionary
        
    Returns:
        Immutable trajectory representation using pyrsistent
    """
    return convert_to_acbo_format(trajectory)


# Legacy compatibility - re-export main functions
__all__ = [
    'run_full_parent_scale_algorithm',
    'run_parent_discovery', 
    'run_batch_expert_demonstrations',
    'convert_trajectory_to_acbo_format'
]