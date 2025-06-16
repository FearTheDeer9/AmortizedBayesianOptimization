#!/usr/bin/env python3
"""
PARENT_SCALE Validation Utilities

This module contains pure functions for validating PARENT_SCALE algorithm
results, comparing trajectories, and ensuring data quality for expert
demonstration collection.
"""

from typing import Dict, Any, List, Tuple, Optional
import statistics
import numpy as np


def validate_trajectory_completeness(trajectory: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a trajectory contains all required fields for training.
    
    Args:
        trajectory: Expert demonstration trajectory dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_fields = [
        'algorithm',
        'target_variable', 
        'iterations',
        'status'
    ]
    
    # Additional fields required for successful trajectories
    success_fields = [
        'intervention_sequence',
        'intervention_values',
        'target_outcomes',
        'global_optimum_trajectory',
        'final_optimum',
        'total_interventions',
        'convergence_rate',
        'exploration_efficiency'
    ]
    
    errors = []
    
    # Check basic required fields
    for field in required_fields:
        if field not in trajectory:
            errors.append(f"Missing required field: {field}")
    
    # Check success-specific fields if trajectory completed
    if trajectory.get('status') == 'completed':
        for field in success_fields:
            if field not in trajectory:
                errors.append(f"Missing success field: {field}")
    
    # Validate data consistency for successful trajectories
    if trajectory.get('status') == 'completed':
        validation_errors = validate_trajectory_consistency(trajectory)
        errors.extend(validation_errors)
    
    return len(errors) == 0, errors


def validate_trajectory_consistency(trajectory: Dict[str, Any]) -> List[str]:
    """
    Validate internal consistency of trajectory data.
    
    Args:
        trajectory: Expert demonstration trajectory dictionary
        
    Returns:
        List of error messages (empty if consistent)
    """
    errors = []
    
    # Extract trajectory components
    intervention_sequence = trajectory.get('intervention_sequence', [])
    intervention_values = trajectory.get('intervention_values', [])
    target_outcomes = trajectory.get('target_outcomes', [])
    global_opt = trajectory.get('global_optimum_trajectory', [])
    
    # Check length consistency
    if len(intervention_sequence) != len(intervention_values):
        errors.append(f"Intervention sequence length ({len(intervention_sequence)}) != values length ({len(intervention_values)})")
    
    if len(intervention_sequence) != len(target_outcomes):
        errors.append(f"Intervention sequence length ({len(intervention_sequence)}) != outcomes length ({len(target_outcomes)})")
    
    if len(global_opt) > 0 and len(global_opt) != len(target_outcomes):
        errors.append(f"Global optimum trajectory length ({len(global_opt)}) != outcomes length ({len(target_outcomes)})")
    
    # Check that final optimum matches trajectory
    final_optimum = trajectory.get('final_optimum')
    if final_optimum is not None and global_opt:
        if abs(final_optimum - global_opt[-1]) > 1e-10:
            errors.append(f"Final optimum ({final_optimum}) != last global optimum ({global_opt[-1]})")
    
    # Check total interventions consistency
    total_interventions = trajectory.get('total_interventions', 0)
    if total_interventions != len(intervention_sequence):
        errors.append(f"Total interventions ({total_interventions}) != sequence length ({len(intervention_sequence)})")
    
    # Check that convergence rate is reasonable
    convergence_rate = trajectory.get('convergence_rate', 0)
    if not (0 <= convergence_rate <= 1):
        errors.append(f"Convergence rate ({convergence_rate}) not in [0, 1]")
    
    # Check that exploration efficiency is reasonable
    exploration_efficiency = trajectory.get('exploration_efficiency', 0)
    if not (0 <= exploration_efficiency <= 1):
        errors.append(f"Exploration efficiency ({exploration_efficiency}) not in [0, 1]")
    
    return errors


def validate_algorithm_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate PARENT_SCALE algorithm configuration parameters.
    
    Args:
        config: Algorithm configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required configuration fields
    required_config_fields = ['nonlinear', 'causal_prior', 'individual', 'use_doubly_robust']
    for field in required_config_fields:
        if field not in config:
            errors.append(f"Missing configuration field: {field}")
    
    # Check that boolean fields are actually boolean
    boolean_fields = ['nonlinear', 'causal_prior', 'individual', 'use_doubly_robust']
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            errors.append(f"Configuration field {field} should be boolean, got {type(config[field])}")
    
    return len(errors) == 0, errors


def compare_trajectories(
    trajectory1: Dict[str, Any], 
    trajectory2: Dict[str, Any],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare two trajectories for similarity (useful for validating integration).
    
    Args:
        trajectory1: First trajectory
        trajectory2: Second trajectory  
        tolerance: Numerical tolerance for comparisons
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'trajectories_valid': True,
        'final_optimum_match': False,
        'intervention_count_match': False,
        'convergence_behavior_match': False,
        'differences': []
    }
    
    # Check if both trajectories are valid
    valid1, errors1 = validate_trajectory_completeness(trajectory1)
    valid2, errors2 = validate_trajectory_completeness(trajectory2)
    
    if not valid1 or not valid2:
        comparison['trajectories_valid'] = False
        comparison['differences'].extend(errors1)
        comparison['differences'].extend(errors2)
        return comparison
    
    # Compare final optimum values
    final1 = trajectory1.get('final_optimum')
    final2 = trajectory2.get('final_optimum')
    
    if final1 is not None and final2 is not None:
        comparison['final_optimum_difference'] = abs(final1 - final2)
        comparison['final_optimum_match'] = comparison['final_optimum_difference'] < tolerance
    else:
        comparison['differences'].append("Cannot compare final optimum - missing values")
    
    # Compare intervention counts
    count1 = trajectory1.get('total_interventions', 0)
    count2 = trajectory2.get('total_interventions', 0)
    comparison['intervention_count_match'] = count1 == count2
    
    if not comparison['intervention_count_match']:
        comparison['differences'].append(f"Intervention counts differ: {count1} vs {count2}")
    
    # Compare convergence behavior
    rate1 = trajectory1.get('convergence_rate', 0)
    rate2 = trajectory2.get('convergence_rate', 0)
    
    convergence_diff = abs(rate1 - rate2)
    comparison['convergence_rate_difference'] = convergence_diff
    comparison['convergence_behavior_match'] = convergence_diff < 0.2  # Allow some variation
    
    # Compare trajectory shapes
    global_opt1 = trajectory1.get('global_optimum_trajectory', [])
    global_opt2 = trajectory2.get('global_optimum_trajectory', [])
    
    if len(global_opt1) == len(global_opt2) and len(global_opt1) > 0:
        trajectory_correlation = np.corrcoef(global_opt1, global_opt2)[0, 1]
        comparison['trajectory_correlation'] = trajectory_correlation
        comparison['trajectories_correlated'] = trajectory_correlation > 0.8
    else:
        comparison['differences'].append("Cannot compare trajectory shapes - different lengths")
    
    return comparison


def compute_trajectory_statistics(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across multiple trajectories.
    
    Args:
        trajectories: List of expert demonstration trajectories
        
    Returns:
        Dictionary with summary statistics
    """
    # Filter successful trajectories
    successful = [t for t in trajectories if t.get('status') == 'completed']
    
    if not successful:
        return {
            'total_trajectories': len(trajectories),
            'successful_trajectories': 0,
            'success_rate': 0.0,
            'error': 'No successful trajectories to analyze'
        }
    
    # Extract metrics from successful trajectories
    final_optimums = [t['final_optimum'] for t in successful]
    convergence_rates = [t['convergence_rate'] for t in successful]
    exploration_efficiencies = [t['exploration_efficiency'] for t in successful]
    total_interventions = [t['total_interventions'] for t in successful]
    
    statistics_dict = {
        'total_trajectories': len(trajectories),
        'successful_trajectories': len(successful),
        'success_rate': len(successful) / len(trajectories),
        
        # Final optimum statistics
        'final_optimum_mean': statistics.mean(final_optimums),
        'final_optimum_std': statistics.stdev(final_optimums) if len(final_optimums) > 1 else 0.0,
        'final_optimum_min': min(final_optimums),
        'final_optimum_max': max(final_optimums),
        
        # Convergence statistics
        'convergence_rate_mean': statistics.mean(convergence_rates),
        'convergence_rate_std': statistics.stdev(convergence_rates) if len(convergence_rates) > 1 else 0.0,
        
        # Exploration statistics
        'exploration_efficiency_mean': statistics.mean(exploration_efficiencies),
        'exploration_efficiency_std': statistics.stdev(exploration_efficiencies) if len(exploration_efficiencies) > 1 else 0.0,
        
        # Intervention statistics
        'interventions_mean': statistics.mean(total_interventions),
        'interventions_std': statistics.stdev(total_interventions) if len(total_interventions) > 1 else 0.0,
        'interventions_min': min(total_interventions),
        'interventions_max': max(total_interventions)
    }
    
    return statistics_dict


def validate_demonstration_quality(
    trajectories: List[Dict[str, Any]],
    min_success_rate: float = 0.9,
    min_convergence_rate: float = 0.3,
    min_exploration_efficiency: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the overall quality of a set of expert demonstrations.
    
    Args:
        trajectories: List of expert demonstration trajectories
        min_success_rate: Minimum required success rate
        min_convergence_rate: Minimum required average convergence rate
        min_exploration_efficiency: Minimum required average exploration efficiency
        
    Returns:
        Tuple of (meets_quality_standards, detailed_assessment)
    """
    stats = compute_trajectory_statistics(trajectories)
    
    if 'error' in stats:
        return False, stats
    
    # Quality checks
    quality_checks = {
        'success_rate_ok': stats['success_rate'] >= min_success_rate,
        'convergence_rate_ok': stats['convergence_rate_mean'] >= min_convergence_rate,
        'exploration_efficiency_ok': stats['exploration_efficiency_mean'] >= min_exploration_efficiency,
        'result_consistency_ok': stats['final_optimum_std'] < 1.0,  # Results shouldn't vary too much
        'sufficient_data_ok': stats['successful_trajectories'] >= 10  # Need minimum sample size
    }
    
    all_checks_pass = all(quality_checks.values())
    
    assessment = {
        **stats,
        'quality_checks': quality_checks,
        'meets_quality_standards': all_checks_pass,
        'recommendations': []
    }
    
    # Generate recommendations based on failed checks
    if not quality_checks['success_rate_ok']:
        assessment['recommendations'].append(
            f"Improve algorithm parameters to increase success rate (current: {stats['success_rate']:.1%}, target: {min_success_rate:.1%})"
        )
    
    if not quality_checks['convergence_rate_ok']:
        assessment['recommendations'].append(
            f"Increase number of iterations or improve convergence (current rate: {stats['convergence_rate_mean']:.1%})"
        )
    
    if not quality_checks['exploration_efficiency_ok']:
        assessment['recommendations'].append(
            f"Improve exploration diversity (current efficiency: {stats['exploration_efficiency_mean']:.1%})"
        )
    
    if not quality_checks['result_consistency_ok']:
        assessment['recommendations'].append(
            f"Results vary too much - check algorithm determinism (std: {stats['final_optimum_std']:.3f})"
        )
    
    if not quality_checks['sufficient_data_ok']:
        assessment['recommendations'].append(
            f"Collect more successful demonstrations (current: {stats['successful_trajectories']}, recommended: ≥10)"
        )
    
    return all_checks_pass, assessment


def diagnose_trajectory_failure(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose why a trajectory failed and suggest fixes.
    
    Args:
        trajectory: Failed trajectory dictionary
        
    Returns:
        Dictionary with diagnosis and recommendations
    """
    diagnosis = {
        'failure_type': 'unknown',
        'error_message': trajectory.get('error', 'No error message provided'),
        'likely_causes': [],
        'recommendations': []
    }
    
    error_message = trajectory.get('error', '').lower()
    
    # Classify failure type based on error message
    if 'import' in error_message or 'parent_scale not available' in error_message:
        diagnosis['failure_type'] = 'missing_dependencies'
        diagnosis['likely_causes'] = ['PARENT_SCALE not properly installed', 'Path configuration issues']
        diagnosis['recommendations'] = [
            'Ensure external/parent_scale directory exists',
            'Check PARENT_SCALE installation',
            'Verify import paths are correct'
        ]
    
    elif 'data' in error_message or 'missing' in error_message:
        diagnosis['failure_type'] = 'data_generation_failure'
        diagnosis['likely_causes'] = ['Insufficient data', 'Data generation parameters too small']
        diagnosis['recommendations'] = [
            'Increase n_observational samples',
            'Increase n_interventional samples',
            'Check data generation seed'
        ]
    
    elif 'algorithm' in error_message or 'optimization' in error_message:
        diagnosis['failure_type'] = 'algorithm_failure'
        diagnosis['likely_causes'] = ['Optimization convergence issues', 'Invalid parameter configuration']
        diagnosis['recommendations'] = [
            'Check algorithm configuration parameters',
            'Try different random seeds',
            'Increase number of iterations'
        ]
    
    elif 'memory' in error_message or 'out of memory' in error_message:
        diagnosis['failure_type'] = 'resource_failure'
        diagnosis['likely_causes'] = ['Insufficient memory', 'Problem size too large']
        diagnosis['recommendations'] = [
            'Reduce sample sizes',
            'Use smaller graphs',
            'Close other applications'
        ]
    
    else:
        diagnosis['failure_type'] = 'unknown_error'
        diagnosis['likely_causes'] = ['Unexpected algorithm behavior']
        diagnosis['recommendations'] = [
            'Check algorithm logs for details',
            'Try with default parameters',
            'Report issue if reproducible'
        ]
    
    return diagnosis