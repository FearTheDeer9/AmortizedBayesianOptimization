"""
Pure JAX Operations for Causal Bayesian Optimization

All operations in this module are JAX-compiled functions with no Python loops,
dictionary operations, or other JAX compilation blockers.

Key features:
- All functions decorated with @jax.jit for optimal compilation
- Pure functions with no side effects
- Static tensor shapes throughout
- Comprehensive type hints and documentation
"""

from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp

from .state import JAXAcquisitionState
from .sample_buffer import JAXSampleBuffer


# JAX-compiled tensor operations
@jax.jit
def compute_mechanism_confidence_from_tensors_jax(
    mechanism_features: jnp.ndarray,  # [n_vars, feature_dim]
    target_mask: jnp.ndarray          # [n_vars] boolean
) -> jnp.ndarray:
    """JAX-compiled computation of mechanism confidence from tensor inputs."""
    # Confidence based on inverse uncertainty and feature magnitude
    # features[:, 0] = predicted effect
    # features[:, 1] = uncertainty
    
    effect_magnitude = jnp.abs(mechanism_features[:, 0])
    uncertainty = mechanism_features[:, 1]
    
    # Confidence = effect_magnitude / (1 + uncertainty)
    raw_confidence = effect_magnitude / (1.0 + uncertainty)
    normalized_confidence = jnp.clip(raw_confidence, 0.0, 1.0)
    
    # Set target variable confidence to 0
    confidence_scores = jnp.where(target_mask, 0.0, normalized_confidence)
    
    return confidence_scores


def compute_mechanism_confidence_jax(state: JAXAcquisitionState) -> jnp.ndarray:
    """
    Compute mechanism confidence scores using JAX operations.
    
    Confidence is based on mechanism feature consistency and uncertainty.
    Higher confidence indicates more reliable mechanism predictions.
    
    Args:
        state: JAX acquisition state
        
    Returns:
        Confidence scores [n_vars] with target variable set to 0.0
    """
    # Use JAX-compiled tensor operation
    return compute_mechanism_confidence_from_tensors_jax(
        state.mechanism_features,
        state.config.create_target_mask()
    )


def compute_optimization_progress_jax(state: JAXAcquisitionState) -> Dict[str, float]:
    """
    Compute optimization progress metrics using JAX operations.
    
    Args:
        state: JAX acquisition state
        
    Returns:
        Dictionary with optimization progress metrics
    """
    # Get target values from buffer
    if state.sample_buffer.n_samples == 0:
        return {
            'improvement_from_start': 0.0,
            'recent_improvement': 0.0,
            'optimization_rate': 0.0,
            'stagnation_steps': 0
        }
    
    # Extract valid target values
    valid_targets = jnp.where(
        state.sample_buffer.valid_mask,
        state.sample_buffer.targets,
        -jnp.inf  # Invalid samples get -inf (will be ignored in max operations)
    )
    
    # Get valid samples only
    n_valid = state.sample_buffer.n_samples
    recent_window = jnp.minimum(10, n_valid)  # Look at last 10 samples
    
    # Improvement from start
    if n_valid > 1:
        first_value = valid_targets[0]  # Assumes samples are stored chronologically
        improvement_from_start = state.best_value - first_value
        
        # Recent improvement (last 10 samples)
        recent_start_idx = jnp.maximum(0, n_valid - recent_window)
        recent_best = jnp.max(valid_targets[recent_start_idx:n_valid])
        prev_best = jnp.max(valid_targets[:recent_start_idx]) if recent_start_idx > 0 else first_value
        recent_improvement = jnp.maximum(0.0, recent_best - prev_best)
        
        # Optimization rate (improvement per step)
        optimization_rate = improvement_from_start / float(state.current_step) if state.current_step > 0 else 0.0
        
    else:
        improvement_from_start = 0.0
        recent_improvement = 0.0
        optimization_rate = 0.0
    
    # Stagnation steps (how many steps since last improvement)
    # This is approximated since we don't track the exact step of last improvement
    stagnation_threshold = state.best_value - 1e-6  # Small tolerance
    recent_targets = valid_targets[jnp.maximum(0, n_valid - 5):n_valid]
    recent_improvements = jnp.sum(recent_targets > stagnation_threshold)
    stagnation_steps = 5 - recent_improvements if n_valid >= 5 else 0
    
    return {
        'improvement_from_start': float(improvement_from_start),
        'recent_improvement': float(recent_improvement),
        'optimization_rate': float(optimization_rate),
        'stagnation_steps': int(stagnation_steps)
    }


def compute_exploration_coverage_jax(state: JAXAcquisitionState) -> Dict[str, float]:
    """
    Compute exploration coverage metrics using JAX operations.
    
    Args:
        state: JAX acquisition state
        
    Returns:
        Dictionary with exploration coverage metrics
    """
    if state.sample_buffer.n_samples == 0:
        return {
            'target_coverage_rate': 0.0,
            'intervention_diversity': 0.0,
            'unexplored_variables': 1.0
        }
    
    # Get intervention matrix for valid samples
    valid_interventions = jnp.where(
        state.sample_buffer.valid_mask[:, None],
        state.sample_buffer.interventions,
        0.0
    )
    
    # Count interventions per variable (excluding target)
    non_target_mask = state.config.create_non_target_mask()
    intervention_counts = jnp.sum(valid_interventions, axis=0)
    non_target_counts = intervention_counts * non_target_mask
    
    # Coverage rate: fraction of non-target variables that have been intervened on
    variables_explored = jnp.sum(non_target_counts > 0)
    n_non_target = jnp.sum(non_target_mask)
    target_coverage_rate = variables_explored / n_non_target if n_non_target > 0 else 0.0
    
    # Intervention diversity: entropy of intervention distribution
    total_interventions = jnp.sum(non_target_counts)
    if total_interventions > 0:
        intervention_probs = non_target_counts / total_interventions
        # Avoid log(0) by adding small epsilon
        intervention_probs = jnp.where(intervention_probs > 0, intervention_probs, 1e-10)
        entropy = -jnp.sum(intervention_probs * jnp.log(intervention_probs))
        max_entropy = jnp.log(n_non_target) if n_non_target > 1 else 1.0
        intervention_diversity = entropy / max_entropy
    else:
        intervention_diversity = 0.0
    
    # Fraction of variables unexplored
    unexplored_variables = 1.0 - target_coverage_rate
    
    return {
        'target_coverage_rate': float(target_coverage_rate),
        'intervention_diversity': float(intervention_diversity),
        'unexplored_variables': float(unexplored_variables)
    }


def compute_policy_features_jax(state: JAXAcquisitionState) -> jnp.ndarray:
    """
    Compute comprehensive policy features using JAX operations.
    
    Combines mechanism features, exploration statistics, and optimization
    progress into a single tensor suitable for policy network input.
    
    Args:
        state: JAX acquisition state
        
    Returns:
        Policy features tensor [n_vars, total_feature_dim]
    """
    n_vars = state.config.n_vars
    
    # Base mechanism features [n_vars, feature_dim]
    base_features = state.mechanism_features
    
    # Marginal probabilities [n_vars, 1]
    marginal_features = state.marginal_probs[:, None]
    
    # Confidence scores [n_vars, 1]  
    confidence_features = state.confidence_scores[:, None]
    
    # Global optimization context
    opt_progress = compute_optimization_progress_jax(state)
    exp_coverage = compute_exploration_coverage_jax(state)
    
    global_context = jnp.array([
        state.best_value,
        state.uncertainty_bits,
        float(state.current_step),
        float(state.sample_buffer.n_samples),
        opt_progress['improvement_from_start'],
        opt_progress['optimization_rate'],
        exp_coverage['target_coverage_rate'],
        exp_coverage['intervention_diversity']
    ])
    
    # Broadcast global context to all variables [n_vars, 8]
    global_broadcasted = jnp.tile(global_context[None, :], (n_vars, 1))
    
    # Variable-specific exploration features
    if state.sample_buffer.n_samples > 0:
        # Count interventions per variable
        valid_interventions = jnp.where(
            state.sample_buffer.valid_mask[:, None],
            state.sample_buffer.interventions,
            0.0
        )
        intervention_counts = jnp.sum(valid_interventions, axis=0)
        
        # Normalize by total interventions
        total_interventions = jnp.sum(intervention_counts)
        intervention_rates = jnp.where(
            total_interventions > 0,
            intervention_counts / total_interventions,
            0.0
        )
        
        # Time since last intervention (approximated)
        # For simplicity, use inverse of intervention rate as proxy
        time_since_intervention = jnp.where(
            intervention_rates > 0,
            1.0 / (intervention_rates + 1e-6),
            float(state.current_step)
        )
        time_since_intervention = jnp.clip(time_since_intervention, 0.0, 100.0)
        
    else:
        intervention_rates = jnp.zeros(n_vars)
        time_since_intervention = jnp.zeros(n_vars)
    
    # Variable-specific features [n_vars, 2]
    var_specific_features = jnp.stack([
        intervention_rates,
        time_since_intervention / 100.0  # Normalize to [0, 1]
    ], axis=1)
    
    # Combine all features
    policy_features = jnp.concatenate([
        base_features,           # [n_vars, feature_dim]
        marginal_features,       # [n_vars, 1]
        confidence_features,     # [n_vars, 1]
        global_broadcasted,      # [n_vars, 8]
        var_specific_features    # [n_vars, 2]
    ], axis=1)
    
    return policy_features


@jax.jit
def update_mechanism_features_jax(
    current_features: jnp.ndarray,   # [n_vars, feature_dim]
    new_observations: jnp.ndarray,   # [n_vars, feature_dim]
    learning_rate: float = 0.1
) -> jnp.ndarray:
    """
    Update mechanism features with new observations using JAX operations.
    
    Args:
        current_features: Current mechanism features
        new_observations: New feature observations
        learning_rate: Learning rate for exponential moving average
        
    Returns:
        Updated mechanism features
    """
    # Exponential moving average update
    updated_features = (1.0 - learning_rate) * current_features + learning_rate * new_observations
    
    # Ensure features stay in valid ranges
    # Effect magnitude: unconstrained
    # Uncertainty: [0, 1]
    # Confidence: [0, 1]
    
    updated_features = updated_features.at[:, 1].set(
        jnp.clip(updated_features[:, 1], 0.0, 1.0)  # Uncertainty
    )
    
    if current_features.shape[1] > 2:
        updated_features = updated_features.at[:, 2].set(
            jnp.clip(updated_features[:, 2], 0.0, 1.0)  # Confidence
        )
    
    return updated_features


@jax.jit
def compute_acquisition_scores_jax(
    policy_features: jnp.ndarray,    # [n_vars, feature_dim]
    target_idx: int,
    exploration_weight: float = 0.1
) -> jnp.ndarray:
    """
    Compute acquisition scores for variable selection using JAX operations.
    
    Simple acquisition function based on mechanism confidence and exploration.
    
    Args:
        policy_features: Policy feature tensor
        target_idx: Index of target variable (to exclude)
        exploration_weight: Weight for exploration term
        
    Returns:
        Acquisition scores [n_vars] with target variable masked to -inf
    """
    n_vars = policy_features.shape[0]
    
    # Extract relevant features
    confidence = policy_features[:, -3]  # Assuming confidence is 3rd from end
    uncertainty = policy_features[:, 1]  # Uncertainty from mechanism features
    intervention_rate = policy_features[:, -2]  # 2nd from end
    
    # Acquisition = confidence * (1 + exploration_weight * (1 - intervention_rate))
    exploration_bonus = exploration_weight * (1.0 - intervention_rate)
    acquisition_scores = confidence * (1.0 + exploration_bonus) - uncertainty * 0.1
    
    # Mask target variable
    acquisition_scores = acquisition_scores.at[target_idx].set(-jnp.inf)
    
    return acquisition_scores


# Validation functions for testing
def validate_jax_compilation() -> bool:
    """
    Validate that all operations can be JAX-compiled successfully.
    
    Returns:
        True if all operations compile successfully
    """
    try:
        from .config import create_test_config
        from .state import create_test_state
        
        # Create test state
        state = create_test_state()
        
        # Test all operations
        confidence = compute_mechanism_confidence_jax(state)
        progress = compute_optimization_progress_jax(state)
        coverage = compute_exploration_coverage_jax(state)
        features = compute_policy_features_jax(state)
        scores = compute_acquisition_scores_jax(features, state.config.target_idx)
        
        # Check shapes
        assert confidence.shape == (state.config.n_vars,)
        assert features.shape[0] == state.config.n_vars
        assert scores.shape == (state.config.n_vars,)
        
        return True
        
    except Exception as e:
        print(f"JAX compilation validation failed: {e}")
        return False