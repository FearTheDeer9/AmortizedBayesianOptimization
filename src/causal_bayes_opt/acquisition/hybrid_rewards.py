#!/usr/bin/env python3
"""
Hybrid Reward System for Mechanism-Aware ACBO

This module implements the enhanced reward system that combines supervised learning
signals (using ground truth during training) with observable signals (no ground truth,
for robustness) to guide mechanism-aware intervention selection.

The hybrid approach provides:
1. Supervised signals: Use ground truth mechanism information during training
2. Observable signals: Use only observable outcomes for deployment robustness
3. Configurable weighting: Balance between different signal types

Architecture Enhancement Pivot - Part B: Hybrid Reward System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, FrozenSet, Tuple, Union
import statistics

import jax.numpy as jnp
import pyrsistent as pyr

from .verifiable_rewards import SimpleRewardComponents
from ..avici_integration.parent_set.mechanism_aware import MechanismPrediction
from ..data_structures.sample import get_values, get_intervention_targets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridRewardConfig:
    """Configuration for hybrid reward system."""
    
    # Signal type controls
    use_supervised_signals: bool = True
    use_observable_signals: bool = True
    
    # Supervised signal weights (use ground truth during training)
    supervised_parent_weight: float = 1.0
    supervised_mechanism_weight: float = 0.8
    
    # Observable signal weights (no ground truth, for robustness)
    posterior_confidence_weight: float = 0.5
    causal_effect_weight: float = 0.6
    mechanism_consistency_weight: float = 0.4
    
    # Thresholds and parameters
    effect_threshold: float = 0.5  # Minimum effect size to be considered significant
    confidence_threshold: float = 0.7  # Minimum confidence for mechanism predictions
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.use_supervised_signals and not self.use_observable_signals:
            raise ValueError("At least one signal type must be enabled")
        
        weights = [
            self.supervised_parent_weight,
            self.supervised_mechanism_weight,
            self.posterior_confidence_weight,
            self.causal_effect_weight,
            self.mechanism_consistency_weight
        ]
        
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")


@dataclass(frozen=True)
class HybridRewardComponents:
    """Decomposed hybrid reward components for analysis."""
    
    # Supervised components (use ground truth)
    supervised_parent_reward: float
    supervised_mechanism_reward: float
    
    # Observable components (no ground truth)
    posterior_confidence_reward: float
    causal_effect_reward: float
    mechanism_consistency_reward: float
    
    # Combined result
    total_reward: float
    
    # Metadata for analysis
    metadata: Dict[str, Any]
    
    def summary(self) -> Dict[str, Any]:
        """Create human-readable summary."""
        return {
            'total_reward': self.total_reward,
            'supervised_total': self.supervised_parent_reward + self.supervised_mechanism_reward,
            'observable_total': (self.posterior_confidence_reward + 
                               self.causal_effect_reward + 
                               self.mechanism_consistency_reward),
            'component_breakdown': {
                'supervised_parent': self.supervised_parent_reward,
                'supervised_mechanism': self.supervised_mechanism_reward,
                'posterior_confidence': self.posterior_confidence_reward,
                'causal_effect': self.causal_effect_reward,
                'mechanism_consistency': self.mechanism_consistency_reward
            },
            'metadata': self.metadata
        }


# ============================================================================
# Supervised Reward Components (Use Ground Truth During Training)
# ============================================================================

def supervised_mechanism_impact_reward(
    intervention_targets: FrozenSet[str],
    intervention_values: Dict[str, float],
    true_mechanism_info: Dict[str, Any],
    target_variable: str,
    impact_threshold: float = 0.5
) -> float:
    """
    Reward interventions based on true mechanism impact.
    
    Uses ground truth mechanism information to weight interventions by their
    actual causal impact on the target variable. This provides a strong 
    learning signal during training.
    
    Args:
        intervention_targets: Variables being intervened upon
        intervention_values: Values for intervention
        true_mechanism_info: Ground truth mechanism information
        target_variable: Target variable name
        impact_threshold: Minimum impact for reward
        
    Returns:
        Reward based on true mechanism impact [0, 1]
    """
    if target_variable not in true_mechanism_info:
        return 0.0
    
    true_mech = true_mechanism_info[target_variable]
    true_parents = true_mech.get('parents', frozenset())
    true_coefficients = true_mech.get('coefficients', {})
    
    # Calculate total impact of intervention
    total_impact = 0.0
    
    for var in intervention_targets:
        if var in true_parents and var in true_coefficients:
            # Impact = |coefficient| * |intervention_value|
            coeff = abs(true_coefficients[var])
            value = abs(intervention_values.get(var, 0.0))
            impact = coeff * value
            total_impact += impact
    
    # Normalize by reasonable maximum impact (current intervention values)
    max_intervention_value = max(abs(v) for v in intervention_values.values()) if intervention_values else 1.0
    max_possible_impact = sum(abs(coeff) for coeff in true_coefficients.values()) * max_intervention_value
    if max_possible_impact == 0:
        return 0.0
    
    normalized_impact = min(total_impact / max_possible_impact, 1.0)
    
    # Apply threshold - reward scales with normalized impact
    if normalized_impact >= impact_threshold:
        return normalized_impact
    else:
        # Give partial credit proportional to impact
        return normalized_impact * 0.8  # 80% credit for below-threshold impacts


def supervised_mechanism_discovery_reward(
    intervention_targets: FrozenSet[str],
    current_predictions: List[MechanismPrediction],
    edge_uncertainty: Dict[str, float],
    true_parents: FrozenSet[str],
    target_variable: str,
    uncertainty_threshold: float = 0.5
) -> float:
    """
    Reward interventions on uncertain edges that are actually causal.
    
    Encourages exploration of high-uncertainty edges that are true parents,
    using ground truth to guide structure discovery.
    
    Args:
        intervention_targets: Variables being intervened upon
        current_predictions: Current mechanism predictions
        edge_uncertainty: Uncertainty about each edge
        true_parents: True parent variables (ground truth)
        target_variable: Target variable name
        uncertainty_threshold: Minimum uncertainty for reward
        
    Returns:
        Reward for exploring uncertain true edges [0, 1]
    """
    total_reward = 0.0
    
    for var in intervention_targets:
        if var == target_variable:
            continue  # Can't intervene on target itself
        
        # Check if this is a true parent with high uncertainty
        is_true_parent = var in true_parents
        uncertainty = edge_uncertainty.get(var, 0.0)
        
        if is_true_parent and uncertainty >= uncertainty_threshold:
            # Reward proportional to uncertainty
            total_reward += uncertainty
    
    # Normalize by number of intervention targets
    if len(intervention_targets) > 0:
        return min(total_reward / len(intervention_targets), 1.0)
    else:
        return 0.0


# ============================================================================
# Observable Reward Components (No Ground Truth, For Robustness)
# ============================================================================

def posterior_confidence_reward(
    current_posterior: jnp.ndarray,
    next_posterior: jnp.ndarray
) -> float:
    """
    Reward interventions that reduce uncertainty in parent set posterior.
    
    Uses only observable posterior distributions, no ground truth required.
    Measures information gain through entropy reduction.
    
    Args:
        current_posterior: Posterior before intervention [k]
        next_posterior: Posterior after intervention [k]
        
    Returns:
        Reward for uncertainty reduction [0, 1]
    """
    # Compute entropy (uncertainty) of both posteriors
    def entropy(probs):
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-8
        return -jnp.sum(probs * jnp.log(probs))
    
    current_entropy = entropy(current_posterior)
    next_entropy = entropy(next_posterior)
    
    # Reward is proportional to entropy reduction
    entropy_reduction = max(0.0, float(current_entropy - next_entropy))
    
    # Normalize by maximum possible entropy reduction
    max_entropy = float(jnp.log(len(current_posterior)))  # Uniform distribution entropy
    
    if max_entropy > 0:
        return float(min(entropy_reduction / max_entropy, 1.0))
    else:
        return 0.0


def causal_effect_discovery_reward(
    intervention_outcome: float,
    baseline_prediction: float,
    predicted_effect: float,
    effect_threshold: float = 0.5
) -> float:
    """
    Reward interventions that reveal strong, predictable causal effects.
    
    Uses only observable outcomes and predictions, no ground truth required.
    Rewards consistency between predicted and observed effects.
    
    Args:
        intervention_outcome: Observed target value after intervention
        baseline_prediction: Predicted target value without intervention
        predicted_effect: Model's predicted effect of intervention
        effect_threshold: Minimum effect size for reward
        
    Returns:
        Reward for discovering predictable effects [0, 1]
    """
    # Calculate observed effect
    observed_effect = abs(intervention_outcome - baseline_prediction)
    
    # If effect is below threshold, give partial reward
    if observed_effect < effect_threshold:
        return observed_effect / effect_threshold * 0.3  # Partial credit
    
    # Calculate prediction accuracy
    prediction_error = abs(observed_effect - abs(predicted_effect))
    
    # Reward is proportional to effect size and prediction accuracy
    effect_reward = min(observed_effect / 10.0, 1.0)  # Normalize by max expected effect
    accuracy_reward = max(0.0, 1.0 - prediction_error / max(observed_effect, 1.0))
    
    return effect_reward * accuracy_reward


def mechanism_consistency_reward(
    predicted_mechanism: MechanismPrediction,
    observed_effect: float,
    intervention_values: Dict[str, float]
) -> float:
    """
    Reward mechanism predictions that are consistent with observed outcomes.
    
    Uses only predicted mechanisms and observed outcomes, no ground truth.
    Validates mechanism predictions through consistency checks.
    
    Args:
        predicted_mechanism: Predicted mechanism for the intervention
        observed_effect: Observed change in target variable
        intervention_values: Values used in intervention
        
    Returns:
        Reward for consistent mechanism prediction [0, 1]
    """
    if predicted_mechanism.confidence < 0.5:
        return 0.0  # Low confidence predictions get no reward
    
    # Extract predicted parameters
    params = predicted_mechanism.parameters
    
    if predicted_mechanism.mechanism_type == 'linear':
        # For linear mechanisms, predict effect based on coefficients
        coefficients = params.get('coefficients', {})
        predicted_effect = 0.0
        
        for var, value in intervention_values.items():
            if var in coefficients:
                predicted_effect += coefficients[var] * value
        
        # Compare with observed effect
        prediction_error = abs(observed_effect - predicted_effect)
        max_error = max(abs(observed_effect), abs(predicted_effect), 1.0)
        
        accuracy = max(0.0, 1.0 - prediction_error / max_error)
        
        # Weight by confidence
        return accuracy * predicted_mechanism.confidence
    
    elif predicted_mechanism.mechanism_type == 'polynomial':
        # Simplified polynomial prediction (would be more complex in practice)
        # For now, just check if effect direction matches
        if 'coefficients' in params:
            coefficients = params['coefficients']
            
            # Simple direction check
            predicted_direction = sum(coeff * intervention_values.get(var, 0) 
                                    for var, coeff in coefficients.items())
            observed_direction = observed_effect
            
            if (predicted_direction > 0 and observed_direction > 0) or \
               (predicted_direction < 0 and observed_direction < 0) or \
               (abs(predicted_direction) < 0.1 and abs(observed_direction) < 0.1):
                return predicted_mechanism.confidence * 0.7  # Partial credit for direction
            else:
                return 0.0
    
    # For other mechanism types, give partial credit based on confidence
    return predicted_mechanism.confidence * 0.5


# ============================================================================
# Hybrid Reward Integration
# ============================================================================

def compute_hybrid_reward(
    current_state: Any,  # AcquisitionState or mock
    intervention: pyr.PMap,
    outcome: pyr.PMap,
    next_state: Any,  # AcquisitionState or mock
    config: HybridRewardConfig,
    ground_truth: Optional[Dict[str, Any]] = None
) -> HybridRewardComponents:
    """
    Compute hybrid reward combining supervised and observable signals.
    
    This is the main reward function that combines multiple signal types
    based on configuration and availability of ground truth information.
    
    Args:
        current_state: State before intervention
        intervention: Intervention that was applied
        outcome: Observed outcome of intervention
        next_state: State after intervention
        config: Hybrid reward configuration
        ground_truth: Optional ground truth information (for supervised signals)
        
    Returns:
        Decomposed reward components with analysis metadata
    """
    # Initialize components
    supervised_parent_reward_val = 0.0
    supervised_mechanism_reward_val = 0.0
    posterior_confidence_reward_val = 0.0
    causal_effect_reward_val = 0.0
    mechanism_consistency_reward_val = 0.0
    
    # Extract common information
    intervention_targets = intervention.get('targets', frozenset())
    intervention_values = intervention.get('values', {})
    outcome_values = get_values(outcome)
    target_variable = current_state.current_target
    
    metadata = {
        'intervention_targets': list(intervention_targets),
        'intervention_values': dict(intervention_values),
        'target_variable': target_variable,
        'config_mode': {
            'supervised': config.use_supervised_signals,
            'observable': config.use_observable_signals
        }
    }
    
    # Compute supervised signals (if enabled and ground truth available)
    if config.use_supervised_signals and ground_truth is not None:
        # Supervised parent intervention reward
        true_mechanism_info = ground_truth.get('mechanism_info', {})
        if true_mechanism_info:
            supervised_parent_reward_val = supervised_mechanism_impact_reward(
                intervention_targets=intervention_targets,
                intervention_values=intervention_values,
                true_mechanism_info=true_mechanism_info,
                target_variable=target_variable
            )
        
        # Supervised mechanism discovery reward
        scm = ground_truth.get('scm')
        if scm and hasattr(scm, 'get'):
            edges = scm.get('edges', frozenset())
            true_parents = frozenset(parent for parent, child in edges if child == target_variable)
            
            # Create mock edge uncertainty (would be computed from actual predictions)
            edge_uncertainty = {}
            for var in ['X', 'Y', 'Z', 'W']:  # Common variable names
                if var != target_variable:
                    edge_uncertainty[var] = 0.5  # Medium uncertainty
            
            supervised_mechanism_reward_val = supervised_mechanism_discovery_reward(
                intervention_targets=intervention_targets,
                current_predictions=getattr(current_state, 'mechanism_predictions', []),
                edge_uncertainty=edge_uncertainty,
                true_parents=true_parents,
                target_variable=target_variable
            )
    
    # Compute observable signals (if enabled)
    if config.use_observable_signals:
        # Posterior confidence reward
        current_posterior = getattr(current_state, 'parent_posterior', jnp.array([1.0]))
        next_posterior = getattr(next_state, 'parent_posterior', jnp.array([1.0]))
        
        posterior_confidence_reward_val = posterior_confidence_reward(
            current_posterior=current_posterior,
            next_posterior=next_posterior
        )
        
        # Causal effect discovery reward
        current_best = getattr(current_state, 'best_target_value', 0.0)
        next_best = getattr(next_state, 'best_target_value', 0.0)
        target_outcome = outcome_values.get(target_variable, current_best)
        
        # Simple predicted effect (would use actual model predictions)
        predicted_effect = 0.5  # Placeholder
        
        causal_effect_reward_val = causal_effect_discovery_reward(
            intervention_outcome=target_outcome,
            baseline_prediction=current_best,
            predicted_effect=predicted_effect,
            effect_threshold=config.effect_threshold
        )
        
        # Mechanism consistency reward
        mechanism_preds = getattr(current_state, 'mechanism_predictions', [])
        if mechanism_preds:
            # Find relevant mechanism prediction
            relevant_pred = None
            for pred in mechanism_preds:
                if len(pred.parent_set & intervention_targets) > 0:
                    relevant_pred = pred
                    break
            
            if relevant_pred:
                observed_effect = target_outcome - current_best
                mechanism_consistency_reward_val = mechanism_consistency_reward(
                    predicted_mechanism=relevant_pred,
                    observed_effect=observed_effect,
                    intervention_values=intervention_values
                )
    
    # Combine components according to configuration
    total_reward = (
        config.supervised_parent_weight * supervised_parent_reward_val +
        config.supervised_mechanism_weight * supervised_mechanism_reward_val +
        config.posterior_confidence_weight * posterior_confidence_reward_val +
        config.causal_effect_weight * causal_effect_reward_val +
        config.mechanism_consistency_weight * mechanism_consistency_reward_val
    )
    
    metadata.update({
        'component_values': {
            'supervised_parent': supervised_parent_reward_val,
            'supervised_mechanism': supervised_mechanism_reward_val,
            'posterior_confidence': posterior_confidence_reward_val,
            'causal_effect': causal_effect_reward_val,
            'mechanism_consistency': mechanism_consistency_reward_val
        }
    })
    
    return HybridRewardComponents(
        supervised_parent_reward=supervised_parent_reward_val,
        supervised_mechanism_reward=supervised_mechanism_reward_val,
        posterior_confidence_reward=posterior_confidence_reward_val,
        causal_effect_reward=causal_effect_reward_val,
        mechanism_consistency_reward=mechanism_consistency_reward_val,
        total_reward=total_reward,
        metadata=metadata
    )


# ============================================================================
# Factory Functions and Configuration
# ============================================================================

def create_hybrid_reward_config(
    mode: str = "training",
    **kwargs
) -> HybridRewardConfig:
    """
    Create hybrid reward configuration for common use cases.
    
    Args:
        mode: Configuration mode ('training', 'deployment', 'research')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured HybridRewardConfig
    """
    if mode == "training":
        # Use both supervised and observable signals
        return HybridRewardConfig(
            use_supervised_signals=True,
            use_observable_signals=True,
            supervised_parent_weight=1.0,
            supervised_mechanism_weight=0.8,
            posterior_confidence_weight=0.5,
            causal_effect_weight=0.6,
            mechanism_consistency_weight=0.4,
            **kwargs
        )
    
    elif mode == "deployment":
        # Use only observable signals (no ground truth available)
        return HybridRewardConfig(
            use_supervised_signals=False,
            use_observable_signals=True,
            posterior_confidence_weight=0.8,
            causal_effect_weight=1.0,
            mechanism_consistency_weight=0.6,
            **kwargs
        )
    
    elif mode == "research":
        # Use only supervised signals (for studying ground truth effects)
        return HybridRewardConfig(
            use_supervised_signals=True,
            use_observable_signals=False,
            supervised_parent_weight=1.2,
            supervised_mechanism_weight=1.0,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'training', 'deployment', or 'research'")


def create_adaptive_hybrid_config(
    training_phase: str,
    **kwargs
) -> HybridRewardConfig:
    """
    Create adaptive configuration based on training phase.
    
    Args:
        training_phase: Phase of training ('early', 'middle', 'late')
        **kwargs: Additional configuration parameters
        
    Returns:
        Phase-appropriate HybridRewardConfig
    """
    if training_phase == "early":
        # Emphasize supervised signals for initial learning
        return HybridRewardConfig(
            use_supervised_signals=True,
            use_observable_signals=True,
            supervised_parent_weight=1.5,
            supervised_mechanism_weight=1.2,
            posterior_confidence_weight=0.3,
            causal_effect_weight=0.4,
            mechanism_consistency_weight=0.2,
            **kwargs
        )
    
    elif training_phase == "middle":
        # Balanced approach
        return create_hybrid_reward_config(mode="training", **kwargs)
    
    elif training_phase == "late":
        # Emphasize observable signals for robustness
        return HybridRewardConfig(
            use_supervised_signals=True,
            use_observable_signals=True,
            supervised_parent_weight=0.6,
            supervised_mechanism_weight=0.5,
            posterior_confidence_weight=0.8,
            causal_effect_weight=1.0,
            mechanism_consistency_weight=0.7,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown training phase: {training_phase}")


# ============================================================================
# Validation and Analysis
# ============================================================================

def validate_hybrid_reward_consistency(
    reward_history: List[HybridRewardComponents],
    window_size: int = 50
) -> Dict[str, Any]:
    """
    Validate hybrid reward consistency and detect gaming patterns.
    
    Args:
        reward_history: Recent reward components
        window_size: Number of recent rewards to analyze
        
    Returns:
        Validation metrics and gaming detection results
    """
    if not reward_history:
        return {'valid': True, 'warning': 'No reward history'}
    
    recent_rewards = reward_history[-window_size:]
    gaming_issues = []
    
    # Extract component values
    supervised_parent_vals = [r.supervised_parent_reward for r in recent_rewards]
    supervised_mech_vals = [r.supervised_mechanism_reward for r in recent_rewards]
    posterior_conf_vals = [r.posterior_confidence_reward for r in recent_rewards]
    causal_effect_vals = [r.causal_effect_reward for r in recent_rewards]
    mech_consistency_vals = [r.mechanism_consistency_reward for r in recent_rewards]
    total_vals = [r.total_reward for r in recent_rewards]
    
    # Check for gaming patterns
    
    # 1. Suspiciously perfect supervised rewards (gaming ground truth)
    if len(supervised_parent_vals) > 0:
        perfect_supervised_rate = sum(1 for v in supervised_parent_vals if v > 0.95) / len(supervised_parent_vals)
        if perfect_supervised_rate > 0.8:
            gaming_issues.append(f"Suspiciously high perfect supervised reward rate: {perfect_supervised_rate:.2f}")
    
    # 2. Complete lack of observable rewards (not learning from experience)
    if len(posterior_conf_vals) > 0:
        zero_observable_rate = sum(1 for v in posterior_conf_vals if v < 0.05) / len(posterior_conf_vals)
        if zero_observable_rate > 0.9:
            gaming_issues.append(f"Very low observable reward rate: {zero_observable_rate:.2f}")
    
    # 3. Extremely low variance (suggesting constant behavior)
    if len(total_vals) > 5:
        total_variance = statistics.variance(total_vals)
        if total_variance < 0.01:
            gaming_issues.append(f"Very low total reward variance: {total_variance:.4f}")
    
    # 4. Imbalanced component contributions
    if len(recent_rewards) > 10:
        avg_total = statistics.mean(total_vals)
        if avg_total > 0:
            supervised_contribution = statistics.mean([r.supervised_parent_reward + r.supervised_mechanism_reward 
                                                     for r in recent_rewards]) / avg_total
            if supervised_contribution > 0.9:
                gaming_issues.append(f"Over-reliance on supervised signals: {supervised_contribution:.2f}")
    
    # Compute summary statistics
    stats = {}
    if total_vals:
        stats.update({
            'mean_total_reward': statistics.mean(total_vals),
            'reward_variance': statistics.variance(total_vals) if len(total_vals) > 1 else 0.0,
            'n_samples': len(recent_rewards)
        })
    
    if supervised_parent_vals:
        stats['supervised_parent_rate'] = statistics.mean(supervised_parent_vals)
    if posterior_conf_vals:
        stats['posterior_confidence_rate'] = statistics.mean(posterior_conf_vals)
    
    return {
        'valid': len(gaming_issues) == 0,
        'gaming_issues': gaming_issues,
        'statistics': stats
    }


def compare_reward_strategies(
    strategy1_rewards: List[HybridRewardComponents],
    strategy2_rewards: List[HybridRewardComponents],
    strategy1_name: str = "Strategy 1",
    strategy2_name: str = "Strategy 2"
) -> Dict[str, Any]:
    """
    Compare two reward strategies for scientific analysis.
    
    Args:
        strategy1_rewards: Reward history for first strategy
        strategy2_rewards: Reward history for second strategy
        strategy1_name: Name of first strategy
        strategy2_name: Name of second strategy
        
    Returns:
        Comparison metrics and statistical analysis
    """
    if not strategy1_rewards or not strategy2_rewards:
        return {'error': 'Both strategies must have reward history'}
    
    # Extract total rewards
    rewards1 = [r.total_reward for r in strategy1_rewards]
    rewards2 = [r.total_reward for r in strategy2_rewards]
    
    # Basic statistics
    mean1 = statistics.mean(rewards1)
    mean2 = statistics.mean(rewards2)
    
    # Statistical significance (simplified t-test approximation)
    def simple_ttest_p_value(x1, x2):
        """Simplified t-test p-value approximation."""
        if len(x1) < 2 or len(x2) < 2:
            return 1.0
        
        mean_diff = abs(statistics.mean(x1) - statistics.mean(x2))
        pooled_std = (statistics.stdev(x1) + statistics.stdev(x2)) / 2
        
        if pooled_std == 0:
            return 0.0 if mean_diff > 0 else 1.0
        
        # Rough approximation
        t_stat = mean_diff / (pooled_std / ((len(x1) + len(x2)) ** 0.5))
        
        # Very rough p-value approximation
        if t_stat > 2.0:
            return 0.05
        elif t_stat > 1.5:
            return 0.15
        else:
            return 0.5
    
    p_value = simple_ttest_p_value(rewards1, rewards2)
    
    return {
        'strategy1_name': strategy1_name,
        'strategy2_name': strategy2_name,
        'strategy1_mean': mean1,
        'strategy2_mean': mean2,
        'difference': mean1 - mean2,
        'strategy1_samples': len(rewards1),
        'strategy2_samples': len(rewards2),
        'statistical_significance': p_value < 0.05,
        'p_value_approx': p_value
    }


# ============================================================================
# Integration with Existing Reward System
# ============================================================================

def integrate_with_simple_rewards(
    simple_reward_components: SimpleRewardComponents,
    hybrid_reward_components: HybridRewardComponents,
    integration_weight: float = 0.5
) -> float:
    """
    Integrate hybrid rewards with existing simple reward system.
    
    Args:
        simple_reward_components: Existing simple reward components
        hybrid_reward_components: New hybrid reward components
        integration_weight: Weight for hybrid components [0, 1]
        
    Returns:
        Combined reward value
    """
    simple_total = simple_reward_components.total_reward
    hybrid_total = hybrid_reward_components.total_reward
    
    # Weighted combination
    combined_reward = (
        (1.0 - integration_weight) * simple_total +
        integration_weight * hybrid_total
    )
    
    return combined_reward