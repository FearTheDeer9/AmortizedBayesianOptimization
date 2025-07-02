"""
Reward Component Validation Framework.

This module provides utilities for validating individual reward components
by selectively disabling them to test optimization pressure and learning.
This addresses the requirement to ensure both structure learning and target
optimization objectives are properly reinforced.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import jax.numpy as jnp
import pyrsistent as pyr

from ..acquisition.rewards import (
    compute_verifiable_reward, 
    create_default_reward_config,
    RewardComponents
)
from ..acquisition.state import AcquisitionState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentValidationResult:
    """Results from validating a single reward component."""
    component_name: str
    enabled_reward: float
    disabled_reward: float
    component_contribution: float
    validation_passed: bool
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RewardValidationSuite:
    """Complete validation results for all reward components."""
    optimization_result: ComponentValidationResult
    structure_result: ComponentValidationResult
    parent_result: ComponentValidationResult
    exploration_result: ComponentValidationResult
    overall_validation_passed: bool
    metadata: Dict[str, Any]


def create_component_test_configs() -> Dict[str, pyr.PMap]:
    """
    Create test configurations with each component zeroed out.
    
    Returns:
        Dictionary mapping component names to configurations with that component disabled
    """
    base_weights = {
        'optimization': 1.0,
        'structure': 0.5, 
        'parent': 0.3,
        'exploration': 0.1
    }
    
    test_configs = {}
    
    for component in base_weights.keys():
        # Create config with this component zeroed out
        test_weights = base_weights.copy()
        test_weights[component] = 0.0
        
        config = create_default_reward_config(
            optimization_weight=test_weights['optimization'],
            structure_weight=test_weights['structure'],
            parent_weight=test_weights['parent'],
            exploration_weight=test_weights['exploration']
        )
        
        test_configs[f"no_{component}"] = config
    
    # Also create a balanced config for comparison
    test_configs["balanced"] = create_default_reward_config(
        optimization_weight=1.0,
        structure_weight=1.0,  # Equal weight for validation
        parent_weight=0.3,
        exploration_weight=0.1
    )
    
    # Create optimization-only and structure-only configs
    test_configs["optimization_only"] = create_default_reward_config(
        optimization_weight=1.0,
        structure_weight=0.0,
        parent_weight=0.0,
        exploration_weight=0.0
    )
    
    test_configs["structure_only"] = create_default_reward_config(
        optimization_weight=0.0,
        structure_weight=1.0,
        parent_weight=0.0,
        exploration_weight=0.0
    )
    
    return test_configs


def validate_single_component(
    component_name: str,
    state_before: AcquisitionState,
    intervention: pyr.PMap,
    outcome: pyr.PMap,
    state_after: AcquisitionState,
    tolerance: float = 0.01
) -> ComponentValidationResult:
    """
    Validate a single reward component by comparing enabled vs disabled rewards.
    
    Args:
        component_name: Name of component to validate ('optimization', 'structure', 'parent', 'exploration')
        state_before: State before intervention
        intervention: Applied intervention
        outcome: Observed outcome
        state_after: State after intervention
        tolerance: Minimum component contribution to consider significant
        
    Returns:
        Validation result for this component
    """
    try:
        # Create configs with component enabled and disabled
        base_config = create_default_reward_config()
        
        # Config with component disabled (zeroed out)
        disabled_weights = {
            'optimization': 1.0 if component_name != 'optimization' else 0.0,
            'structure': 0.5 if component_name != 'structure' else 0.0,
            'parent': 0.3 if component_name != 'parent' else 0.0,
            'exploration': 0.1 if component_name != 'exploration' else 0.0
        }
        
        disabled_config = create_default_reward_config(
            optimization_weight=disabled_weights['optimization'],
            structure_weight=disabled_weights['structure'],
            parent_weight=disabled_weights['parent'],
            exploration_weight=disabled_weights['exploration']
        )
        
        # Compute rewards with component enabled and disabled
        enabled_result = compute_verifiable_reward(
            state_before, intervention, outcome, state_after, base_config
        )
        
        disabled_result = compute_verifiable_reward(
            state_before, intervention, outcome, state_after, disabled_config
        )
        
        enabled_reward = enabled_result.total_reward
        disabled_reward = disabled_result.total_reward
        
        # Compute component contribution
        component_contribution = abs(enabled_reward - disabled_reward)
        
        # Validation passes if component makes a significant contribution
        validation_passed = component_contribution >= tolerance
        
        # Extract component-specific reward for analysis
        component_specific_reward = 0.0
        if component_name == 'optimization':
            component_specific_reward = enabled_result.optimization_reward
        elif component_name == 'structure':
            component_specific_reward = enabled_result.structure_discovery_reward
        elif component_name == 'parent':
            component_specific_reward = enabled_result.parent_intervention_reward
        elif component_name == 'exploration':
            component_specific_reward = enabled_result.exploration_bonus
        
        metadata = {
            'component_specific_reward': component_specific_reward,
            'enabled_components': enabled_result,
            'disabled_components': disabled_result,
            'expected_contribution': component_specific_reward * disabled_weights.get(component_name, 0.0),
            'tolerance_used': tolerance
        }
        
        logger.debug(
            f"Component {component_name}: enabled={enabled_reward:.3f}, "
            f"disabled={disabled_reward:.3f}, contribution={component_contribution:.3f}, "
            f"passed={validation_passed}"
        )
        
        return ComponentValidationResult(
            component_name=component_name,
            enabled_reward=enabled_reward,
            disabled_reward=disabled_reward,
            component_contribution=component_contribution,
            validation_passed=validation_passed,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error validating component {component_name}: {e}")
        return ComponentValidationResult(
            component_name=component_name,
            enabled_reward=0.0,
            disabled_reward=0.0,
            component_contribution=0.0,
            validation_passed=False,
            metadata={'error': str(e)}
        )


def validate_all_components(
    state_before: AcquisitionState,
    intervention: pyr.PMap,
    outcome: pyr.PMap,
    state_after: AcquisitionState,
    tolerance: float = 0.01
) -> RewardValidationSuite:
    """
    Validate all reward components to ensure proper optimization pressure.
    
    This implements the zero-out test mentioned in requirements: train with
    one reward component zeroed out and verify optimization pressure works
    correctly for the remaining components.
    
    Args:
        state_before: State before intervention
        intervention: Applied intervention
        outcome: Observed outcome
        state_after: State after intervention
        tolerance: Minimum component contribution to consider significant
        
    Returns:
        Complete validation suite results
    """
    try:
        # Validate each component
        optimization_result = validate_single_component(
            'optimization', state_before, intervention, outcome, state_after, tolerance
        )
        
        structure_result = validate_single_component(
            'structure', state_before, intervention, outcome, state_after, tolerance
        )
        
        parent_result = validate_single_component(
            'parent', state_before, intervention, outcome, state_after, tolerance
        )
        
        exploration_result = validate_single_component(
            'exploration', state_before, intervention, outcome, state_after, tolerance
        )
        
        # Overall validation passes if critical components work
        critical_components = [optimization_result, structure_result]
        overall_passed = all(result.validation_passed for result in critical_components)
        
        # Additional validation: ensure components have different impacts
        component_contributions = [
            optimization_result.component_contribution,
            structure_result.component_contribution,
            parent_result.component_contribution,
            exploration_result.component_contribution
        ]
        
        # Check that contributions are meaningfully different
        contribution_variance = float(jnp.var(jnp.array(component_contributions)))
        diversity_threshold = 0.001  # Minimum variance in contributions
        diverse_contributions = contribution_variance > diversity_threshold
        
        metadata = {
            'component_contributions': component_contributions,
            'contribution_variance': contribution_variance,
            'diverse_contributions': diverse_contributions,
            'critical_components_passed': all(r.validation_passed for r in critical_components),
            'all_components_passed': all(r.validation_passed for r in [
                optimization_result, structure_result, parent_result, exploration_result
            ]),
            'tolerance_used': tolerance
        }
        
        overall_passed = overall_passed and diverse_contributions
        
        logger.info(
            f"Reward validation: optimization={optimization_result.validation_passed}, "
            f"structure={structure_result.validation_passed}, "
            f"parent={parent_result.validation_passed}, "
            f"exploration={exploration_result.validation_passed}, "
            f"overall={overall_passed}"
        )
        
        return RewardValidationSuite(
            optimization_result=optimization_result,
            structure_result=structure_result,
            parent_result=parent_result,
            exploration_result=exploration_result,
            overall_validation_passed=overall_passed,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error in complete reward validation: {e}")
        # Return failed validation suite
        failed_result = ComponentValidationResult(
            component_name="failed",
            enabled_reward=0.0,
            disabled_reward=0.0,
            component_contribution=0.0,
            validation_passed=False,
            metadata={'error': str(e)}
        )
        
        return RewardValidationSuite(
            optimization_result=failed_result,
            structure_result=failed_result,
            parent_result=failed_result,
            exploration_result=failed_result,
            overall_validation_passed=False,
            metadata={'error': str(e)}
        )


def run_component_isolation_test(
    reward_computer: Callable,
    test_scenarios: List[Tuple[AcquisitionState, pyr.PMap, pyr.PMap, AcquisitionState]],
    component_name: str,
    expected_behavior: str = "increase"
) -> Dict[str, Any]:
    """
    Run component isolation test to verify specific component behavior.
    
    This creates scenarios where only one component should provide signal
    and verifies that the reward behaves as expected.
    
    Args:
        reward_computer: Function to compute rewards
        test_scenarios: List of (state_before, intervention, outcome, state_after) tuples
        component_name: Component to isolate ('optimization', 'structure', etc.)
        expected_behavior: Expected behavior ("increase", "decrease", "stable")
        
    Returns:
        Test results with pass/fail status
    """
    try:
        # Create config with only this component enabled
        isolated_weights = {
            'optimization': 1.0 if component_name == 'optimization' else 0.0,
            'structure': 1.0 if component_name == 'structure' else 0.0,
            'parent': 1.0 if component_name == 'parent' else 0.0,
            'exploration': 1.0 if component_name == 'exploration' else 0.0
        }
        
        isolated_config = create_default_reward_config(**isolated_weights)
        
        # Run scenarios and collect rewards
        rewards = []
        for state_before, intervention, outcome, state_after in test_scenarios:
            result = reward_computer(
                state_before, intervention, outcome, state_after, isolated_config
            )
            rewards.append(result.total_reward)
        
        # Analyze behavior
        if len(rewards) < 2:
            return {
                'component_name': component_name,
                'test_passed': False,
                'reason': 'Insufficient scenarios for trend analysis',
                'rewards': rewards
            }
        
        # Check trend
        reward_trend = jnp.diff(jnp.array(rewards))
        mean_trend = float(jnp.mean(reward_trend))
        
        if expected_behavior == "increase":
            test_passed = mean_trend > 0
        elif expected_behavior == "decrease":
            test_passed = mean_trend < 0
        else:  # stable
            test_passed = abs(mean_trend) < 0.1
        
        return {
            'component_name': component_name,
            'test_passed': test_passed,
            'expected_behavior': expected_behavior,
            'mean_trend': mean_trend,
            'rewards': rewards,
            'reward_trend': reward_trend.tolist(),
            'config_used': isolated_weights
        }
        
    except Exception as e:
        return {
            'component_name': component_name,
            'test_passed': False,
            'error': str(e),
            'rewards': []
        }


# Export main functions
__all__ = [
    'ComponentValidationResult',
    'RewardValidationSuite', 
    'create_component_test_configs',
    'validate_single_component',
    'validate_all_components',
    'run_component_isolation_test'
]