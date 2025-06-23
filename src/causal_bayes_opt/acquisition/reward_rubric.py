"""Modular reward rubric system inspired by the verifiers repository.

This module provides a flexible framework for composing multiple reward signals
into a unified reward system. The design allows for:
- Modular composition of reward components
- Support for both supervised and observable signals
- Diversity monitoring to prevent mode collapse
- Easy switching between training and deployment configurations
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import pyrsistent as pyr

from ..jax_native.state import JAXAcquisitionState
from .hybrid_rewards import (
    mechanism_consistency_reward,
    posterior_confidence_reward,
    supervised_mechanism_discovery_reward,
    supervised_mechanism_impact_reward,
)
from .verifiable_rewards import (
    exploration_diversity_reward,
    target_improvement_reward,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardComponent:
    """Single reward signal with metadata.
    
    Inspired by the verifiers repository's modular approach to rewards.
    Each component encapsulates a specific reward signal that can be
    composed with others.
    
    Args:
        name: Unique identifier for this component
        compute_fn: Function (state, action, outcome, ground_truth) -> float
        weight: Relative importance of this component
        requires_ground_truth: Whether this component needs ground truth data
    """
    name: str
    compute_fn: Callable[[Any, Any, Any, Optional[Dict]], float]
    weight: float
    requires_ground_truth: bool


@dataclass(frozen=True)
class RewardResult:
    """Result of reward computation with detailed breakdown.
    
    Args:
        total_reward: Weighted sum of all component rewards
        component_rewards: Individual reward values by component name
        metadata: Additional information (skipped components, warnings, etc.)
    """
    total_reward: float
    component_rewards: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CausalRewardRubric:
    """Modular reward composition for causal optimization.
    
    This class provides a flexible framework for combining multiple reward
    signals, inspired by the verifiers repository's Rubric pattern.
    
    Args:
        components: Tuple of reward components to combine
        diversity_threshold: Minimum reward variance to maintain
        normalize_weights: Whether to normalize weights to sum to 1
    """
    components: Tuple[RewardComponent, ...]
    diversity_threshold: float = 0.3
    normalize_weights: bool = True

    def get_normalized_weights(self) -> Dict[str, float]:
        """Get normalized component weights."""
        total_weight = sum(c.weight for c in self.components)
        if self.normalize_weights and total_weight > 0:
            return {c.name: c.weight / total_weight for c in self.components}
        return {c.name: c.weight for c in self.components}

    def compute_reward(
        self,
        state: JAXAcquisitionState,
        action: pyr.PMap,
        outcome: pyr.PMap,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute total reward from all components.
        
        Args:
            state: Current acquisition state
            action: Intervention performed
            outcome: Observed outcome
            ground_truth: Optional ground truth data for supervised components
            
        Returns:
            RewardResult with total reward and component breakdown
        """
        component_rewards = {}
        skipped_components = []
        normalized_weights = self.get_normalized_weights()

        # Compute each component
        for component in self.components:
            try:
                if component.requires_ground_truth and ground_truth is None:
                    # Skip supervised components when no ground truth
                    component_rewards[component.name] = 0.0
                    skipped_components.append(component.name)
                else:
                    # Compute component reward
                    reward = component.compute_fn(state, action, outcome, ground_truth)
                    component_rewards[component.name] = float(reward)
            except Exception as e:
                logger.warning(f"Error computing {component.name}: {e}")
                component_rewards[component.name] = 0.0
                skipped_components.append(component.name)

        # Compute weighted total
        total_reward = sum(
            component_rewards[name] * normalized_weights[name]
            for name in component_rewards
        )

        # Prepare metadata
        metadata = {
            "normalized_weights": normalized_weights,
            "skipped_components": skipped_components,
        }

        return RewardResult(
            total_reward=total_reward,
            component_rewards=component_rewards,
            metadata=metadata
        )

    def compute_diversity(self, reward_results: List[RewardResult]) -> Dict[str, Any]:
        """Compute diversity metrics for a batch of rewards.
        
        This helps detect mode collapse and reward hacking.
        
        Args:
            reward_results: List of reward results from a batch
            
        Returns:
            Dictionary with diversity metrics
        """
        if not reward_results:
            return {
                "reward_variance": 0.0,
                "component_variances": {},
                "below_threshold": True,
            }

        # Extract total rewards
        total_rewards = jnp.array([r.total_reward for r in reward_results])
        reward_variance = float(jnp.var(total_rewards))

        # Compute per-component variances
        component_variances = {}
        for name in self.components[0].name if self.components else []:
            values = jnp.array([r.component_rewards.get(name, 0.0) for r in reward_results])
            component_variances[name] = float(jnp.var(values))

        # Check if below threshold
        below_threshold = reward_variance < self.diversity_threshold

        return {
            "reward_variance": reward_variance,
            "component_variances": component_variances,
            "below_threshold": below_threshold,
        }

    def to_config(self) -> Dict[str, Any]:
        """Convert rubric to configuration dictionary.
        
        Note: Functions cannot be serialized, so only metadata is included.
        """
        return {
            "components": [
                {
                    "name": c.name,
                    "weight": c.weight,
                    "requires_ground_truth": c.requires_ground_truth,
                }
                for c in self.components
            ],
            "diversity_threshold": self.diversity_threshold,
            "normalize_weights": self.normalize_weights,
        }


def create_training_rubric(
    improvement_weight: float = 2.0,
    mechanism_impact_weight: float = 1.5,
    mechanism_discovery_weight: float = 1.0,
    exploration_weight: float = 0.5,
    confidence_weight: float = 0.8,
) -> CausalRewardRubric:
    """Create a rubric for training with both supervised and observable signals.
    
    This configuration uses ground truth during training for better guidance
    while also incorporating observable signals for robustness.
    """
    components = []

    # Observable components (no ground truth needed)
    components.append(RewardComponent(
        name="target_improvement",
        compute_fn=lambda s, a, o, g=None: target_improvement_reward(
            current_value=o.get(s.get_target_name(), 0.0),
            best_value=s.best_value,
            improvement_threshold=0.01
        ),
        weight=improvement_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="exploration_diversity",
        compute_fn=lambda s, a, o, g=None: exploration_diversity_reward(
            intervention=a,
            buffer=s.sample_buffer,
            diversity_threshold=0.3
        ),
        weight=exploration_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="posterior_confidence",
        compute_fn=lambda s, a, o, g=None: posterior_confidence_reward(
            state=s,
            intervention=a,
            outcome=o,
            predictions=None  # Will use state's built-in info
        ),
        weight=confidence_weight,
        requires_ground_truth=False
    ))

    # Supervised components (need ground truth)
    components.append(RewardComponent(
        name="mechanism_impact",
        compute_fn=lambda s, a, o, g=None: supervised_mechanism_impact_reward(
            state=s,
            intervention=a,
            outcome=o,
            true_scm=g.get("scm") if g else None,
            predictions=g.get("predictions") if g else None
        ),
        weight=mechanism_impact_weight,
        requires_ground_truth=True
    ))

    components.append(RewardComponent(
        name="mechanism_discovery",
        compute_fn=lambda s, a, o, g=None: supervised_mechanism_discovery_reward(
            state=s,
            intervention=a,
            outcome=o,
            true_scm=g.get("scm") if g else None,
            predictions=g.get("predictions") if g else None
        ),
        weight=mechanism_discovery_weight,
        requires_ground_truth=True
    ))

    return CausalRewardRubric(
        components=tuple(components),
        diversity_threshold=0.3,
        normalize_weights=True
    )


def create_deployment_rubric(
    improvement_weight: float = 3.0,
    exploration_weight: float = 1.0,
    confidence_weight: float = 1.5,
    consistency_weight: float = 1.0,
) -> CausalRewardRubric:
    """Create a rubric for deployment using only observable signals.
    
    This configuration does not require ground truth and is suitable
    for real-world deployment where the true causal structure is unknown.
    """
    components = []

    # All components are observable
    components.append(RewardComponent(
        name="target_improvement",
        compute_fn=lambda s, a, o, g=None: target_improvement_reward(
            current_value=o.get(s.get_target_name(), 0.0),
            best_value=s.best_value,
            improvement_threshold=0.01
        ),
        weight=improvement_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="exploration_diversity",
        compute_fn=lambda s, a, o, g=None: exploration_diversity_reward(
            intervention=a,
            buffer=s.sample_buffer,
            diversity_threshold=0.3
        ),
        weight=exploration_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="posterior_confidence",
        compute_fn=lambda s, a, o, g=None: posterior_confidence_reward(
            state=s,
            intervention=a,
            outcome=o,
            predictions=None
        ),
        weight=confidence_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="mechanism_consistency",
        compute_fn=lambda s, a, o, g=None: mechanism_consistency_reward(
            state=s,
            intervention=a,
            outcome=o,
            predictions=None  # Will use state's mechanism features
        ),
        weight=consistency_weight,
        requires_ground_truth=False
    ))

    return CausalRewardRubric(
        components=tuple(components),
        diversity_threshold=0.3,
        normalize_weights=True
    )


def create_research_rubric(
    improvement_weight: float = 1.0,
    discovery_weight: float = 2.0,
    exploration_weight: float = 1.5,
    consistency_weight: float = 1.0,
    confidence_weight: float = 0.8,
    diversity_threshold: float = 0.5,
) -> CausalRewardRubric:
    """Create a rubric optimized for research and experimentation.
    
    This configuration emphasizes discovery and exploration more than
    immediate target improvement, making it suitable for research scenarios
    where understanding the causal structure is the primary goal.
    
    Args:
        improvement_weight: Weight for target improvement
        discovery_weight: Weight for mechanism discovery
        exploration_weight: Weight for exploration diversity
        consistency_weight: Weight for mechanism consistency
        confidence_weight: Weight for posterior confidence
        diversity_threshold: Higher threshold for research diversity
        
    Returns:
        Research-optimized CausalRewardRubric
    """
    components = []

    # Observable components with research focus
    components.append(RewardComponent(
        name="target_improvement",
        compute_fn=lambda s, a, o, g=None: target_improvement_reward(
            current_value=o.get(s.get_target_name(), 0.0),
            best_value=s.best_value,
            improvement_threshold=0.005  # Lower threshold for research
        ),
        weight=improvement_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="exploration_diversity",
        compute_fn=lambda s, a, o, g=None: exploration_diversity_reward(
            intervention=a,
            buffer=s.sample_buffer,
            diversity_threshold=0.5  # Higher diversity requirement
        ),
        weight=exploration_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="posterior_confidence",
        compute_fn=lambda s, a, o, g=None: posterior_confidence_reward(
            state=s,
            intervention=a,
            outcome=o,
            predictions=None
        ),
        weight=confidence_weight,
        requires_ground_truth=False
    ))

    components.append(RewardComponent(
        name="mechanism_consistency",
        compute_fn=lambda s, a, o, g=None: mechanism_consistency_reward(
            state=s,
            intervention=a,
            outcome=o,
            predictions=None
        ),
        weight=consistency_weight,
        requires_ground_truth=False
    ))

    # Supervised components for research validation
    components.append(RewardComponent(
        name="mechanism_discovery",
        compute_fn=lambda s, a, o, g=None: supervised_mechanism_discovery_reward(
            state=s,
            intervention=a,
            outcome=o,
            true_scm=g.get("scm") if g else None,
            predictions=g.get("predictions") if g else None
        ),
        weight=discovery_weight,
        requires_ground_truth=True
    ))

    return CausalRewardRubric(
        components=tuple(components),
        diversity_threshold=diversity_threshold,
        normalize_weights=True
    )


def create_ablation_rubric(
    use_supervised: bool = True,
    use_observable: bool = True,
    diversity_threshold: float = 0.3,
) -> CausalRewardRubric:
    """Create a rubric for ablation studies.
    
    This allows testing different combinations of reward signals
    to understand their individual contributions.
    
    Args:
        use_supervised: Include supervised reward components
        use_observable: Include observable reward components
        diversity_threshold: Minimum reward variance threshold
    """
    components = []

    if use_observable:
        # Add observable components
        components.extend([
            RewardComponent(
                name="target_improvement",
                compute_fn=lambda s, a, o, g=None: target_improvement_reward(
                    current_value=o.get(s.get_target_name(), 0.0),
                    best_value=s.best_value,
                    improvement_threshold=0.01
                ),
                weight=2.0,
                requires_ground_truth=False
            ),
            RewardComponent(
                name="exploration_diversity",
                compute_fn=lambda s, a, o, g=None: exploration_diversity_reward(
                    intervention=a,
                    buffer=s.sample_buffer,
                    diversity_threshold=0.3
                ),
                weight=0.5,
                requires_ground_truth=False
            ),
        ])

    if use_supervised:
        # Add supervised components
        components.extend([
            RewardComponent(
                name="mechanism_impact",
                compute_fn=lambda s, a, o, g=None: supervised_mechanism_impact_reward(
                    state=s,
                    intervention=a,
                    outcome=o,
                    true_scm=g.get("scm") if g else None,
                    predictions=g.get("predictions") if g else None
                ),
                weight=1.5,
                requires_ground_truth=True
            ),
            RewardComponent(
                name="mechanism_discovery",
                compute_fn=lambda s, a, o, g=None: supervised_mechanism_discovery_reward(
                    state=s,
                    intervention=a,
                    outcome=o,
                    true_scm=g.get("scm") if g else None,
                    predictions=g.get("predictions") if g else None
                ),
                weight=1.0,
                requires_ground_truth=True
            ),
        ])

    if not components:
        raise ValueError("At least one reward type must be enabled")

    return CausalRewardRubric(
        components=tuple(components),
        diversity_threshold=diversity_threshold,
        normalize_weights=True
    )
