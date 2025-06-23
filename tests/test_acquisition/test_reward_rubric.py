"""Tests for the modular reward rubric system.

This module tests the CausalRewardRubric and RewardComponent abstractions
inspired by the verifiers repository for flexible reward composition.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr
from typing import Dict, Any

from causal_bayes_opt.acquisition.reward_rubric import (
    RewardComponent,
    CausalRewardRubric,
    RewardResult,
    create_training_rubric,
    create_deployment_rubric,
    create_ablation_rubric,
)
from causal_bayes_opt.jax_native.state import JAXAcquisitionState
from causal_bayes_opt.jax_native.config import JAXConfig
from causal_bayes_opt.jax_native.sample_buffer import JAXSampleBuffer


class TestRewardComponent:
    """Test individual reward component functionality."""
    
    def test_reward_component_creation(self):
        """Test creating a reward component."""
        def dummy_reward(state, action, outcome, ground_truth=None):
            return 1.0
            
        component = RewardComponent(
            name="test_reward",
            compute_fn=dummy_reward,
            weight=0.5,
            requires_ground_truth=False
        )
        
        assert component.name == "test_reward"
        assert component.weight == 0.5
        assert not component.requires_ground_truth
        assert component.compute_fn(None, None, None) == 1.0
    
    def test_reward_component_immutability(self):
        """Test that reward components are immutable."""
        component = RewardComponent(
            name="test",
            compute_fn=lambda s, a, o, g=None: 1.0,
            weight=1.0,
            requires_ground_truth=False
        )
        
        with pytest.raises(AttributeError):
            component.weight = 2.0
    
    def test_reward_component_with_ground_truth(self):
        """Test component that requires ground truth."""
        def gt_reward(state, action, outcome, ground_truth=None):
            if ground_truth is None:
                raise ValueError("Ground truth required")
            return float(ground_truth.get("correct", 0))
            
        component = RewardComponent(
            name="supervised",
            compute_fn=gt_reward,
            weight=1.0,
            requires_ground_truth=True
        )
        
        assert component.requires_ground_truth
        
        # Should raise without ground truth
        with pytest.raises(ValueError, match="Ground truth required"):
            component.compute_fn(None, None, None, None)
            
        # Should work with ground truth
        result = component.compute_fn(None, None, None, {"correct": 1})
        assert result == 1.0


class TestCausalRewardRubric:
    """Test the causal reward rubric system."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock reward components."""
        return (
            RewardComponent(
                name="improvement",
                compute_fn=lambda s, a, o, g=None: 0.8,
                weight=2.0,
                requires_ground_truth=False
            ),
            RewardComponent(
                name="diversity",
                compute_fn=lambda s, a, o, g=None: 0.6,
                weight=1.0,
                requires_ground_truth=False
            ),
            RewardComponent(
                name="supervised",
                compute_fn=lambda s, a, o, g=None: 0.9 if g else 0.0,
                weight=1.5,
                requires_ground_truth=True
            ),
        )
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock JAX acquisition state."""
        config = JAXConfig(
            n_vars=3,
            target_idx=2,
            max_samples=100,
            max_history=50,
            variable_names=("X", "Y", "Z"),
            mechanism_types=(0, 1, 0),
            feature_dim=3
        )
        
        buffer = JAXSampleBuffer(
            values=jnp.zeros((100, 3)),
            interventions=jnp.zeros((100, 3)),
            targets=jnp.zeros(100),
            valid_mask=jnp.zeros(100, dtype=bool),
            write_idx=0,
            n_samples=0,
            config=config
        )
        
        return JAXAcquisitionState(
            sample_buffer=buffer,
            mechanism_features=jnp.ones((3, 3)),
            marginal_probs=jnp.array([0.5, 0.8, 0.0]),
            confidence_scores=jnp.array([0.7, 0.9, 0.6]),
            best_value=0.0,
            current_step=0,
            uncertainty_bits=1.5,
            config=config
        )
    
    def test_rubric_creation(self, mock_components):
        """Test creating a reward rubric."""
        rubric = CausalRewardRubric(
            components=mock_components,
            diversity_threshold=0.3,
            normalize_weights=True
        )
        
        assert len(rubric.components) == 3
        assert rubric.diversity_threshold == 0.3
        assert rubric.normalize_weights
    
    def test_rubric_compute_reward(self, mock_components, mock_state):
        """Test computing rewards with rubric."""
        rubric = CausalRewardRubric(
            components=mock_components,
            diversity_threshold=0.3,
            normalize_weights=True
        )
        
        action = pyr.pmap({"X": 1.0})
        outcome = pyr.pmap({"X": 1.0, "Y": 2.0, "Z": 3.0})
        ground_truth = {"correct": True}
        
        result = rubric.compute_reward(
            state=mock_state,
            action=action,
            outcome=outcome,
            ground_truth=ground_truth
        )
        
        assert isinstance(result, RewardResult)
        assert result.total_reward > 0
        assert len(result.component_rewards) == 3
        assert "improvement" in result.component_rewards
        assert "diversity" in result.component_rewards
        assert "supervised" in result.component_rewards
        assert result.metadata is not None
    
    def test_rubric_without_ground_truth(self, mock_components, mock_state):
        """Test rubric handles missing ground truth gracefully."""
        rubric = CausalRewardRubric(
            components=mock_components,
            diversity_threshold=0.3,
            normalize_weights=True
        )
        
        action = pyr.pmap({"X": 1.0})
        outcome = pyr.pmap({"X": 1.0, "Y": 2.0, "Z": 3.0})
        
        # Should work but skip supervised component
        result = rubric.compute_reward(
            state=mock_state,
            action=action,
            outcome=outcome,
            ground_truth=None
        )
        
        assert result.total_reward > 0
        assert result.component_rewards["supervised"] == 0.0
        assert "skipped_components" in result.metadata
        assert "supervised" in result.metadata["skipped_components"]
    
    def test_rubric_weight_normalization(self):
        """Test weight normalization."""
        components = (
            RewardComponent("a", lambda s, a, o, g=None: 1.0, 2.0, False),
            RewardComponent("b", lambda s, a, o, g=None: 1.0, 3.0, False),
        )
        
        rubric = CausalRewardRubric(
            components=components,
            normalize_weights=True
        )
        
        # Weights should sum to 1 after normalization
        total_weight = sum(c.weight for c in components)
        normalized_weights = rubric.get_normalized_weights()
        
        assert abs(sum(normalized_weights.values()) - 1.0) < 1e-6
        assert normalized_weights["a"] == 2.0 / total_weight
        assert normalized_weights["b"] == 3.0 / total_weight
    
    def test_rubric_diversity_computation(self, mock_state):
        """Test diversity metric computation."""
        # Create components with varying rewards
        components = (
            RewardComponent("c1", lambda s, a, o, g=None: 0.1, 1.0, False),
            RewardComponent("c2", lambda s, a, o, g=None: 0.5, 1.0, False),
            RewardComponent("c3", lambda s, a, o, g=None: 0.9, 1.0, False),
        )
        
        rubric = CausalRewardRubric(
            components=components,
            diversity_threshold=0.3
        )
        
        # Collect multiple rewards to compute diversity
        rewards = []
        for i in range(10):
            action = pyr.pmap({"X": float(i)})
            outcome = pyr.pmap({"X": float(i), "Y": 0.0, "Z": 0.0})
            result = rubric.compute_reward(mock_state, action, outcome)
            rewards.append(result)
        
        diversity = rubric.compute_diversity(rewards)
        
        assert "reward_variance" in diversity
        assert "component_variances" in diversity
        assert "below_threshold" in diversity
        assert isinstance(diversity["below_threshold"], bool)


class TestRubricFactoryFunctions:
    """Test factory functions for creating standard rubrics."""
    
    def test_create_training_rubric(self):
        """Test creating a training rubric with all components."""
        rubric = create_training_rubric()
        
        # Should have both supervised and observable components
        component_names = {c.name for c in rubric.components}
        assert "target_improvement" in component_names
        assert "mechanism_impact" in component_names  # Supervised
        assert "exploration_diversity" in component_names
        
        # Check some components require ground truth
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        assert len(gt_required) > 0
    
    def test_create_deployment_rubric(self):
        """Test creating a deployment rubric without ground truth."""
        rubric = create_deployment_rubric()
        
        # Should only have observable components
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        assert len(gt_required) == 0
        
        # Should still have core components
        component_names = {c.name for c in rubric.components}
        assert "target_improvement" in component_names
        assert "exploration_diversity" in component_names
    
    def test_create_ablation_rubric(self):
        """Test creating ablation study rubrics."""
        # Test with only supervised signals
        supervised_rubric = create_ablation_rubric(
            use_supervised=True,
            use_observable=False
        )
        
        gt_required = [c for c in supervised_rubric.components if c.requires_ground_truth]
        gt_not_required = [c for c in supervised_rubric.components if not c.requires_ground_truth]
        
        assert len(gt_required) > 0
        assert len(gt_not_required) == 0
        
        # Test with only observable signals
        observable_rubric = create_ablation_rubric(
            use_supervised=False,
            use_observable=True
        )
        
        gt_required = [c for c in observable_rubric.components if c.requires_ground_truth]
        gt_not_required = [c for c in observable_rubric.components if not c.requires_ground_truth]
        
        assert len(gt_required) == 0
        assert len(gt_not_required) > 0


class TestRewardRubricIntegration:
    """Test integration with existing reward systems."""
    
    def test_rubric_with_hybrid_rewards(self):
        """Test rubric can use hybrid reward functions."""
        from causal_bayes_opt.acquisition.hybrid_rewards import (
            supervised_mechanism_impact_reward,
            posterior_confidence_reward,
        )
        
        # Create rubric with actual reward functions
        components = (
            RewardComponent(
                name="mechanism_impact",
                compute_fn=lambda s, a, o, g=None: supervised_mechanism_impact_reward(
                    s, a, o, g.get("scm") if g else None, g.get("predictions") if g else None
                ) if g else 0.0,
                weight=1.0,
                requires_ground_truth=True
            ),
            RewardComponent(
                name="confidence",
                compute_fn=lambda s, a, o, g=None: posterior_confidence_reward(
                    s, a, o, None  # No predictions needed for observable
                ),
                weight=0.5,
                requires_ground_truth=False
            ),
        )
        
        rubric = CausalRewardRubric(components=components)
        assert len(rubric.components) == 2
    
    def test_rubric_performance(self):
        """Test rubric computation performance."""
        import time
        
        # Create a mock state
        config = JAXConfig(
            n_vars=3,
            target_idx=2,
            max_samples=100,
            max_history=50,
            variable_names=("X", "Y", "Z"),
            mechanism_types=(0, 1, 0),
            feature_dim=3
        )
        
        buffer = JAXSampleBuffer(
            values=jnp.zeros((100, 3)),
            interventions=jnp.zeros((100, 3)),
            targets=jnp.zeros(100),
            valid_mask=jnp.zeros(100, dtype=bool),
            write_idx=0,
            n_samples=0,
            config=config
        )
        
        mock_state = JAXAcquisitionState(
            sample_buffer=buffer,
            mechanism_features=jnp.ones((3, 3)),
            marginal_probs=jnp.array([0.5, 0.8, 0.0]),
            confidence_scores=jnp.array([0.7, 0.9, 0.6]),
            best_value=0.0,
            current_step=0,
            uncertainty_bits=1.5,
            config=config
        )
        
        # Create a rubric with multiple components
        rubric = create_training_rubric()
        
        action = pyr.pmap({"X": 1.0})
        outcome = pyr.pmap({"X": 1.0, "Y": 2.0, "Z": 3.0})
        ground_truth = {"correct": True}
        
        # Measure computation time
        start = time.time()
        for _ in range(100):
            rubric.compute_reward(mock_state, action, outcome, ground_truth)
        elapsed = time.time() - start
        
        # Should be fast (< 10ms per computation)
        assert elapsed / 100 < 0.01
    
    def test_rubric_serialization(self):
        """Test rubric can be saved and loaded."""
        rubric = create_training_rubric()
        
        # Convert to config dict
        config = rubric.to_config()
        
        assert "components" in config
        assert "diversity_threshold" in config
        assert "normalize_weights" in config
        
        # Should be JSON serializable (functions excluded)
        import json
        json_str = json.dumps(config)
        assert len(json_str) > 0