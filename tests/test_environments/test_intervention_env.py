"""Tests for intervention environment abstraction.

This module tests the InterventionEnvironment and related classes that provide
a clean abstraction for SCM interactions inspired by the verifiers repository.
"""

import pytest
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.environments.intervention_env import (
    EnvironmentConfig,
    EnvironmentInfo,
    InterventionEnvironment,
    create_intervention_environment,
    create_batch_environments,
)
from causal_bayes_opt.acquisition.reward_rubric import (
    create_training_rubric,
    create_deployment_rubric,
)
from causal_bayes_opt.jax_native.config import JAXConfig
from causal_bayes_opt.jax_native.state import JAXAcquisitionState


class TestEnvironmentConfig:
    """Test environment configuration."""
    
    def test_config_creation(self):
        """Test creating environment config."""
        config = EnvironmentConfig(
            difficulty=0.8,
            max_interventions=50,
            intervention_budget=5.0,
            target_threshold=1.5,
            noise_level=0.05
        )
        
        assert config.difficulty == 0.8
        assert config.max_interventions == 50
        assert config.intervention_budget == 5.0
        assert config.target_threshold == 1.5
        assert config.noise_level == 0.05
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = EnvironmentConfig()
        
        assert config.difficulty == 0.5
        assert config.max_interventions == 100
        assert config.intervention_budget == 10.0
        assert config.enable_early_stopping is True
        assert config.target_threshold == 2.0
        assert config.noise_level == 0.1
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = EnvironmentConfig()
        
        with pytest.raises(AttributeError):
            config.difficulty = 0.8


class TestEnvironmentInfo:
    """Test environment information structure."""
    
    def test_info_creation(self):
        """Test creating environment info."""
        info = EnvironmentInfo(
            intervention_count=5,
            budget_remaining=7.5,
            target_achieved=True,
            early_stopped=False,
            episode_complete=True,
            best_value_so_far=2.3,
            metadata={"test": "value"}
        )
        
        assert info.intervention_count == 5
        assert info.budget_remaining == 7.5
        assert info.target_achieved is True
        assert info.early_stopped is False
        assert info.episode_complete is True
        assert info.best_value_so_far == 2.3
        assert info.metadata["test"] == "value"
    
    def test_info_immutability(self):
        """Test that info is immutable."""
        info = EnvironmentInfo(
            intervention_count=0, budget_remaining=10.0, target_achieved=False,
            early_stopped=False, episode_complete=False, best_value_so_far=0.0,
            metadata={}
        )
        
        with pytest.raises(AttributeError):
            info.intervention_count = 5


class TestInterventionEnvironment:
    """Test the main intervention environment."""
    
    @pytest.fixture
    def sample_scm(self):
        """Create a sample SCM for testing."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        mechanisms = {
            "X": lambda parents, key: jax.random.normal(key),
            "Y": lambda parents, key: parents.get("X", 0.0) + parents.get("Z", 0.0) + 0.1 * jax.random.normal(key),
            "Z": lambda parents, key: jax.random.normal(key),
        }
        
        return create_scm(
            variables=frozenset({"X", "Y", "Z"}),
            edges=frozenset([("X", "Y"), ("Z", "Y")]),
            mechanisms=mechanisms,
            target="Y"
        )
    
    @pytest.fixture
    def sample_rubric(self):
        """Create a sample reward rubric."""
        return create_deployment_rubric()  # Observable-only for testing
    
    @pytest.fixture
    def test_environment(self, sample_scm, sample_rubric):
        """Create a test environment."""
        return create_intervention_environment(
            scm=sample_scm,
            rubric=sample_rubric,
            difficulty=0.5,
            max_interventions=20,
            target_threshold=1.0
        )
    
    def test_environment_creation(self, sample_scm, sample_rubric):
        """Test creating intervention environment."""
        env = create_intervention_environment(
            scm=sample_scm,
            rubric=sample_rubric,
            difficulty=0.7,
            max_interventions=50
        )
        
        assert env.scm == sample_scm
        assert env.rubric == sample_rubric
        assert env.config.difficulty == 0.7
        assert env.config.max_interventions == 50
        assert env.jax_config.n_vars == 3  # X, Y, Z
        assert env.jax_config.get_target_name() == "Y"
    
    def test_environment_reset(self, test_environment):
        """Test environment reset functionality."""
        key = jax.random.PRNGKey(42)
        
        state = test_environment.reset(key)
        
        assert isinstance(state, JAXAcquisitionState)
        assert state.current_step == 0
        assert state.best_value == -1000.0  # Initial large negative value
        assert state.sample_buffer.n_samples == 0
        assert state.mechanism_features.shape == (3, 3)  # n_vars x feature_dim
        assert state.uncertainty_bits > 0
    
    def test_mechanism_features_initialization(self, test_environment):
        """Test mechanism features are properly initialized."""
        key = jax.random.PRNGKey(42)
        
        state = test_environment.reset(key)
        features = state.mechanism_features
        
        # Should have proper shape
        assert features.shape == (3, 3)
        
        # Feature 0 should be mechanism types (normalized)
        assert jnp.all(features[:, 0] >= 0.0)
        assert jnp.all(features[:, 0] <= 1.0)
        
        # Feature 1 should be difficulty-based
        assert jnp.allclose(features[:, 1], test_environment.config.difficulty)
    
    def test_environment_step(self, test_environment):
        """Test environment step functionality."""
        key = jax.random.PRNGKey(42)
        
        # Reset environment
        state = test_environment.reset(key)
        
        # Create intervention
        action = pyr.pmap({"X": 1.0})
        
        # Take step
        key, step_key = jax.random.split(key)
        next_state, reward_result, env_info = test_environment.step(state, action, step_key)
        
        # Check state update
        assert isinstance(next_state, JAXAcquisitionState)
        assert next_state.current_step == state.current_step + 1
        assert next_state.sample_buffer.n_samples == 1
        
        # Check reward
        assert hasattr(reward_result, 'total_reward')
        assert hasattr(reward_result, 'component_rewards')
        assert isinstance(reward_result.total_reward, float)
        
        # Check environment info
        assert isinstance(env_info, EnvironmentInfo)
        assert env_info.intervention_count == 1
        assert env_info.budget_remaining < test_environment.config.intervention_budget
    
    def test_state_update_with_sample(self, test_environment):
        """Test state updating with sample data."""
        key = jax.random.PRNGKey(42)
        
        # Reset and take multiple steps
        state = test_environment.reset(key)
        
        for i in range(3):
            action = pyr.pmap({"X": float(i)})
            key, step_key = jax.random.split(key)
            state, _, _ = test_environment.step(state, action, step_key)
        
        # Should have 3 samples
        assert state.sample_buffer.n_samples == 3
        assert state.current_step == 3
    
    def test_early_stopping(self, sample_scm, sample_rubric):
        """Test early stopping when target is achieved."""
        # Create environment with low threshold for easy achievement
        env = create_intervention_environment(
            scm=sample_scm,
            rubric=sample_rubric,
            target_threshold=0.5,  # Low threshold
            enable_early_stopping=True
        )
        
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # Keep taking steps until early stopping or max reached
        for i in range(10):
            action = pyr.pmap({"X": 2.0, "Z": 2.0})  # High intervention values
            key, step_key = jax.random.split(key)
            state, reward_result, env_info = env.step(state, action, step_key)
            
            if env_info.early_stopped:
                assert env_info.target_achieved
                assert env_info.episode_complete
                break
    
    def test_budget_exhaustion(self, sample_scm, sample_rubric):
        """Test budget exhaustion termination."""
        env = create_intervention_environment(
            scm=sample_scm,
            rubric=sample_rubric,
            intervention_budget=3.0,  # Small budget
            max_interventions=10
        )
        
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # Take steps until budget exhausted
        for i in range(5):
            action = pyr.pmap({"X": 1.0})
            key, step_key = jax.random.split(key)
            state, reward_result, env_info = env.step(state, action, step_key)
            
            if env_info.budget_remaining <= 0:
                assert env_info.episode_complete
                break
    
    def test_noise_addition(self, sample_scm, sample_rubric):
        """Test that environment noise is properly added."""
        # Create environment with high noise
        env = create_intervention_environment(
            scm=sample_scm,
            rubric=sample_rubric,
            noise_level=0.5
        )
        
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # Take same action multiple times, should get different outcomes due to noise
        action = pyr.pmap({"X": 1.0})
        outcomes = []
        
        for i in range(5):
            key, step_key = jax.random.split(key)
            outcome = env._sample_outcome(action, step_key)
            outcomes.append(float(outcome["values"]["Y"]))
        
        # Should have variation due to noise
        assert len(set(outcomes)) > 1  # Not all identical
    
    def test_curriculum_metrics(self, test_environment):
        """Test curriculum learning metrics."""
        metrics = test_environment.get_curriculum_metrics()
        
        assert "difficulty" in metrics
        assert "max_interventions" in metrics
        assert "target_threshold" in metrics
        assert "noise_level" in metrics
        
        assert metrics["difficulty"] == test_environment.config.difficulty
        assert metrics["max_interventions"] == float(test_environment.config.max_interventions)


class TestEnvironmentFactory:
    """Test environment factory functions."""
    
    @pytest.fixture
    def sample_scms(self):
        """Create sample SCMs for batch testing."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        scm1 = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        scm2 = create_scm(
            variables=frozenset({"A", "B", "C"}),
            edges=frozenset([("A", "C"), ("B", "C")]),
            mechanisms={
                "A": lambda parents, key: jax.random.normal(key),
                "B": lambda parents, key: jax.random.normal(key),
                "C": lambda parents, key: parents.get("A", 0.0) + parents.get("B", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="C"
        )
        
        return [scm1, scm2]
    
    def test_create_intervention_environment(self, sample_scms):
        """Test creating single intervention environment."""
        rubric = create_deployment_rubric()
        
        env = create_intervention_environment(
            scm=sample_scms[0],
            rubric=rubric,
            difficulty=0.8,
            max_interventions=30,
            target_threshold=1.5
        )
        
        assert env.config.difficulty == 0.8
        assert env.config.max_interventions == 30
        assert env.config.target_threshold == 1.5
        assert env.jax_config.n_vars == 2  # X, Y
    
    def test_create_batch_environments(self, sample_scms):
        """Test creating batch of environments."""
        rubric = create_training_rubric()
        
        environments = create_batch_environments(
            scms=sample_scms,
            rubric=rubric,
            difficulty_range=(0.3, 0.9),
            max_interventions=25
        )
        
        assert len(environments) == 2
        
        # Check difficulty progression (allow small floating point errors)
        assert abs(environments[0].config.difficulty - 0.3) < 1e-10  # Start of range
        assert abs(environments[1].config.difficulty - 0.9) < 1e-10  # End of range
        
        # Check shared config
        for env in environments:
            assert env.config.max_interventions == 25
            assert env.rubric == rubric
    
    def test_create_batch_single_scm(self, sample_scms):
        """Test creating batch with single SCM."""
        rubric = create_deployment_rubric()
        
        environments = create_batch_environments(
            scms=[sample_scms[0]],  # Single SCM
            rubric=rubric,
            difficulty_range=(0.5, 0.5)
        )
        
        assert len(environments) == 1
        assert environments[0].config.difficulty == 0.5
    
    def test_factory_kwargs_passthrough(self, sample_scms):
        """Test that factory functions pass through kwargs."""
        rubric = create_deployment_rubric()
        
        env = create_intervention_environment(
            scm=sample_scms[0],
            rubric=rubric,
            intervention_budget=15.0,
            noise_level=0.2,
            enable_early_stopping=False
        )
        
        assert env.config.intervention_budget == 15.0
        assert env.config.noise_level == 0.2
        assert env.config.enable_early_stopping is False


class TestEnvironmentIntegration:
    """Test integration scenarios."""
    
    def test_full_episode_execution(self):
        """Test executing a full episode from start to finish."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        # Create simple SCM
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: 2.0 * parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        rubric = create_deployment_rubric()
        env = create_intervention_environment(
            scm=scm,
            rubric=rubric,
            max_interventions=10,
            target_threshold=3.0
        )
        
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        episode_rewards = []
        episode_complete = False
        step_count = 0
        
        while not episode_complete and step_count < 15:  # Safety limit
            # Simple strategy: increase X to increase Y
            action = pyr.pmap({"X": 2.0})
            
            key, step_key = jax.random.split(key)
            state, reward_result, env_info = env.step(state, action, step_key)
            
            episode_rewards.append(reward_result.total_reward)
            episode_complete = env_info.episode_complete
            step_count += 1
        
        # Should have completed the episode
        assert episode_complete
        assert len(episode_rewards) > 0
        assert state.current_step > 0
    
    def test_environment_with_training_rubric(self):
        """Test environment with training rubric (requires ground truth)."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        rubric = create_training_rubric()  # Needs ground truth
        env = create_intervention_environment(
            scm=scm,
            rubric=rubric,
            difficulty=0.6
        )
        
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        action = pyr.pmap({"X": 1.0})
        
        # Should work with ground truth provided
        key, step_key = jax.random.split(key)
        next_state, reward_result, env_info = env.step(state, action, step_key)
        
        assert isinstance(reward_result.total_reward, float)
        # Some components might be skipped if no ground truth predictions available
        assert len(reward_result.component_rewards) > 0
    
    def test_async_compatibility(self):
        """Test that environment is compatible with async training patterns."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        rubric = create_deployment_rubric()
        
        # Create multiple environments (batch)
        environments = create_batch_environments(
            scms=[scm] * 4,
            rubric=rubric,
            difficulty_range=(0.2, 0.8)
        )
        
        # Test parallel reset
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, len(environments))
        
        states = []
        for env, env_key in zip(environments, keys):
            state = env.reset(env_key)
            states.append(state)
        
        assert len(states) == 4
        
        # Test parallel step execution
        action = pyr.pmap({"X": 1.0})
        results = []
        
        for env, state, env_key in zip(environments, states, keys):
            result = env.step(state, action, env_key)
            results.append(result)
        
        assert len(results) == 4
        # Each result should be (state, reward, info)
        for result in results:
            assert len(result) == 3