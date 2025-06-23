"""Tests for async training infrastructure.

This module tests the AsyncTrainingManager and related classes that provide
parallel environment execution and JAX-compiled training loops.
"""

import pytest
import asyncio
import time
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.training.async_training import (
    AsyncTrainingConfig,
    TrainingBatch,
    TrainingProgress,
    AsyncTrainingManager,
    create_async_training_manager,
    estimate_batch_memory_usage,
    optimize_batch_size_for_memory,
)
from causal_bayes_opt.environments.intervention_env import (
    create_intervention_environment,
    create_batch_environments,
)
from causal_bayes_opt.acquisition.reward_rubric import (
    create_deployment_rubric,
    RewardResult,
)
from causal_bayes_opt.training.diversity_monitor import create_diversity_monitor
from causal_bayes_opt.jax_native.state import JAXAcquisitionState


class TestAsyncTrainingConfig:
    """Test async training configuration."""
    
    def test_config_creation(self):
        """Test creating async training config."""
        config = AsyncTrainingConfig(
            max_parallel_envs=32,
            batch_size=128,
            num_workers=16,
            checkpoint_interval=500,
            max_memory_mb=8192,
            enable_compilation=False,
            device_placement='cpu',
            progress_logging_interval=50
        )
        
        assert config.max_parallel_envs == 32
        assert config.batch_size == 128
        assert config.num_workers == 16
        assert config.checkpoint_interval == 500
        assert config.max_memory_mb == 8192
        assert config.enable_compilation is False
        assert config.device_placement == 'cpu'
        assert config.progress_logging_interval == 50
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = AsyncTrainingConfig()
        
        assert config.max_parallel_envs == 16
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.checkpoint_interval == 1000
        assert config.max_memory_mb == 4096
        assert config.enable_compilation is True
        assert config.device_placement == 'auto'
        assert config.progress_logging_interval == 100
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = AsyncTrainingConfig()
        
        with pytest.raises(AttributeError):
            config.batch_size = 256


class TestTrainingBatch:
    """Test training batch data structure."""
    
    @pytest.fixture
    def sample_states(self):
        """Create sample JAX acquisition states."""
        from causal_bayes_opt.jax_native.state import create_test_state
        return [create_test_state() for _ in range(3)]
    
    @pytest.fixture
    def sample_actions(self):
        """Create sample actions."""
        return [
            pyr.pmap({"X": 1.0}),
            pyr.pmap({"Y": 0.5}),
            pyr.pmap({"X": -1.0})
        ]
    
    @pytest.fixture
    def sample_rewards(self):
        """Create sample reward results."""
        return [
            RewardResult(1.0, {"r1": 0.8, "r2": 0.2}, {}),
            RewardResult(0.5, {"r1": 0.3, "r2": 0.2}, {}),
            RewardResult(0.8, {"r1": 0.6, "r2": 0.2}, {}),
        ]
    
    def test_batch_creation(self, sample_states, sample_actions, sample_rewards):
        """Test creating training batch."""
        from causal_bayes_opt.training.diversity_monitor import DiversityMetrics
        from causal_bayes_opt.environments.intervention_env import EnvironmentInfo
        
        env_infos = [
            EnvironmentInfo(1, 9.0, False, False, False, 1.0, {}),
            EnvironmentInfo(1, 9.0, False, False, False, 0.5, {}),
            EnvironmentInfo(1, 9.0, False, False, False, 0.8, {}),
        ]
        
        diversity_metrics = DiversityMetrics(
            reward_variance=0.4,
            component_entropies={"r1": 1.0, "r2": 0.2},
            mode_collapse_risk=0.3,
            below_threshold=False,
            batch_size=3,
            timestamp=time.time()
        )
        
        batch = TrainingBatch(
            states=sample_states,
            actions=sample_actions,
            next_states=sample_states,  # Reuse for simplicity
            rewards=sample_rewards,
            env_infos=env_infos,
            diversity_metrics=diversity_metrics,
            metadata={"step": 10, "episode": 3}
        )
        
        assert len(batch.states) == 3
        assert len(batch.actions) == 3
        assert len(batch.rewards) == 3
        assert batch.diversity_metrics.batch_size == 3
        assert batch.metadata["step"] == 10
    
    def test_batch_immutability(self, sample_states, sample_actions, sample_rewards):
        """Test that batch is immutable."""
        from causal_bayes_opt.training.diversity_monitor import DiversityMetrics
        from causal_bayes_opt.environments.intervention_env import EnvironmentInfo
        
        env_infos = [EnvironmentInfo(0, 10.0, False, False, False, 0.0, {}) for _ in range(3)]
        diversity_metrics = DiversityMetrics(0.0, {}, 0.0, False, 3, 0.0)
        
        batch = TrainingBatch(
            states=sample_states,
            actions=sample_actions,
            next_states=sample_states,
            rewards=sample_rewards,
            env_infos=env_infos,
            diversity_metrics=diversity_metrics,
            metadata={}
        )
        
        with pytest.raises(AttributeError):
            batch.metadata = {"new": "value"}


class TestTrainingProgress:
    """Test training progress tracking."""
    
    def test_progress_creation(self):
        """Test creating training progress."""
        progress = TrainingProgress(
            step=1000,
            episode=500,
            total_samples=32000,
            total_time=3600.0,
            avg_reward=1.5,
            diversity_health=0.8,
            curriculum_difficulty=0.6,
            checkpoint_path="/path/to/checkpoint",
            metadata={"version": "1.0"}
        )
        
        assert progress.step == 1000
        assert progress.episode == 500
        assert progress.total_samples == 32000
        assert progress.total_time == 3600.0
        assert progress.avg_reward == 1.5
        assert progress.diversity_health == 0.8
        assert progress.curriculum_difficulty == 0.6
        assert progress.checkpoint_path == "/path/to/checkpoint"
        assert progress.metadata["version"] == "1.0"
    
    def test_progress_immutability(self):
        """Test that progress is immutable."""
        progress = TrainingProgress(0, 0, 0, 0.0, 0.0, 0.0, 0.0, None, {})
        
        with pytest.raises(AttributeError):
            progress.step = 100


class TestAsyncTrainingManager:
    """Test the main async training manager."""
    
    @pytest.fixture
    def sample_environments(self):
        """Create sample intervention environments."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        # Create simple SCM
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
        
        # Create multiple environments with different difficulties
        environments = create_batch_environments(
            scms=[scm] * 3,
            rubric=rubric,
            difficulty_range=(0.3, 0.7),
            max_interventions=10
        )
        
        return environments
    
    @pytest.fixture
    def diversity_monitor(self):
        """Create diversity monitor."""
        return create_diversity_monitor()
    
    @pytest.fixture
    def async_config(self):
        """Create async training config for testing."""
        return AsyncTrainingConfig(
            max_parallel_envs=3,
            batch_size=3,
            num_workers=2,
            checkpoint_interval=5,
            max_memory_mb=1024,
            enable_compilation=False,  # Disable for testing
            progress_logging_interval=2
        )
    
    @pytest.fixture
    def training_manager(self, sample_environments, diversity_monitor, async_config):
        """Create training manager for testing."""
        return AsyncTrainingManager(
            environments=sample_environments,
            diversity_monitor=diversity_monitor,
            config=async_config,
            training_config=None  # Will use defaults
        )
    
    def test_manager_creation(self, sample_environments, diversity_monitor, async_config):
        """Test creating async training manager."""
        manager = AsyncTrainingManager(
            environments=sample_environments,
            diversity_monitor=diversity_monitor,
            config=async_config,
            training_config=None
        )
        
        assert len(manager.environments) == 3
        assert manager.diversity_monitor == diversity_monitor
        assert manager.config == async_config
        assert manager._training_step == 0
        assert manager._episode_count == 0
    
    @pytest.mark.asyncio
    async def test_collect_training_batch(self, training_manager):
        """Test collecting training batch from environments."""
        def simple_policy(state):
            """Simple policy that always chooses X=1.0."""
            return pyr.pmap({"X": 1.0})
        
        batch = await training_manager._collect_training_batch(simple_policy)
        
        assert isinstance(batch, TrainingBatch)
        assert len(batch.states) == 3  # Batch size
        assert len(batch.actions) == 3
        assert len(batch.next_states) == 3
        assert len(batch.rewards) == 3
        assert len(batch.env_infos) == 3
        
        # Check that all actions are the expected policy output
        for action in batch.actions:
            assert action["X"] == 1.0
        
        # Check diversity metrics
        assert batch.diversity_metrics.batch_size == 3
        assert isinstance(batch.diversity_metrics.reward_variance, float)
    
    @pytest.mark.asyncio
    async def test_run_environment_step(self, training_manager):
        """Test running single environment step."""
        def simple_policy(state):
            return pyr.pmap({"X": 0.5})
        
        env = training_manager.environments[0]
        result = await training_manager._run_environment_step(env, simple_policy)
        
        state, action, next_state, reward, env_info = result
        
        assert isinstance(state, JAXAcquisitionState)
        assert isinstance(next_state, JAXAcquisitionState)
        assert action["X"] == 0.5
        assert isinstance(reward.total_reward, float)
        assert env_info.intervention_count == 1
    
    def test_update_progress(self, training_manager):
        """Test progress updating."""
        # Create mock batch
        rewards = [
            RewardResult(1.0, {"r1": 1.0}, {}),
            RewardResult(0.5, {"r1": 0.5}, {}),
            RewardResult(0.8, {"r1": 0.8}, {}),
        ]
        
        from causal_bayes_opt.training.diversity_monitor import DiversityMetrics
        from causal_bayes_opt.environments.intervention_env import EnvironmentInfo
        
        batch = TrainingBatch(
            states=[], actions=[], next_states=[], rewards=rewards,
            env_infos=[], diversity_metrics=DiversityMetrics(0.0, {}, 0.0, False, 3, 0.0),
            metadata={}
        )
        
        initial_samples = training_manager._total_samples
        training_manager._update_progress(batch)
        
        assert training_manager._total_samples == initial_samples + 3
        assert len(training_manager._recent_rewards) == 3
        assert 0.5 in training_manager._recent_rewards
        assert 1.0 in training_manager._recent_rewards
        assert 0.8 in training_manager._recent_rewards
    
    def test_get_current_progress(self, training_manager):
        """Test getting current progress."""
        # Add some recent rewards
        training_manager._recent_rewards = [1.0, 0.5, 0.8]
        training_manager._training_step = 10
        training_manager._episode_count = 5
        training_manager._total_samples = 30
        
        progress = training_manager._get_current_progress()
        
        assert progress.step == 10
        assert progress.episode == 5
        assert progress.total_samples == 30
        assert progress.avg_reward == (1.0 + 0.5 + 0.8) / 3
        assert progress.total_time > 0
        assert isinstance(progress.diversity_health, float)
        assert isinstance(progress.curriculum_difficulty, float)
    
    @pytest.mark.asyncio
    async def test_short_training_loop(self, training_manager):
        """Test running a short training loop."""
        def simple_policy(state):
            return pyr.pmap({"X": 1.0})
        
        def simple_update(batch):
            return {"loss": 0.1}
        
        # Run for just 2 steps
        progress = await training_manager.run_training_loop(
            max_steps=2,
            policy_fn=simple_policy,
            update_fn=simple_update
        )
        
        assert progress.step == 2
        assert progress.episode > 0  # Should have run some episodes
        assert progress.total_samples > 0
        assert progress.total_time > 0


class TestAsyncTrainingFactory:
    """Test factory functions for async training."""
    
    def test_create_async_training_manager_defaults(self):
        """Test creating manager with default configuration."""
        from causal_bayes_opt.data_structures.scm import create_scm
        from causal_bayes_opt.environments.intervention_env import create_intervention_environment
        
        # Create minimal environment
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        env = create_intervention_environment(scm, create_deployment_rubric())
        diversity_monitor = create_diversity_monitor()
        
        manager = create_async_training_manager([env], diversity_monitor)
        
        assert isinstance(manager, AsyncTrainingManager)
        assert len(manager.environments) == 1
        assert manager.diversity_monitor == diversity_monitor
        assert isinstance(manager.config, AsyncTrainingConfig)
    
    def test_create_async_training_manager_custom(self):
        """Test creating manager with custom configuration."""
        from causal_bayes_opt.data_structures.scm import create_scm
        from causal_bayes_opt.environments.intervention_env import create_intervention_environment
        from causal_bayes_opt.training.acquisition_training import AcquisitionTrainingConfig
        
        # Create minimal environment
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        env = create_intervention_environment(scm, create_deployment_rubric())
        diversity_monitor = create_diversity_monitor()
        
        async_config = AsyncTrainingConfig(batch_size=32, num_workers=4)
        training_config = AcquisitionTrainingConfig()
        
        manager = create_async_training_manager(
            [env], diversity_monitor, async_config, training_config
        )
        
        assert manager.config.batch_size == 32
        assert manager.config.num_workers == 4
        assert manager.training_config == training_config


class TestMemoryOptimization:
    """Test memory optimization utilities."""
    
    def test_estimate_batch_memory_usage(self):
        """Test memory usage estimation."""
        memory_mb = estimate_batch_memory_usage(
            batch_size=32,
            n_vars=5,
            max_samples=100,
            feature_dim=3
        )
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
        # Should be reasonable (not too large for small batch)
        assert memory_mb < 1000  # Less than 1GB for small batch
    
    def test_memory_scaling(self):
        """Test that memory usage scales with batch size."""
        small_batch = estimate_batch_memory_usage(
            batch_size=8, n_vars=3, max_samples=50, feature_dim=3
        )
        
        large_batch = estimate_batch_memory_usage(
            batch_size=64, n_vars=3, max_samples=50, feature_dim=3
        )
        
        # Large batch should use more memory
        assert large_batch > small_batch
        # Should scale roughly linearly
        assert large_batch / small_batch >= 7  # Should be close to 8x
    
    def test_optimize_batch_size_for_memory(self):
        """Test batch size optimization."""
        optimal_size = optimize_batch_size_for_memory(
            max_memory_mb=100.0,
            n_vars=3,
            max_samples=50,
            feature_dim=3
        )
        
        assert isinstance(optimal_size, int)
        assert optimal_size > 0
        assert optimal_size <= 1024  # Reasonable upper bound
        
        # Check that the optimal size fits within memory
        estimated_memory = estimate_batch_memory_usage(
            optimal_size, n_vars=3, max_samples=50, feature_dim=3
        )
        assert estimated_memory <= 100.0
    
    def test_memory_constraint_respected(self):
        """Test that memory constraints are respected."""
        # Very small memory limit
        small_optimal = optimize_batch_size_for_memory(
            max_memory_mb=1.0,  # 1 MB
            n_vars=10,
            max_samples=100,
            feature_dim=5
        )
        
        # Large memory limit
        large_optimal = optimize_batch_size_for_memory(
            max_memory_mb=1000.0,  # 1 GB
            n_vars=10,
            max_samples=100,
            feature_dim=5
        )
        
        # Large limit should allow larger batch size
        assert large_optimal > small_optimal


class TestAsyncTrainingIntegration:
    """Test integration scenarios for async training."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_training_workflow(self):
        """Test complete training workflow."""
        from causal_bayes_opt.data_structures.scm import create_scm
        
        # Create test SCM
        scm = create_scm(
            variables=frozenset({"X", "Y"}),
            edges=frozenset([("X", "Y")]),
            mechanisms={
                "X": lambda parents, key: jax.random.normal(key),
                "Y": lambda parents, key: 2.0 * parents.get("X", 0.0) + 0.1 * jax.random.normal(key),
            },
            target="Y"
        )
        
        # Create environments
        environments = create_batch_environments(
            scms=[scm] * 2,
            rubric=create_deployment_rubric(),
            difficulty_range=(0.4, 0.8),
            max_interventions=5
        )
        
        # Create manager
        config = AsyncTrainingConfig(
            max_parallel_envs=2,
            batch_size=2,
            num_workers=1,
            checkpoint_interval=2,
            enable_compilation=False,
            progress_logging_interval=1
        )
        
        manager = create_async_training_manager(
            environments=environments,
            diversity_monitor=create_diversity_monitor(),
            async_config=config
        )
        
        # Simple policy and update functions
        def policy(state):
            return pyr.pmap({"X": 1.0})  # Always intervene with X=1.0
        
        def update(batch):
            return {"average_reward": sum(r.total_reward for r in batch.rewards) / len(batch.rewards)}
        
        # Run short training
        progress = await manager.run_training_loop(
            max_steps=3,
            policy_fn=policy,
            update_fn=update
        )
        
        # Verify training completed
        assert progress.step == 3
        assert progress.episode > 0
        assert progress.total_samples > 0
        assert progress.avg_reward is not None
        assert progress.diversity_health >= 0.0
    
    def test_memory_optimization_integration(self):
        """Test memory optimization in realistic scenario."""
        # Simulate medium-scale training scenario
        optimal_batch = optimize_batch_size_for_memory(
            max_memory_mb=2048.0,  # 2 GB
            n_vars=10,
            max_samples=200,
            feature_dim=8
        )
        
        # Should get reasonable batch size (allowing for large optimal size with lots of memory)
        assert 10 <= optimal_batch <= 1024
        
        # Verify memory usage is within bounds
        estimated_memory = estimate_batch_memory_usage(
            optimal_batch, n_vars=10, max_samples=200, feature_dim=8
        )
        assert estimated_memory <= 2048.0
        
        # Should utilize memory efficiently (but our estimation may be conservative)
        assert estimated_memory > 10.0  # Should use some reasonable amount