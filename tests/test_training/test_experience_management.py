"""Tests for experience management system.

This module tests the ExperienceManager and related classes for efficient
experience storage, sampling, and replay in GRPO training.
"""

import pytest
import time
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.training.experience_management import (
    ExperienceConfig,
    Experience,
    ExperienceBatch,
    ExperienceManager,
    SumTree,
    create_experience_manager,
    create_high_capacity_experience_manager,
    create_memory_efficient_experience_manager,
)
from causal_bayes_opt.training.grpo_core import GRPOConfig, create_default_grpo_config
from causal_bayes_opt.jax_native.state import create_test_state
from causal_bayes_opt.acquisition.reward_rubric import RewardResult
from causal_bayes_opt.environments.intervention_env import EnvironmentInfo


class TestExperienceConfig:
    """Test experience management configuration."""
    
    def test_config_creation(self):
        """Test creating experience config."""
        config = ExperienceConfig(
            max_buffer_size=5000,
            batch_size=16,
            min_replay_size=500,
            prioritized_replay=True,
            priority_alpha=0.7,
            importance_beta=0.5,
            max_trajectory_length=150,
            enable_compression=True,
            memory_limit_mb=2048
        )
        
        assert config.max_buffer_size == 5000
        assert config.batch_size == 16
        assert config.min_replay_size == 500
        assert config.prioritized_replay is True
        assert config.priority_alpha == 0.7
        assert config.importance_beta == 0.5
        assert config.max_trajectory_length == 150
        assert config.enable_compression is True
        assert config.memory_limit_mb == 2048
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ExperienceConfig()
        
        assert config.max_buffer_size == 10000
        assert config.batch_size == 32
        assert config.min_replay_size == 1000
        assert config.prioritized_replay is False
        assert config.priority_alpha == 0.6
        assert config.importance_beta == 0.4
        assert config.max_trajectory_length == 100
        assert config.enable_compression is False
        assert config.memory_limit_mb == 1024
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = ExperienceConfig()
        
        with pytest.raises(AttributeError):
            config.batch_size = 64


class TestExperience:
    """Test experience data structure."""
    
    @pytest.fixture
    def sample_experience(self):
        """Create sample experience."""
        state = create_test_state()
        next_state = create_test_state()
        action = pyr.pmap({"X": 1.0})
        reward = RewardResult(0.8, {"r1": 0.6, "r2": 0.2}, {})
        env_info = EnvironmentInfo(1, 10.0, False, False, False, 0.8, {})
        
        return Experience(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=False,
            log_prob=-0.5,
            value=0.7,
            env_info=env_info,
            timestamp=time.time(),
            priority=1.0
        )
    
    def test_experience_creation(self, sample_experience):
        """Test creating experience."""
        exp = sample_experience
        
        assert exp.state is not None
        assert exp.action["X"] == 1.0
        assert exp.next_state is not None
        assert exp.reward.total_reward == 0.8
        assert exp.done is False
        assert exp.log_prob == -0.5
        assert exp.value == 0.7
        assert exp.env_info.intervention_count == 1
        assert exp.priority == 1.0
        assert isinstance(exp.timestamp, float)
    
    def test_experience_immutability(self, sample_experience):
        """Test that experience is immutable."""
        exp = sample_experience
        
        with pytest.raises(AttributeError):
            exp.reward = RewardResult(1.0, {}, {})


class TestExperienceBatch:
    """Test experience batch data structure."""
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences."""
        experiences = []
        for i in range(3):
            state = create_test_state()
            action = pyr.pmap({"X": float(i)})
            reward = RewardResult(0.5 + i * 0.1, {"r1": 0.3 + i * 0.1}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.5 + i * 0.1, {})
            
            exp = Experience(
                state=state,
                action=action,
                next_state=state,  # Reuse for simplicity
                reward=reward,
                done=(i == 2),  # Last experience is terminal
                log_prob=-0.1 * i,
                value=0.6 + i * 0.1,
                env_info=env_info,
                timestamp=time.time(),
                priority=1.0 + i * 0.2
            )
            experiences.append(exp)
        
        return experiences
    
    def test_batch_creation(self, sample_experiences):
        """Test creating experience batch."""
        from causal_bayes_opt.training.grpo_core import GRPOTrajectory
        
        # Create simple trajectory for testing
        trajectory = GRPOTrajectory(
            states=jnp.ones((3, 5)),
            actions=jnp.ones((3, 2)),
            rewards=jnp.array([0.5, 0.6, 0.7]),
            values=jnp.array([0.6, 0.7, 0.8]),
            log_probs=jnp.array([0.0, -0.1, -0.2]),
            dones=jnp.array([0.0, 0.0, 1.0]),
            advantages=jnp.array([0.1, 0.0, -0.1]),
            returns=jnp.array([0.7, 0.7, 0.7])
        )
        
        batch = ExperienceBatch(
            experiences=sample_experiences,
            trajectory=trajectory,
            importance_weights=jnp.array([1.0, 1.0, 1.0]),
            indices=[0, 1, 2],
            metadata={"method": "test"}
        )
        
        assert len(batch.experiences) == 3
        assert batch.trajectory.states.shape == (3, 5)
        assert batch.importance_weights.shape == (3,)
        assert batch.indices == [0, 1, 2]
        assert batch.metadata["method"] == "test"
    
    def test_batch_immutability(self, sample_experiences):
        """Test that batch is immutable."""
        from causal_bayes_opt.training.grpo_core import GRPOTrajectory
        
        trajectory = GRPOTrajectory(
            states=jnp.ones((3, 5)), actions=jnp.ones((3, 2)),
            rewards=jnp.ones(3), values=jnp.ones(3), log_probs=jnp.ones(3),
            dones=jnp.zeros(3), advantages=jnp.ones(3), returns=jnp.ones(3)
        )
        
        batch = ExperienceBatch(
            experiences=sample_experiences, trajectory=trajectory,
            importance_weights=jnp.ones(3), indices=[0, 1, 2], metadata={}
        )
        
        with pytest.raises(AttributeError):
            batch.indices = [3, 4, 5]


class TestSumTree:
    """Test sum tree for prioritized experience replay."""
    
    def test_sumtree_creation(self):
        """Test creating sum tree."""
        tree = SumTree(capacity=4)
        
        assert tree.capacity == 4
        assert len(tree.tree) == 7  # 2 * capacity - 1
    
    def test_sumtree_update(self):
        """Test updating priorities in sum tree."""
        tree = SumTree(capacity=4)
        
        # Update some priorities
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 0.5)
        
        # Root should have sum of all priorities
        assert float(tree.tree[0]) == 3.5
    
    def test_sumtree_sampling(self):
        """Test sampling from sum tree."""
        tree = SumTree(capacity=4)
        
        # Set some priorities
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 1.0)
        tree.update(3, 0.5)
        
        # Sample multiple times
        samples = []
        for _ in range(100):
            idx, priority = tree.sample()
            assert 0 <= idx < 4
            assert priority > 0
            samples.append(idx)
        
        # Index 1 should be sampled most frequently (highest priority)
        assert samples.count(1) > samples.count(0)
        assert samples.count(1) > samples.count(2)


class TestExperienceManager:
    """Test the main experience manager."""
    
    @pytest.fixture
    def experience_config(self):
        """Create test experience config."""
        return ExperienceConfig(
            max_buffer_size=100,
            batch_size=8,
            min_replay_size=10,
            prioritized_replay=False
        )
    
    @pytest.fixture
    def grpo_config(self):
        """Create test GRPO config."""
        return create_default_grpo_config()
    
    @pytest.fixture
    def experience_manager(self, experience_config, grpo_config):
        """Create test experience manager."""
        return ExperienceManager(experience_config, grpo_config)
    
    @pytest.fixture
    def sample_experience(self):
        """Create sample experience."""
        state = create_test_state()
        action = pyr.pmap({"X": 1.0})
        reward = RewardResult(0.8, {"r1": 0.8}, {})
        env_info = EnvironmentInfo(1, 10.0, False, False, False, 0.8, {})
        
        return Experience(
            state=state,
            action=action,
            next_state=state,
            reward=reward,
            done=False,
            log_prob=-0.5,
            value=0.7,
            env_info=env_info,
            timestamp=time.time()
        )
    
    def test_manager_creation(self, experience_manager):
        """Test creating experience manager."""
        assert experience_manager.config.max_buffer_size == 100
        assert experience_manager.config.batch_size == 8
        assert experience_manager._size == 0
        assert len(experience_manager._experiences) == 0
    
    def test_add_single_experience(self, experience_manager, sample_experience):
        """Test adding single experience."""
        initial_size = experience_manager._size
        
        experience_manager.add_experience(sample_experience)
        
        assert experience_manager._size == initial_size + 1
        assert len(experience_manager._experiences) == 1
    
    def test_add_multiple_experiences(self, experience_manager):
        """Test adding multiple experiences."""
        experiences = []
        for i in range(5):
            state = create_test_state()
            action = pyr.pmap({"X": float(i)})
            reward = RewardResult(0.5 + i * 0.1, {"r1": 0.5 + i * 0.1}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.5, {})
            
            exp = Experience(
                state=state, action=action, next_state=state, reward=reward,
                done=False, log_prob=-0.1 * i, value=0.6, env_info=env_info,
                timestamp=time.time()
            )
            experiences.append(exp)
            experience_manager.add_experience(exp)
        
        assert experience_manager._size == 5
        assert len(experience_manager._experiences) == 5
    
    def test_add_trajectory(self, experience_manager):
        """Test adding complete trajectory."""
        trajectory_experiences = []
        for i in range(10):
            state = create_test_state()
            action = pyr.pmap({"X": float(i)})
            reward = RewardResult(0.5, {"r1": 0.5}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.5, {})
            
            exp = Experience(
                state=state, action=action, next_state=state, reward=reward,
                done=(i == 9), log_prob=-0.1, value=0.6, env_info=env_info,
                timestamp=time.time()
            )
            trajectory_experiences.append(exp)
        
        experience_manager.add_trajectory(trajectory_experiences)
        
        assert experience_manager._size == 10
    
    def test_sample_insufficient_experiences(self, experience_manager):
        """Test sampling when insufficient experiences."""
        # Add only a few experiences (less than min_replay_size)
        for i in range(5):
            state = create_test_state()
            exp = Experience(
                state=state, action=pyr.pmap({"X": 1.0}), next_state=state,
                reward=RewardResult(0.5, {}, {}), done=False, log_prob=-0.1,
                value=0.6, env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            experience_manager.add_experience(exp)
        
        batch = experience_manager.sample_batch()
        assert batch is None  # Not enough experiences
    
    def test_sample_sufficient_experiences(self, experience_manager):
        """Test sampling when sufficient experiences available."""
        # Add enough experiences
        for i in range(15):  # More than min_replay_size (10)
            state = create_test_state()
            action = pyr.pmap({"X": float(i % 3)})
            reward = RewardResult(0.5 + i * 0.01, {"r1": 0.5}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.5, {})
            
            exp = Experience(
                state=state, action=action, next_state=state, reward=reward,
                done=(i % 5 == 4), log_prob=-0.1 * i, value=0.6 + i * 0.01,
                env_info=env_info, timestamp=time.time()
            )
            experience_manager.add_experience(exp)
        
        batch = experience_manager.sample_batch()
        
        assert batch is not None
        assert len(batch.experiences) == 8  # batch_size
        assert isinstance(batch.trajectory, object)  # GRPO trajectory
        assert batch.importance_weights.shape == (8,)
        assert len(batch.indices) == 8
        assert batch.metadata["sampling_method"] == "uniform"
    
    def test_buffer_overflow(self, experience_manager):
        """Test buffer behavior when exceeding capacity."""
        # Add more than max_buffer_size
        for i in range(150):  # More than capacity (100)
            state = create_test_state()
            exp = Experience(
                state=state, action=pyr.pmap({"X": 1.0}), next_state=state,
                reward=RewardResult(0.5, {}, {}), done=False, log_prob=-0.1,
                value=0.6, env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            experience_manager.add_experience(exp)
        
        # Buffer should not exceed capacity
        assert experience_manager._size == 100
        assert len(experience_manager._experiences) == 100
    
    def test_get_statistics(self, experience_manager):
        """Test getting manager statistics."""
        # Add some experiences
        for i in range(5):
            state = create_test_state()
            exp = Experience(
                state=state, action=pyr.pmap({"X": 1.0}), next_state=state,
                reward=RewardResult(0.5, {}, {}), done=False, log_prob=-0.1,
                value=0.6, env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            experience_manager.add_experience(exp)
        
        stats = experience_manager.get_statistics()
        
        assert stats["buffer_size"] == 5
        assert stats["buffer_capacity"] == 100
        assert stats["utilization"] == 0.05
        assert stats["can_sample"] is False  # Not enough for min_replay_size
        assert stats["prioritized_replay"] is False
        assert isinstance(stats["memory_usage_mb"], float)
    
    def test_clear_buffer(self, experience_manager, sample_experience):
        """Test clearing experience buffer."""
        # Add some experiences
        for _ in range(5):
            experience_manager.add_experience(sample_experience)
        
        assert experience_manager._size == 5
        
        experience_manager.clear()
        
        assert experience_manager._size == 0
        assert len(experience_manager._experiences) == 0


class TestPrioritizedExperienceManager:
    """Test experience manager with prioritized replay."""
    
    @pytest.fixture
    def prioritized_config(self):
        """Create prioritized experience config."""
        return ExperienceConfig(
            max_buffer_size=100,
            batch_size=8,
            min_replay_size=10,
            prioritized_replay=True,
            priority_alpha=0.6,
            importance_beta=0.4
        )
    
    @pytest.fixture
    def prioritized_manager(self, prioritized_config):
        """Create prioritized experience manager."""
        grpo_config = create_default_grpo_config()
        return ExperienceManager(prioritized_config, grpo_config)
    
    def test_prioritized_manager_creation(self, prioritized_manager):
        """Test creating prioritized experience manager."""
        assert prioritized_manager.config.prioritized_replay is True
        assert prioritized_manager._priority_tree is not None
    
    def test_prioritized_sampling(self, prioritized_manager):
        """Test prioritized experience sampling."""
        # Add experiences with different priorities
        for i in range(15):
            state = create_test_state()
            action = pyr.pmap({"X": float(i % 3)})
            reward = RewardResult(0.5, {"r1": 0.5}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.5, {})
            
            exp = Experience(
                state=state, action=action, next_state=state, reward=reward,
                done=False, log_prob=-0.1, value=0.6, env_info=env_info,
                timestamp=time.time(), priority=1.0 + i * 0.1  # Increasing priorities
            )
            prioritized_manager.add_experience(exp)
        
        batch = prioritized_manager.sample_batch()
        
        assert batch is not None
        assert len(batch.experiences) == 8
        assert batch.metadata["sampling_method"] == "prioritized"
        # Importance weights should not all be equal for prioritized sampling
        assert not jnp.allclose(batch.importance_weights, batch.importance_weights[0])
    
    def test_update_priorities(self, prioritized_manager):
        """Test updating experience priorities."""
        # Add some experiences
        for i in range(15):
            state = create_test_state()
            exp = Experience(
                state=state, action=pyr.pmap({"X": 1.0}), next_state=state,
                reward=RewardResult(0.5, {}, {}), done=False, log_prob=-0.1,
                value=0.6, env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time(), priority=1.0
            )
            prioritized_manager.add_experience(exp)
        
        # Update priorities for some experiences
        indices = [0, 1, 2]
        new_priorities = jnp.array([2.0, 3.0, 1.5])
        
        prioritized_manager.update_priorities(indices, new_priorities)
        
        # This should not raise an error
        assert True  # Successful execution


class TestExperienceManagerFactories:
    """Test factory functions for experience managers."""
    
    def test_create_experience_manager(self):
        """Test basic experience manager factory."""
        grpo_config = create_default_grpo_config()
        manager = create_experience_manager(grpo_config, buffer_size=5000, batch_size=16)
        
        assert manager.config.max_buffer_size == 5000
        assert manager.config.batch_size == 16
        assert manager.config.prioritized_replay is False
    
    def test_create_high_capacity_manager(self):
        """Test high-capacity experience manager factory."""
        grpo_config = create_default_grpo_config()
        manager = create_high_capacity_experience_manager(grpo_config, memory_limit_mb=2048)
        
        assert manager.config.max_buffer_size == 50000
        assert manager.config.batch_size == 64
        assert manager.config.prioritized_replay is True
        assert manager.config.memory_limit_mb == 2048
    
    def test_create_memory_efficient_manager(self):
        """Test memory-efficient experience manager factory."""
        grpo_config = create_default_grpo_config()
        manager = create_memory_efficient_experience_manager(grpo_config)
        
        assert manager.config.max_buffer_size == 2000
        assert manager.config.batch_size == 16
        assert manager.config.memory_limit_mb == 256
        assert manager.config.prioritized_replay is False


class TestExperienceManagerIntegration:
    """Test integration scenarios for experience management."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle with experience management."""
        from causal_bayes_opt.training.experience_management import ExperienceConfig, ExperienceManager
        
        grpo_config = create_default_grpo_config()
        exp_config = ExperienceConfig(
            max_buffer_size=50,
            batch_size=4,
            min_replay_size=10  # Lower threshold for testing
        )
        manager = ExperienceManager(exp_config, grpo_config)
        
        # Simulate training episodes
        for episode in range(3):
            trajectory = []
            for step in range(5):
                state = create_test_state()
                action = pyr.pmap({"X": float(step)})
                reward = RewardResult(0.5 + step * 0.1, {"r1": 0.5}, {})
                env_info = EnvironmentInfo(step, 10.0, False, False, False, 0.5, {})
                
                exp = Experience(
                    state=state, action=action, next_state=state, reward=reward,
                    done=(step == 4), log_prob=-0.1 * step, value=0.6 + step * 0.05,
                    env_info=env_info, timestamp=time.time()
                )
                trajectory.append(exp)
            
            manager.add_trajectory(trajectory)
        
        # Should be able to sample after adding enough experiences
        assert manager._size == 15
        
        batch = manager.sample_batch()
        assert batch is not None
        assert len(batch.experiences) == 4
        
        # Simulate training update
        stats = manager.get_statistics()
        assert stats["can_sample"] is True
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        config = ExperienceConfig(
            max_buffer_size=20,
            memory_limit_mb=1,  # Very low limit to trigger cleanup
            batch_size=4
        )
        grpo_config = create_default_grpo_config()
        manager = ExperienceManager(config, grpo_config)
        
        # Add many experiences to trigger memory management
        for i in range(25):
            state = create_test_state()
            exp = Experience(
                state=state, action=pyr.pmap({"X": 1.0}), next_state=state,
                reward=RewardResult(0.5, {}, {}), done=False, log_prob=-0.1,
                value=0.6, env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            manager.add_experience(exp)
        
        # Buffer should respect capacity limits
        assert manager._size <= config.max_buffer_size
        stats = manager.get_statistics()
        assert isinstance(stats["memory_usage_mb"], float)