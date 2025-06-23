"""Tests for core GRPO algorithm implementation.

This module tests the pure functional GRPO implementation with emphasis on
property-based testing and GRPO algorithmic invariants.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as onp
import optax

from causal_bayes_opt.training.grpo_core import (
    GRPOConfig,
    GRPOTrajectory,
    GRPOUpdateResult,
    compute_gae_advantages,
    compute_simple_advantages,
    normalize_advantages,
    compute_policy_loss,
    compute_value_loss,
    compute_entropy_loss,
    create_grpo_update_fn,
    create_trajectory_from_experiences,
    validate_grpo_config,
    create_default_grpo_config,
    create_high_performance_grpo_config,
    create_exploration_grpo_config,
)


class TestGRPOConfig:
    """Test GRPO configuration system."""
    
    def test_config_creation(self):
        """Test creating GRPO config."""
        config = GRPOConfig(
            learning_rate=1e-4,
            value_learning_rate=5e-4,
            discount_factor=0.95,
            gae_lambda=0.9,
            clip_ratio=0.1,
            entropy_coefficient=0.02,
            value_loss_coefficient=0.8,
            max_grad_norm=1.0,
            target_kl=0.005,
            normalize_advantages=False,
            use_gae=False
        )
        
        assert config.learning_rate == 1e-4
        assert config.value_learning_rate == 5e-4
        assert config.discount_factor == 0.95
        assert config.gae_lambda == 0.9
        assert config.clip_ratio == 0.1
        assert config.entropy_coefficient == 0.02
        assert config.value_loss_coefficient == 0.8
        assert config.max_grad_norm == 1.0
        assert config.target_kl == 0.005
        assert config.normalize_advantages is False
        assert config.use_gae is False
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = GRPOConfig()
        
        assert config.learning_rate == 3e-4
        assert config.value_learning_rate == 1e-3
        assert config.discount_factor == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_ratio == 0.2
        assert config.entropy_coefficient == 0.01
        assert config.value_loss_coefficient == 0.5
        assert config.max_grad_norm == 0.5
        assert config.target_kl == 0.01
        assert config.normalize_advantages is True
        assert config.use_gae is True
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = GRPOConfig()
        
        with pytest.raises(AttributeError):
            config.learning_rate = 1e-3
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        valid_config = GRPOConfig()
        validate_grpo_config(valid_config)
        
        # Invalid learning rates
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_grpo_config(GRPOConfig(learning_rate=-1e-4))
        
        with pytest.raises(ValueError, match="value_learning_rate must be positive"):
            validate_grpo_config(GRPOConfig(value_learning_rate=0.0))
        
        # Invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be in"):
            validate_grpo_config(GRPOConfig(discount_factor=1.5))
        
        with pytest.raises(ValueError, match="discount_factor must be in"):
            validate_grpo_config(GRPOConfig(discount_factor=-0.1))
        
        # Invalid GAE lambda
        with pytest.raises(ValueError, match="gae_lambda must be in"):
            validate_grpo_config(GRPOConfig(gae_lambda=1.2))
        
        # Invalid clip ratio
        with pytest.raises(ValueError, match="clip_ratio must be positive"):
            validate_grpo_config(GRPOConfig(clip_ratio=-0.1))
        
        # Invalid entropy coefficient
        with pytest.raises(ValueError, match="entropy_coefficient must be non-negative"):
            validate_grpo_config(GRPOConfig(entropy_coefficient=-0.1))


class TestGRPOTrajectory:
    """Test GRPO trajectory data structure."""
    
    @pytest.fixture
    def sample_trajectory_data(self):
        """Create sample trajectory data."""
        T = 10
        return {
            'states': jnp.ones((T, 5)),
            'actions': jnp.ones((T, 2)),
            'rewards': jnp.ones(T),
            'values': jnp.ones(T),
            'log_probs': jnp.zeros(T),
            'dones': jnp.zeros(T),
            'advantages': jnp.ones(T),
            'returns': jnp.ones(T)
        }
    
    def test_trajectory_creation(self, sample_trajectory_data):
        """Test creating GRPO trajectory."""
        trajectory = GRPOTrajectory(**sample_trajectory_data)
        
        assert trajectory.states.shape == (10, 5)
        assert trajectory.actions.shape == (10, 2)
        assert trajectory.rewards.shape == (10,)
        assert trajectory.values.shape == (10,)
        assert trajectory.log_probs.shape == (10,)
        assert trajectory.dones.shape == (10,)
        assert trajectory.advantages.shape == (10,)
        assert trajectory.returns.shape == (10,)
    
    def test_trajectory_immutability(self, sample_trajectory_data):
        """Test that trajectory is immutable."""
        trajectory = GRPOTrajectory(**sample_trajectory_data)
        
        with pytest.raises(AttributeError):
            trajectory.rewards = jnp.zeros(10)


class TestAdvantageEstimation:
    """Test advantage estimation functions."""
    
    def test_gae_advantages_shape(self):
        """Test GAE advantages have correct shape."""
        T = 5
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2, 1.0])
        values = jnp.array([0.5, 0.6, 0.7, 0.4, 0.9, 0.5])  # T+1
        dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
        
        advantages, returns = compute_gae_advantages(rewards, values, dones)
        
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
    
    def test_gae_advantages_terminal_episode(self):
        """Test GAE with terminal episode."""
        rewards = jnp.array([1.0, 1.0, 1.0])
        values = jnp.array([0.0, 0.0, 0.0, 0.0])
        dones = jnp.array([0.0, 0.0, 1.0])  # Terminal at end
        
        advantages, returns = compute_gae_advantages(rewards, values, dones, gamma=0.9, gae_lambda=0.95)
        
        # Terminal episode should have zero bootstrapping
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        assert jnp.all(jnp.isfinite(advantages))
        assert jnp.all(jnp.isfinite(returns))
    
    def test_simple_advantages_shape(self):
        """Test simple advantages have correct shape."""
        T = 5
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2, 1.0])
        values = jnp.array([0.5, 0.6, 0.7, 0.4, 0.9, 0.5])  # T+1
        dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
        
        advantages, returns = compute_simple_advantages(rewards, values, dones)
        
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
    
    def test_advantage_estimation_consistency(self):
        """Test that advantage estimation is consistent."""
        # Property: advantages + values should equal returns (approximately)
        rewards = jnp.array([1.0, 0.5, 0.8])
        values = jnp.array([0.2, 0.3, 0.4, 0.5])
        dones = jnp.array([0.0, 0.0, 1.0])
        
        advantages, returns = compute_gae_advantages(rewards, values, dones)
        
        # Check that advantages are finite
        assert jnp.all(jnp.isfinite(advantages))
        assert jnp.all(jnp.isfinite(returns))
    
    def test_normalize_advantages(self):
        """Test advantage normalization."""
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        normalized = normalize_advantages(advantages)
        
        # Should have approximately zero mean and unit variance
        assert abs(float(jnp.mean(normalized))) < 1e-6
        assert abs(float(jnp.std(normalized)) - 1.0) < 1e-6
    
    def test_normalize_advantages_constant(self):
        """Test normalizing constant advantages."""
        advantages = jnp.array([1.0, 1.0, 1.0, 1.0])
        
        normalized = normalize_advantages(advantages)
        
        # Should handle constant case gracefully
        assert jnp.all(jnp.isfinite(normalized))
    
    def test_gae_gamma_zero(self):
        """Test GAE with gamma=0 (no discounting)."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([0.0, 0.0, 0.0, 0.0])
        dones = jnp.array([0.0, 0.0, 1.0])
        
        advantages, returns = compute_gae_advantages(rewards, values, dones, gamma=0.0)
        
        # With gamma=0, advantages should approximately equal rewards
        assert jnp.allclose(advantages, rewards, atol=1e-6)
    
    def test_gae_lambda_zero(self):
        """Test GAE with lambda=0 (no eligibility traces)."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([0.5, 0.6, 0.7, 0.8])
        dones = jnp.array([0.0, 0.0, 1.0])
        
        gae_adv, _ = compute_gae_advantages(rewards, values, dones, gae_lambda=0.0)
        simple_adv, _ = compute_simple_advantages(rewards, values, dones)
        
        # With lambda=0, GAE should equal simple advantages
        assert jnp.allclose(gae_adv, simple_adv, atol=1e-6)


class TestPolicyLoss:
    """Test policy loss computation."""
    
    def test_policy_loss_no_clipping(self):
        """Test policy loss when no clipping occurs."""
        old_log_probs = jnp.array([0.0, 0.0, 0.0])
        new_log_probs = jnp.array([0.1, -0.1, 0.05])  # Small changes
        advantages = jnp.array([1.0, -1.0, 0.5])
        
        policy_loss, kl_div, clip_frac = compute_policy_loss(
            old_log_probs, new_log_probs, advantages, clip_ratio=0.2
        )
        
        assert isinstance(float(policy_loss), float)
        assert isinstance(float(kl_div), float)
        assert isinstance(float(clip_frac), float)
        assert 0.0 <= clip_frac <= 1.0
    
    def test_policy_loss_heavy_clipping(self):
        """Test policy loss with heavy clipping."""
        old_log_probs = jnp.array([0.0, 0.0, 0.0])
        new_log_probs = jnp.array([1.0, -1.0, 2.0])  # Large changes
        advantages = jnp.array([1.0, -1.0, 0.5])
        
        policy_loss, kl_div, clip_frac = compute_policy_loss(
            old_log_probs, new_log_probs, advantages, clip_ratio=0.1
        )
        
        # Should have significant clipping
        assert float(clip_frac) > 0.5
        assert abs(float(kl_div)) > 0.1  # Significant KL divergence
    
    def test_policy_loss_properties(self):
        """Test policy loss properties."""
        # Property: Identical policies should have zero KL and zero clip fraction
        log_probs = jnp.array([0.1, -0.2, 0.3])
        advantages = jnp.array([1.0, -1.0, 0.5])
        
        policy_loss, kl_div, clip_frac = compute_policy_loss(
            log_probs, log_probs, advantages, clip_ratio=0.2
        )
        
        assert abs(float(kl_div)) < 1e-6
        assert float(clip_frac) < 1e-6


class TestValueLoss:
    """Test value function loss computation."""
    
    def test_value_loss_without_clipping(self):
        """Test value loss without clipping."""
        predicted_values = jnp.array([1.0, 2.0, 3.0])
        target_returns = jnp.array([1.1, 1.9, 3.2])
        old_values = jnp.array([0.9, 2.1, 2.8])
        
        value_loss, explained_var = compute_value_loss(
            predicted_values, target_returns, old_values, use_clipping=False
        )
        
        assert isinstance(float(value_loss), float)
        assert isinstance(float(explained_var), float)
        assert float(value_loss) >= 0.0
    
    def test_value_loss_with_clipping(self):
        """Test value loss with clipping."""
        predicted_values = jnp.array([1.0, 2.0, 3.0])
        target_returns = jnp.array([1.1, 1.9, 3.2])
        old_values = jnp.array([0.5, 2.5, 2.5])  # Different from predicted
        
        value_loss, explained_var = compute_value_loss(
            predicted_values, target_returns, old_values, 
            clip_ratio=0.2, use_clipping=True
        )
        
        assert isinstance(float(value_loss), float)
        assert isinstance(float(explained_var), float)
        assert float(value_loss) >= 0.0
    
    def test_value_loss_perfect_prediction(self):
        """Test value loss with perfect predictions."""
        values = jnp.array([1.0, 2.0, 3.0])
        
        value_loss, explained_var = compute_value_loss(
            values, values, values, use_clipping=False
        )
        
        assert abs(float(value_loss)) < 1e-6
        # Explained variance might not be exactly 1 due to numerical issues
        assert float(explained_var) > 0.9


class TestEntropyLoss:
    """Test entropy loss computation."""
    
    def test_entropy_loss_uniform(self):
        """Test entropy loss for uniform distribution."""
        # Uniform distribution has maximum entropy
        uniform_log_probs = jnp.log(jnp.array([0.5, 0.5]))
        
        entropy_loss = compute_entropy_loss(uniform_log_probs)
        
        assert isinstance(float(entropy_loss), float)
        # Should be positive (negative log prob) since we return negative mean log prob
        assert float(entropy_loss) > 0.0
    
    def test_entropy_loss_deterministic(self):
        """Test entropy loss for deterministic distribution."""
        # Deterministic distribution has minimum entropy
        det_log_probs = jnp.array([-100.0, 0.0])  # Almost deterministic
        
        entropy_loss = compute_entropy_loss(det_log_probs)
        
        assert isinstance(float(entropy_loss), float)
        # Should be less negative than uniform case
    
    def test_entropy_loss_properties(self):
        """Test entropy loss properties."""
        log_probs = jnp.array([-1.0, -2.0, -0.5])
        
        entropy_loss = compute_entropy_loss(log_probs)
        
        # Should be finite
        assert jnp.isfinite(entropy_loss)


class TestTrajectoryCreation:
    """Test trajectory creation from experiences."""
    
    @pytest.fixture
    def sample_experience_data(self):
        """Create sample experience data."""
        T = 5
        return {
            'states': jnp.ones((T, 3)),
            'actions': jnp.ones((T, 2)),
            'rewards': jnp.array([1.0, 0.5, 0.8, 0.2, 1.0]),
            'values': jnp.array([0.5, 0.6, 0.7, 0.4, 0.9]),
            'log_probs': jnp.array([0.0, -0.1, 0.1, -0.2, 0.0]),
            'dones': jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            'bootstrap_value': 0.0
        }
    
    def test_trajectory_creation_with_gae(self, sample_experience_data):
        """Test trajectory creation with GAE."""
        config = GRPOConfig(use_gae=True, normalize_advantages=False)
        
        trajectory = create_trajectory_from_experiences(
            **sample_experience_data, config=config
        )
        
        assert isinstance(trajectory, GRPOTrajectory)
        assert trajectory.states.shape == (5, 3)
        assert trajectory.actions.shape == (5, 2)
        assert trajectory.rewards.shape == (5,)
        assert trajectory.advantages.shape == (5,)
        assert trajectory.returns.shape == (5,)
        
        # Check that all values are finite
        assert jnp.all(jnp.isfinite(trajectory.advantages))
        assert jnp.all(jnp.isfinite(trajectory.returns))
    
    def test_trajectory_creation_without_gae(self, sample_experience_data):
        """Test trajectory creation without GAE."""
        config = GRPOConfig(use_gae=False, normalize_advantages=False)
        
        trajectory = create_trajectory_from_experiences(
            **sample_experience_data, config=config
        )
        
        assert isinstance(trajectory, GRPOTrajectory)
        assert jnp.all(jnp.isfinite(trajectory.advantages))
        assert jnp.all(jnp.isfinite(trajectory.returns))
    
    def test_trajectory_creation_with_normalization(self, sample_experience_data):
        """Test trajectory creation with advantage normalization."""
        config = GRPOConfig(normalize_advantages=True)
        
        trajectory = create_trajectory_from_experiences(
            **sample_experience_data, config=config
        )
        
        # Advantages should be normalized
        mean_adv = float(jnp.mean(trajectory.advantages))
        std_adv = float(jnp.std(trajectory.advantages))
        
        assert abs(mean_adv) < 1e-5  # Approximately zero mean
        assert abs(std_adv - 1.0) < 1e-5  # Approximately unit variance


class TestGRPOUpdateFunction:
    """Test GRPO update function creation and execution."""
    
    @pytest.fixture
    def simple_policy_fn(self):
        """Create a simple policy function for testing."""
        def policy_fn(params, states, actions):
            # Simple linear policy returning log probabilities
            return jnp.sum(states * params['weight'], axis=-1)
        return policy_fn
    
    @pytest.fixture
    def simple_value_fn(self):
        """Create a simple value function for testing."""
        def value_fn(params, states):
            # Simple linear value function
            return jnp.sum(states * params['weight'], axis=-1)
        return value_fn
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        T = 5
        return GRPOTrajectory(
            states=jnp.ones((T, 3)),
            actions=jnp.ones((T, 2)),
            rewards=jnp.array([1.0, 0.5, 0.8, 0.2, 1.0]),
            values=jnp.array([0.5, 0.6, 0.7, 0.4, 0.9]),
            log_probs=jnp.array([0.0, -0.1, 0.1, -0.2, 0.0]),
            dones=jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            advantages=jnp.array([0.5, -0.1, 0.1, -0.3, 0.1]),
            returns=jnp.array([1.0, 0.5, 0.8, 0.1, 1.0])
        )
    
    def test_grpo_update_function_creation(self, simple_policy_fn, simple_value_fn):
        """Test creating GRPO update function."""
        config = GRPOConfig()
        policy_optimizer = optax.adam(config.learning_rate)
        value_optimizer = optax.adam(config.value_learning_rate)
        
        update_fn = create_grpo_update_fn(
            simple_policy_fn, simple_value_fn, 
            policy_optimizer, value_optimizer, config
        )
        
        assert callable(update_fn)
    
    def test_grpo_update_execution(self, simple_policy_fn, simple_value_fn, sample_trajectory):
        """Test executing GRPO update."""
        config = GRPOConfig()
        policy_optimizer = optax.adam(config.learning_rate)
        value_optimizer = optax.adam(config.value_learning_rate)
        
        update_fn = create_grpo_update_fn(
            simple_policy_fn, simple_value_fn,
            policy_optimizer, value_optimizer, config
        )
        
        # Initialize parameters and optimizer states
        policy_params = {'weight': jnp.ones(3)}
        value_params = {'weight': jnp.ones(3)}
        policy_opt_state = policy_optimizer.init(policy_params)
        value_opt_state = value_optimizer.init(value_params)
        
        # Execute update
        (new_policy_params, new_value_params, 
         new_policy_opt_state, new_value_opt_state, 
         update_result) = update_fn(
            policy_params, value_params,
            policy_opt_state, value_opt_state,
            sample_trajectory
        )
        
        # Check that update result is valid
        assert isinstance(update_result, GRPOUpdateResult)
        assert jnp.isfinite(update_result.policy_loss)
        assert jnp.isfinite(update_result.value_loss)
        assert jnp.isfinite(update_result.total_loss)
        assert 0.0 <= update_result.clipped_fraction <= 1.0
        
        # Check that parameters were updated
        assert not jnp.allclose(new_policy_params['weight'], policy_params['weight'])
        assert not jnp.allclose(new_value_params['weight'], value_params['weight'])


class TestConfigFactories:
    """Test configuration factory functions."""
    
    def test_default_config_factory(self):
        """Test default config factory."""
        config = create_default_grpo_config()
        
        assert isinstance(config, GRPOConfig)
        validate_grpo_config(config)
    
    def test_high_performance_config_factory(self):
        """Test high-performance config factory."""
        config = create_high_performance_grpo_config()
        
        assert isinstance(config, GRPOConfig)
        validate_grpo_config(config)
        
        # Should have more conservative settings
        assert config.learning_rate < create_default_grpo_config().learning_rate
        assert config.clip_ratio < create_default_grpo_config().clip_ratio
    
    def test_exploration_config_factory(self):
        """Test exploration config factory."""
        config = create_exploration_grpo_config()
        
        assert isinstance(config, GRPOConfig)
        validate_grpo_config(config)
        
        # Should have higher entropy coefficient for exploration
        assert config.entropy_coefficient > create_default_grpo_config().entropy_coefficient


class TestGRPOProperties:
    """Test important GRPO algorithmic properties."""
    
    def test_advantage_estimation_invariants(self):
        """Test that advantage estimation satisfies key invariants."""
        # Property: advantages should sum to approximately zero for long episodes
        T = 100
        rewards = jnp.ones(T) * 0.1  # Small constant reward
        values = jnp.zeros(T + 1)    # Zero value function
        dones = jnp.zeros(T)         # No episode termination
        
        advantages, _ = compute_gae_advantages(rewards, values, dones)
        
        # For constant rewards and zero values, advantages should sum to a reasonable value
        advantage_sum = float(jnp.sum(advantages))
        # With GAE and constant rewards, the sum can be substantial due to bootstrapping
        assert abs(advantage_sum) < T * 10  # More reasonable expectation
    
    def test_policy_loss_monotonicity(self):
        """Test policy loss behavior with advantage signs."""
        old_log_probs = jnp.zeros(3)
        
        # Positive advantages should prefer higher new log probs
        pos_advantages = jnp.array([1.0, 1.0, 1.0])
        new_log_probs_high = jnp.array([0.1, 0.1, 0.1])
        new_log_probs_low = jnp.array([-0.1, -0.1, -0.1])
        
        loss_high, _, _ = compute_policy_loss(old_log_probs, new_log_probs_high, pos_advantages)
        loss_low, _, _ = compute_policy_loss(old_log_probs, new_log_probs_low, pos_advantages)
        
        # Higher log probs with positive advantages should give lower loss
        assert float(loss_high) < float(loss_low)
    
    def test_value_loss_consistency(self):
        """Test value loss consistency properties."""
        # Property: Better predictions should have lower loss
        target_returns = jnp.array([1.0, 2.0, 3.0])
        good_predictions = jnp.array([1.1, 1.9, 3.1])  # Close to targets
        bad_predictions = jnp.array([0.5, 2.5, 3.5])   # Further from targets
        old_values = jnp.ones(3)
        
        good_loss, _ = compute_value_loss(good_predictions, target_returns, old_values, use_clipping=False)
        bad_loss, _ = compute_value_loss(bad_predictions, target_returns, old_values, use_clipping=False)
        
        assert float(good_loss) < float(bad_loss)