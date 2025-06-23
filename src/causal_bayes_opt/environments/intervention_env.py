"""Intervention environment abstraction inspired by verifiers repository.

This module provides a clean abstraction for SCM interactions following the
verifiers repository's environment design: instructions + tasks + interaction + rubric.

For causal optimization, this translates to:
- SCM: The causal model (instructions)
- Target optimization: The task
- Intervention protocol: The interaction
- Reward rubric: The evaluation mechanism
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from ..jax_native.state import JAXAcquisitionState
from ..jax_native.config import JAXConfig  
from ..jax_native.sample_buffer import JAXSampleBuffer
from ..acquisition.reward_rubric import CausalRewardRubric, RewardResult
from ..bridges.legacy_to_jax import convert_legacy_to_jax
from .sampling import sample_with_intervention

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for intervention environment.
    
    Args:
        difficulty: Environment difficulty level (0.0 = easy, 1.0 = hard)
        max_interventions: Maximum number of interventions allowed
        intervention_budget: Budget for intervention costs
        enable_early_stopping: Whether to stop early on target achievement
        target_threshold: Target value threshold for success
        noise_level: Observation noise level
    """
    difficulty: float = 0.5
    max_interventions: int = 100
    intervention_budget: float = 10.0
    enable_early_stopping: bool = True
    target_threshold: float = 2.0
    noise_level: float = 0.1


@dataclass(frozen=True)
class EnvironmentInfo:
    """Information returned from environment step.
    
    Args:
        intervention_count: Number of interventions performed
        budget_remaining: Remaining intervention budget
        target_achieved: Whether target threshold was reached
        early_stopped: Whether episode ended due to early stopping
        episode_complete: Whether episode is finished
        best_value_so_far: Best target value achieved
        metadata: Additional environment-specific information
    """
    intervention_count: int
    budget_remaining: float
    target_achieved: bool
    early_stopped: bool
    episode_complete: bool
    best_value_so_far: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class InterventionEnvironment:
    """Clean abstraction for SCM interactions inspired by verifiers.
    
    This environment provides a standardized interface for causal intervention
    experiments, handling SCM interactions, reward computation, and episode
    management in a way that's compatible with async training.
    
    Args:
        scm: The structural causal model
        rubric: Reward computation system
        config: Environment configuration
        jax_config: JAX state configuration
        state_converter: Function to convert legacy states to JAX
    """
    scm: pyr.PMap
    rubric: CausalRewardRubric
    config: EnvironmentConfig
    jax_config: JAXConfig
    state_converter: Callable[[Any], JAXAcquisitionState]
    
    def reset(self, key: jax.Array) -> JAXAcquisitionState:
        """Initialize environment with JAX state.
        
        Args:
            key: JAX random key for initialization
            
        Returns:
            Initial JAX acquisition state
        """
        # Create empty sample buffer
        buffer = JAXSampleBuffer(
            values=jnp.zeros((self.jax_config.max_samples, self.jax_config.n_vars)),
            interventions=jnp.zeros((self.jax_config.max_samples, self.jax_config.n_vars)),
            targets=jnp.zeros(self.jax_config.max_samples),
            valid_mask=jnp.zeros(self.jax_config.max_samples, dtype=bool),
            write_idx=0,
            n_samples=0,
            config=self.jax_config
        )
        
        # Initialize mechanism features based on SCM
        mechanism_features = self._initialize_mechanism_features(key)
        
        # Create initial state
        return JAXAcquisitionState(
            sample_buffer=buffer,
            mechanism_features=mechanism_features,
            marginal_probs=jnp.zeros(self.jax_config.n_vars),
            confidence_scores=jnp.ones(self.jax_config.n_vars) * 0.5,  # Neutral confidence
            best_value=-1000.0,  # Large negative but finite value
            current_step=0,
            uncertainty_bits=jnp.log(float(self.jax_config.n_vars)),  # Max uncertainty
            config=self.jax_config
        )
    
    def step(
        self,
        state: JAXAcquisitionState,
        action: pyr.PMap,
        key: jax.Array
    ) -> Tuple[JAXAcquisitionState, RewardResult, EnvironmentInfo]:
        """Execute intervention and compute rewards.
        
        Args:
            state: Current acquisition state
            action: Intervention to perform
            key: JAX random key for sampling
            
        Returns:
            Tuple of (next_state, reward_result, environment_info)
        """
        # Sample outcome from SCM with intervention
        outcome = self._sample_outcome(action, key)
        
        # Compute reward using rubric
        ground_truth = self._create_ground_truth_data()
        reward_result = self.rubric.compute_reward(
            state=state,
            action=action,
            outcome=outcome,
            ground_truth=ground_truth
        )
        
        # Update state with new sample
        next_state = self._update_state_with_sample(state, action, outcome)
        
        # Compute environment info
        env_info = self._compute_environment_info(next_state, reward_result)
        
        return next_state, reward_result, env_info
    
    def _initialize_mechanism_features(self, key: jax.Array) -> jnp.ndarray:
        """Initialize mechanism features based on SCM properties."""
        n_vars, feature_dim = self.jax_config.n_vars, self.jax_config.feature_dim
        
        # Extract mechanism types from config
        mechanism_types = jnp.array(self.jax_config.mechanism_types)
        
        # Create basic feature representation
        features = jnp.zeros((n_vars, feature_dim))
        
        # Feature 0: mechanism type (normalized)
        features = features.at[:, 0].set(mechanism_types / 3.0)  # Assume 4 types (0-3)
        
        # Feature 1: difficulty-based complexity
        complexity = self.config.difficulty * jnp.ones(n_vars)
        features = features.at[:, 1].set(complexity)
        
        # Feature 2: random initialization based on difficulty
        key, subkey = jax.random.split(key)
        noise_scale = 0.1 + self.config.difficulty * 0.2
        random_features = jax.random.normal(subkey, (n_vars,)) * noise_scale
        features = features.at[:, 2].set(random_features)
        
        return features
    
    def _sample_outcome(self, action: pyr.PMap, key: jax.Array) -> pyr.PMap:
        """Sample outcome from SCM given intervention."""
        # Convert JAX key to integer seed for sampling function
        seed = int(key[0]) % (2**31 - 1)  # Ensure valid int32 range
        
        # Convert action to proper intervention specification
        from ..interventions.handlers import create_perfect_intervention
        
        intervention = create_perfect_intervention(
            targets=frozenset(action.keys()),
            values=dict(action)
        )
        
        # Sample outcome from SCM with intervention
        base_outcome = sample_with_intervention(
            self.scm, intervention, n_samples=1, seed=seed
        )[0]
        
        # Add environment noise
        if self.config.noise_level > 0:
            target_var = self.jax_config.get_target_name()
            if target_var in base_outcome:
                key, noise_key = jax.random.split(key)
                noise = jax.random.normal(noise_key) * self.config.noise_level
                base_outcome = base_outcome.set(
                    target_var, 
                    base_outcome[target_var] + float(noise)
                )
        
        return base_outcome
    
    def _create_ground_truth_data(self) -> Optional[Dict[str, Any]]:
        """Create ground truth data for supervised reward components."""
        return {
            "scm": self.scm,
            "predictions": None,  # Would come from surrogate model in real training
            "difficulty": self.config.difficulty,
        }
    
    def _update_state_with_sample(
        self,
        state: JAXAcquisitionState,
        action: pyr.PMap,
        outcome: pyr.PMap
    ) -> JAXAcquisitionState:
        """Update state with new sample data."""
        from ..jax_native.sample_buffer import add_sample_jax
        
        # Convert action and outcome to tensor format
        intervention_mask = jnp.zeros(self.jax_config.n_vars, dtype=bool)
        values_tensor = jnp.zeros(self.jax_config.n_vars)
        
        # Extract values from outcome (it's a complex nested structure)
        outcome_values = outcome['values']  # This should contain the variable values
        
        # Fill tensors
        for i, var_name in enumerate(self.jax_config.variable_names):
            # Check if this variable was intervened on
            if var_name in action:
                intervention_mask = intervention_mask.at[i].set(True)
            
            # Get the actual observed value from outcome
            if var_name in outcome_values:
                values_tensor = values_tensor.at[i].set(float(outcome_values[var_name]))
        
        # Get target value
        target_value = float(outcome_values.get(self.jax_config.get_target_name(), 0.0))
        
        # Update buffer with new sample
        new_buffer = add_sample_jax(
            state.sample_buffer, 
            values_tensor, 
            intervention_mask, 
            target_value
        )
        
        # Update best value if this is better
        new_best_value = jnp.maximum(state.best_value, target_value)
        
        # Create new state with updated buffer and step count
        return JAXAcquisitionState(
            sample_buffer=new_buffer,
            mechanism_features=state.mechanism_features,
            marginal_probs=state.marginal_probs,
            confidence_scores=state.confidence_scores,
            best_value=new_best_value,
            current_step=state.current_step + 1,
            uncertainty_bits=state.uncertainty_bits,
            config=state.config
        )
    
    def _compute_environment_info(
        self,
        state: JAXAcquisitionState,
        reward_result: RewardResult
    ) -> EnvironmentInfo:
        """Compute environment information for this step."""
        intervention_count = state.current_step
        budget_remaining = max(0.0, self.config.intervention_budget - intervention_count)
        target_achieved = state.best_value >= self.config.target_threshold
        
        # Check for early stopping
        early_stopped = (
            self.config.enable_early_stopping and target_achieved
        )
        
        # Check if episode is complete
        episode_complete = (
            early_stopped or
            intervention_count >= self.config.max_interventions or
            budget_remaining <= 0
        )
        
        return EnvironmentInfo(
            intervention_count=intervention_count,
            budget_remaining=budget_remaining,
            target_achieved=target_achieved,
            early_stopped=early_stopped,
            episode_complete=episode_complete,
            best_value_so_far=state.best_value,
            metadata={
                "difficulty": self.config.difficulty,
                "total_reward": reward_result.total_reward,
                "reward_components": reward_result.component_rewards,
            }
        )
    
    def get_curriculum_metrics(self) -> Dict[str, float]:
        """Get metrics for curriculum learning advancement."""
        return {
            "difficulty": self.config.difficulty,
            "max_interventions": float(self.config.max_interventions),
            "target_threshold": self.config.target_threshold,
            "noise_level": self.config.noise_level,
        }


def create_intervention_environment(
    scm: pyr.PMap,
    rubric: CausalRewardRubric,
    difficulty: float = 0.5,
    max_interventions: int = 100,
    target_threshold: float = 2.0,
    **kwargs
) -> InterventionEnvironment:
    """Create an intervention environment with default configuration.
    
    Args:
        scm: Structural causal model
        rubric: Reward computation rubric
        difficulty: Environment difficulty (0.0-1.0)
        max_interventions: Maximum number of interventions
        target_threshold: Target value for success
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured InterventionEnvironment
    """
    # Create environment config
    env_config = EnvironmentConfig(
        difficulty=difficulty,
        max_interventions=max_interventions,
        target_threshold=target_threshold,
        **kwargs
    )
    
    # Extract variable information from SCM
    variables = tuple(sorted(scm.get("variables", set())))
    n_vars = len(variables)
    
    # Find target variable (assume last one if not specified)
    target_var = scm.get("target", variables[-1] if variables else "Y")
    target_idx = variables.index(target_var) if target_var in variables else 0
    
    # Create JAX config
    jax_config = JAXConfig(
        n_vars=n_vars,
        target_idx=target_idx,
        max_samples=max_interventions * 2,  # Buffer size
        max_history=max_interventions,
        variable_names=variables,
        mechanism_types=tuple(0 for _ in variables),  # Default to linear
        feature_dim=3
    )
    
    # Create state converter using existing bridge
    def state_converter(legacy_state):
        return convert_legacy_to_jax(
            legacy_state, 
            config=jax_config,
            max_samples=jax_config.max_samples,
            max_history=jax_config.max_history
        )
    
    return InterventionEnvironment(
        scm=scm,
        rubric=rubric,
        config=env_config,
        jax_config=jax_config,
        state_converter=state_converter
    )


def create_batch_environments(
    scms: List[pyr.PMap],
    rubric: CausalRewardRubric,
    difficulty_range: Tuple[float, float] = (0.3, 0.8),
    max_interventions: int = 100,
    **kwargs
) -> List[InterventionEnvironment]:
    """Create a batch of environments with varying difficulty.
    
    Args:
        scms: List of structural causal models
        rubric: Shared reward computation rubric
        difficulty_range: Range of difficulty levels to sample from
        max_interventions: Maximum interventions per environment
        **kwargs: Additional configuration parameters
        
    Returns:
        List of configured InterventionEnvironments
    """
    environments = []
    
    for i, scm in enumerate(scms):
        # Sample difficulty from range
        difficulty = difficulty_range[0] + (
            (difficulty_range[1] - difficulty_range[0]) * i / max(1, len(scms) - 1)
        )
        
        # Create environment
        env = create_intervention_environment(
            scm=scm,
            rubric=rubric,
            difficulty=difficulty,
            max_interventions=max_interventions,
            **kwargs
        )
        
        environments.append(env)
    
    return environments