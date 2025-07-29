"""
GRPO Policy Loader for Two-Phase Training

This module provides utilities to load trained GRPO policies and convert them
to intervention functions suitable for Phase 2 active learning.

Key functions:
- load_grpo_policy(): Loads checkpoint and creates policy inference function
- create_grpo_intervention_fn(): Converts policy to intervention selection function
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import pyrsistent as pyr

from ..acquisition.enriched.policy_heads import EnrichedAcquisitionPolicyNetwork
from ..acquisition.enriched.state_enrichment import EnrichedHistoryBuilder
from ..data_structures.scm import get_variables, get_target
from ..data_structures.buffer import ExperienceBuffer
from ..interventions import create_perfect_intervention
from ..surrogate.bootstrap import create_bootstrap_surrogate_features
from ..surrogate.phase_manager import PhaseConfig, BootstrapConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedGRPOPolicy:
    """Container for loaded GRPO policy components."""
    policy_params: Any
    policy_config: Dict[str, Any]
    apply_fn: Callable
    variables: Optional[List[str]]  # None for variable-agnostic policies
    target_variable: Optional[str]  # None for variable-agnostic policies
    is_enriched: bool = True
    is_variable_agnostic: bool = False


def load_grpo_policy(checkpoint_path: str) -> LoadedGRPOPolicy:
    """
    Load GRPO policy from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or policy_params.pkl file
        
    Returns:
        LoadedGRPOPolicy containing policy components
        
    Raises:
        ValueError: If checkpoint is invalid or missing required data
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Handle both directory and direct file paths
    if checkpoint_path.is_dir():
        policy_file = checkpoint_path / "policy_params.pkl"
    else:
        policy_file = checkpoint_path
    
    if not policy_file.exists():
        raise ValueError(f"Policy file not found: {policy_file}")
    
    # Load policy data
    with open(policy_file, 'rb') as f:
        policy_data = pickle.load(f)
    
    # Validate checkpoint
    required_keys = ['policy_params', 'policy_config']
    for key in required_keys:
        if key not in policy_data:
            raise ValueError(f"Checkpoint missing required key: {key}")
    
    # Check if this is an enriched architecture
    if not policy_data.get('enriched_architecture', False):
        raise ValueError("This loader only supports enriched GRPO policies")
    
    # Extract policy configuration
    policy_config = policy_data['policy_config']
    policy_params = policy_data['policy_params']
    
    # Check if this is a variable-agnostic policy
    is_variable_agnostic = policy_config.get('architecture', {}).get('variable_agnostic', False)
    
    if is_variable_agnostic:
        # For variable-agnostic policies, we don't need specific variables/target
        # These will be provided when creating the intervention function
        variables = None
        target_variable = None
        logger.info("Loaded variable-agnostic GRPO policy")
    else:
        # Legacy: Extract variables from config for fixed-variable policies
        if 'variables' not in policy_config or 'target_variable' not in policy_config:
            raise ValueError("Policy config missing variables or target_variable")
        
        variables = policy_config['variables']
        target_variable = policy_config['target_variable']
        logger.info(f"Loaded fixed-variable GRPO policy for {len(variables)} variables")
    
    # Create Haiku transformed policy network
    # IMPORTANT: Parameter names must match training exactly for Haiku module compatibility
    def policy_fn(enriched_history: jnp.ndarray, 
                 target_variable_idx: int = 0,
                 is_training: bool = False) -> Dict[str, jnp.ndarray]:
        # Extract architecture config
        arch_config = policy_config.get('architecture', {})
        
        # EnrichedAcquisitionPolicyNetwork doesn't need n_vars - it infers from input shape
        # The network dynamically adapts to the number of variables in the input
        policy = EnrichedAcquisitionPolicyNetwork(
            num_layers=arch_config.get('num_layers', 4),
            num_heads=arch_config.get('num_heads', 8),
            hidden_dim=arch_config.get('hidden_dim', 128),
            key_size=arch_config.get('key_size', 32),
            widening_factor=arch_config.get('widening_factor', 4),
            policy_intermediate_dim=arch_config.get('policy_intermediate_dim', None),
            dropout=arch_config.get('dropout', 0.1),
            use_role_based_projection=arch_config.get('use_role_based_projection', True)
        )
        return policy(
            enriched_history=enriched_history,
            target_variable_idx=target_variable_idx,
            is_training=is_training
        )
    
    # Transform to pure function
    policy_net = hk.transform(policy_fn)
    
    logger.info(f"Loaded GRPO policy from {policy_file}")
    logger.info(f"Policy config: {policy_config}")
    if not is_variable_agnostic:
        logger.info(f"Variables: {variables}, Target: {target_variable}")
    
    return LoadedGRPOPolicy(
        policy_params=policy_params,
        policy_config=policy_config,
        apply_fn=policy_net.apply,
        variables=variables,
        target_variable=target_variable,
        is_enriched=True,
        is_variable_agnostic=is_variable_agnostic
    )


def create_grpo_intervention_fn(
    loaded_policy: LoadedGRPOPolicy,
    scm: pyr.PMap,
    phase_config: Optional[PhaseConfig] = None,
    bootstrap_config: Optional[BootstrapConfig] = None,
    intervention_range: tuple = (-2.0, 2.0)
) -> Callable[[Any], pyr.PMap]:
    """
    Create intervention function from loaded GRPO policy.
    
    This function creates a callable that uses the trained GRPO policy to
    select interventions, suitable for use in Phase 2 active learning.
    
    Args:
        loaded_policy: Loaded GRPO policy components
        scm: Structural causal model for bootstrap features
        phase_config: Phase configuration for bootstrap (uses defaults if None)
        bootstrap_config: Bootstrap configuration (uses defaults if None)
        intervention_range: Range for intervention values
        
    Returns:
        Intervention function with signature: fn(key) -> intervention
    """
    # Use default configs if not provided
    if phase_config is None:
        phase_config = PhaseConfig(bootstrap_steps=100)
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig()
    
    # Extract components
    if loaded_policy.is_variable_agnostic:
        # For variable-agnostic policies, get variables from the SCM
        variables = list(get_variables(scm))
        target_variable = get_target(scm)
    else:
        # Use the fixed variables from the policy
        variables = loaded_policy.variables
        target_variable = loaded_policy.target_variable
    
    target_idx = variables.index(target_variable)
    n_vars = len(variables)
    
    # Create a state buffer to track history
    state_buffer = ExperienceBuffer()
    step_counter = [0]  # Mutable counter in list
    
    def select_intervention(state: Optional[Any] = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """
        Select intervention using trained GRPO policy.
        
        Args:
            state: Optional state (for compatibility, but we track internally)
            key: JAX random key
            
        Returns:
            Intervention decision as PMap
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Split keys
        key1, key2 = random.split(key)
        
        # Get current step
        current_step = step_counter[0]
        
        # Create bootstrap surrogate features
        bootstrap_features = create_bootstrap_surrogate_features(
            scm=scm,
            step=current_step,
            config=phase_config,
            bootstrap_config=bootstrap_config,
            rng_key=key1
        )
        
        # Create enriched history using bootstrap features
        # Build enriched history directly using the builder
        
        # Create minimal state-like object with required attributes
        class MinimalState:
            def __init__(self, buffer, current_target):
                self.buffer = buffer
                self.current_target = current_target
        
        state_for_enrichment = MinimalState(
            buffer=state_buffer,
            current_target=target_variable
        )
        
        # Create builder and build history
        builder = EnrichedHistoryBuilder(
            standardize_values=True,
            include_temporal_features=True,
            max_history_size=100,
            support_variable_scms=True
        )
        
        # Build enriched history (returns tuple of history and mask)
        enriched_history, variable_mask = builder.build_enriched_history(state_for_enrichment)
        
        # Ensure history has correct shape [T, n_vars, 5]
        if enriched_history.ndim == 2:
            enriched_history = enriched_history[jnp.newaxis, :, :]  # Add time dimension
        
        # Apply policy network (inference mode)
        policy_output = loaded_policy.apply_fn(
            loaded_policy.policy_params,
            key2,
            enriched_history,
            target_idx,
            False  # is_training=False for inference
        )
        
        # Extract intervention decision
        intervention_logits = policy_output['variable_logits']  # [n_vars]
        
        # Mask out target variable
        mask = jnp.ones(n_vars, dtype=bool).at[target_idx].set(False)
        masked_logits = jnp.where(mask, intervention_logits, -jnp.inf)
        
        # Sample intervention variable
        intervention_probs = jax.nn.softmax(masked_logits)
        var_idx = random.choice(key2, n_vars, p=intervention_probs)
        
        # Skip if target was somehow selected
        if var_idx == target_idx:
            # Fallback to random non-target variable
            non_target_indices = [i for i in range(n_vars) if i != target_idx]
            var_idx = random.choice(key2, jnp.array(non_target_indices))
        
        selected_var = variables[int(var_idx)]
        
        # For value, use policy's value prediction if available
        if 'value_params' in policy_output:
            # Use value network prediction
            # value_params has shape [n_vars, 2] where 2 = [mean, log_std]
            value_params = policy_output['value_params'][int(var_idx)]  # [mean, log_std]
            predicted_value = float(value_params[0])  # Extract just the mean
            # Clip to intervention range
            intervention_value = jnp.clip(predicted_value, intervention_range[0], intervention_range[1])
        else:
            # Fallback to sampling from range
            value_key = random.fold_in(key2, 1)
            intervention_value = random.uniform(
                value_key, 
                minval=intervention_range[0], 
                maxval=intervention_range[1]
            )
        
        # Increment step counter
        step_counter[0] += 1
        
        # Create intervention
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        logger.debug(f"GRPO selected intervention: {selected_var} = {intervention_value:.3f}")
        
        return intervention
    
    # Add method to update buffer (for tracking history)
    def update_buffer(intervention: pyr.PMap, outcome: pyr.PMap):
        """Update internal buffer with intervention outcome."""
        state_buffer.add_intervention(intervention, outcome)
    
    # Attach update method as attribute
    select_intervention.update_buffer = update_buffer
    
    return select_intervention


def create_grpo_intervention_policy(
    checkpoint_path: str,
    scm: pyr.PMap,
    intervention_range: tuple = (-2.0, 2.0)
) -> Callable:
    """
    Convenience function to create GRPO intervention policy from checkpoint.
    
    Args:
        checkpoint_path: Path to GRPO checkpoint
        scm: Structural causal model
        intervention_range: Range for intervention values
        
    Returns:
        Intervention policy function
    """
    # Load policy
    loaded_policy = load_grpo_policy(checkpoint_path)
    
    # Verify SCM compatibility
    if not loaded_policy.is_variable_agnostic:
        # Only verify for fixed-variable policies
        scm_variables = list(get_variables(scm))
        scm_target = get_target(scm)
        
        if set(scm_variables) != set(loaded_policy.variables):
            logger.warning(f"SCM variables {scm_variables} != policy variables {loaded_policy.variables}")
        
        if scm_target != loaded_policy.target_variable:
            raise ValueError(f"SCM target {scm_target} != policy target {loaded_policy.target_variable}")
    else:
        # Variable-agnostic policies work with any SCM
        logger.debug("Using variable-agnostic policy - no SCM compatibility check needed")
    
    # Create intervention function
    return create_grpo_intervention_fn(
        loaded_policy=loaded_policy,
        scm=scm,
        intervention_range=intervention_range
    )