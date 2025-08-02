"""
Enriched GRPO Policy Integration for ACBO Experiments

This module provides integration functions for loading and using trained
enriched GRPO policies within the ACBO experiment framework. It bridges
the gap between the enriched policy architecture and the existing
acquisition state representation.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import pyrsistent as pyr

from .enriched.state_enrichment import EnrichedHistoryBuilder, create_enriched_history_tensor
from .enriched.policy_heads import EnrichedAcquisitionPolicyNetwork
from ..data_structures.scm import get_variables, get_target

logger = logging.getLogger(__name__)


class EnrichedPolicyWrapper:
    """
    Wrapper for trained enriched GRPO policy that provides intervention recommendations.
    
    This class handles:
    - Loading trained policy checkpoints
    - Converting acquisition states to enriched representation
    - Generating intervention recommendations
    - Fallback behavior for edge cases
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 fallback_to_random: bool = True,
                 intervention_value_range: Tuple[float, float] = (-2.0, 2.0)):
        """
        Initialize enriched policy wrapper.
        
        Args:
            checkpoint_path: Path to trained policy checkpoint
            fallback_to_random: Whether to fallback to random if policy fails
            intervention_value_range: Range for intervention values
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback_to_random = fallback_to_random
        self.intervention_value_range = intervention_value_range
        
        # Load checkpoint and initialize policy
        self.checkpoint_data = self._load_checkpoint()
        self.policy_fn = self._create_policy_function()
        self.policy_params = self.checkpoint_data['policy_params']
        self.policy_config = self.checkpoint_data['policy_config']
        
        # Create enriched history builder with variable-agnostic support
        # Use 5 channels to match the trained model
        self.history_builder = EnrichedHistoryBuilder(
            standardize_values=True,
            include_temporal_features=True,
            max_history_size=100,
            support_variable_scms=True,  # Enable variable-agnostic processing
            num_channels=5  # Match trained model's 5-channel format
        )
        
        logger.info(f"Loaded enriched policy from {checkpoint_path}")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load policy checkpoint from file."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint_file = self.checkpoint_path / "checkpoint.pkl"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        # Validate checkpoint
        required_keys = ['policy_params', 'policy_config', 'enriched_architecture']
        for key in required_keys:
            if key not in checkpoint_data:
                raise ValueError(f"Invalid checkpoint: missing key '{key}'")
        
        if not checkpoint_data.get('enriched_architecture', False):
            warnings.warn(
                f"Checkpoint at {self.checkpoint_path} may not be from enriched architecture training. "
                "This may cause compatibility issues."
            )
        
        return checkpoint_data
    
    def _create_policy_function(self) -> Callable:
        """Recreate policy function from checkpoint configuration."""
        policy_config = self.checkpoint_data['policy_config']
        architecture_config = policy_config.get('architecture', {})
        
        # Store checkpoint variable count for dynamic handling
        self.checkpoint_num_variables = policy_config.get('num_variables', 3)
        
        def policy_fn(enriched_history: jnp.ndarray, 
                     target_variable_idx: int = 0,
                     is_training: bool = False) -> Dict[str, jnp.ndarray]:
            
            # Determine number of variables from enriched history shape
            current_num_variables = enriched_history.shape[1] if len(enriched_history.shape) >= 2 else self.checkpoint_num_variables
            
            # Handle variable count mismatch by reshaping enriched history
            processed_history = self._handle_variable_count_mismatch(
                enriched_history, current_num_variables, self.checkpoint_num_variables
            )
            
            # Use enriched policy network architecture
            network = EnrichedAcquisitionPolicyNetwork(
                num_layers=architecture_config.get('num_layers', 4),
                num_heads=architecture_config.get('num_heads', 8),
                hidden_dim=architecture_config.get('hidden_dim', 128),
                key_size=architecture_config.get('key_size', 32),
                widening_factor=architecture_config.get('widening_factor', 4),
                dropout=architecture_config.get('dropout', 0.1),
                policy_intermediate_dim=architecture_config.get('policy_intermediate_dim', None)
            )
            
            # Process enriched history through the variable-agnostic network
            outputs = network(
                enriched_history=processed_history,
                target_variable_idx=min(target_variable_idx, self.checkpoint_num_variables - 1),
                is_training=is_training
            )
            
            return outputs
        
        return hk.transform(policy_fn)
    
    def _handle_variable_count_mismatch(self, 
                                      enriched_history: jnp.ndarray,
                                      current_num_variables: int,
                                      checkpoint_num_variables: int) -> jnp.ndarray:
        """
        Handle mismatch between current SCM variable count and checkpoint training size.
        
        Args:
            enriched_history: Current enriched history tensor [time, variables, channels]
            current_num_variables: Number of variables in current SCM
            checkpoint_num_variables: Number of variables the checkpoint was trained on
            
        Returns:
            Processed enriched history tensor with correct variable dimension
        """
        if current_num_variables == checkpoint_num_variables:
            return enriched_history
        
        time_steps, _, num_channels = enriched_history.shape
        
        if current_num_variables < checkpoint_num_variables:
            # Pad with zeros for missing variables
            padding_size = checkpoint_num_variables - current_num_variables
            padding = jnp.zeros((time_steps, padding_size, num_channels))
            processed_history = jnp.concatenate([enriched_history, padding], axis=1)
            
            logger.debug(f"Padded enriched history from {current_num_variables} to {checkpoint_num_variables} variables")
            
        else:
            # Truncate to checkpoint size (take first N variables)
            processed_history = enriched_history[:, :checkpoint_num_variables, :]
            
            logger.warning(
                f"Truncated enriched history from {current_num_variables} to {checkpoint_num_variables} variables. "
                f"Some variable information will be lost."
            )
        
        return processed_history
    
    def get_intervention_recommendation(self, 
                                     state, 
                                     scm: pyr.PMap,
                                     key: jax.random.PRNGKey) -> pyr.PMap:
        """
        Get intervention recommendation from enriched policy.
        
        Args:
            state: AcquisitionState object
            scm: Structural causal model
            key: JAX random key
            
        Returns:
            Intervention recommendation as pyr.PMap
        """
        try:
            # Convert state to enriched representation
            enriched_history = self._convert_state_to_enriched(state)
            
            # Get target variable index
            variables = list(get_variables(scm))
            target = get_target(scm)
            target_idx = variables.index(target) if target in variables else 0
            
            # Get policy output
            policy_output = self.policy_fn.apply(
                self.policy_params, key, enriched_history, target_idx, False
            )
            
            # Convert policy output to intervention
            intervention = self._policy_output_to_intervention(
                policy_output, variables, target, scm
            )
            
            return intervention
            
        except Exception as e:
            logger.warning(f"Policy inference failed: {e}")
            if self.fallback_to_random:
                return self._create_random_intervention(scm, key)
            else:
                raise
    
    def _convert_state_to_enriched(self, state) -> jnp.ndarray:
        """Convert acquisition state to enriched representation."""
        try:
            # Use enriched history builder to create multi-channel input
            enriched_history, variable_mask = self.history_builder.build_enriched_history(state)
            
            # Validate enriched history
            if not self.history_builder.validate_enriched_history(enriched_history):
                raise ValueError("Invalid enriched history tensor")
            
            # For now, we only return the enriched history (variable mask is for future use)
            return enriched_history
            
        except Exception as e:
            logger.error(f"State conversion to enriched format failed: {e}")
            
            # Create fallback enriched history
            return self._create_fallback_enriched_history(state)
    
    def _create_fallback_enriched_history(self, state) -> jnp.ndarray:
        """Create fallback enriched history when conversion fails."""
        # Get basic state information
        try:
            variables = sorted(state.buffer.get_variable_coverage())
            n_vars = len(variables)
        except:
            # Fallback to checkpoint variable count if we can't determine from state
            n_vars = getattr(self, 'checkpoint_num_variables', 3)
        
        # Ensure we have at least the minimum required variables
        n_vars = max(n_vars, 3)
        
        # Create empty enriched history with correct shape
        max_history_size = 100
        # Use the same number of channels as the history builder
        num_channels = self.history_builder.num_channels
        fallback_history = jnp.zeros((max_history_size, n_vars, num_channels))
        
        # Fill with basic state information if available
        try:
            # Channel 0: Use best value as proxy for variable values
            best_value = getattr(state, 'best_value', 0.0)
            fallback_history = fallback_history.at[-1, :, 0].set(best_value)
            
            # Channel 1: No interventions in fallback
            # Channel 2: Target indicators - mark all as potential targets
            fallback_history = fallback_history.at[-1, :, 2].set(1.0)
            
            # Channel 3: Marginal parent probabilities (if available)
            if hasattr(state, 'marginal_parent_probs'):
                for i, var in enumerate(variables[:n_vars]):
                    if var in state.marginal_parent_probs:
                        prob = float(state.marginal_parent_probs[var])
                        fallback_history = fallback_history.at[-1, i, 3].set(prob)
            
            # Channel 4: Uncertainty
            if num_channels > 4:
                uncertainty = getattr(state, 'uncertainty_bits', 1.0)
                fallback_history = fallback_history.at[-1, :, 4].set(uncertainty)
            
        except Exception as e:
            logger.warning(f"Could not populate fallback history: {e}")
        
        logger.info(f"Created fallback enriched history with shape {fallback_history.shape}")
        return fallback_history
    
    def _policy_output_to_intervention(self, 
                                     policy_output: Dict[str, jnp.ndarray],
                                     variables: List[str],
                                     target: str,
                                     scm: pyr.PMap) -> pyr.PMap:
        """Convert policy output to intervention format."""
        # Get variable selection logits and value parameters from enriched policy
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # Handle variable count mismatch between policy output and current SCM
        current_num_variables = len(variables)
        policy_output_size = len(variable_logits)
        
        if current_num_variables != policy_output_size:
            if current_num_variables < policy_output_size:
                # Truncate policy output to match current variables
                variable_logits = variable_logits[:current_num_variables]
                value_params = value_params[:current_num_variables]
                logger.debug(f"Truncated policy output from {policy_output_size} to {current_num_variables} variables")
            else:
                # Pad policy output with zeros for missing outputs
                padding_size = current_num_variables - policy_output_size
                logit_padding = jnp.full(padding_size, -1e9)  # Large negative for non-selection
                value_padding = jnp.zeros((padding_size, 2))
                variable_logits = jnp.concatenate([variable_logits, logit_padding])
                value_params = jnp.concatenate([value_params, value_padding], axis=0)
                logger.debug(f"Padded policy output from {policy_output_size} to {current_num_variables} variables")
        
        # Convert variable selection probabilities and value parameters
        variable_probs = jax.nn.softmax(variable_logits)
        intervention_means = value_params[:, 0]
        intervention_log_stds = value_params[:, 1]
        
        # Select intervention variable based on probabilities (excluding target)
        target_idx = variables.index(target) if target in variables else -1
        
        # Mask target variable from selection
        masked_probs = jnp.where(
            jnp.arange(len(variables)) == target_idx,
            0.0,  # Zero probability for target variable
            variable_probs
        )
        
        # Renormalize probabilities after masking
        if jnp.sum(masked_probs) > 0:
            masked_probs = masked_probs / jnp.sum(masked_probs)
        
        # Select intervention targets and values
        intervention_targets = set()
        intervention_value_dict = {}
        
        # Use threshold-based selection for multiple interventions
        selection_threshold = 0.1  # Minimum probability to consider intervention
        
        for i, var in enumerate(variables):
            if var != target and masked_probs[i] > selection_threshold:
                # Use mean intervention value (could add noise sampling later)
                intervention_value = float(intervention_means[i])
                
                # Clip intervention values to reasonable range
                intervention_value = jnp.clip(
                    intervention_value, 
                    self.intervention_value_range[0], 
                    self.intervention_value_range[1]
                )
                
                intervention_targets.add(var)
                intervention_value_dict[var] = float(intervention_value)
        
        # Ensure at least one intervention if possible
        if not intervention_targets and len(variables) > 1:
            # Choose variable with highest probability (excluding target)
            non_target_indices = [i for i, var in enumerate(variables) if var != target]
            if non_target_indices:
                best_idx = non_target_indices[jnp.argmax(masked_probs[non_target_indices])]
                var = variables[best_idx]
                intervention_value = float(jnp.clip(
                    intervention_means[best_idx],
                    self.intervention_value_range[0], 
                    self.intervention_value_range[1]
                ))
                intervention_targets.add(var)
                intervention_value_dict[var] = intervention_value
        
        return pyr.m(
            type="perfect",
            targets=intervention_targets,
            values=intervention_value_dict
        )
    
    def _create_random_intervention(self, scm: pyr.PMap, key: jax.random.PRNGKey) -> pyr.PMap:
        """Create random intervention as fallback."""
        variables = list(get_variables(scm))
        target = get_target(scm)
        
        # Select random non-target variable
        non_target_vars = [var for var in variables if var != target]
        if not non_target_vars:
            # No valid intervention targets
            return pyr.m(type="perfect", targets=set(), values={})
        
        # Choose random variable and value
        key1, key2 = random.split(key)
        var_idx = random.randint(key1, (), 0, len(non_target_vars))
        intervention_var = non_target_vars[var_idx]
        
        intervention_value = random.uniform(
            key2, (), 
            minval=self.intervention_value_range[0],
            maxval=self.intervention_value_range[1]
        )
        
        return pyr.m(
            type="perfect",
            targets={intervention_var},
            values={intervention_var: float(intervention_value)}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'architecture': self.policy_config.get('architecture', {}),
            'training_config': self.checkpoint_data.get('training_config', {}),
            'episode': self.checkpoint_data.get('episode', -1),
            'is_final': self.checkpoint_data.get('is_final', False),
            'enriched_architecture': self.checkpoint_data.get('enriched_architecture', False),
            'checkpoint_num_variables': getattr(self, 'checkpoint_num_variables', 3),
            'supports_variable_scms': True,
            'fallback_to_random': self.fallback_to_random,
            'intervention_value_range': self.intervention_value_range
        }


def load_enriched_policy_for_acbo(checkpoint_path: str, 
                                 intervention_value_range: Tuple[float, float] = (-2.0, 2.0)) -> EnrichedPolicyWrapper:
    """
    Load trained enriched GRPO policy for ACBO experiments.
    
    Args:
        checkpoint_path: Path to policy checkpoint directory
        intervention_value_range: Range for intervention values
        
    Returns:
        EnrichedPolicyWrapper ready for use in ACBO experiments
    """
    return EnrichedPolicyWrapper(
        checkpoint_path=checkpoint_path,
        fallback_to_random=True,
        intervention_value_range=intervention_value_range
    )


def create_enriched_policy_intervention_function(checkpoint_path: str,
                                               intervention_value_range: Tuple[float, float] = (-2.0, 2.0)) -> Callable:
    """
    Create intervention function for ACBO experiments using enriched policy.
    
    Args:
        checkpoint_path: Path to policy checkpoint
        intervention_value_range: Range for intervention values
        
    Returns:
        Function that takes (state, scm, key) and returns intervention
    """
    policy_wrapper = load_enriched_policy_for_acbo(checkpoint_path, intervention_value_range)
    
    def intervention_fn(state, scm: pyr.PMap, key: jax.random.PRNGKey) -> pyr.PMap:
        """
        Generate intervention using enriched policy.
        
        Args:
            state: AcquisitionState
            scm: Structural causal model
            key: JAX random key
            
        Returns:
            Intervention recommendation
        """
        return policy_wrapper.get_intervention_recommendation(state, scm, key)
    
    return intervention_fn


def validate_enriched_policy_integration(checkpoint_path: str) -> bool:
    """
    Validate that enriched policy integration works correctly.
    
    Args:
        checkpoint_path: Path to policy checkpoint
        
    Returns:
        True if integration validation passes, False otherwise
    """
    try:
        # Load policy
        policy_wrapper = load_enriched_policy_for_acbo(checkpoint_path)
        
        # Create mock state and SCM for testing
        from unittest.mock import Mock
        
        # Mock SCM with variable size testing
        mock_scm = pyr.m(
            variables={'X', 'Y', 'Z', 'W'},  # Test with 4 variables
            edges={('X', 'Y'), ('Z', 'Y'), ('W', 'Z')},
            target='Y'
        )
        
        # Mock state
        mock_state = Mock()
        mock_buffer = Mock()
        mock_buffer.get_all_samples.return_value = []
        mock_buffer.get_variable_coverage.return_value = ['X', 'Y', 'Z', 'W']
        
        mock_state.buffer = mock_buffer
        mock_state.current_target = 'Y'
        mock_state.step = 5
        mock_state.best_value = 1.0
        mock_state.uncertainty_bits = 1.5
        mock_state.marginal_parent_probs = {'X': 0.8, 'Z': 0.6, 'W': 0.7}
        mock_state.mechanism_confidence = {'X': 0.7, 'Y': 0.8, 'Z': 0.6, 'W': 0.5}
        
        def mock_mechanism_insights():
            return {
                'predicted_effects': {'X': 1.2, 'Y': 0.0, 'Z': -0.8, 'W': 0.5},
                'mechanism_types': {'X': 'linear', 'Y': 'linear', 'Z': 'linear', 'W': 'linear'}
            }
        
        def mock_optimization_progress():
            return {'best_value': 1.0, 'steps_since_improvement': 2}
        
        mock_state.get_mechanism_insights = mock_mechanism_insights
        mock_state.get_optimization_progress = mock_optimization_progress
        
        # Test intervention generation
        key = random.PRNGKey(42)
        intervention = policy_wrapper.get_intervention_recommendation(mock_state, mock_scm, key)
        
        # Validate intervention format
        if not isinstance(intervention, pyr.PMap):
            logger.error("Intervention is not a pyrsistent Map")
            return False
        
        required_keys = ['type', 'targets', 'values']
        for req_key in required_keys:
            if req_key not in intervention:
                logger.error(f"Intervention missing required key: {req_key}")
                return False
        
        # Check intervention values are within range
        values = intervention.get('values', {})
        for var, value in values.items():
            if not isinstance(value, (int, float)):
                logger.error(f"Intervention value for {var} is not numeric: {value}")
                return False
            
            if not (-3.0 <= value <= 3.0):  # Allow some margin
                logger.error(f"Intervention value for {var} out of reasonable range: {value}")
                return False
        
        logger.info("Enriched policy integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Enriched policy integration validation failed: {e}")
        return False


# Convenience functions for common use cases
def create_demo_enriched_policy() -> Callable:
    """
    Create a demo enriched policy for testing (returns random interventions).
    
    Returns:
        Function that generates random interventions
    """
    def demo_intervention_fn(state, scm: pyr.PMap, key: jax.random.PRNGKey) -> pyr.PMap:
        variables = list(get_variables(scm))
        target = get_target(scm)
        
        non_target_vars = [var for var in variables if var != target]
        if not non_target_vars:
            return pyr.m(type="perfect", targets=set(), values={})
        
        # Random intervention
        key1, key2 = random.split(key)
        var_idx = random.randint(key1, (), 0, len(non_target_vars))
        intervention_var = non_target_vars[var_idx]
        
        intervention_value = random.uniform(key2, (), minval=-2.0, maxval=2.0)
        
        return pyr.m(
            type="perfect",
            targets={intervention_var},
            values={intervention_var: float(intervention_value)}
        )
    
    return demo_intervention_fn