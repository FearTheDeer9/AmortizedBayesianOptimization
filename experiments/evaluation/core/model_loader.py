"""
Simplified model loading wrapper that uses existing infrastructure.

This module provides a unified interface for loading trained models,
baselines, and creating untrained models for comparison.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Note: model_interfaces don't exist, implementing directly here
# from src.causal_bayes_opt.evaluation.model_interfaces import (
#     create_grpo_acquisition,
#     create_bc_acquisition,
#     create_random_acquisition,
#     create_optimal_oracle_acquisition
# )
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy
from src.causal_bayes_opt.policies.clean_bc_policy_factory import create_clean_bc_policy
from src.causal_bayes_opt.avici_integration.enhanced_surrogate import (
    create_enhanced_surrogate_for_grpo, EnhancedSurrogateFactory
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Unified interface for loading trained models and baselines."""
    
    @staticmethod
    def detect_checkpoint_architecture(checkpoint: Dict[str, Any]) -> Tuple[str, bool, float]:
        """
        Detect the actual architecture and settings from checkpoint.
        
        Returns:
            Tuple of (architecture_name, use_fixed_std, fixed_std_value)
        """
        architecture = checkpoint.get('architecture', {})
        params = checkpoint.get('params', {})
        
        # Get the architecture type from metadata
        arch_type = architecture.get('architecture_type', 'permutation_invariant')
        
        # IMPORTANT: Map old "permutation_invariant" to new standard
        if arch_type == 'permutation_invariant':
            arch_type = 'simple_permutation_invariant'
            logger.info("Mapping 'permutation_invariant' to 'simple_permutation_invariant' (new standard)")
        
        # DEFAULT: use_fixed_std=True (this is the standard in training)
        # Only set to False if we find evidence of learned std
        use_fixed_std = True  # Default to True
        fixed_std = 0.5  # Standard value
        
        # Check for val_mlp_output parameter to detect learned std
        # Parameters might be stored as nested dicts or flat with '/w' suffix
        val_output_key = None
        val_output_shape = None
        
        # First check for flat key with '/w'
        for key in params.keys():
            if 'val_mlp_output' in key and key.endswith('/w'):
                val_output_key = key
                val_output_shape = params[key].shape
                break
        
        # If not found, check for nested dict structure
        if val_output_key is None and 'val_mlp_output' in params:
            val_output_key = 'val_mlp_output'
            param = params[val_output_key]
            if isinstance(param, dict) and 'w' in param:
                val_output_shape = param['w'].shape
            elif hasattr(param, 'shape'):
                val_output_shape = param.shape
        
        if val_output_shape is not None:
            logger.info(f"Found val_mlp_output parameter: {val_output_key} with shape {val_output_shape}")
            
            # If output dimension is 2, then we have learned std (mean and log_std)
            if len(val_output_shape) >= 2 and val_output_shape[-1] == 2:
                use_fixed_std = False
                logger.info(f"Detected use_fixed_std=False (learned std, output shape: {val_output_shape})")
            else:
                # Shape is 1, confirming fixed std
                logger.info(f"Confirmed use_fixed_std=True (fixed std, output shape: {val_output_shape})")
        else:
            # No val_mlp_output found, keep default of True
            logger.info("No val_mlp_output found, defaulting to use_fixed_std=True")
        
        return arch_type, use_fixed_std, fixed_std
    
    @staticmethod
    def load_policy(checkpoint_path: Path, policy_type: str = 'auto', seed: int = 42) -> Callable:
        """
        Load policy model and return acquisition function.
        
        Args:
            checkpoint_path: Path to policy checkpoint
            policy_type: Type of policy ('auto', 'grpo', 'bc')
            seed: Random seed for stochastic policies
            
        Returns:
            Acquisition function that maps (tensor, posterior, target, variables) -> intervention
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Auto-detect type from checkpoint
        if policy_type == 'auto':
            checkpoint = load_checkpoint(checkpoint_path)
            policy_type = checkpoint.get('model_subtype', 'grpo')
            logger.info(f"Auto-detected policy type: {policy_type}")
        
        # Load the full checkpoint to detect architecture
        checkpoint = load_checkpoint(checkpoint_path)
        arch_type, use_fixed_std, fixed_std = ModelLoader.detect_checkpoint_architecture(checkpoint)
        
        if policy_type == 'grpo':
            logger.info(f"Loading GRPO policy from {checkpoint_path}")
            logger.info(f"  Architecture: {arch_type}, use_fixed_std: {use_fixed_std}")
            
            # Create the policy with detected settings
            from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy
            policy_fn = create_clean_grpo_policy(
                hidden_dim=checkpoint.get('architecture', {}).get('hidden_dim', 256),
                architecture=arch_type,
                use_fixed_std=use_fixed_std,
                fixed_std=fixed_std
            )
            
            # Transform and load params
            net = hk.without_apply_rng(hk.transform(policy_fn))
            params = checkpoint['params']
            
            # Create acquisition function
            return ModelLoader._create_acquisition_fn(net, params, seed, use_fixed_std, fixed_std)
            
        elif policy_type == 'bc':
            logger.info(f"Loading BC policy from {checkpoint_path}")
            # Similar for BC if needed
            raise NotImplementedError("BC loading not yet implemented")
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    @staticmethod
    def load_surrogate(checkpoint_path: Path) -> Tuple[Dict, Dict]:
        """
        Load surrogate model parameters and architecture.
        
        Args:
            checkpoint_path: Path to surrogate checkpoint
            
        Returns:
            Tuple of (params, architecture)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading surrogate from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Verify it's a surrogate model
        if checkpoint.get('model_type') != 'surrogate':
            raise ValueError(f"Expected surrogate model, got {checkpoint.get('model_type')}")
        
        return checkpoint['params'], checkpoint['architecture']
    
    @staticmethod
    def load_joint_models(checkpoint_dir: Path) -> Tuple[Callable, Tuple[Dict, Dict]]:
        """
        Load both policy and surrogate from a joint training checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing policy.pkl and surrogate.pkl
            
        Returns:
            Tuple of (policy_acquisition_fn, (surrogate_params, surrogate_architecture))
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        policy_path = checkpoint_dir / 'policy.pkl'
        surrogate_path = checkpoint_dir / 'surrogate.pkl'
        
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
        if not surrogate_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {surrogate_path}")
        
        policy_fn = ModelLoader.load_policy(policy_path)
        surrogate_data = ModelLoader.load_surrogate(surrogate_path)
        
        logger.info(f"Loaded joint models from {checkpoint_dir}")
        return policy_fn, surrogate_data
    
    @staticmethod
    def create_random_acquisition(seed: int = 42) -> Callable:
        """Create random baseline acquisition function."""
        rng_key = random.PRNGKey(seed)
        
        def random_acquisition(tensor: jnp.ndarray,
                              posterior: Optional[Dict[str, Any]],
                              target: str,
                              variables: list) -> Dict[str, Any]:
            nonlocal rng_key
            
            var_list = list(variables) if not isinstance(variables, list) else variables
            
            # Random variable selection
            rng_key, var_key = random.split(rng_key)
            selected_idx = random.choice(var_key, len(var_list))
            selected_var = var_list[int(selected_idx)]
            
            # Random value
            rng_key, val_key = random.split(rng_key)
            value = random.normal(val_key) * 2.0
            
            return {
                'targets': frozenset([selected_var]),
                'values': {selected_var: float(value)}
            }
        
        return random_acquisition
    
    @staticmethod
    def create_optimal_oracle_acquisition(scm: Any,
                                         optimization_direction: str = 'MINIMIZE') -> Callable:
        """
        Create optimal oracle acquisition that uses exact SCM coefficients.
        
        This oracle has perfect knowledge of the SCM structure AND coefficients.
        It selects the parent with maximum effect considering:
        - The coefficient value
        - The allowed intervention range
        - The optimization direction
        
        Args:
            scm: The full structural causal model with mechanisms
            optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
            
        Returns:
            Optimal oracle acquisition function
        """
        from src.causal_bayes_opt.data_structures.scm import get_parents, get_target, get_mechanisms
        
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var))
        mechanisms = get_mechanisms(scm)
        metadata = scm.get('metadata', {})
        variable_ranges = metadata.get('variable_ranges', {})
        coefficients_info = metadata.get('coefficients', {})
        
        def oracle_acquisition(tensor: jnp.ndarray,
                              posterior: Optional[Dict[str, Any]],
                              target: str,
                              variables: list) -> Dict[str, Any]:
            var_list = list(variables) if not isinstance(variables, list) else variables
            
            if not true_parents:
                # Target is a root node - no parents to intervene on
                # Return a dummy intervention
                non_target = [v for v in var_list if v != target][0]
                return {
                    'targets': frozenset([non_target]),
                    'values': {non_target: 0.0}
                }
            
            # Find parent with largest possible effect
            best_parent = None
            best_effect = 0.0
            best_value = 0.0
            
            for parent in true_parents:
                # Get coefficient from metadata
                edge = (parent, target)
                coeff = coefficients_info.get(edge, 0.0)
                
                if coeff == 0.0:
                    continue  # Skip if no coefficient info
                
                # Get intervention range for this parent
                parent_range = variable_ranges.get(parent, (-2.0, 2.0))
                min_val, max_val = parent_range
                
                # Determine optimal intervention value based on coefficient sign
                if optimization_direction == 'MINIMIZE':
                    # We want to minimize target (most negative contribution)
                    if coeff > 0:
                        # Positive coeff: use minimum value to get negative contribution
                        optimal_value = min_val
                        effect = abs(coeff * min_val)
                    else:
                        # Negative coeff: use maximum value to get negative contribution
                        optimal_value = max_val
                        effect = abs(coeff * max_val)
                else:  # MAXIMIZE
                    # We want to maximize target (most positive contribution)
                    if coeff > 0:
                        # Positive coeff: use maximum value to get positive contribution
                        optimal_value = max_val
                        effect = abs(coeff * max_val)
                    else:
                        # Negative coeff: use minimum value to get positive contribution
                        optimal_value = min_val
                        effect = abs(coeff * min_val)
                
                # Track parent with largest effect magnitude
                if effect > best_effect:
                    best_effect = effect
                    best_parent = parent
                    best_value = optimal_value
            
            # If we found a good parent, use it
            if best_parent is not None:
                return {
                    'targets': frozenset([best_parent]),
                    'values': {best_parent: float(best_value)}
                }
            
            # Fallback: use first parent with default range
            best_parent = true_parents[0]
            parent_range = variable_ranges.get(best_parent, (-2.0, 2.0))
            best_value = parent_range[0] if optimization_direction == 'MINIMIZE' else parent_range[1]
            
            return {
                'targets': frozenset([best_parent]),
                'values': {best_parent: float(best_value)}
            }
        
        return oracle_acquisition
    
    @staticmethod
    def load_baseline(baseline_type: str, **kwargs) -> Callable:
        """
        Load baseline acquisition function.
        
        Args:
            baseline_type: Type of baseline ('random', 'oracle', etc.)
            **kwargs: Additional arguments for the baseline
            
        Returns:
            Acquisition function
        """
        if baseline_type == 'random':
            logger.info(f"Creating random baseline with seed={kwargs.get('seed', 42)}")
            return ModelLoader.create_random_acquisition(kwargs.get('seed', 42))
        elif baseline_type == 'oracle':
            if 'scm' not in kwargs:
                raise ValueError("Oracle baseline requires 'scm' parameter")
            logger.info("Creating sophisticated oracle baseline with coefficient optimization")
            optimization_direction = kwargs.get('optimization_direction', 'MINIMIZE')
            return ModelLoader.create_optimal_oracle_acquisition(
                kwargs['scm'], 
                optimization_direction=optimization_direction
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")
    
    @staticmethod
    def _create_acquisition_fn(net, params, seed: int, use_fixed_std: bool, fixed_std: float) -> Callable:
        """Create acquisition function from network and params."""
        rng_key = random.PRNGKey(seed)
        
        def acquisition_fn(tensor: jnp.ndarray,
                          posterior: Optional[Dict[str, Any]],
                          target: str,
                          variables: list) -> Dict[str, Any]:
            nonlocal rng_key
            
            var_list = list(variables) if not isinstance(variables, list) else variables
            target_idx = var_list.index(target) if target in var_list else 0
            
            # Get policy output
            policy_output = net.apply(params, tensor, target_idx)
            
            # Extract outputs
            variable_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            # Sample variable
            rng_key, var_key = random.split(rng_key)
            var_probs = jax.nn.softmax(variable_logits)
            selected_idx = random.choice(var_key, len(var_list), p=var_probs)
            selected_var = var_list[int(selected_idx)]
            
            # Sample value
            rng_key, val_key = random.split(rng_key)
            if use_fixed_std:
                # Fixed std - value_params is [n_vars] for simple_permutation_invariant
                # or [n_vars, 1] for other architectures
                if value_params.ndim == 1:
                    # Shape is [n_vars] - direct indexing
                    value_mean = value_params[selected_idx]
                else:
                    # Shape is [n_vars, 1] - need to squeeze
                    value_mean = value_params[selected_idx, 0]
                value = value_mean + fixed_std * random.normal(val_key)
            else:
                # Learned std - value_params has mean and log_std [n_vars, 2]
                value_mean = value_params[selected_idx, 0]
                value_log_std = value_params[selected_idx, 1]
                value = value_mean + jnp.exp(value_log_std) * random.normal(val_key)
            
            # Ensure value is a scalar (handle potential shape issues)
            if hasattr(value, 'shape') and value.shape != ():
                value = value.item()  # Convert to Python scalar
            else:
                value = float(value)
            
            return {
                'targets': frozenset([selected_var]),
                'values': {selected_var: value}
            }
        
        return acquisition_fn
    
    @staticmethod
    def create_untrained_policy(architecture: str = 'permutation_invariant',
                               hidden_dim: int = 256,
                               policy_type: str = 'grpo',
                               seed: int = 42,
                               checkpoint_reference: Optional[Path] = None) -> Callable:
        """
        Create untrained policy for comparison.
        
        Args:
            architecture: Policy architecture
            hidden_dim: Hidden dimension size
            policy_type: Type of policy ('grpo' or 'bc')
            seed: Random seed for initialization
            checkpoint_reference: Optional checkpoint to match architecture settings
            
        Returns:
            Acquisition function with random initialization
        """
        # If we have a reference checkpoint, match its architecture
        use_fixed_std = True  # Default to match training
        fixed_std = 0.5
        
        if checkpoint_reference and checkpoint_reference.exists():
            try:
                ref_checkpoint = load_checkpoint(checkpoint_reference)
                architecture, use_fixed_std, fixed_std = ModelLoader.detect_checkpoint_architecture(ref_checkpoint)
                hidden_dim = ref_checkpoint.get('architecture', {}).get('hidden_dim', hidden_dim)
                logger.info(f"Matching architecture from reference: {architecture}")
            except Exception as e:
                logger.warning(f"Could not load reference checkpoint: {e}")
        
        # Map old architecture name to new standard
        if architecture == 'permutation_invariant':
            architecture = 'simple_permutation_invariant'
            logger.info("Using 'simple_permutation_invariant' (new standard)")
        
        logger.info(f"Creating untrained {policy_type} policy")
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Use fixed std: {use_fixed_std} (std={fixed_std})")
        
        # Create policy network with proper settings
        if policy_type == 'grpo':
            policy_fn = create_clean_grpo_policy(hidden_dim, architecture, use_fixed_std, fixed_std)
        elif policy_type == 'bc':
            policy_fn = create_clean_bc_policy(hidden_dim, architecture)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Transform and initialize with random params
        net = hk.without_apply_rng(hk.transform(policy_fn))
        
        # Create dummy input for initialization
        # Use 4 channels as that's what the network expects internally after normalization
        dummy_input = jnp.zeros((10, 5, 4))  # [T, n_vars, channels]
        init_key = random.PRNGKey(seed)
        params = net.init(init_key, dummy_input)
        
        # Create acquisition function with proper settings
        return ModelLoader._create_acquisition_fn(net, params, seed, use_fixed_std, fixed_std)
    
    @staticmethod
    def create_untrained_surrogate(variables: list,
                                  target_variable: str,
                                  model_complexity: str = "medium",
                                  use_continuous: bool = True,
                                  performance_mode: str = "balanced",
                                  seed: int = 42,
                                  checkpoint_reference: Optional[Path] = None) -> Tuple[Callable, Dict[str, Any]]:
        """
        Create untrained surrogate model for comparison.
        
        Args:
            variables: List of variable names
            target_variable: Target variable name
            model_complexity: "simple", "medium", or "full"
            use_continuous: Whether to use continuous parent set prediction
            performance_mode: "fast", "balanced", or "quality"
            seed: Random seed for initialization
            checkpoint_reference: Optional checkpoint to match architecture
            
        Returns:
            Tuple of (surrogate_function, configuration)
        """
        # If we have a reference checkpoint, try to match its settings
        if checkpoint_reference and checkpoint_reference.exists():
            try:
                ref_checkpoint = load_checkpoint(checkpoint_reference)
                if ref_checkpoint.get('model_type') == 'surrogate':
                    ref_arch = ref_checkpoint.get('architecture', {})
                    model_complexity = ref_arch.get('model_complexity', model_complexity)
                    use_continuous = ref_arch.get('use_continuous', use_continuous)
                    performance_mode = ref_arch.get('performance_mode', performance_mode)
                    logger.info(f"Matched surrogate architecture from reference: {model_complexity}")
            except Exception as e:
                logger.warning(f"Could not load reference surrogate checkpoint: {e}")
        
        logger.info(f"Creating untrained surrogate model")
        logger.info(f"  Variables: {len(variables)} ({target_variable} is target)")
        logger.info(f"  Complexity: {model_complexity}")
        logger.info(f"  Use continuous: {use_continuous}")
        logger.info(f"  Performance mode: {performance_mode}")
        
        # Create surrogate using enhanced factory
        surrogate_fn, config = create_enhanced_surrogate_for_grpo(
            variables=variables,
            target_variable=target_variable,
            model_complexity=model_complexity,
            use_continuous=use_continuous,
            performance_mode=performance_mode
        )
        
        return surrogate_fn, config
    
    @staticmethod
    def verify_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """
        Verify and return information about a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {'exists': False, 'path': str(checkpoint_path)}
        
        try:
            checkpoint = load_checkpoint(checkpoint_path)
            return {
                'exists': True,
                'path': str(checkpoint_path),
                'model_type': checkpoint.get('model_type'),
                'model_subtype': checkpoint.get('model_subtype'),
                'architecture': checkpoint.get('architecture'),
                'version': checkpoint.get('version'),
                'timestamp': checkpoint.get('timestamp')
            }
        except Exception as e:
            return {
                'exists': True,
                'path': str(checkpoint_path),
                'error': str(e)
            }