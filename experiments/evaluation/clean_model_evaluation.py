#!/usr/bin/env python3
"""
Clean Model Evaluation Script

Direct evaluation of trained GRPO models without complex wrappers.
Uses the exact same interface as training to ensure compatibility.
"""

import argparse
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.training.five_channel_converter import create_uniform_posterior
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy, create_quantile_policy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_policy_checkpoint(checkpoint_path: Path) -> Tuple[Any, Dict[str, Any], str]:
    """
    Load policy checkpoint and return params, config, and architecture type.
    
    Returns:
        Tuple of (params, config, architecture_type)
    """
    logger.info(f"\n📦 Loading policy checkpoint: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    
    # Extract architecture info
    architecture = checkpoint.get('architecture', {})
    arch_type = architecture.get('architecture_type', 
                                architecture.get('policy_architecture', 'quantile'))
    hidden_dim = architecture.get('hidden_dim', 256)
    
    # Detect if fixed std is used (for non-quantile policies)
    use_fixed_std = True  # Default
    fixed_std = 0.5
    
    logger.info(f"  Architecture: {arch_type}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Params keys: {list(params.keys())[:5]}...")
    
    config = {
        'architecture': arch_type,
        'hidden_dim': hidden_dim,
        'use_fixed_std': use_fixed_std,
        'fixed_std': fixed_std
    }
    
    return params, config, arch_type


def load_surrogate_checkpoint(checkpoint_path: Path) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Load surrogate checkpoint and return params, config, and network.
    
    Returns:
        Tuple of (params, config, network)
    """
    logger.info(f"\n🔮 Loading surrogate checkpoint: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    architecture = checkpoint.get('architecture', {})
    
    logger.info(f"  Model type: {checkpoint.get('model_type')}")
    logger.info(f"  Architecture keys: {list(architecture.keys())}")
    
    # Create the surrogate network following train_avici_style.py pattern
    from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
    
    # Extract architecture parameters
    hidden_dim = architecture.get('hidden_dim', 128)
    num_layers = architecture.get('num_layers', 8)
    num_heads = architecture.get('num_heads', 8)
    key_size = architecture.get('key_size', 32)
    
    # Create model function
    def surrogate_model_fn(data, target_idx, variables=None, training: bool = False):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=key_size,
            dropout=0.0,  # No dropout during evaluation
            use_weighted_loss=False
        )
        return model(data, variables, target_idx, training)
    
    # Transform with Haiku
    surrogate_net = hk.transform(surrogate_model_fn)
    
    logger.info(f"  Created surrogate network: hidden_dim={hidden_dim}, layers={num_layers}")
    
    return params, architecture, surrogate_net


def create_policy_function(config: Dict[str, Any], arch_type: str) -> Any:
    """
    Create the policy function based on architecture type.
    """
    if arch_type == 'quantile':
        policy_fn = create_quantile_policy(
            hidden_dim=config['hidden_dim']
        )
    else:
        policy_fn = create_clean_grpo_policy(
            hidden_dim=config['hidden_dim'],
            architecture=arch_type,
            use_fixed_std=config.get('use_fixed_std', True),
            fixed_std=config.get('fixed_std', 0.5)
        )
    
    # Transform with Haiku
    return hk.without_apply_rng(hk.transform(policy_fn))


def run_single_intervention(
    buffer: ExperienceBuffer,
    scm: Any,
    policy_net: Any,
    policy_params: Any,
    policy_config: Dict[str, Any],
    rng_key: Any,
    surrogate_net: Optional[Any] = None,
    surrogate_params: Optional[Any] = None,
    intervention_idx: int = 0
) -> Dict[str, Any]:
    """
    Run a single intervention using the policy.
    
    This mirrors the exact interface used during training.
    """
    # Get SCM info
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    
    # Initialize posterior to None (will be set if surrogate is used)
    posterior = None
    
    # Convert buffer to tensor (exactly as in training)
    tensor_3ch, mapper = buffer_to_three_channel_tensor(
        buffer, target_var, max_history_size=100, standardize=True
    )
    
    # Get surrogate predictions and create 4-channel tensor
    tensor = tensor_3ch  # Default to 3-channel
    parent_probs_for_channel = None
    
    if surrogate_net is not None and surrogate_params is not None:
        rng_key, surrogate_key = random.split(rng_key)
        try:
            # Call surrogate with exact training interface
            surrogate_output = surrogate_net.apply(
                surrogate_params, surrogate_key, tensor_3ch, mapper.target_idx, mapper.variables, False
            )
            
            if 'parent_probabilities' in surrogate_output:
                parent_probs = surrogate_output['parent_probabilities']
                
                # Create 4-channel tensor with parent probabilities
                tensor_4ch = jnp.zeros((tensor_3ch.shape[0], tensor_3ch.shape[1], 4))
                tensor_4ch = tensor_4ch.at[:, :, :3].set(tensor_3ch)
                
                # Add parent probabilities as 4th channel
                # parent_probs should be [n_vars] array
                for t in range(tensor_3ch.shape[0]):
                    tensor_4ch = tensor_4ch.at[t, :, 3].set(parent_probs)
                
                tensor = tensor_4ch
                parent_probs_for_channel = parent_probs
                
                # Calculate F1 score for surrogate predictions
                predicted_parents = set()
                for var_idx, prob in enumerate(parent_probs):
                    var_name = mapper.variables[var_idx]
                    if prob > 0.5 and var_name != target_var:
                        predicted_parents.add(var_name)
                
                tp = len(true_parents & predicted_parents)
                fp = len(predicted_parents - true_parents)
                fn = len(true_parents - predicted_parents)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                logger.info(f"\n🔮 Surrogate Predictions (Intervention {intervention_idx+1}):")
                logger.info(f"  F1 Score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")
                logger.info(f"  True parents: {true_parents}")
                logger.info(f"  Predicted parents: {predicted_parents}")
                logger.info(f"  Parent probabilities:")
                for var_idx, var_name in enumerate(mapper.variables):
                    is_parent = var_name in true_parents
                    prob = float(parent_probs[var_idx])
                    logger.info(f"    {var_name}: {prob:.3f} {'✓' if is_parent else ''}")
                
        except Exception as e:
            logger.warning(f"Surrogate prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Debug: Check what's in the 4th channel (parent probabilities)
    logger.info("\n📊 Tensor Channel Analysis:")
    logger.info(f"  Tensor shape: {tensor.shape}")
    logger.info(f"  Number of channels: {tensor.shape[2]}")
    
    # Show the last timestep's values for each channel
    last_timestep = tensor[-1]  # [n_vars, n_channels]
    logger.info(f"\n  Last timestep channel values:")
    for var_idx, var_name in enumerate(mapper.variables):
        is_parent = var_name in get_parents(scm, target_var)
        is_target = var_name == target_var
        if tensor.shape[2] >= 3:
            logger.info(f"    {var_name}: value={last_timestep[var_idx, 0]:.3f}, "
                       f"intervened={last_timestep[var_idx, 1]:.0f}, "
                       f"is_target={last_timestep[var_idx, 2]:.0f}, "
                       f"is_parent={is_parent}")
        if tensor.shape[2] >= 4:
            logger.info(f"      Channel 4 (parent prob?): {last_timestep[var_idx, 3]:.3f}")
    
    # Get policy action (exactly as in training)
    rng_key, policy_key = random.split(rng_key)
    policy_output = policy_net.apply(
        policy_params, tensor, mapper.target_idx
    )
    
    # Handle quantile vs regular policy output
    if 'quantile_scores' in policy_output:
        # Quantile policy
        quantile_scores = policy_output['quantile_scores']
        
        # Debug: Show all quantile scores
        logger.info("\n🎯 Quantile Policy Scores:")
        logger.info(f"  Raw scores shape: {quantile_scores.shape}")  # Should be [n_vars, 3]
        
        for var_idx, var_name in enumerate(mapper.variables):
            is_parent = var_name in get_parents(scm, target_var)
            scores_for_var = quantile_scores[var_idx]
            logger.info(f"  {var_name} (parent={is_parent}): "
                       f"Q25={scores_for_var[0]:.3f}, "
                       f"Q50={scores_for_var[1]:.3f}, "
                       f"Q75={scores_for_var[2]:.3f}")
        
        # Sample from quantile distribution
        flat_scores = quantile_scores.flatten()
        rng_key, sample_key = random.split(rng_key)
        probs = jax.nn.softmax(flat_scores)
        
        # Debug: Show top probabilities
        top_k = 5
        top_indices = jnp.argsort(probs)[-top_k:][::-1]
        logger.info("\n  Top 5 selection probabilities:")
        for idx in top_indices:
            var_idx = int(idx // 3)
            quantile_idx = int(idx % 3)
            var_name = mapper.variables[var_idx]
            quantile_name = ['Q25', 'Q50', 'Q75'][quantile_idx]
            is_parent = var_name in get_parents(scm, target_var)
            logger.info(f"    {var_name}_{quantile_name} (parent={is_parent}): {probs[idx]:.4f}")
        
        selected_flat_idx = random.choice(sample_key, len(flat_scores), p=probs)
        
        # Convert to variable and quantile
        n_vars = len(variables)
        selected_var_idx = int(selected_flat_idx // 3)
        selected_quantile_idx = int(selected_flat_idx % 3)
        
        # Debug: What was actually selected
        selected_var_name = mapper.variables[selected_var_idx]
        quantile_name = ['Q25', 'Q50', 'Q75'][selected_quantile_idx]
        is_parent = selected_var_name in get_parents(scm, target_var)
        logger.info(f"\n  ✓ Selected: {selected_var_name}_{quantile_name} (parent={is_parent})")
        
        # Map quantile to value
        quantile_values = {0: -1.0, 1: 0.0, 2: 1.0}  # Simple mapping
        intervention_value = quantile_values[selected_quantile_idx]
        
        # Add noise for exploration
        rng_key, noise_key = random.split(rng_key)
        intervention_value += random.normal(noise_key) * 0.2
        
    else:
        # Regular policy with variable and value heads
        var_logits = policy_output['variable_logits']
        value_params = policy_output.get('value_params')
        
        # Sample variable
        rng_key, var_key = random.split(rng_key)
        var_probs = jax.nn.softmax(var_logits)
        selected_var_idx = int(random.choice(var_key, len(variables), p=var_probs))
        
        # Sample value
        if value_params is not None:
            rng_key, val_key = random.split(rng_key)
            if value_params.ndim == 1:
                value_mean = value_params[selected_var_idx]
            else:
                value_mean = value_params[selected_var_idx, 0]
            intervention_value = float(value_mean + random.normal(val_key) * policy_config.get('fixed_std', 0.5))
        else:
            # Fallback to random value
            rng_key, val_key = random.split(rng_key)
            intervention_value = float(random.normal(val_key))
    
    # Get selected variable name
    selected_var = mapper.variables[selected_var_idx]
    
    # Debug: print what was selected
    true_parents = set(get_parents(scm, target_var))
    is_parent = selected_var in true_parents
    logger.debug(f"  Selected: {selected_var} (idx={selected_var_idx}, is_parent={is_parent}), value={intervention_value:.2f}")
    
    # Create and execute intervention
    intervention = create_perfect_intervention(
        targets=frozenset([selected_var]),
        values={selected_var: intervention_value}
    )
    
    # Sample outcome
    samples = sample_with_intervention(scm, intervention, n_samples=10, seed=42)
    outcome = samples[0]
    
    # Get target value
    target_value = get_values(outcome).get(target_var, 0.0)
    
    return {
        'intervention': intervention,
        'outcome': outcome,
        'target_value': target_value,
        'selected_var': selected_var,
        'selected_var_idx': selected_var_idx,
        'intervention_value': intervention_value,
        'posterior': posterior
    }


def evaluate_on_scm(
    scm: Any,
    scm_name: str,
    policy_net: Any,
    policy_params: Any,
    policy_config: Dict[str, Any],
    n_episodes: int = 5,
    n_interventions: int = 10,
    n_obs: int = 20,
    surrogate_net: Optional[Any] = None,
    surrogate_params: Optional[Any] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate policy on a single SCM over multiple episodes.
    """
    # Get SCM info
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on: {scm_name}")
    logger.info(f"  Variables: {len(variables)}")
    logger.info(f"  Target: {target_var}")
    logger.info(f"  True parents: {true_parents if true_parents else 'None (root)'}")
    
    episode_results = []
    
    for episode in range(n_episodes):
        # Initialize RNG for episode
        rng_key = random.PRNGKey(seed + episode)
        
        # Create buffer with observations
        buffer = ExperienceBuffer()
        obs_samples = sample_from_linear_scm(scm, n_obs, seed=seed + episode * 100)
        
        # Add observations with uniform posterior
        uniform_posterior = create_uniform_posterior(variables, target_var)
        for sample in obs_samples:
            buffer.add_observation(sample, posterior=uniform_posterior)
        
        # Track intervention results
        intervention_values = []
        
        # Run interventions
        for i in range(n_interventions):
            rng_key, intervention_key = random.split(rng_key)
            
            result = run_single_intervention(
                buffer, scm, policy_net, policy_params, policy_config,
                intervention_key, surrogate_net, surrogate_params,
                intervention_idx=i
            )
            
            # Add to buffer
            buffer.add_intervention(
                result['intervention'],
                result['outcome'],
                posterior=result.get('posterior')
            )
            
            # Track target value
            intervention_values.append(result['target_value'])
        
        # Episode metrics
        episode_metrics = {
            'episode': episode,
            'initial_value': intervention_values[0] if intervention_values else 0.0,
            'final_value': intervention_values[-1] if intervention_values else 0.0,
            'best_value': min(intervention_values) if intervention_values else 0.0,
            'mean_value': np.mean(intervention_values) if intervention_values else 0.0,
            'improvement': intervention_values[0] - intervention_values[-1] if len(intervention_values) >= 2 else 0.0
        }
        
        episode_results.append(episode_metrics)
        
        logger.info(f"  Episode {episode+1}: "
                   f"best={episode_metrics['best_value']:.3f}, "
                   f"final={episode_metrics['final_value']:.3f}, "
                   f"improvement={episode_metrics['improvement']:+.3f}")
    
    # Aggregate metrics
    aggregated = {
        'scm_name': scm_name,
        'n_variables': len(variables),
        'n_parents': len(true_parents),
        'mean_best': np.mean([e['best_value'] for e in episode_results]),
        'mean_final': np.mean([e['final_value'] for e in episode_results]),
        'mean_improvement': np.mean([e['improvement'] for e in episode_results]),
        'episodes': episode_results
    }
    
    return aggregated


def create_test_scms(seed: int = 42) -> List[Tuple[str, Any]]:
    """
    Create a set of test SCMs including training types and novel types.
    """
    factory = VariableSCMFactory(seed=seed, noise_scale=0.1)
    
    scms = []
    
    # Training types (fork and chain)
    scms.append(("fork_5vars", factory.create_variable_scm(
        num_variables=5, structure_type='fork'
    )))
    scms.append(("chain_5vars", factory.create_variable_scm(
        num_variables=5, structure_type='chain'
    )))
    scms.append(("fork_8vars", factory.create_variable_scm(
        num_variables=8, structure_type='fork'
    )))
    scms.append(("chain_8vars", factory.create_variable_scm(
        num_variables=8, structure_type='chain'
    )))
    
    # Novel types
    scms.append(("random_5vars", factory.create_variable_scm(
        num_variables=5, structure_type='random', edge_density=0.4
    )))
    scms.append(("collider_5vars", factory.create_variable_scm(
        num_variables=5, structure_type='collider'
    )))
    
    return scms


def create_random_baseline(seed: int = 42) -> Tuple[Any, Any]:
    """
    Create a random policy for baseline comparison.
    """
    class RandomPolicy:
        def apply(self, params, tensor, target_idx):
            n_vars = tensor.shape[1]
            key = random.PRNGKey(seed + int(jnp.sum(tensor)))
            
            # Random variable logits
            var_key, val_key = random.split(key)
            var_logits = random.normal(var_key, (n_vars,))
            
            # Mask target
            var_logits = var_logits.at[target_idx].set(-1e10)
            
            # Random value params
            value_params = random.normal(val_key, (n_vars, 2))
            
            return {
                'variable_logits': var_logits,
                'value_params': value_params
            }
    
    # Return instance with apply method
    return RandomPolicy(), {}


def main():
    parser = argparse.ArgumentParser(description="Clean Model Evaluation")
    parser.add_argument('--policy-path', type=Path, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--surrogate-path', type=Path, default=None,
                       help='Path to surrogate checkpoint (optional)')
    parser.add_argument('--n-episodes', type=int, default=5,
                       help='Number of episodes per SCM')
    parser.add_argument('--n-interventions', type=int, default=10,
                       help='Number of interventions per episode')
    parser.add_argument('--n-obs', type=int, default=20,
                       help='Number of initial observations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test-random', action='store_true',
                       help='Also test random baseline')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("CLEAN MODEL EVALUATION")
    logger.info("="*70)
    
    # Load policy checkpoint
    if not args.policy_path.exists():
        logger.error(f"Policy checkpoint not found: {args.policy_path}")
        return 1
    
    policy_params, policy_config, arch_type = load_policy_checkpoint(args.policy_path)
    
    # Create policy network
    policy_net = create_policy_function(policy_config, arch_type)
    
    # Load surrogate if provided
    surrogate_net = None
    surrogate_params = None
    if args.surrogate_path and args.surrogate_path.exists():
        surrogate_params, surrogate_config, surrogate_net = load_surrogate_checkpoint(args.surrogate_path)
        logger.info(f"✅ Surrogate loaded successfully")
    
    # Create test SCMs
    logger.info("\n📊 Creating test SCMs...")
    test_scms = create_test_scms(args.seed)
    
    # Evaluate on each SCM
    all_results = []
    
    for scm_name, scm in test_scms:
        results = evaluate_on_scm(
            scm, scm_name,
            policy_net, policy_params, policy_config,
            n_episodes=args.n_episodes,
            n_interventions=args.n_interventions,
            n_obs=args.n_obs,
            surrogate_net=surrogate_net,
            surrogate_params=surrogate_params,
            seed=args.seed
        )
        all_results.append(results)
    
    # Test random baseline if requested
    if args.test_random:
        logger.info("\n" + "="*70)
        logger.info("RANDOM BASELINE")
        logger.info("="*70)
        
        random_net, random_params = create_random_baseline(args.seed)
        random_results = []
        
        for scm_name, scm in test_scms[:2]:  # Just test on first 2 SCMs
            results = evaluate_on_scm(
                scm, f"RANDOM_{scm_name}",
                random_net, random_params, {'fixed_std': 0.5},
                n_episodes=args.n_episodes,
                n_interventions=args.n_interventions,
                n_obs=args.n_obs,
                seed=args.seed + 1000
            )
            random_results.append(results)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    # Separate training vs novel types
    training_types = [r for r in all_results if 'fork' in r['scm_name'] or 'chain' in r['scm_name']]
    novel_types = [r for r in all_results if 'random' in r['scm_name'] or 'collider' in r['scm_name']]
    
    logger.info("\n📈 Training Types (fork/chain):")
    for result in training_types:
        logger.info(f"  {result['scm_name']:15s}: "
                   f"best={result['mean_best']:.3f}, "
                   f"improvement={result['mean_improvement']:+.3f}")
    
    logger.info("\n🆕 Novel Types (random/collider):")
    for result in novel_types:
        logger.info(f"  {result['scm_name']:15s}: "
                   f"best={result['mean_best']:.3f}, "
                   f"improvement={result['mean_improvement']:+.3f}")
    
    if args.test_random:
        logger.info("\n🎲 Random Baseline:")
        for result in random_results:
            logger.info(f"  {result['scm_name']:15s}: "
                       f"best={result['mean_best']:.3f}, "
                       f"improvement={result['mean_improvement']:+.3f}")
    
    # Overall performance
    overall_improvement = np.mean([r['mean_improvement'] for r in all_results])
    training_improvement = np.mean([r['mean_improvement'] for r in training_types])
    novel_improvement = np.mean([r['mean_improvement'] for r in novel_types])
    
    logger.info("\n📊 Overall Performance:")
    logger.info(f"  All SCMs:      improvement={overall_improvement:+.3f}")
    logger.info(f"  Training types: improvement={training_improvement:+.3f}")
    logger.info(f"  Novel types:    improvement={novel_improvement:+.3f}")
    
    if args.test_random:
        random_improvement = np.mean([r['mean_improvement'] for r in random_results])
        logger.info(f"  Random baseline: improvement={random_improvement:+.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"clean_eval_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'policy_path': str(args.policy_path),
                'surrogate_path': str(args.surrogate_path) if args.surrogate_path else None,
                'n_episodes': args.n_episodes,
                'n_interventions': args.n_interventions,
                'n_obs': args.n_obs,
                'seed': args.seed
            },
            'results': all_results,
            'random_results': random_results if args.test_random else None,
            'summary': {
                'overall_improvement': overall_improvement,
                'training_improvement': training_improvement,
                'novel_improvement': novel_improvement
            }
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    
    # Verdict
    logger.info("\n" + "="*70)
    if overall_improvement > 0.1:
        logger.info("✅ Model shows improvement over episodes")
    elif overall_improvement > -0.1:
        logger.info("➖ Model shows no clear improvement")
    else:
        logger.info("❌ Model performance degrades over episodes")
    
    if training_improvement > novel_improvement + 0.2:
        logger.info("✅ Model performs better on training types (expected)")
    elif abs(training_improvement - novel_improvement) < 0.1:
        logger.info("➖ Similar performance on training and novel types")
    else:
        logger.info("⚠️  Model performs worse on training types (unexpected)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())