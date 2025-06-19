#!/usr/bin/env python3
"""
Complete Acquisition Training Demo with Enhanced GRPO and Verifiable Rewards.

This demo shows the complete training pipeline for acquisition policies in ACBO:
1. Expert demonstration collection from PARENT_SCALE
2. Behavioral cloning warm-start phase
3. Enhanced GRPO fine-tuning with verifiable rewards
4. Performance evaluation and comparison

Key Features:
- End-to-end training pipeline validation
- 2024 GRPO enhancements (zero KL penalty, adaptive advantage scaling)
- Mathematically verifiable rewards (no human feedback)
- Comprehensive performance metrics and analysis
- Integration with existing ACBO infrastructure

Usage:
    python examples/complete_acquisition_training_demo.py
    
Or with custom configuration:
    python examples/complete_acquisition_training_demo.py --config production
"""

# Standard library imports
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party imports
import jax
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from src.causal_bayes_opt.training.acquisition_training import train_acquisition_model, TrainingResults
from src.causal_bayes_opt.training.acquisition_config import (
    TrainingConfig,
    create_standard_config,
    create_high_performance_config,
    create_memory_efficient_config,
    validate_config_compatibility,
    get_recommended_config_for_problem_size
)
from src.causal_bayes_opt.training.expert_collection.collector import (
    ExpertDemonstrationCollector,
    ParentScaleTrajectory
)
from src.causal_bayes_opt.acquisition.policy import AcquisitionPolicyNetwork, PolicyConfig
from src.causal_bayes_opt.experiments.test_scms import create_fork_scm, create_chain_scm
from src.causal_bayes_opt.data_structures.scm import get_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete acquisition training demo."""
    parser = argparse.ArgumentParser(description="Complete acquisition training demonstration")
    parser.add_argument(
        '--config', 
        choices=['standard', 'production', 'memory_efficient', 'debug'], 
        default='standard',
        help='Training configuration preset'
    )
    parser.add_argument(
        '--n_expert_trajectories', 
        type=int, 
        default=50,
        help='Number of expert trajectories to collect'
    )
    parser.add_argument(
        '--n_variables', 
        type=int, 
        default=5,
        help='Number of variables in test SCMs'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./training_output',
        help='Directory to save training results'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize random key
    key = jax.random.PRNGKey(args.seed)
    
    logger.info("Starting Complete Acquisition Training Demo")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Expert trajectories: {args.n_expert_trajectories}")
    logger.info(f"Problem size: {args.n_variables} variables")
    
    try:
        # Run complete demo
        results = run_complete_training_demo(
            config_type=args.config,
            n_expert_trajectories=args.n_expert_trajectories,
            n_variables=args.n_variables,
            output_dir=output_dir,
            key=key
        )
        
        # Save and display results
        save_training_results(results, output_dir)
        display_training_summary(results)
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


def run_complete_training_demo(
    config_type: str = 'standard',
    n_expert_trajectories: int = 50,
    n_variables: int = 5,
    output_dir: Optional[Path] = None,
    key: Optional[jax.Array] = None
) -> Dict[str, Any]:
    """
    Run complete acquisition training demonstration.
    
    Args:
        config_type: Training configuration preset
        n_expert_trajectories: Number of expert trajectories to collect
        n_variables: Number of variables in test SCMs
        output_dir: Directory to save results
        key: JAX random key
        
    Returns:
        Dictionary with complete training results and analysis
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    demo_start_time = time.time()
    
    # Step 1: Create training configuration
    logger.info("Step 1: Creating training configuration")
    config = create_training_configuration(config_type, n_variables, n_expert_trajectories)
    
    # Validate configuration
    warnings = validate_config_compatibility(config)
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    # Step 2: Generate test SCMs and collect expert demonstrations
    logger.info("Step 2: Collecting expert demonstrations")
    key, demo_key = jax.random.split(key)
    
    expert_trajectories, demo_metrics = collect_expert_demonstrations(
        n_trajectories=n_expert_trajectories,
        n_variables=n_variables,
        key=demo_key
    )
    
    # Step 3: Create surrogate model (placeholder for demo)
    logger.info("Step 3: Creating surrogate model")
    key, surrogate_key = jax.random.split(key)
    
    surrogate_model, surrogate_params = create_demo_surrogate_model(
        expert_trajectories, key=surrogate_key
    )
    
    # Step 4: Train acquisition model
    logger.info("Step 4: Training acquisition model")
    key, training_key = jax.random.split(key)
    
    training_results = train_acquisition_model(
        expert_trajectories=expert_trajectories,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        config=config,
        key=training_key
    )
    
    # Step 5: Evaluate trained model
    logger.info("Step 5: Evaluating trained model")
    key, eval_key = jax.random.split(key)
    
    evaluation_results = evaluate_trained_model(
        training_results,
        expert_trajectories,
        surrogate_model,
        surrogate_params,
        config,
        eval_key
    )
    
    demo_total_time = time.time() - demo_start_time
    
    # Compile complete results
    complete_results = {
        'config': config,
        'demo_metrics': demo_metrics,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'demo_total_time': demo_total_time,
        'expert_trajectory_count': len(expert_trajectories),
        'config_warnings': warnings
    }
    
    logger.info(f"Complete training demo finished in {demo_total_time:.2f} seconds")
    
    return complete_results


def create_training_configuration(
    config_type: str,
    n_variables: int,
    n_expert_trajectories: int
) -> TrainingConfig:
    """Create training configuration based on preset and problem characteristics."""
    
    if config_type == 'standard':
        config = create_standard_config()
    elif config_type == 'production':
        config = create_high_performance_config()
    elif config_type == 'memory_efficient':
        config = create_memory_efficient_config()
    elif config_type == 'debug':
        config = create_standard_config().get_debug_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Adjust based on problem characteristics
    recommended_config = get_recommended_config_for_problem_size(n_variables, n_expert_trajectories)
    
    # Use recommended settings for key parameters
    config.policy_config.hidden_dim = recommended_config.policy_config.hidden_dim
    config.policy_config.num_layers = recommended_config.policy_config.num_layers
    config.grpo_config.group_size = recommended_config.grpo_config.group_size
    
    # Adjust data config
    config.data_config.target_expert_trajectories = n_expert_trajectories
    config.data_config.min_expert_trajectories = min(20, n_expert_trajectories // 5)
    
    return config


def collect_expert_demonstrations(
    n_trajectories: int,
    n_variables: int,
    key: jax.Array
) -> tuple[List[ParentScaleTrajectory], Dict[str, Any]]:
    """Collect expert demonstrations from PARENT_SCALE."""
    
    logger.info(f"Collecting {n_trajectories} expert trajectories for {n_variables}-variable SCMs")
    
    # Create diverse test SCMs
    test_scms = create_diverse_test_scms(n_variables, n_trajectories // 2, key)
    
    # Initialize expert demonstration collector
    collector = ExpertDemonstrationCollector()
    
    # Collect demonstrations
    start_time = time.time()
    expert_trajectories = []
    
    for i, scm in enumerate(test_scms):
        trajectory_key = jax.random.fold_in(key, i)
        
        try:
            # Collect trajectory from this SCM
            trajectory = collector.collect_full_trajectory(
                scm=scm,
                n_observational=20,  # Start with some observational data
                n_interventional=10,  # Collect 10 interventions
                key=trajectory_key
            )
            
            expert_trajectories.append(trajectory)
            
            if len(expert_trajectories) >= n_trajectories:
                break
                
        except Exception as e:
            logger.warning(f"Failed to collect trajectory {i}: {e}")
            continue
    
    collection_time = time.time() - start_time
    
    metrics = {
        'collection_time': collection_time,
        'trajectories_collected': len(expert_trajectories),
        'trajectories_requested': n_trajectories,
        'success_rate': len(expert_trajectories) / n_trajectories,
        'avg_trajectory_length': jnp.mean(jnp.array([len(t.states) for t in expert_trajectories])),
        'scms_used': len(test_scms)
    }
    
    logger.info(f"Collected {len(expert_trajectories)} trajectories in {collection_time:.2f}s")
    
    return expert_trajectories, metrics


def create_diverse_test_scms(n_variables: int, n_scms: int, key: jax.Array) -> List[Any]:
    """Create diverse test SCMs for expert demonstration collection."""
    
    scms = []
    keys = jax.random.split(key, n_scms)
    
    for i, scm_key in enumerate(keys):
        seed = int(scm_key[0]) % 10000
        
        if i % 3 == 0:
            # Fork structure: X → Y ← Z (good for testing)
            scm = create_fork_scm(seed=seed)
        elif i % 3 == 1:
            # Chain structure: X → Y → Z
            scm = create_chain_scm(n_variables=n_variables, seed=seed)
        else:
            # More complex structure (if available)
            scm = create_fork_scm(seed=seed)  # Fallback to fork
        
        scms.append(scm)
    
    return scms


def create_demo_surrogate_model(
    expert_trajectories: List[ParentScaleTrajectory],
    key: jax.Array
) -> tuple[Any, Any]:
    """Create demo surrogate model for training (placeholder implementation)."""
    
    logger.info("Creating demo surrogate model")
    
    # For demo purposes, create a simple placeholder model
    # In real implementation, this would be a trained ParentSetPredictionModel
    
    class DemoSurrogateModel:
        """Demo surrogate model that returns reasonable posterior predictions."""
        
        def apply(self, params, data, target_idx):
            """Return demo posterior predictions."""
            n_samples, n_vars, _ = data.shape
            n_parent_sets = min(10, 2**n_vars)  # Reasonable number of parent sets
            
            # Return random but reasonable posteriors
            key = jax.random.PRNGKey(42)
            logits = jax.random.normal(key, (n_parent_sets,))
            return jax.nn.softmax(logits)
    
    surrogate_model = DemoSurrogateModel()
    surrogate_params = {}  # Empty params for demo
    
    return surrogate_model, surrogate_params


def evaluate_trained_model(
    training_results: TrainingResults,
    expert_trajectories: List[ParentScaleTrajectory],
    surrogate_model: Any,
    surrogate_params: Any,
    config: TrainingConfig,
    key: jax.Array
) -> Dict[str, Any]:
    """Comprehensive evaluation of the trained acquisition model."""
    
    logger.info("Evaluating trained acquisition model")
    
    # Create policy network for evaluation
    example_state = expert_trajectories[0].states[0]
    from src.causal_bayes_opt.acquisition.policy import create_acquisition_policy
    
    policy_network = create_acquisition_policy(
        PolicyConfig(
            hidden_dim=config.policy_config.hidden_dim,
            num_layers=config.policy_config.num_layers,
            num_heads=config.policy_config.num_heads
        ),
        example_state
    )
    
    evaluation_results = {}
    
    # 1. Expert Similarity Evaluation
    logger.info("  Computing expert similarity metrics")
    expert_similarity = evaluate_expert_similarity(
        training_results.final_params,
        policy_network,
        expert_trajectories[:5],  # Use first 5 for evaluation
        key
    )
    evaluation_results['expert_similarity'] = expert_similarity
    
    # 2. Reward Performance Evaluation
    logger.info("  Computing reward performance metrics")
    reward_performance = evaluate_reward_performance(
        training_results.final_params,
        policy_network,
        surrogate_model,
        surrogate_params,
        expert_trajectories[:3],  # Use first 3 SCMs
        config.reward_config,
        key
    )
    evaluation_results['reward_performance'] = reward_performance
    
    # 3. Intervention Diversity Analysis
    logger.info("  Analyzing intervention diversity")
    diversity_analysis = analyze_intervention_diversity(
        training_results.final_params,
        policy_network,
        expert_trajectories[0].states[:10],  # First 10 states from first trajectory
        key
    )
    evaluation_results['diversity_analysis'] = diversity_analysis
    
    # 4. Training Convergence Analysis
    logger.info("  Analyzing training convergence")
    convergence_analysis = analyze_training_convergence(training_results)
    evaluation_results['convergence_analysis'] = convergence_analysis
    
    return evaluation_results


def evaluate_expert_similarity(
    params: Any,
    policy_network: Any,
    expert_trajectories: List[ParentScaleTrajectory],
    key: jax.Array
) -> Dict[str, float]:
    """Evaluate how similar the trained policy is to expert behavior."""
    
    total_comparisons = 0
    matching_actions = 0
    log_prob_similarities = []
    
    for trajectory in expert_trajectories:
        for state, expert_action in zip(trajectory.states, trajectory.actions):
            # Get policy prediction
            policy_output = policy_network.apply(params, state, is_training=False)
            
            # Sample action from policy
            action_key = jax.random.fold_in(key, total_comparisons)
            from src.causal_bayes_opt.acquisition.policy import sample_intervention_from_policy
            
            predicted_action = sample_intervention_from_policy(
                policy_output, state, action_key, PolicyConfig()
            )
            
            # Compare actions (simplified comparison)
            if predicted_action.get('targets') == expert_action.get('targets'):
                matching_actions += 1
            
            # Compute log probability of expert action
            from src.causal_bayes_opt.acquisition.policy import compute_action_log_probability
            log_prob = compute_action_log_probability(
                policy_output, expert_action, state, PolicyConfig()
            )
            
            if jnp.isfinite(log_prob):
                log_prob_similarities.append(float(log_prob))
            
            total_comparisons += 1
    
    if total_comparisons == 0:
        return {'accuracy': 0.0, 'avg_log_prob': -float('inf'), 'n_comparisons': 0}
    
    accuracy = matching_actions / total_comparisons
    avg_log_prob = jnp.mean(jnp.array(log_prob_similarities)) if log_prob_similarities else -float('inf')
    
    return {
        'accuracy': accuracy,
        'avg_log_prob': float(avg_log_prob),
        'n_comparisons': total_comparisons,
        'log_prob_std': float(jnp.std(jnp.array(log_prob_similarities))) if log_prob_similarities else 0.0
    }


def evaluate_reward_performance(
    params: Any,
    policy_network: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    expert_trajectories: List[ParentScaleTrajectory],
    reward_config: Any,
    key: jax.Array
) -> Dict[str, float]:
    """Evaluate reward performance of the trained model."""
    
    total_rewards = []
    optimization_rewards = []
    structure_rewards = []
    
    for i, trajectory in enumerate(expert_trajectories):
        if not trajectory.states:
            continue
            
        # Use initial state
        initial_state = trajectory.states[0]
        eval_key = jax.random.fold_in(key, i)
        
        # Get policy action
        policy_output = policy_network.apply(params, initial_state, is_training=False)
        from src.causal_bayes_opt.acquisition.policy import sample_intervention_from_policy
        
        intervention = sample_intervention_from_policy(
            policy_output, initial_state, eval_key, PolicyConfig()
        )
        
        # Simulate intervention (simplified for demo)
        if hasattr(trajectory, 'scm'):
            scm = trajectory.scm
            from src.causal_bayes_opt.environments.sampling import sample_with_intervention
            
            try:
                outcome = sample_with_intervention(scm, intervention, n_samples=1, key=eval_key)[0]
                
                # Create next state (simplified)
                next_buffer = initial_state.buffer.copy()
                next_buffer.add_intervention(intervention, outcome)
                
                from src.causal_bayes_opt.acquisition.services import create_acquisition_state
                next_state = create_acquisition_state(
                    scm=scm,
                    buffer=next_buffer,
                    surrogate_model=surrogate_model,
                    surrogate_params=surrogate_params,
                    target_variable=initial_state.current_target,
                    step=initial_state.step + 1
                )
                
                # Compute reward
                from src.causal_bayes_opt.acquisition.rewards import compute_verifiable_reward
                reward_components = compute_verifiable_reward(
                    initial_state, intervention, outcome, next_state, reward_config.to_pyrsistent()
                )
                
                total_rewards.append(reward_components.total_reward)
                optimization_rewards.append(reward_components.optimization_reward)
                structure_rewards.append(reward_components.structure_discovery_reward)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate reward for trajectory {i}: {e}")
                continue
    
    if not total_rewards:
        return {'mean_total_reward': 0.0, 'mean_optimization_reward': 0.0, 'mean_structure_reward': 0.0}
    
    return {
        'mean_total_reward': float(jnp.mean(jnp.array(total_rewards))),
        'std_total_reward': float(jnp.std(jnp.array(total_rewards))),
        'mean_optimization_reward': float(jnp.mean(jnp.array(optimization_rewards))),
        'mean_structure_reward': float(jnp.mean(jnp.array(structure_rewards))),
        'n_evaluations': len(total_rewards)
    }


def analyze_intervention_diversity(
    params: Any,
    policy_network: Any,
    states: List[Any],
    key: jax.Array
) -> Dict[str, float]:
    """Analyze diversity of interventions produced by the trained policy."""
    
    intervention_targets = []
    intervention_values = []
    
    for i, state in enumerate(states[:20]):  # Limit to 20 for demo
        eval_key = jax.random.fold_in(key, i)
        
        # Get policy action
        policy_output = policy_network.apply(params, state, is_training=False)
        from src.causal_bayes_opt.acquisition.policy import sample_intervention_from_policy
        
        intervention = sample_intervention_from_policy(
            policy_output, state, eval_key, PolicyConfig()
        )
        
        # Extract intervention characteristics
        targets = intervention.get('targets', set())
        values = intervention.get('values', {})
        
        intervention_targets.extend(list(targets))
        intervention_values.extend(list(values.values()))
    
    # Analyze diversity
    unique_targets = len(set(intervention_targets)) if intervention_targets else 0
    total_interventions = len(intervention_targets)
    
    target_diversity = unique_targets / max(1, total_interventions)
    value_diversity = float(jnp.std(jnp.array(intervention_values))) if intervention_values else 0.0
    
    return {
        'target_diversity': target_diversity,
        'value_diversity': value_diversity,
        'unique_targets': unique_targets,
        'total_interventions': total_interventions
    }


def analyze_training_convergence(training_results: TrainingResults) -> Dict[str, Any]:
    """Analyze training convergence patterns."""
    
    bc_metrics = training_results.bc_metrics
    grpo_metrics = training_results.grpo_metrics
    
    convergence_analysis = {}
    
    # BC convergence
    if 'train_loss' in bc_metrics and bc_metrics['train_loss']:
        bc_losses = bc_metrics['train_loss']
        bc_convergence_rate = (bc_losses[0] - bc_losses[-1]) / max(1, len(bc_losses))
        bc_final_loss = bc_losses[-1]
        
        convergence_analysis['bc_convergence_rate'] = float(bc_convergence_rate)
        convergence_analysis['bc_final_loss'] = float(bc_final_loss)
        convergence_analysis['bc_epochs'] = len(bc_losses)
    
    # GRPO convergence
    if 'mean_reward' in grpo_metrics and grpo_metrics['mean_reward']:
        rewards = grpo_metrics['mean_reward']
        reward_trend = (rewards[-1] - rewards[0]) / max(1, len(rewards))
        final_reward = rewards[-1]
        
        convergence_analysis['grpo_reward_trend'] = float(reward_trend)
        convergence_analysis['grpo_final_reward'] = float(final_reward)
        convergence_analysis['grpo_episodes'] = len(rewards)
    
    # Training times
    convergence_analysis['bc_training_time'] = training_results.bc_training_time
    convergence_analysis['grpo_training_time'] = training_results.grpo_training_time
    convergence_analysis['total_training_time'] = training_results.total_training_time
    
    return convergence_analysis


def save_training_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save training results to disk."""
    
    import json
    import pickle
    
    # Save human-readable summary
    summary = create_results_summary(results)
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results (pickle for complex objects)
    with open(output_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {output_dir}")


def create_results_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create human-readable summary of training results."""
    
    training_results = results['training_results']
    evaluation_results = results['evaluation_results']
    
    summary = {
        'demo_info': {
            'total_time': results['demo_total_time'],
            'expert_trajectories': results['expert_trajectory_count'],
            'config_warnings': results['config_warnings']
        },
        'training_performance': {
            'bc_training_time': training_results.bc_training_time,
            'grpo_training_time': training_results.grpo_training_time,
            'total_training_time': training_results.total_training_time
        },
        'model_performance': {
            'expert_similarity': evaluation_results.get('expert_similarity', {}),
            'reward_performance': evaluation_results.get('reward_performance', {}),
            'diversity_analysis': evaluation_results.get('diversity_analysis', {}),
            'convergence_analysis': evaluation_results.get('convergence_analysis', {})
        },
        'final_evaluation': training_results.final_evaluation
    }
    
    return summary


def display_training_summary(results: Dict[str, Any]) -> None:
    """Display training summary to console."""
    
    print("\n" + "="*60)
    print("ACQUISITION TRAINING DEMO RESULTS")
    print("="*60)
    
    training_results = results['training_results']
    evaluation_results = results['evaluation_results']
    
    print(f"\n📊 TRAINING PERFORMANCE:")
    print(f"  • Total training time: {training_results.total_training_time:.2f}s")
    print(f"  • BC phase time: {training_results.bc_training_time:.2f}s")
    print(f"  • GRPO phase time: {training_results.grpo_training_time:.2f}s")
    
    if 'expert_similarity' in evaluation_results:
        similarity = evaluation_results['expert_similarity']
        print(f"\n🎯 EXPERT SIMILARITY:")
        print(f"  • Action matching accuracy: {similarity.get('accuracy', 0):.3f}")
        print(f"  • Average log probability: {similarity.get('avg_log_prob', 0):.3f}")
        print(f"  • Comparisons made: {similarity.get('n_comparisons', 0)}")
    
    if 'reward_performance' in evaluation_results:
        reward_perf = evaluation_results['reward_performance']
        print(f"\n🏆 REWARD PERFORMANCE:")
        print(f"  • Mean total reward: {reward_perf.get('mean_total_reward', 0):.3f}")
        print(f"  • Mean optimization reward: {reward_perf.get('mean_optimization_reward', 0):.3f}")
        print(f"  • Mean structure reward: {reward_perf.get('mean_structure_reward', 0):.3f}")
    
    if 'diversity_analysis' in evaluation_results:
        diversity = evaluation_results['diversity_analysis']
        print(f"\n🎨 INTERVENTION DIVERSITY:")
        print(f"  • Target diversity: {diversity.get('target_diversity', 0):.3f}")
        print(f"  • Value diversity (std): {diversity.get('value_diversity', 0):.3f}")
        print(f"  • Unique targets: {diversity.get('unique_targets', 0)}")
    
    if 'convergence_analysis' in evaluation_results:
        convergence = evaluation_results['convergence_analysis']
        print(f"\n📈 CONVERGENCE ANALYSIS:")
        if 'bc_final_loss' in convergence:
            print(f"  • BC final loss: {convergence['bc_final_loss']:.4f}")
        if 'grpo_final_reward' in convergence:
            print(f"  • GRPO final reward: {convergence['grpo_final_reward']:.3f}")
        if 'grpo_reward_trend' in convergence:
            print(f"  • GRPO reward trend: {convergence['grpo_reward_trend']:.4f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   Expert trajectories: {results['expert_trajectory_count']}")
    print(f"   Total demo time: {results['demo_total_time']:.2f}s")
    
    if results['config_warnings']:
        print(f"\n⚠️  Configuration warnings: {len(results['config_warnings'])}")
        for warning in results['config_warnings']:
            print(f"     • {warning}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()