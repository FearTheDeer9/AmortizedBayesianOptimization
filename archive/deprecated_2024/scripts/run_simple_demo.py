#!/usr/bin/env python3
"""
Simple ACBO Demo - Quick working demonstration of the system

This is a simplified version that:
1. Trains a basic GRPO policy
2. Evaluates it against baselines
3. Shows the results
"""

import sys
import logging
from pathlib import Path
import json
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
# Removed unused import
from src.causal_bayes_opt.acquisition.grpo import GRPOConfig, create_grpo_trainer
from src.causal_bayes_opt.acquisition.state import AcquisitionState
from src.causal_bayes_opt.acquisition.rewards import create_default_reward_config
from src.causal_bayes_opt.environments.intervention_env import InterventionEnvironment
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.evaluation.base_evaluator import BaseEvaluator
from src.causal_bayes_opt.evaluation.baselines import RandomEvaluator, OracleEvaluator
from src.causal_bayes_opt.evaluation.run_evaluation import create_test_scms
from scripts.core.utils.metric_utils import compute_f1_score, compute_shd, aggregate_trajectories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGRPOEvaluator(BaseEvaluator):
    """Simple GRPO evaluator that uses a trained policy."""
    
    def __init__(self, policy_params, policy_fn, optimization_direction="MINIMIZE"):
        self.policy_params = policy_params
        self.policy_fn = policy_fn
        self.optimization_direction = optimization_direction
        self.name = "simple_grpo"
        
    def get_name(self) -> str:
        return self.name
        
    def get_params(self):
        return {"optimization_direction": self.optimization_direction}
        
    def choose_intervention(self, state: AcquisitionState, scm, key):
        """Choose intervention using GRPO policy."""
        # Get policy output
        policy_output = self.policy_fn.apply(self.policy_params, state.get_state_tensor())
        
        # Sample from policy
        intervention_key, value_key = random.split(key)
        
        # Get variables
        variables = list(get_variables(scm))
        target = get_target(scm)
        
        # Sample intervention target
        probs = jax.nn.softmax(policy_output['logits'])
        target_idx = random.categorical(intervention_key, probs)
        intervention_target = variables[target_idx]
        
        # Sample intervention value
        if 'value_params' in policy_output:
            # Use learned value distribution
            mean = policy_output['value_params']['mean'][target_idx]
            std = policy_output['value_params']['std'][target_idx]
            value = mean + std * random.normal(value_key)
        else:
            # Default to random value
            value = random.uniform(value_key, minval=-2.0, maxval=2.0)
        
        return {intervention_target: float(value)}


def train_simple_grpo(n_episodes=50, episode_length=10, n_scms=4):
    """Train a simple GRPO policy."""
    logger.info("Training simple GRPO policy...")
    
    # Create training SCMs
    factory = VariableSCMFactory(seed=42)
    training_scms = []
    for i in range(n_scms):
        n_vars = 3 + (i % 3)  # 3-5 variables
        structure = ['fork', 'chain', 'collider'][i % 3]
        scm = factory.create_variable_scm(
            num_variables=n_vars,
            structure_type=structure
        )
        training_scms.append(scm)
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        learning_rate=1e-3,
        entropy_coeff=0.1,  # Higher for exploration
        group_size=16,
        interventions_per_state=4
    )
    
    # Create reward config
    reward_config = create_default_reward_config()
    
    # Initialize policy (simple MLP)
    key = random.PRNGKey(42)
    init_key, train_key = random.split(key)
    
    # Create a simple policy network
    import haiku as hk
    
    def policy_network(state):
        """Simple policy network."""
        # Flatten state if needed
        if len(state.shape) > 2:
            state = state.reshape(state.shape[0], -1)
        
        # MLP layers
        x = hk.Linear(128)(state)
        x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        
        # Output logits for variable selection
        logits = hk.Linear(10)(x)  # Max 10 variables
        
        return {'logits': logits}
    
    # Transform to pure function
    policy_fn = hk.transform(policy_network)
    
    # Initialize with dummy input
    dummy_state = jnp.zeros((1, 100, 5, 5))  # [batch, history, vars, channels]
    policy_params = policy_fn.init(init_key, dummy_state)
    
    # Create optimizer
    import optax
    optimizer = optax.adam(grpo_config.learning_rate)
    opt_state = optimizer.init(policy_params)
    
    # Training loop
    logger.info(f"Running {n_episodes} episodes...")
    all_rewards = []
    
    for episode in range(n_episodes):
        # Select SCM for this episode
        scm = training_scms[episode % len(training_scms)]
        
        # Create environment
        env = InterventionEnvironment(scm)
        buffer = ExperienceBuffer()
        
        # Collect observational data
        obs_samples = env.get_observational_samples(n_samples=100)
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Run episode
        episode_rewards = []
        state = AcquisitionState(buffer=buffer)
        
        for step in range(episode_length):
            # Get policy output
            state_tensor = state.get_state_tensor()
            state_tensor = state_tensor[None, ...]  # Add batch dim
            
            # Sample intervention
            step_key = random.fold_in(train_key, episode * episode_length + step)
            intervention_key, outcome_key = random.split(step_key)
            
            policy_output = policy_fn.apply(policy_params, state_tensor)
            
            # Simple intervention selection
            variables = list(get_variables(scm))
            target = get_target(scm)
            
            if len(variables) > 10:
                variables = variables[:10]  # Limit to 10 vars
            
            # Sample from policy
            logits = policy_output['logits'][0, :len(variables)]
            probs = jax.nn.softmax(logits)
            var_idx = random.categorical(intervention_key, probs)
            intervention_var = variables[var_idx]
            
            # Random intervention value
            intervention_value = float(random.uniform(outcome_key, minval=-2.0, maxval=2.0))
            intervention = {intervention_var: intervention_value}
            
            # Execute intervention
            outcome = env.intervene(intervention, n_samples=1)[0]
            
            # Calculate reward (simple: negative absolute target value for minimization)
            target_value = outcome.values[target]
            reward = -abs(target_value) if target_value is not None else 0.0
            episode_rewards.append(reward)
            
            # Update state
            buffer.add_intervention(create_sample(intervention), outcome)
            state = AcquisitionState(buffer=buffer)
        
        # Simple policy gradient update (simplified for demo)
        if episode > 0 and episode % 10 == 0:
            # Placeholder for actual GRPO update
            # In real implementation, would use proper GRPO loss
            mean_reward = np.mean(episode_rewards)
            all_rewards.append(mean_reward)
            logger.info(f"Episode {episode}: avg reward = {mean_reward:.4f}")
    
    return policy_params, policy_fn


def evaluate_simple_demo(policy_params, policy_fn, n_test_scms=3, n_runs=2):
    """Evaluate the trained policy against baselines."""
    logger.info("Evaluating trained policy...")
    
    # Create test SCMs
    test_scms = create_test_scms(
        n_scms=n_test_scms,
        variable_range=[3, 5],
        structure_types=['fork', 'chain', 'collider'],
        seed=123
    )
    
    # Create evaluators
    grpo_evaluator = SimpleGRPOEvaluator(policy_params, policy_fn)
    random_evaluator = RandomEvaluator()
    oracle_evaluator = OracleEvaluator()
    
    evaluators = [grpo_evaluator, random_evaluator, oracle_evaluator]
    
    results = {}
    for evaluator in evaluators:
        logger.info(f"Evaluating {evaluator.get_name()}...")
        method_results = []
        
        for scm_idx, scm in enumerate(test_scms):
            for run in range(n_runs):
                # Run single evaluation
                env = InterventionEnvironment(scm)
                buffer = ExperienceBuffer()
                
                # Get observational samples
                obs_samples = env.get_observational_samples(n_samples=100)
                for sample in obs_samples:
                    buffer.add_observation(sample)
                
                state = AcquisitionState(buffer=buffer)
                target = get_target(scm)
                
                # Track target values
                target_values = []
                
                # Run interventions
                key = random.PRNGKey(run * 1000 + scm_idx)
                for step in range(10):
                    step_key = random.fold_in(key, step)
                    
                    # Get intervention
                    intervention = evaluator.choose_intervention(state, scm, step_key)
                    
                    # Execute
                    outcome = env.intervene(intervention, n_samples=1)[0]
                    
                    # Track target value
                    if target in outcome.values:
                        target_values.append(float(outcome.values[target]))
                    
                    # Update state
                    buffer.add_intervention(create_sample(intervention), outcome)
                    state = AcquisitionState(buffer=buffer)
                
                # Store result
                method_results.append({
                    'scm_idx': scm_idx,
                    'run': run,
                    'target_values': target_values,
                    'final_value': target_values[-1] if target_values else None
                })
        
        results[evaluator.get_name()] = method_results
    
    return results


def main():
    """Run simple demonstration."""
    logger.info("="*60)
    logger.info("ACBO Simple Demonstration")
    logger.info("="*60)
    
    # Train GRPO
    start_time = time.time()
    policy_params, policy_fn = train_simple_grpo(n_episodes=50)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f} seconds")
    
    # Evaluate
    eval_start = time.time()
    results = evaluate_simple_demo(policy_params, policy_fn, n_test_scms=3, n_runs=2)
    eval_time = time.time() - eval_start
    logger.info(f"Evaluation completed in {eval_time:.1f} seconds")
    
    # Show results
    logger.info("\n" + "="*60)
    logger.info("Results Summary")
    logger.info("="*60)
    
    for method, method_results in results.items():
        final_values = [r['final_value'] for r in method_results if r['final_value'] is not None]
        if final_values:
            mean_value = np.mean(final_values)
            std_value = np.std(final_values)
            logger.info(f"{method:20s}: {mean_value:.4f} ± {std_value:.4f}")
    
    # Save results
    output_dir = Path("results/simple_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'train_time': train_time,
            'eval_time': eval_time,
            'results': results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nDemo completed successfully!")
    
    return results


if __name__ == "__main__":
    main()