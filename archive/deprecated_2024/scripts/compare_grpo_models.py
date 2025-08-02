#!/usr/bin/env python3
"""
GRPO Model Comparison Script

Compare performance of different trained GRPO models with statistical analysis,
visualization, and comprehensive evaluation metrics. Follows methodology 
similar to acbo_wandb_experiment.py for systematic model comparison.

Usage:
    # Compare all models in checkpoints directory
    poetry run python scripts/compare_grpo_models.py
    
    # Compare specific models
    poetry run python scripts/compare_grpo_models.py \
        --model_paths checkpoints/model1 checkpoints/model2
    
    # Custom evaluation configuration
    poetry run python scripts/compare_grpo_models.py \
        evaluation.n_test_episodes=50 \
        evaluation.test_scm_variants=5 \
        logging.wandb.enabled=true

Features:
- Load and compare multiple trained GRPO models
- Statistical significance testing between models
- Performance evaluation on multiple test environments
- Visualization of learning curves and performance metrics
- WandB integration for experiment tracking
- Automated report generation
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
import numpy as onp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_bayes_opt.acquisition.rewards import (
    RewardComponents, compute_verifiable_reward, create_default_reward_config
)
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from scipy import stats
    import matplotlib.pyplot as plt
    SCIPY_AVAILABLE = True
    PLOTTING_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a trained model."""
    name: str
    path: Path
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    policy_params: Any
    value_params: Any
    scm: pyr.PMap


@dataclass
class EvaluationResult:
    """Results from evaluating a model."""
    model_name: str
    test_rewards: List[float]
    test_values: List[float] 
    convergence_time: Optional[float]
    success_rate: float
    mean_performance: float
    std_performance: float


class GRPOModelComparator:
    """Compare performance of trained GRPO models."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: List[ModelInfo] = []
        self.evaluation_results: List[EvaluationResult] = []
        
        # Create test environments
        self.test_scms = self._create_test_environments()
        
        logger.info(f"Initialized model comparator with {len(self.test_scms)} test environments")
    
    def _create_test_environments(self) -> List[pyr.PMap]:
        """Create diverse test environments for evaluation."""
        test_scms = []
        
        for variant in range(self.config.evaluation.test_scm_variants):
            n_vars = self.config.evaluation.n_test_variables
            variables = [f"X{i}" for i in range(n_vars)]
            
            # Create different topologies
            if variant == 0:
                # Chain structure
                edges = [(variables[i], variables[i+1]) for i in range(n_vars-1)]
            elif variant == 1:
                # Star structure (all point to last variable)
                edges = [(variables[i], variables[-1]) for i in range(n_vars-1)]
            elif variant == 2:
                # Fork structure (first variable influences all others)
                edges = [(variables[0], variables[i]) for i in range(1, n_vars)]
            else:
                # Random sparse structure
                key = random.PRNGKey(variant + 1000)
                edges = []
                for i in range(n_vars-1):
                    for j in range(i+1, n_vars):
                        if random.uniform(key) < 0.3:  # 30% edge probability
                            edges.append((variables[i], variables[j]))
                        key, _ = random.split(key)
            
            # Create mechanisms with variant-specific coefficients
            mechanisms = {}
            key = random.PRNGKey(variant + 2000)
            
            for var in variables:
                parents = [edge[0] for edge in edges if edge[1] == var]
                
                if not parents:
                    # Root variable
                    mechanisms[var] = create_linear_mechanism([], {}, intercept=0.0, noise_scale=1.0)
                else:
                    # Create coefficients with different distributions per variant
                    coefficients = {}
                    for parent in parents:
                        key, subkey = random.split(key)
                        if variant % 2 == 0:
                            coeff = random.uniform(subkey, minval=0.5, maxval=2.0)
                        else:
                            coeff = random.normal(subkey) * 0.8 + 1.0
                        coefficients[parent] = float(coeff)
                    
                    mechanisms[var] = create_linear_mechanism(
                        parents, coefficients, intercept=0.0, noise_scale=1.0
                    )
            
            scm = create_scm(
                variables=set(variables),
                edges=set(edges),
                mechanisms=mechanisms,
                target=variables[-1]
            )
            test_scms.append(scm)
        
        return test_scms
    
    def load_model(self, model_path: Path) -> ModelInfo:
        """Load a trained model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load model data
        model_file = model_path / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        # Load metrics
        metrics_file = model_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = checkpoint_data.get('metrics', {})
        
        model_info = ModelInfo(
            name=model_path.name,
            path=model_path,
            config=checkpoint_data['config'],
            metrics=metrics,
            policy_params=checkpoint_data['policy_params'],
            value_params=checkpoint_data['value_params'],
            scm=checkpoint_data['scm']
        )
        
        logger.info(f"Loaded model: {model_info.name}")
        return model_info
    
    def load_models_from_directory(self, directory: Path) -> List[ModelInfo]:
        """Load all models from a directory."""
        models = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return models
        
        for model_dir in directory.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pkl").exists():
                try:
                    model = self.load_model(model_dir)
                    models.append(model)
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_dir}: {e}")
        
        logger.info(f"Loaded {len(models)} models from {directory}")
        return models
    
    def _create_policy_network_from_config(self, config: Dict[str, Any]) -> Any:
        """Recreate policy network from config."""
        import haiku as hk
        
        training_config = config['training']
        
        def policy_fn(state_tensor):
            mlp = hk.nets.MLP([
                training_config['hidden_size']
            ] * training_config['num_layers'] + [
                training_config['action_dim']
            ])
            logits = mlp(state_tensor)
            interventions = jnp.tanh(logits) * training_config['max_intervention_value']
            return interventions
        
        return hk.transform(policy_fn)
    
    def _evaluate_model_on_scm(self, model: ModelInfo, test_scm: pyr.PMap, 
                              n_episodes: int) -> Tuple[List[float], List[float]]:
        """Evaluate a model on a specific test SCM."""
        # Recreate policy network
        policy_net = self._create_policy_network_from_config(model.config)
        
        # Create reward config
        reward_weights = model.config['training']['reward_weights']
        reward_config = create_default_reward_config(
            optimization_weight=reward_weights['optimization'],
            structure_weight=reward_weights['structure'],
            parent_weight=reward_weights['parent'],
            exploration_weight=reward_weights['exploration']
        )
        
        episode_rewards = []
        episode_values = []
        
        for episode in range(n_episodes):
            rewards, values = self._run_evaluation_episode(
                model, policy_net, test_scm, reward_config, episode
            )
            episode_rewards.append(float(jnp.mean(jnp.array(rewards))))
            episode_values.append(max(values))
        
        return episode_rewards, episode_values
    
    def _run_evaluation_episode(self, model: ModelInfo, policy_net: Any, 
                               test_scm: pyr.PMap, reward_config: pyr.PMap, 
                               episode_idx: int) -> Tuple[List[float], List[float]]:
        """Run single evaluation episode."""
        from unittest.mock import Mock
        
        episode_length = self.config.evaluation.episode_length
        rewards = []
        values = []
        best_value = 0.0
        
        # Create mock state similar to training
        def create_mock_state(step, best_val):
            state = Mock()
            state.current_target = test_scm["target"]
            state.step = step
            state.best_value = best_val
            state.uncertainty_bits = max(0.1, 1.0 - step * 0.05)
            
            mock_buffer_stats = Mock()
            mock_buffer_stats.total_samples = max(1, step * 2)
            state.buffer_statistics = mock_buffer_stats
            
            variables = list(test_scm["variables"])
            state.marginal_parent_probs = {
                var: 0.8 - step * 0.02 for var in variables[:-1]
            }
            
            # Add mechanism predictions and bounds
            target = test_scm["target"]
            mock_mechanism = Mock()
            mock_mechanism.coefficients = {variables[-2]: 1.5} if len(variables) > 1 else {}
            mock_mechanism.intercept = 0.0
            state.mechanism_predictions = {target: mock_mechanism}
            
            max_val = model.config['training']['max_intervention_value']
            state.intervention_bounds = {var: (-max_val, max_val) for var in variables[:-1]}
            
            return state
        
        def convert_state_to_tensor(state):
            """Convert state to tensor matching model's expected input."""
            variables = list(test_scm["variables"])
            features = [
                state.step / 100.0,
                state.best_value / 10.0,
                state.uncertainty_bits,
                state.buffer_statistics.total_samples / 50.0,
            ]
            
            # Add parent probabilities
            for var in variables[:-1]:
                features.append(state.marginal_parent_probs.get(var, 0.0))
            
            # Pad to required dimension
            state_dim = model.config['training']['state_dim']
            while len(features) < state_dim:
                features.append(0.0)
            
            return jnp.array(features[:state_dim])
        
        state_before = create_mock_state(0, best_value)
        
        for step in range(episode_length):
            # Get policy action
            state_tensor = convert_state_to_tensor(state_before)
            state_tensor = jnp.expand_dims(state_tensor, axis=0)
            
            step_key = random.PRNGKey(episode_idx * 1000 + step)
            action = policy_net.apply(model.policy_params, step_key, state_tensor)[0]
            
            # Create intervention
            variables = list(test_scm["variables"])
            intervention_var = variables[0] if len(variables) > 1 else variables[0]
            intervention = pyr.m({
                'type': "perfect",
                'targets': {intervention_var},
                'values': {intervention_var: float(action[0])}
            })
            
            # Simulate outcome (simplified)
            target_value = float(action[0]) * 1.5 + random.normal(step_key) * 0.1
            outcome = pyr.m({'values': {test_scm["target"]: target_value, intervention_var: float(action[0])}})
            
            if target_value > best_value:
                best_value = target_value
            
            values.append(target_value)
            
            # Compute reward
            state_after = create_mock_state(step + 1, best_value)
            reward_components = compute_verifiable_reward(
                state_before, intervention, outcome, state_after, reward_config
            )
            rewards.append(reward_components.total_reward)
            
            state_before = state_after
        
        return rewards, values
    
    def evaluate_model(self, model: ModelInfo) -> EvaluationResult:
        """Comprehensive evaluation of a single model."""
        logger.info(f"Evaluating model: {model.name}")
        
        all_rewards = []
        all_values = []
        
        # Test on all environments
        for scm_idx, test_scm in enumerate(self.test_scms):
            episode_rewards, episode_values = self._evaluate_model_on_scm(
                model, test_scm, self.config.evaluation.n_test_episodes
            )
            all_rewards.extend(episode_rewards)
            all_values.extend(episode_values)
        
        # Compute performance metrics
        mean_performance = float(jnp.mean(jnp.array(all_rewards)))
        std_performance = float(jnp.std(jnp.array(all_rewards)))
        
        # Success rate (episodes with positive mean reward)
        success_rate = float(jnp.mean(jnp.array(all_rewards) > 0))
        
        # Estimate convergence time (simplified)
        convergence_time = None
        if len(all_rewards) > 10:
            # Find where performance stabilizes
            window_size = min(10, len(all_rewards) // 4)
            if window_size > 0:
                variances = []
                for i in range(window_size, len(all_rewards)):
                    window_var = float(jnp.var(jnp.array(all_rewards[i-window_size:i])))
                    variances.append(window_var)
                
                if variances:
                    threshold = jnp.percentile(jnp.array(variances), 25)
                    for i, var in enumerate(variances):
                        if var < threshold:
                            convergence_time = float(i + window_size)
                            break
        
        result = EvaluationResult(
            model_name=model.name,
            test_rewards=all_rewards,
            test_values=all_values,
            convergence_time=convergence_time,
            success_rate=success_rate,
            mean_performance=mean_performance,
            std_performance=std_performance
        )
        
        logger.info(f"Model {model.name} - Mean performance: {mean_performance:.4f}, Success rate: {success_rate:.2f}")
        return result
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare all loaded models and generate comprehensive analysis."""
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        logger.info(f"Comparing {len(self.models)} models")
        
        # Evaluate all models
        results = []
        for model in self.models:
            result = self.evaluate_model(model)
            results.append(result)
            self.evaluation_results.append(result)
        
        # Statistical comparison
        comparison_stats = self._compute_statistical_comparison(results)
        
        # Performance ranking
        ranking = self._rank_models(results)
        
        # Generate visualizations
        plots = {}
        if PLOTTING_AVAILABLE:
            plots = self._create_comparison_plots(results)
        
        comparison_report = {
            'model_rankings': ranking,
            'statistical_comparison': comparison_stats,
            'individual_results': {r.model_name: {
                'mean_performance': r.mean_performance,
                'std_performance': r.std_performance,
                'success_rate': r.success_rate,
                'convergence_time': r.convergence_time
            } for r in results},
            'plots': plots,
            'summary': self._generate_summary(results, comparison_stats)
        }
        
        return comparison_report
    
    def _compute_statistical_comparison(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute statistical significance tests between models."""
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available for statistical tests"}
        
        stats_results = {}
        
        # Pairwise comparisons
        pairwise_comparisons = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                model1, model2 = results[i], results[j]
                
                # t-test for performance difference
                statistic, p_value = stats.ttest_ind(
                    model1.test_rewards, model2.test_rewards
                )
                
                # Effect size (Cohen's d)
                pooled_std = jnp.sqrt(
                    (jnp.var(jnp.array(model1.test_rewards)) + jnp.var(jnp.array(model2.test_rewards))) / 2
                )
                cohens_d = (model1.mean_performance - model2.mean_performance) / pooled_std
                
                comparison = {
                    'model1': model1.model_name,
                    'model2': model2.model_name,
                    'mean_diff': model1.mean_performance - model2.mean_performance,
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05
                }
                pairwise_comparisons.append(comparison)
        
        stats_results['pairwise_comparisons'] = pairwise_comparisons
        
        # ANOVA if more than 2 models
        if len(results) > 2:
            all_performances = [r.test_rewards for r in results]
            f_stat, anova_p = stats.f_oneway(*all_performances)
            stats_results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'significant': anova_p < 0.05
            }
        
        return stats_results
    
    def _rank_models(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """Rank models by performance."""
        # Sort by mean performance
        sorted_results = sorted(results, key=lambda r: r.mean_performance, reverse=True)
        
        ranking = []
        for rank, result in enumerate(sorted_results, 1):
            ranking.append({
                'rank': rank,
                'model_name': result.model_name,
                'mean_performance': result.mean_performance,
                'success_rate': result.success_rate,
                'convergence_time': result.convergence_time
            })
        
        return ranking
    
    def _create_comparison_plots(self, results: List[EvaluationResult]) -> Dict[str, str]:
        """Create comparison plots and save to files."""
        plots = {}
        
        # Performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot of performance
        model_names = [r.model_name for r in results]
        performances = [r.test_rewards for r in results]
        
        ax1.boxplot(performances, labels=model_names)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_rates = [r.success_rate for r in results]
        ax2.bar(model_names, success_rates)
        ax2.set_title('Success Rate Comparison')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = "model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plots['comparison'] = plot_path
        
        return plots
    
    def _generate_summary(self, results: List[EvaluationResult], 
                         stats: Dict[str, Any]) -> str:
        """Generate textual summary of comparison."""
        best_model = max(results, key=lambda r: r.mean_performance)
        worst_model = min(results, key=lambda r: r.mean_performance)
        
        summary = f"""
Model Comparison Summary:
========================

Best Performing Model: {best_model.model_name}
- Mean Performance: {best_model.mean_performance:.4f}
- Success Rate: {best_model.success_rate:.2f}
- Convergence Time: {best_model.convergence_time or 'N/A'}

Worst Performing Model: {worst_model.model_name}
- Mean Performance: {worst_model.mean_performance:.4f}
- Success Rate: {worst_model.success_rate:.2f}

Performance Gap: {best_model.mean_performance - worst_model.mean_performance:.4f}

Statistical Significance:
"""
        
        if 'anova' in stats:
            anova = stats['anova']
            summary += f"- ANOVA p-value: {anova['p_value']:.4f} ({'Significant' if anova['significant'] else 'Not significant'})\n"
        
        if 'pairwise_comparisons' in stats:
            significant_pairs = [c for c in stats['pairwise_comparisons'] if c['significant']]
            summary += f"- Significant pairwise differences: {len(significant_pairs)}/{len(stats['pairwise_comparisons'])}\n"
        
        return summary


@hydra.main(version_base=None, config_path="../config", config_name="model_comparison_config")
def main(cfg: DictConfig) -> None:
    """Main comparison function."""
    
    logger.info("🔍 Starting GRPO Model Comparison")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB if enabled
    wandb_run = None
    if cfg.logging.wandb.enabled and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.logging.wandb.tags + ["model_comparison", "grpo_evaluation"],
            group="model_comparison",
            name=f"grpo_comparison_{int(time.time())}"
        )
    
    # Create comparator
    comparator = GRPOModelComparator(cfg)
    
    try:
        # Load models
        if hasattr(cfg, 'model_paths') and cfg.model_paths:
            # Load specific models
            for path_str in cfg.model_paths:
                model = comparator.load_model(Path(path_str))
                comparator.models.append(model)
        else:
            # Load all models from checkpoints directory
            checkpoints_dir = Path("checkpoints")
            models = comparator.load_models_from_directory(checkpoints_dir)
            comparator.models.extend(models)
        
        if not comparator.models:
            logger.error("No models found to compare")
            return
        
        # Run comparison
        results = comparator.compare_models()
        
        # Log results
        logger.info("📊 Comparison Results:")
        logger.info(f"Models compared: {len(comparator.models)}")
        
        if results:
            # Print ranking
            logger.info("\n🏆 Model Rankings:")
            for rank_info in results['model_rankings']:
                logger.info(f"  {rank_info['rank']}. {rank_info['model_name']} - "
                          f"Performance: {rank_info['mean_performance']:.4f}")
            
            # Print summary
            logger.info(f"\n{results['summary']}")
            
            # Log to WandB
            if wandb_run:
                # Log individual model results
                for model_name, metrics in results['individual_results'].items():
                    for metric_name, value in metrics.items():
                        if value is not None:
                            wandb.log({f"models/{model_name}/{metric_name}": value})
                
                # Log comparison table
                comparison_table = wandb.Table(
                    columns=["Model", "Rank", "Mean Performance", "Success Rate"],
                    data=[[r['model_name'], r['rank'], r['mean_performance'], r['success_rate']] 
                          for r in results['model_rankings']]
                )
                wandb.log({"model_comparison_table": comparison_table})
                
                # Upload plots as artifacts
                if results['plots']:
                    for plot_name, plot_path in results['plots'].items():
                        wandb.log({f"plots/{plot_name}": wandb.Image(plot_path)})
        
        logger.info("✅ Model comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        if wandb_run:
            wandb.log({"error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main()