"""
Trajectory analysis for target optimization experiments.

This module implements core trajectory analysis for evaluating
optimization performance across different methods.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add paths for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.core.metric_collector import MetricCollector
from experiments.evaluation.core.pairing_manager import (
    PairingManager, PairingConfig, ModelSpec, ModelType, create_standard_pairings
)

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result of a single optimization trajectory."""
    method_name: str
    scm_size: int
    scm_id: str
    interventions: List[Dict[str, Any]]
    final_target_value: float
    best_target_value: float
    convergence_rate: float
    parent_accuracy: float
    exploration_diversity: float
    success: bool
    error_message: Optional[str] = None


class TrajectoryAnalyzer:
    """Analyzes optimization trajectories across different methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exp_config = config['experiment']
        self.eval_config = config['evaluation']
        
        self.scm_factory = VariableSCMFactory(seed=self.exp_config['seed'])
        self.results = []
        
        logger.info(f"Initialized TrajectoryAnalyzer for {self.exp_config['name']}")
    
    def create_test_pairings(self, experiments_dir: Path) -> List[PairingConfig]:
        """
        Create test pairings for trajectory analysis.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            List of pairing configurations to test
        """
        manager = PairingManager(base_seed=self.exp_config['seed'])
        
        # Add baselines
        manager.add_baseline_pairing("Random", "random")
        manager.add_baseline_pairing("Oracle", "oracle")
        
        # Add untrained policy
        manager.add_untrained_policy_pairing("Untrained Policy")
        
        # Try to add trained models
        try:
            # Look for joint training checkpoints
            joint_dir = experiments_dir / 'joint-training' / 'checkpoints'
            if joint_dir.exists():
                joint_checkpoints = [d for d in joint_dir.iterdir() 
                                   if d.is_dir() and (d / 'policy.pkl').exists()]
                if joint_checkpoints:
                    # Use the most recent one (joint_ep2 if available)
                    best_checkpoint = None
                    for cp in joint_checkpoints:
                        if 'ep2' in cp.name:
                            best_checkpoint = cp
                            break
                    if not best_checkpoint:
                        best_checkpoint = joint_checkpoints[0]
                    
                    manager.add_trained_policy_pairing(
                        "Joint Trained",
                        policy_checkpoint=best_checkpoint / 'policy.pkl',
                        surrogate_checkpoint=best_checkpoint / 'surrogate.pkl',
                        description=f"Joint trained from {best_checkpoint.name}"
                    )
                    logger.info(f"Added joint trained model: {best_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not add trained models: {e}")
        
        pairings = manager.get_pairings()
        logger.info(f"Created {len(pairings)} test pairings")
        return pairings
    
    def run_single_trajectory(self,
                            pairing: PairingConfig,
                            scm: Dict[str, Any],
                            scm_id: str) -> TrajectoryResult:
        """
        Run optimization trajectory for a single method on a single SCM.
        
        Args:
            pairing: Method configuration
            scm: SCM to optimize on
            scm_id: Identifier for the SCM
            
        Returns:
            TrajectoryResult with trajectory data and metrics
        """
        start_time = time.time()
        
        try:
            # Setup
            variables = list(get_variables(scm))
            target_var = get_target(scm)
            true_parents = get_parents(scm, target_var)
            
            logger.debug(f"Running {pairing.name} on SCM {scm_id} "
                        f"({len(variables)} vars, target: {target_var})")
            
            # Load models based on pairing spec
            acquisition_fn = self._load_acquisition_function(pairing, scm)
            surrogate_model = self._load_surrogate_model(pairing, variables, target_var)
            
            # Initialize with observational data
            buffer = ExperienceBuffer()
            n_obs = self.config['data_generation']['n_observational_samples']
            obs_samples = sample_from_linear_scm(scm, n_samples=n_obs, seed=42)
            for sample in obs_samples:
                buffer.add_observation(sample)
            
            # Track trajectory
            interventions = []
            target_values = []
            parent_interventions = []
            
            # Run interventions
            n_interventions = self.eval_config['n_interventions']
            for step in range(n_interventions):
                # Convert buffer to tensor
                tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
                
                # Get intervention
                intervention_dict = acquisition_fn(tensor, None, target_var, variables)
                
                # Extract intervention details
                if isinstance(intervention_dict['targets'], frozenset):
                    selected_var = list(intervention_dict['targets'])[0]
                else:
                    selected_var = intervention_dict['targets'][0]
                intervention_value = intervention_dict['values'][selected_var]
                
                # Apply intervention
                intervention = create_perfect_intervention(
                    targets=frozenset([selected_var]),
                    values={selected_var: float(intervention_value)}
                )
                
                # Sample post-intervention
                post_samples = sample_with_intervention(scm, intervention, 1, seed=step+42)
                
                if post_samples:
                    post_sample = post_samples[0]
                    buffer.add_intervention(intervention, post_sample)
                    
                    # Record trajectory point
                    target_value = float(post_sample['values'][target_var])
                    is_parent = selected_var in true_parents
                    
                    interventions.append({
                        'step': step,
                        'variable': selected_var,
                        'value': float(intervention_value),
                        'target_value': target_value,
                        'is_parent': is_parent
                    })
                    
                    target_values.append(target_value)
                    parent_interventions.append(is_parent)
            
            # Compute metrics
            final_target = target_values[-1] if target_values else 0.0
            best_target = min(target_values) if target_values else 0.0
            
            # Convergence rate (improvement per intervention)
            if len(target_values) >= 2:
                improvements = [target_values[i] - target_values[i+1] 
                              for i in range(len(target_values)-1)]
                positive_improvements = [imp for imp in improvements if imp > 0]
                convergence_rate = np.mean(positive_improvements) if positive_improvements else 0.0
            else:
                convergence_rate = 0.0
            
            # Parent accuracy
            parent_accuracy = np.mean(parent_interventions) if parent_interventions else 0.0
            
            # Exploration diversity (unique variables tried)
            unique_vars = len(set(inv['variable'] for inv in interventions))
            exploration_diversity = unique_vars / len(variables) if variables else 0.0
            
            execution_time = time.time() - start_time
            
            logger.debug(f"{pairing.name} completed: final={final_target:.3f}, "
                        f"best={best_target:.3f}, parent_acc={parent_accuracy:.1%}, "
                        f"time={execution_time:.1f}s")
            
            return TrajectoryResult(
                method_name=pairing.name,
                scm_size=len(variables),
                scm_id=scm_id,
                interventions=interventions,
                final_target_value=final_target,
                best_target_value=best_target,
                convergence_rate=convergence_rate,
                parent_accuracy=parent_accuracy,
                exploration_diversity=exploration_diversity,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Trajectory failed for {pairing.name} on {scm_id}: {e}")
            
            return TrajectoryResult(
                method_name=pairing.name,
                scm_size=len(get_variables(scm)) if scm else 0,
                scm_id=scm_id,
                interventions=[],
                final_target_value=0.0,
                best_target_value=0.0,
                convergence_rate=0.0,
                parent_accuracy=0.0,
                exploration_diversity=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _load_acquisition_function(self, pairing: PairingConfig, scm: Dict[str, Any]):
        """Load acquisition function based on pairing spec."""
        if pairing.policy_spec.model_type == ModelType.TRAINED:
            return ModelLoader.load_policy(pairing.policy_spec.checkpoint_path, seed=42)
        elif pairing.policy_spec.model_type == ModelType.UNTRAINED:
            return ModelLoader.create_untrained_policy(seed=42)
        elif pairing.policy_spec.model_type == ModelType.RANDOM:
            return ModelLoader.load_baseline('random', seed=42)
        elif pairing.policy_spec.model_type == ModelType.ORACLE:
            return ModelLoader.load_baseline('oracle', scm=scm)
        else:
            raise ValueError(f"Unknown policy type: {pairing.policy_spec.model_type}")
    
    def _load_surrogate_model(self, pairing: PairingConfig, variables: list, target_var: str):
        """Load surrogate model based on pairing spec."""
        if pairing.surrogate_spec.model_type == ModelType.TRAINED:
            return ModelLoader.load_surrogate(pairing.surrogate_spec.checkpoint_path)
        elif pairing.surrogate_spec.model_type == ModelType.UNTRAINED:
            return ModelLoader.create_untrained_surrogate(
                variables=variables,
                target_variable=target_var,
                seed=42
            )
        else:
            return None
    
    def run_trajectory_experiment(self, experiments_dir: Path) -> List[TrajectoryResult]:
        """
        Run complete trajectory experiment.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            List of trajectory results
        """
        logger.info("Starting trajectory optimization experiment")
        
        # Create test SCMs
        test_scms = []
        sizes = self.eval_config['graph_sizes']
        n_per_size = self.eval_config['n_graphs_per_size']
        
        for size in sizes:
            for i in range(n_per_size):
                scm = self.scm_factory.create_variable_scm(
                    num_variables=size,
                    structure_type="random",
                    edge_density=0.3
                )
                test_scms.append((size, f"scm_{size}_{i}", scm))
        
        logger.info(f"Created {len(test_scms)} test SCMs")
        
        # Create pairings
        pairings = self.create_test_pairings(experiments_dir)
        
        # Run all combinations
        all_results = []
        total_experiments = len(pairings) * len(test_scms)
        experiment_count = 0
        
        logger.info(f"Running {total_experiments} trajectory experiments")
        
        for pairing in pairings:
            for size, scm_id, scm in test_scms:
                experiment_count += 1
                logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                           f"{pairing.name} on {scm_id}")
                
                result = self.run_single_trajectory(pairing, scm, scm_id)
                all_results.append(result)
        
        logger.info(f"Completed {len(all_results)} trajectory experiments")
        return all_results
    
    def analyze_trajectories(self, results: List[TrajectoryResult]) -> Dict[str, Any]:
        """
        Analyze trajectory results for insights.
        
        Args:
            results: List of trajectory results
            
        Returns:
            Dictionary with analysis results
        """
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {'error': 'No successful trajectories to analyze'}
        
        # Group by method
        by_method = {}
        for result in successful_results:
            if result.method_name not in by_method:
                by_method[result.method_name] = []
            by_method[result.method_name].append(result)
        
        # Compute method statistics
        method_stats = {}
        for method, method_results in by_method.items():
            final_values = [r.final_target_value for r in method_results]
            best_values = [r.best_target_value for r in method_results]
            convergence_rates = [r.convergence_rate for r in method_results]
            parent_accuracies = [r.parent_accuracy for r in method_results]
            
            method_stats[method] = {
                'n_trajectories': len(method_results),
                'mean_final_target': np.mean(final_values),
                'std_final_target': np.std(final_values),
                'mean_best_target': np.mean(best_values),
                'std_best_target': np.std(best_values),
                'mean_convergence_rate': np.mean(convergence_rates),
                'mean_parent_accuracy': np.mean(parent_accuracies),
                'exploration_diversity': np.mean([r.exploration_diversity for r in method_results])
            }
        
        # Ranking analysis
        method_ranking = self._rank_methods(method_stats)
        
        # Scaling analysis
        scaling_analysis = self._analyze_scaling_behavior(successful_results)
        
        return {
            'method_statistics': method_stats,
            'method_ranking': method_ranking,
            'scaling_analysis': scaling_analysis,
            'total_trajectories': len(successful_results),
            'failed_trajectories': len(results) - len(successful_results)
        }
    
    def _rank_methods(self, method_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Rank methods by different criteria."""
        rankings = {}
        
        # Rank by final target value (lower is better for minimization)
        final_target_ranking = sorted(
            method_stats.items(),
            key=lambda x: x[1]['mean_final_target']
        )
        rankings['by_final_target'] = [method for method, _ in final_target_ranking]
        
        # Rank by best target value
        best_target_ranking = sorted(
            method_stats.items(), 
            key=lambda x: x[1]['mean_best_target']
        )
        rankings['by_best_target'] = [method for method, _ in best_target_ranking]
        
        # Rank by parent accuracy (higher is better)
        parent_acc_ranking = sorted(
            method_stats.items(),
            key=lambda x: x[1]['mean_parent_accuracy'],
            reverse=True
        )
        rankings['by_parent_accuracy'] = [method for method, _ in parent_acc_ranking]
        
        # Overall composite ranking (weighted score)
        composite_scores = {}
        for method, stats in method_stats.items():
            # Normalize metrics (lower final target is better, higher parent accuracy is better)
            final_score = -stats['mean_final_target']  # Negate for minimization
            parent_score = stats['mean_parent_accuracy']
            convergence_score = stats['mean_convergence_rate']
            
            # Weighted combination
            composite_scores[method] = (0.5 * final_score + 
                                      0.3 * parent_score + 
                                      0.2 * convergence_score)
        
        composite_ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['composite'] = [method for method, _ in composite_ranking]
        rankings['composite_scores'] = composite_scores
        
        return rankings
    
    def _analyze_scaling_behavior(self, results: List[TrajectoryResult]) -> Dict[str, Any]:
        """Analyze how methods perform across different graph sizes."""
        scaling_data = {}
        
        # Group by method and size
        for result in results:
            method = result.method_name
            size = result.scm_size
            
            if method not in scaling_data:
                scaling_data[method] = {}
            if size not in scaling_data[method]:
                scaling_data[method][size] = []
            
            scaling_data[method][size].append(result)
        
        # Compute scaling statistics
        scaling_stats = {}
        for method, size_data in scaling_data.items():
            method_scaling = {}
            
            for size, size_results in size_data.items():
                final_values = [r.final_target_value for r in size_results]
                parent_accs = [r.parent_accuracy for r in size_results]
                
                method_scaling[size] = {
                    'mean_final_target': np.mean(final_values),
                    'std_final_target': np.std(final_values),
                    'mean_parent_accuracy': np.mean(parent_accs),
                    'n_trajectories': len(size_results)
                }
            
            scaling_stats[method] = method_scaling
        
        return scaling_stats
    
    def export_results(self, 
                      results: List[TrajectoryResult],
                      analysis: Dict[str, Any], 
                      output_dir: Path) -> None:
        """
        Export trajectory results and analysis.
        
        Args:
            results: Trajectory results
            analysis: Analysis results
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export raw trajectory data
        trajectory_data = []
        for result in results:
            base_data = {
                'method': result.method_name,
                'scm_size': result.scm_size,
                'scm_id': result.scm_id,
                'final_target': result.final_target_value,
                'best_target': result.best_target_value,
                'convergence_rate': result.convergence_rate,
                'parent_accuracy': result.parent_accuracy,
                'exploration_diversity': result.exploration_diversity,
                'success': result.success,
                'error': result.error_message
            }
            
            # Add intervention details
            for i, intervention in enumerate(result.interventions):
                intervention_data = base_data.copy()
                intervention_data.update({
                    'intervention_step': intervention['step'],
                    'intervention_variable': intervention['variable'],
                    'intervention_value': intervention['value'],
                    'target_value_at_step': intervention['target_value'],
                    'is_parent_intervention': intervention['is_parent']
                })
                trajectory_data.append(intervention_data)
        
        # Save as CSV
        df = pd.DataFrame(trajectory_data)
        df.to_csv(output_dir / 'trajectory_results.csv', index=False)
        
        # Save analysis results
        import json
        with open(output_dir / 'trajectory_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_trajectory_report(results, analysis, output_dir)
        
        logger.info(f"Trajectory results exported to {output_dir}")
    
    def _generate_trajectory_report(self, 
                                  results: List[TrajectoryResult],
                                  analysis: Dict[str, Any],
                                  output_dir: Path) -> None:
        """Generate human-readable trajectory report."""
        lines = ["=" * 80]
        lines.append("TARGET OPTIMIZATION TRAJECTORY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary statistics
        successful = len([r for r in results if r.success])
        total = len(results)
        lines.append(f"Total trajectories: {total}")
        lines.append(f"Successful trajectories: {successful}")
        lines.append(f"Success rate: {successful/total:.1%}")
        lines.append("")
        
        # Method performance
        if 'method_statistics' in analysis:
            lines.append("METHOD PERFORMANCE SUMMARY")
            lines.append("-" * 50)
            
            method_stats = analysis['method_statistics']
            for method, stats in method_stats.items():
                lines.append(f"\n{method}:")
                lines.append(f"  Final target: {stats['mean_final_target']:.3f} ± {stats['std_final_target']:.3f}")
                lines.append(f"  Best target: {stats['mean_best_target']:.3f} ± {stats['std_best_target']:.3f}")
                lines.append(f"  Parent accuracy: {stats['mean_parent_accuracy']:.1%}")
                lines.append(f"  Convergence rate: {stats['mean_convergence_rate']:.3f}")
                lines.append(f"  Exploration diversity: {stats['exploration_diversity']:.1%}")
        
        # Method ranking
        if 'method_ranking' in analysis:
            lines.append("\n\nMETHOD RANKINGS")
            lines.append("-" * 50)
            
            ranking = analysis['method_ranking']
            lines.append(f"By final target value: {' > '.join(ranking['by_final_target'])}")
            lines.append(f"By best target value: {' > '.join(ranking['by_best_target'])}")
            lines.append(f"By parent accuracy: {' > '.join(ranking['by_parent_accuracy'])}")
            lines.append(f"Overall composite: {' > '.join(ranking['composite'])}")
        
        # Scaling insights
        if 'scaling_analysis' in analysis:
            lines.append("\n\nSCALING BEHAVIOR")
            lines.append("-" * 50)
            
            scaling = analysis['scaling_analysis']
            for method, size_stats in scaling.items():
                lines.append(f"\n{method} scaling:")
                sizes = sorted(size_stats.keys())
                for size in sizes:
                    stats = size_stats[size]
                    lines.append(f"  Size {size}: {stats['mean_final_target']:.3f} "
                                f"(n={stats['n_trajectories']})")
        
        lines.append("\n" + "=" * 80)
        
        # Write report
        with open(output_dir / 'trajectory_report.txt', 'w') as f:
            f.write('\n'.join(lines))