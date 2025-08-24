"""
Unified experiment runner with systematic policy+surrogate pairing support.

This module provides a principled framework for running experiments with
arbitrary policy+surrogate combinations while collecting both optimization
and structure learning metrics.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add paths for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.core.metric_collector import MetricCollector
from experiments.evaluation.core.pairing_manager import PairingConfig, ModelType
from experiments.evaluation.initial_comparison.src.graph_metrics import evaluate_graph_discovery

from src.causal_bayes_opt.data_structures.scm import create_scm
from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
import pyrsistent as pyr
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention

logger = logging.getLogger(__name__)


@dataclass
class UnifiedExperimentConfig:
    """Configuration for unified experiments."""
    scm_sizes: List[int]
    n_scms_per_size: int
    n_observational_samples: int
    n_interventions: int
    structure_types: List[str]
    edge_density: float
    seed: int
    optimization_metrics: List[str]
    structure_learning_metrics: List[str]


@dataclass
class ExperimentResult:
    """Comprehensive result from a single experiment."""
    pairing_name: str
    scm_size: int
    scm_id: str
    
    # Optimization metrics
    final_target_value: float
    best_target_value: float
    convergence_rate: float
    cumulative_regret: float
    sample_efficiency: float
    
    # Structure learning metrics
    parent_f1_score: float
    parent_precision: float
    parent_recall: float
    structural_hamming_distance: float
    posterior_calibration_error: float
    
    # Trajectory data
    intervention_trajectory: List[Dict[str, Any]]
    posterior_trajectory: List[Dict[str, Any]]
    
    # Metadata
    success: bool
    execution_time: float
    error_message: Optional[str] = None


class UnifiedExperimentRunner:
    """
    Unified experiment runner with systematic policy+surrogate pairing support.
    
    This runner provides a principled framework for evaluating arbitrary
    policy+surrogate combinations while collecting comprehensive metrics.
    """
    
    def __init__(self, config: UnifiedExperimentConfig):
        """
        Initialize unified experiment runner.
        
        Args:
            config: Unified experiment configuration
        """
        self.config = config
        self.seed = config.seed
        
        # Initialize metric collectors for dual metrics
        self.optimization_collector = MetricCollector(config.optimization_metrics)
        self.structure_collector = MetricCollector(config.structure_learning_metrics)
        
        self.results = []
        
        logger.info(f"Initialized UnifiedExperimentRunner")
        logger.info(f"  Optimization metrics: {config.optimization_metrics}")
        logger.info(f"  Structure learning metrics: {config.structure_learning_metrics}")
    
    def generate_test_scms(self) -> List[Tuple[int, str, Any]]:
        """
        Generate test SCMs for evaluation using proper pyr.pmap structures.
        
        Returns:
            List of (size, scm_id, scm) tuples
        """
        import jax.random as jrandom
        rng_key = jrandom.PRNGKey(self.seed)
        scms = []
        
        for size in self.config.scm_sizes:
            for i in range(self.config.n_scms_per_size):
                # Rotate through structure types
                structure_idx = i % len(self.config.structure_types)
                structure_type = self.config.structure_types[structure_idx]
                
                rng_key, scm_key = jrandom.split(rng_key)
                scm = self._create_proper_scm(scm_key, size, structure_type)
                
                scm_id = f"scm_{size}_{structure_type}_{i}"
                scms.append((size, scm_id, scm))
        
        logger.info(f"Generated {len(scms)} test SCMs")
        return scms
    
    def _create_proper_scm(self, rng_key, num_vars: int, structure_type: str):
        """Create SCM using proper pyr.pmap structure like in working experiments."""
        import jax.random as jrandom
        
        # Generate variable names
        variables = frozenset([f'X{i}' for i in range(num_vars)])
        target_var = f'X{num_vars-1}'  # Last variable as target
        
        # Generate edges based on structure type
        edges = set()
        if structure_type == "random":
            # Random edges with specified density
            for i in range(num_vars-1):
                for j in range(i+1, num_vars):
                    edge_key = jrandom.split(rng_key)[0]
                    if jrandom.uniform(edge_key) < self.config.edge_density:
                        edges.add((f'X{i}', f'X{j}'))
        elif structure_type == "chain":
            # Chain structure
            for i in range(num_vars-1):
                edges.add((f'X{i}', f'X{i+1}'))
        
        edges = frozenset(edges)
        
        # Create mechanisms for all variables
        mechanisms = {}
        
        # First create root mechanisms for all variables
        for var in variables:
            mechanisms[var] = create_linear_mechanism(
                parents=[], 
                coefficients={}, 
                intercept=0.0, 
                noise_scale=0.1
            )
        
        # Then override with edge-based mechanisms
        for edge in edges:
            parent, child = edge
            mechanisms[child] = create_linear_mechanism(
                parents=[parent],
                coefficients={parent: np.random.normal(0, 1)},
                intercept=0.0,
                noise_scale=0.1
            )
        
        # Create SCM using proper interface
        scm = create_scm(
            variables=variables,
            edges=edges,
            mechanisms=pyr.pmap(mechanisms),
            target=target_var,
            metadata=pyr.pmap({
                'structure': structure_type,
                'edge_density': self.config.edge_density,
                'num_variables': num_vars
            })
        )
        
        return scm
    
    def load_pairing_models(self, 
                           pairing: PairingConfig,
                           variables: List[str],
                           target_var: str,
                           scm: Dict[str, Any]) -> Tuple[Callable, Optional[Callable]]:
        """
        Load models based on pairing configuration.
        
        Args:
            pairing: Pairing configuration
            variables: List of variable names
            target_var: Target variable name
            scm: SCM (needed for oracle)
            
        Returns:
            Tuple of (acquisition_function, surrogate_function)
        """
        # Load policy/acquisition function
        if pairing.policy_spec.model_type == ModelType.TRAINED:
            acquisition_fn = ModelLoader.load_policy(
                pairing.policy_spec.checkpoint_path,
                seed=pairing.policy_spec.seed
            )
        elif pairing.policy_spec.model_type == ModelType.UNTRAINED:
            acquisition_fn = ModelLoader.create_untrained_policy(
                architecture=pairing.policy_spec.architecture or 'simple_permutation_invariant',
                hidden_dim=pairing.policy_spec.hidden_dim or 256,
                seed=pairing.policy_spec.seed
            )
        elif pairing.policy_spec.model_type == ModelType.RANDOM:
            acquisition_fn = ModelLoader.load_baseline('random', seed=pairing.policy_spec.seed)
        elif pairing.policy_spec.model_type == ModelType.ORACLE:
            acquisition_fn = ModelLoader.load_baseline('oracle', scm=scm)
        else:
            raise ValueError(f"Unknown policy type: {pairing.policy_spec.model_type}")
        
        # Load surrogate function
        surrogate_fn = None
        if pairing.surrogate_spec.model_type == ModelType.TRAINED:
            surrogate_params, surrogate_arch = ModelLoader.load_surrogate(
                pairing.surrogate_spec.checkpoint_path
            )
            # Create surrogate inference function
            surrogate_fn = self._create_surrogate_inference_fn(surrogate_params, surrogate_arch)
            
        elif pairing.surrogate_spec.model_type == ModelType.UNTRAINED:
            surrogate_model, surrogate_config = ModelLoader.create_untrained_surrogate(
                variables=variables,
                target_variable=target_var,
                seed=pairing.surrogate_spec.seed
            )
            surrogate_fn = self._create_surrogate_inference_fn(surrogate_model, surrogate_config)
            
        # Note: ModelType.NONE means no surrogate (surrogate_fn remains None)
        
        return acquisition_fn, surrogate_fn
    
    def _create_surrogate_inference_fn(self, surrogate_model, surrogate_config) -> Callable:
        """
        Create surrogate inference function from model and config.
        
        Args:
            surrogate_model: Surrogate model parameters or function
            surrogate_config: Surrogate configuration
            
        Returns:
            Function that takes (buffer, target_var) and returns posterior predictions
        """
        def surrogate_inference(buffer: ExperienceBuffer, target_var: str) -> Dict[str, np.ndarray]:
            """
            Run surrogate inference to get posterior predictions.
            
            Args:
                buffer: Current experience buffer
                target_var: Target variable name
                
            Returns:
                Dictionary with posterior predictions (parent probabilities, etc.)
            """
            # TODO: This needs actual surrogate inference implementation
            # For now, return dummy predictions for framework testing
            
            # Get variable coverage from buffer
            variable_coverage = buffer.get_variable_coverage()
            variables = list(variable_coverage.keys())
            n_vars = len(variables)
            
            # Create dummy parent probability predictions
            parent_probs = {}
            for var in variables:
                # Random probabilities for testing (replace with actual inference)
                parent_probs[var] = np.random.random(n_vars)
                parent_probs[var][variables.index(var)] = 0.0  # No self-loops
            
            return {
                'parent_probabilities': parent_probs,
                'confidence_scores': {var: np.random.random() for var in variables},
                'prediction_metadata': {
                    'model_type': 'surrogate',
                    'inference_time': 0.001  # Placeholder
                }
            }
        
        return surrogate_inference
    
    def run_single_experiment(self,
                            pairing: PairingConfig,
                            scm: Dict[str, Any],
                            scm_id: str) -> ExperimentResult:
        """
        Run a single experiment with comprehensive dual metric collection.
        
        Args:
            pairing: Model pairing configuration
            scm: SCM to evaluate on
            scm_id: SCM identifier
            
        Returns:
            ExperimentResult with dual metrics
        """
        start_time = time.time()
        
        try:
            # Setup
            variables = list(get_variables(scm))
            target_var = get_target(scm)
            true_parents = {var: get_parents(scm, var) for var in variables}
            
            logger.debug(f"Running {pairing.name} on {scm_id} "
                        f"({len(variables)} vars, target: {target_var})")
            
            # Load models based on pairing
            acquisition_fn, surrogate_fn = self.load_pairing_models(
                pairing, variables, target_var, scm
            )
            
            # Initialize buffer with observational data
            buffer = ExperienceBuffer()
            obs_samples = sample_from_linear_scm(
                scm, 
                n_samples=self.config.n_observational_samples,
                seed=42
            )
            for sample in obs_samples:
                buffer.add_observation(sample)
            
            # Track trajectories
            intervention_trajectory = []
            posterior_trajectory = []
            target_values = []
            parent_interventions = []
            
            # Run interventions with dual metric collection
            for step in range(self.config.n_interventions):
                # Convert buffer to tensor
                tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
                
                # Get surrogate predictions if available
                posterior_predictions = None
                if surrogate_fn is not None:
                    posterior_predictions = surrogate_fn(buffer, target_var)
                    posterior_trajectory.append({
                        'step': step,
                        'predictions': posterior_predictions
                    })
                
                # Get policy intervention
                intervention_dict = acquisition_fn(tensor, posterior_predictions, target_var, variables)
                
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
                    
                    # Record intervention
                    target_value = float(post_sample['values'][target_var])
                    is_parent = selected_var in true_parents[target_var]
                    
                    intervention_record = {
                        'step': step,
                        'variable': selected_var,
                        'value': float(intervention_value),
                        'target_value': target_value,
                        'is_parent': is_parent,
                        'has_surrogate_prediction': posterior_predictions is not None
                    }
                    
                    intervention_trajectory.append(intervention_record)
                    target_values.append(target_value)
                    parent_interventions.append(is_parent)
            
            # Compute optimization metrics
            final_target = target_values[-1] if target_values else 0.0
            best_target = min(target_values) if target_values else 0.0
            
            # Convergence rate
            if len(target_values) >= 2:
                improvements = [target_values[i] - target_values[i+1] 
                              for i in range(len(target_values)-1)]
                positive_improvements = [imp for imp in improvements if imp > 0]
                convergence_rate = np.mean(positive_improvements) if positive_improvements else 0.0
            else:
                convergence_rate = 0.0
            
            # Sample efficiency (interventions to reach threshold)
            threshold = -5.0  # Standard threshold
            sample_efficiency = len(target_values)  # Default to all interventions
            for i, val in enumerate(target_values):
                if val <= threshold:
                    sample_efficiency = i + 1
                    break
            
            # Cumulative regret (assuming optimal is known)
            optimal_value = min(target_values) if target_values else 0.0
            cumulative_regret = sum(val - optimal_value for val in target_values)
            
            # Compute structure learning metrics
            parent_f1, parent_precision, parent_recall = 0.0, 0.0, 0.0
            shd = 0.0
            calibration_error = 0.0
            
            if posterior_trajectory and posterior_trajectory[-1]['predictions']:
                # Use final posterior predictions for structure metrics
                final_predictions = posterior_trajectory[-1]['predictions']
                parent_probs = final_predictions.get('parent_probabilities', {})
                
                if parent_probs:
                    # Compute structure learning metrics
                    graph_metrics = evaluate_graph_discovery(
                        true_parents, parent_probs, variables, threshold=0.5
                    )
                    
                    parent_f1 = graph_metrics.get('f1', 0.0)
                    parent_precision = graph_metrics.get('precision', 0.0)
                    parent_recall = graph_metrics.get('recall', 0.0)
                    shd = graph_metrics.get('shd', 0.0)
                    
                    # TODO: Implement posterior calibration error
                    calibration_error = 0.0
            
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                pairing_name=pairing.name,
                scm_size=len(variables),
                scm_id=scm_id,
                
                # Optimization metrics
                final_target_value=final_target,
                best_target_value=best_target,
                convergence_rate=convergence_rate,
                cumulative_regret=cumulative_regret,
                sample_efficiency=sample_efficiency,
                
                # Structure learning metrics
                parent_f1_score=parent_f1,
                parent_precision=parent_precision,
                parent_recall=parent_recall,
                structural_hamming_distance=shd,
                posterior_calibration_error=calibration_error,
                
                # Trajectory data
                intervention_trajectory=intervention_trajectory,
                posterior_trajectory=posterior_trajectory,
                
                # Metadata
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Experiment failed for {pairing.name} on {scm_id}: {e}")
            
            return ExperimentResult(
                pairing_name=pairing.name,
                scm_size=len(get_variables(scm)) if scm else 0,
                scm_id=scm_id,
                
                # Zero metrics for failed experiments
                final_target_value=0.0,
                best_target_value=0.0,
                convergence_rate=0.0,
                cumulative_regret=0.0,
                sample_efficiency=0.0,
                parent_f1_score=0.0,
                parent_precision=0.0,
                parent_recall=0.0,
                structural_hamming_distance=0.0,
                posterior_calibration_error=0.0,
                
                intervention_trajectory=[],
                posterior_trajectory=[],
                
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def run_all_experiments(self, 
                          pairings: List[PairingConfig]) -> List[ExperimentResult]:
        """
        Run experiments for all pairings on all SCMs.
        
        Args:
            pairings: List of pairing configurations to evaluate
            
        Returns:
            List of experiment results
        """
        # Generate test SCMs
        test_scms = self.generate_test_scms()
        total_experiments = len(pairings) * len(test_scms)
        
        logger.info(f"Running {total_experiments} experiments "
                   f"({len(pairings)} pairings × {len(test_scms)} SCMs)")
        
        all_results = []
        experiment_count = 0
        
        for pairing in pairings:
            logger.info(f"Evaluating pairing: {pairing.name}")
            
            for size, scm_id, scm in test_scms:
                experiment_count += 1
                logger.debug(f"Experiment {experiment_count}/{total_experiments}: "
                           f"{pairing.name} on {scm_id}")
                
                result = self.run_single_experiment(pairing, scm, scm_id)
                all_results.append(result)
        
        logger.info(f"Completed {len(all_results)} experiments")
        return all_results
    
    def analyze_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Analyze experiment results for both optimization and structure learning.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary with comprehensive analysis
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        # Group by pairing
        by_pairing = {}
        for result in successful_results:
            if result.pairing_name not in by_pairing:
                by_pairing[result.pairing_name] = []
            by_pairing[result.pairing_name].append(result)
        
        # Compute pairing statistics
        pairing_stats = {}
        for pairing_name, pairing_results in by_pairing.items():
            # Optimization metrics
            opt_metrics = {
                'mean_final_target': np.mean([r.final_target_value for r in pairing_results]),
                'std_final_target': np.std([r.final_target_value for r in pairing_results]),
                'mean_best_target': np.mean([r.best_target_value for r in pairing_results]),
                'mean_convergence_rate': np.mean([r.convergence_rate for r in pairing_results]),
                'mean_sample_efficiency': np.mean([r.sample_efficiency for r in pairing_results]),
                'mean_cumulative_regret': np.mean([r.cumulative_regret for r in pairing_results])
            }
            
            # Structure learning metrics
            struct_metrics = {
                'mean_parent_f1': np.mean([r.parent_f1_score for r in pairing_results]),
                'std_parent_f1': np.std([r.parent_f1_score for r in pairing_results]),
                'mean_parent_precision': np.mean([r.parent_precision for r in pairing_results]),
                'mean_parent_recall': np.mean([r.parent_recall for r in pairing_results]),
                'mean_shd': np.mean([r.structural_hamming_distance for r in pairing_results]),
                'mean_calibration_error': np.mean([r.posterior_calibration_error for r in pairing_results])
            }
            
            # Combined metrics
            pairing_stats[pairing_name] = {
                **opt_metrics,
                **struct_metrics,
                'n_experiments': len(pairing_results)
            }
        
        # Component contribution analysis
        component_analysis = self._analyze_component_contributions(pairing_stats)
        
        # Scaling analysis
        scaling_analysis = self._analyze_scaling_behavior(successful_results)
        
        return {
            'pairing_statistics': pairing_stats,
            'component_analysis': component_analysis,
            'scaling_analysis': scaling_analysis,
            'total_experiments': len(successful_results),
            'failed_experiments': len(results) - len(successful_results)
        }
    
    def _analyze_component_contributions(self, pairing_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze relative contributions of policy vs surrogate components.
        
        This compares pairings to understand which component contributes more
        to optimization vs structure learning performance.
        """
        analysis = {
            'policy_contribution': {},
            'surrogate_contribution': {},
            'interaction_effects': {}
        }
        
        # TODO: Implement systematic component contribution analysis
        # This would compare things like:
        # - Trained Policy + Random Surrogate vs Random Policy + Trained Surrogate
        # - Same policy with different surrogates
        # - Same surrogate with different policies
        
        return analysis
    
    def _analyze_scaling_behavior(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze how pairings perform across different graph sizes."""
        scaling_data = {}
        
        # Group by pairing and size
        for result in results:
            pairing = result.pairing_name
            size = result.scm_size
            
            if pairing not in scaling_data:
                scaling_data[pairing] = {}
            if size not in scaling_data[pairing]:
                scaling_data[pairing][size] = []
            
            scaling_data[pairing][size].append(result)
        
        # Compute scaling statistics
        scaling_stats = {}
        for pairing, size_data in scaling_data.items():
            pairing_scaling = {}
            
            for size, size_results in size_data.items():
                # Optimization metrics by size
                final_targets = [r.final_target_value for r in size_results]
                parent_f1s = [r.parent_f1_score for r in size_results]
                
                pairing_scaling[size] = {
                    'mean_final_target': np.mean(final_targets),
                    'std_final_target': np.std(final_targets),
                    'mean_parent_f1': np.mean(parent_f1s),
                    'std_parent_f1': np.std(parent_f1s),
                    'n_experiments': len(size_results)
                }
            
            scaling_stats[pairing] = pairing_scaling
        
        return scaling_stats
    
    def export_results(self, 
                      results: List[ExperimentResult],
                      analysis: Dict[str, Any],
                      output_dir: Path) -> None:
        """
        Export comprehensive results with dual metrics.
        
        Args:
            results: Experiment results
            analysis: Analysis results
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export raw results
        results_data = []
        for result in results:
            result_dict = {
                'pairing_name': result.pairing_name,
                'scm_size': result.scm_size,
                'scm_id': result.scm_id,
                
                # Optimization metrics
                'final_target_value': result.final_target_value,
                'best_target_value': result.best_target_value,
                'convergence_rate': result.convergence_rate,
                'cumulative_regret': result.cumulative_regret,
                'sample_efficiency': result.sample_efficiency,
                
                # Structure learning metrics
                'parent_f1_score': result.parent_f1_score,
                'parent_precision': result.parent_precision,
                'parent_recall': result.parent_recall,
                'structural_hamming_distance': result.structural_hamming_distance,
                'posterior_calibration_error': result.posterior_calibration_error,
                
                # Metadata
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            }
            results_data.append(result_dict)
        
        # Save as CSV
        df = pd.DataFrame(results_data)
        df.to_csv(output_dir / 'unified_results.csv', index=False)
        
        # Save analysis
        import json
        with open(output_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate dual metric report
        self._generate_dual_metric_report(results, analysis, output_dir)
        
        logger.info(f"Unified experiment results exported to {output_dir}")
    
    def _generate_dual_metric_report(self,
                                   results: List[ExperimentResult],
                                   analysis: Dict[str, Any],
                                   output_dir: Path) -> None:
        """Generate comprehensive report covering both metric families."""
        lines = ["=" * 80]
        lines.append("UNIFIED DUAL-METRIC EXPERIMENT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary statistics
        successful = len([r for r in results if r.success])
        total = len(results)
        lines.append(f"Total experiments: {total}")
        lines.append(f"Successful experiments: {successful}")
        lines.append(f"Success rate: {successful/total:.1%}")
        lines.append("")
        
        # Pairing performance (dual metrics)
        if 'pairing_statistics' in analysis:
            lines.append("PAIRING PERFORMANCE (DUAL METRICS)")
            lines.append("-" * 60)
            
            pairing_stats = analysis['pairing_statistics']
            for pairing, stats in pairing_stats.items():
                lines.append(f"\n{pairing}:")
                lines.append("  Optimization Performance:")
                lines.append(f"    Final target: {stats['mean_final_target']:.3f} ± {stats['std_final_target']:.3f}")
                lines.append(f"    Best target: {stats['mean_best_target']:.3f}")
                lines.append(f"    Convergence rate: {stats['mean_convergence_rate']:.3f}")
                lines.append(f"    Sample efficiency: {stats['mean_sample_efficiency']:.1f} interventions")
                
                lines.append("  Structure Learning Performance:")
                lines.append(f"    Parent F1: {stats['mean_parent_f1']:.3f} ± {stats['std_parent_f1']:.3f}")
                lines.append(f"    Parent precision: {stats['mean_parent_precision']:.3f}")
                lines.append(f"    Parent recall: {stats['mean_parent_recall']:.3f}")
                lines.append(f"    SHD: {stats['mean_shd']:.1f}")
        
        # Scaling behavior
        if 'scaling_analysis' in analysis:
            lines.append("\n\nSCALING BEHAVIOR")
            lines.append("-" * 60)
            
            scaling = analysis['scaling_analysis']
            for pairing, size_stats in scaling.items():
                lines.append(f"\n{pairing} scaling:")
                sizes = sorted(size_stats.keys())
                for size in sizes:
                    stats = size_stats[size]
                    lines.append(f"  Size {size}: opt={stats['mean_final_target']:.3f}, "
                                f"struct_f1={stats['mean_parent_f1']:.3f}")
        
        lines.append("\n" + "=" * 80)
        
        # Write report
        with open(output_dir / 'dual_metric_report.txt', 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Dual metric report saved to {output_dir / 'dual_metric_report.txt'}")