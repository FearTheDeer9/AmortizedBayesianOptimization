"""
Unified Evaluation Entry Point

This module provides the single, principled way to run evaluations in the
causal Bayesian optimization framework. It handles all evaluation types
(GRPO, BC, baselines) through a clean, consistent interface.

Usage:
    # From script with Hydra config
    results = run_evaluation(
        checkpoint_path=Path("checkpoints/grpo/checkpoint"),
        output_dir=Path("results/my_evaluation"),
        config=hydra_config
    )
    
    # From notebook with dict config
    results = run_evaluation(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config={'num_scms': 5, 'n_seeds': 3}
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import time
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from .unified_runner import UnifiedEvaluationRunner, MethodRegistry
from .grpo_evaluator import GRPOEvaluator
from .grpo_evaluator_fixed import GRPOEvaluatorFixed
from .simplified_grpo_evaluator import SimplifiedGRPOEvaluator
from .bc_evaluator import BCEvaluator
from .simplified_bc_evaluator import SimplifiedBCEvaluator
from .baseline_evaluators import (
    RandomBaselineEvaluator,
    OracleBaselineEvaluator,
    LearningBaselineEvaluator
)
from .result_types import ComparisonResults

logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    output_dir: Path = Path("./results"),
    config: Union[DictConfig, Dict[str, Any]] = None,
    test_scms: Optional[List[Any]] = None,
    methods: Optional[List[str]] = None
) -> ComparisonResults:
    """
    Run evaluation with specified configuration.
    
    This is the single entry point for all evaluation workflows. It handles:
    - GRPO checkpoint evaluation
    - BC checkpoint evaluation
    - Baseline method comparisons
    - Mixed method comparisons
    
    Args:
        checkpoint_path: Path to checkpoint (for GRPO/BC evaluation)
        output_dir: Directory to save results (will be created if needed)
        config: Configuration dict or DictConfig with evaluation parameters
        test_scms: Optional pre-generated test SCMs
        methods: Optional list of methods to evaluate (auto-detected if None)
        
    Returns:
        ComparisonResults with complete evaluation data
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If checkpoint doesn't exist
    """
    start_time = time.time()
    
    # Convert DictConfig to dict for easier handling
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    elif config is None:
        config = {}
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Extract evaluation parameters with defaults
    eval_config = {
        'experiment': config.get('experiment', {
            'target': {
                'max_interventions': 10,
                'n_observational_samples': 100,
                'intervention_value_range': (-2.0, 2.0),
                'learning_rate': 1e-3
            }
        }),
        'n_scms': config.get('n_scms', 3),
        'n_seeds': config.get('experiment', {}).get('runs_per_method', 3),
        # Check for parallel in multiple places for backward compatibility
        'parallel': config.get('parallel', config.get('performance', {}).get('parallel_execution', True))
    }
    
    # Auto-detect methods if not specified
    if methods is None:
        methods = _detect_methods(checkpoint_path, config)
    
    logger.info(f"Running evaluation with methods: {methods}")
    
    # Create runner
    runner = UnifiedEvaluationRunner(output_dir, parallel=eval_config['parallel'])
    
    # Register evaluators based on methods
    _register_evaluators(runner.registry, methods, checkpoint_path, config)
    
    # Generate or use provided test SCMs
    if test_scms is None:
        test_scms = _generate_test_scms(eval_config['n_scms'], config)
    
    # Run evaluation
    logger.info(f"Starting evaluation on {len(test_scms)} SCMs with {eval_config['n_seeds']} seeds each")
    
    results = runner.run_comparison(
        test_scms=test_scms,
        config=eval_config,
        n_runs_per_scm=eval_config['n_seeds'],
        base_seed=42
    )
    
    # Save results to standard location
    results_file = output_dir / "comparison_results.json"
    _save_results(results, results_file)
    
    # Generate visualizations if enabled
    if config.get('visualization', {}).get('enabled', True):
        _generate_visualizations(results, output_dir, config)
    
    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time/60:.1f} minutes")
    logger.info(f"Results saved to: {results_file}")
    
    return results


def _detect_methods(checkpoint_path: Optional[Path], config: Dict[str, Any]) -> List[str]:
    """Auto-detect which methods to evaluate based on inputs."""
    methods = []
    
    # Check if explicit methods are specified in config
    if 'experiment' in config and 'methods' in config['experiment']:
        # Use methods from config (for compatibility with run_acbo_comparison.py)
        method_mapping = config['experiment']['methods']
        return list(method_mapping.keys())
    
    # Otherwise, detect based on checkpoint
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        
        if 'grpo' in str(checkpoint_path).lower():
            methods.append('grpo')
            # Also add baselines for comparison
            methods.extend(['random', 'learning', 'oracle'])
        elif 'bc' in str(checkpoint_path).lower():
            # Detect BC configuration from checkpoint
            if checkpoint_path.name == 'surrogate_checkpoint':
                methods.append('bc_surrogate')
            elif checkpoint_path.name == 'acquisition_checkpoint':
                methods.append('bc_acquisition')
            else:
                methods.append('bc_both')
            # Add baselines
            methods.extend(['random', 'learning'])
    else:
        # No checkpoint - just run baselines
        methods = ['random', 'learning', 'oracle']
    
    return methods


def _register_evaluators(
    registry: MethodRegistry,
    methods: List[str],
    checkpoint_path: Optional[Path],
    config: Dict[str, Any]
) -> None:
    """Register evaluators based on method list."""
    
    # Map method names to evaluators
    method_mapping = {
        'random': lambda: RandomBaselineEvaluator(),
        'Random + Untrained': lambda: RandomBaselineEvaluator(name="Random + Untrained"),
        'oracle': lambda: OracleBaselineEvaluator(),
        'Oracle + Learning': lambda: OracleBaselineEvaluator(name="Oracle + Learning"),
        'learning': lambda: LearningBaselineEvaluator(),
        'Random + Learning': lambda: LearningBaselineEvaluator(name="Random + Learning"),
    }
    
    # Register simple methods
    for method in methods:
        if method in method_mapping:
            evaluator = method_mapping[method]()
            registry.register(evaluator)
            logger.info(f"Registered {method} evaluator")
    
    # Handle checkpoint-based methods
    if checkpoint_path and 'grpo' in methods:
        # Check if this is a simplified trainer checkpoint
        use_simplified = _is_simplified_checkpoint(checkpoint_path)
        
        if use_simplified:
            evaluator = SimplifiedGRPOEvaluator(checkpoint_path)
            logger.info(f"Using SIMPLIFIED GRPO evaluator for new checkpoint format")
        else:
            # Check if we should use the fixed evaluator
            use_fixed_evaluator = config.get('use_fixed_grpo_evaluator', True)
            
            if use_fixed_evaluator:
                evaluator = GRPOEvaluatorFixed(checkpoint_path)
                logger.info(f"Using FIXED GRPO evaluator with proper surrogate integration")
            else:
                evaluator = GRPOEvaluator(checkpoint_path)
                logger.info(f"Using original GRPO evaluator (no surrogate integration)")
            
        registry.register(evaluator)
        logger.info(f"Registered GRPO evaluator with checkpoint: {checkpoint_path}")
    
    if checkpoint_path and any(m.startswith('bc_') for m in methods):
        # Check if using simplified checkpoints
        use_simplified = _is_simplified_checkpoint(checkpoint_path)
        
        # Handle BC variants
        if 'bc_surrogate' in methods:
            if use_simplified:
                evaluator = SimplifiedBCEvaluator(
                    surrogate_checkpoint=checkpoint_path,
                    name="BC_Surrogate"
                )
                logger.info(f"Using SIMPLIFIED BC surrogate evaluator")
            else:
                evaluator = BCEvaluator(
                    surrogate_checkpoint=checkpoint_path,
                    name="BC_Surrogate_Random"
                )
            registry.register(evaluator)
        elif 'bc_acquisition' in methods:
            if use_simplified:
                evaluator = SimplifiedBCEvaluator(
                    acquisition_checkpoint=checkpoint_path,
                    name="BC_Acquisition"
                )
                logger.info(f"Using SIMPLIFIED BC acquisition evaluator")
            else:
                evaluator = BCEvaluator(
                    acquisition_checkpoint=checkpoint_path,
                    name="BC_Acquisition_Learning"
                )
            registry.register(evaluator)
        elif 'bc_both' in methods:
            # Expect both checkpoints in config
            surrogate_path = config.get('bc_surrogate_checkpoint', checkpoint_path)
            acquisition_path = config.get('bc_acquisition_checkpoint', checkpoint_path)
            
            if use_simplified or (_is_simplified_checkpoint(Path(surrogate_path)) if surrogate_path else False):
                evaluator = SimplifiedBCEvaluator(
                    surrogate_checkpoint=Path(surrogate_path) if surrogate_path else None,
                    acquisition_checkpoint=Path(acquisition_path) if acquisition_path else None,
                    name="BC_Combined"
                )
                logger.info(f"Using SIMPLIFIED BC combined evaluator")
            else:
                evaluator = BCEvaluator(
                    surrogate_checkpoint=Path(surrogate_path),
                    acquisition_checkpoint=Path(acquisition_path),
                    name="BC_Both"
                )
            registry.register(evaluator)
    
    # Special handling for "Trained Policy + Learning" from ACBO comparison
    if "Trained Policy + Learning" in methods:
        if checkpoint_path:
            use_simplified = _is_simplified_checkpoint(checkpoint_path)
            
            if use_simplified:
                evaluator = SimplifiedGRPOEvaluator(checkpoint_path, name="Trained Policy + Learning")
                logger.info(f"Using SIMPLIFIED Trained Policy evaluator")
            else:
                use_fixed_evaluator = config.get('use_fixed_grpo_evaluator', True)
                
                if use_fixed_evaluator:
                    evaluator = GRPOEvaluatorFixed(checkpoint_path, name="Trained Policy + Learning")
                    logger.info(f"Using FIXED Trained Policy evaluator with proper surrogate integration")
                else:
                    evaluator = GRPOEvaluator(checkpoint_path, name="Trained Policy + Learning")
                
            registry.register(evaluator)
            logger.info(f"Registered Trained Policy evaluator with checkpoint: {checkpoint_path}")


def _is_simplified_checkpoint(checkpoint_path: Path) -> bool:
    """
    Detect if a checkpoint is from the simplified trainers.
    
    Simplified checkpoints are identified by:
    - Being a .pkl file (not a directory)
    - Containing specific keys in the pickle
    - File naming patterns
    """
    import pickle
    
    checkpoint_path = Path(checkpoint_path)
    
    # If it's a .pkl file, check its contents
    if checkpoint_path.suffix == '.pkl' and checkpoint_path.is_file():
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check for simplified trainer markers
            if 'trainer_type' in data and 'simplified' in data['trainer_type']:
                return True
                
            # Check for specific config structure
            if 'config' in data and isinstance(data['config'], dict):
                # Simplified trainers have flat config
                config = data['config']
                if 'architecture_level' in config and 'learning_rate' in config:
                    # Simplified format has these at top level
                    return True
                    
            # Check for model_type (BC simplified format)
            if 'model_type' in data and data['model_type'] in ['surrogate', 'acquisition']:
                return True
                
        except Exception:
            # If we can't load it, assume it's not simplified
            pass
            
    # Check file name patterns
    if 'simplified' in str(checkpoint_path).lower():
        return True
        
    if checkpoint_path.name in ['grpo_final.pkl', 'surrogate_final.pkl', 'acquisition_final.pkl']:
        # These are typical names from simplified trainers
        return True
        
    return False


def _generate_test_scms(n_scms: int, config: Dict[str, Any]) -> List[Any]:
    """Generate test SCMs based on configuration."""
    from ..experiments.variable_scm_factory import VariableSCMFactory
    
    # Extract SCM generation config
    scm_config = config.get('experiment', {}).get('scm_generation', {})
    
    # Use variable structure generation if specified
    if scm_config.get('use_variable_factory', False):
        variable_range = scm_config.get('variable_range', [3, 6])
        structure_types = scm_config.get('structure_types', ['fork', 'chain', 'collider'])
        
        # Generate variable structure SCMs
        factory = VariableSCMFactory(
            noise_scale=0.5,
            coefficient_range=(-2.0, 2.0),
            seed=config.get('seed', 42)
        )
        
        scms = []
        scm_idx = 0
        while len(scms) < n_scms:
            for structure_type in structure_types:
                for n_vars in range(variable_range[0], variable_range[1] + 1):
                    if len(scms) >= n_scms:
                        break
                    scm = factory.create_variable_scm(
                        num_variables=n_vars,
                        structure_type=structure_type,
                        target_variable=None
                    )
                    scms.append(scm)
                    scm_idx += 1
        logger.info(f"Generated {len(scms)} test SCMs with variable structure")
        return scms
    else:
        # Use default generation
        from examples.demo_scms import create_easy_scm_base, create_medium_scm, create_hard_scm
        
        # Simple rotation through difficulty levels
        scm_generators = [create_easy_scm_base, create_medium_scm, create_hard_scm]
        scms = []
        for i in range(n_scms):
            scm = scm_generators[i % len(scm_generators)]()
            scms.append(scm)
        
        logger.info(f"Generated {len(scms)} test SCMs with fixed structure")
        return scms


def _save_results(results: ComparisonResults, results_file: Path) -> None:
    """Save results to JSON file with proper serialization."""
    
    # Convert results to serializable format
    results_dict = {
        'config': results.config,
        'method_results': {},
        'raw_results': {},
        'statistical_tests': getattr(results, 'statistical_tests', {}),
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert method results
    for method_name, method_metrics in results.method_metrics.items():
        results_dict['method_results'][method_name] = {
            'mean_improvement': method_metrics.mean_improvement,
            'std_improvement': method_metrics.std_improvement,
            'mean_final_value': method_metrics.mean_final_value,
            'std_final_value': method_metrics.std_final_value,
            'mean_steps': method_metrics.mean_steps,
            'mean_time': method_metrics.mean_time,
            'n_runs': method_metrics.n_runs,
            'n_successful': method_metrics.n_successful
        }
    
    # Add SCM metadata
    results_dict['scm_metadata'] = results.scm_metadata
    
    # Add execution metadata for compatibility
    results_dict['execution_metadata'] = {
        'methods_tested': len(results.method_metrics),
        'scms_tested': results.config.get('n_scms', 0),
        'runs_per_method': results.config.get('n_seeds', 0),
        'total_experiments': len(results.method_metrics) * results.config.get('n_scms', 0) * results.config.get('n_seeds', 0),
        'total_time': sum(m.mean_time for m in results.method_metrics.values())
    }
    
    # Save with proper formatting
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")


def _generate_visualizations(
    results: ComparisonResults,
    output_dir: Path,
    config: Dict[str, Any]
) -> None:
    """Generate visualization plots based on configuration."""
    try:
        from ..visualization.plots import plot_baseline_comparison
        from .notebook_helpers import plot_learning_curves, results_to_dataframe
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate trajectory comparison
        if 'target_trajectory' in config.get('visualization', {}).get('plot_types', []):
            try:
                # Get learning curves data
                curves_data = results.get_learning_curves()
                
                # Generate trajectory plots
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot 1: Target value trajectory
                for method_name, data in curves_data.items():
                    if 'target_mean' in data and data['target_mean']:
                        steps = data.get('steps', list(range(len(data['target_mean']))))
                        ax1.plot(steps, data['target_mean'], label=method_name, marker='o')
                        if 'target_std' in data and data['target_std']:
                            ax1.fill_between(steps,
                                           np.array(data['target_mean']) - np.array(data['target_std']),
                                           np.array(data['target_mean']) + np.array(data['target_std']),
                                           alpha=0.2)
                
                ax1.set_xlabel('Intervention Step')
                ax1.set_ylabel('Target Value')
                ax1.set_title('Target Value Optimization Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: F1 Score trajectory
                has_f1_data = False
                for method_name, data in curves_data.items():
                    if 'f1_mean' in data and data['f1_mean']:
                        has_f1_data = True
                        steps = data.get('steps', list(range(len(data['f1_mean']))))
                        ax2.plot(steps, data['f1_mean'], label=method_name, marker='o')
                        if 'f1_std' in data and data['f1_std']:
                            ax2.fill_between(steps,
                                           np.array(data['f1_mean']) - np.array(data['f1_std']),
                                           np.array(data['f1_mean']) + np.array(data['f1_std']),
                                           alpha=0.2)
                
                if has_f1_data:
                    ax2.set_xlabel('Intervention Step')
                    ax2.set_ylabel('F1 Score')
                    ax2.set_title('Structure Learning (F1) Over Time')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(-0.1, 1.1)
                else:
                    ax2.text(0.5, 0.5, 'F1 trajectory data not available', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Structure Learning (F1) Over Time')
                
                # Plot 3: SHD trajectory
                has_shd_data = False
                for method_name, data in curves_data.items():
                    if 'shd_mean' in data and data['shd_mean']:
                        has_shd_data = True
                        steps = data.get('steps', list(range(len(data['shd_mean']))))
                        ax3.plot(steps, data['shd_mean'], label=method_name, marker='o')
                        if 'shd_std' in data and data['shd_std']:
                            ax3.fill_between(steps,
                                           np.array(data['shd_mean']) - np.array(data['shd_std']),
                                           np.array(data['shd_mean']) + np.array(data['shd_std']),
                                           alpha=0.2)
                
                if has_shd_data:
                    ax3.set_xlabel('Intervention Step')
                    ax3.set_ylabel('SHD')
                    ax3.set_title('Structural Hamming Distance Over Time')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'SHD trajectory data not available', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Structural Hamming Distance Over Time')
                
                plt.tight_layout()
                trajectory_path = plots_dir / "trajectory_comparison.png"
                plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Generated trajectory plot: {trajectory_path}")
                
            except Exception as e:
                logger.warning(f"Could not generate target trajectory plot: {e}")
        
        # Generate other visualizations as configured
        plot_types = config.get('visualization', {}).get('plot_types', [])
        
        if 'method_comparison' in plot_types:
            # Convert results to dataframe and create comparison plot
            df = results_to_dataframe(results)
            logger.info(f"Method comparison data:\n{df}")
        
        logger.info(f"Generated visualizations in: {plots_dir}")
        
    except ImportError:
        logger.warning("Visualization modules not available")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")