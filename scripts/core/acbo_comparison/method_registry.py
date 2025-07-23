"""
Method Registry for ACBO Comparison Framework

This module provides a clean registry pattern for managing different ACBO methods.
It standardizes method interfaces and enables easy addition of new methods.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path
import sys

# Note: Proper imports should be handled via PYTHONPATH or package installation
# Avoid modifying sys.path in production code

import jax.random as random
import pyrsistent as pyr
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMethod:
    """Standardized representation of an experiment method."""
    name: str
    type: str
    description: str
    run_function: Callable
    config: Dict[str, Any]
    requires_checkpoint: bool = False
    checkpoint_path: Optional[str] = None


@dataclass
class MethodResult:
    """Standardized result format for all methods."""
    method_name: str
    method_type: str
    final_target_value: float
    target_improvement: float
    structure_accuracy: float
    sample_efficiency: float
    intervention_count: int
    convergence_steps: int
    detailed_results: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class MethodRegistry:
    """Registry for managing ACBO experiment methods."""
    
    def __init__(self):
        self.methods: Dict[str, ExperimentMethod] = {}
        self._initialize_default_methods()
    
    def register_method(self, method: ExperimentMethod) -> None:
        """Register a new experiment method."""
        if method.type in self.methods:
            logger.warning(f"Overriding existing method: {method.type}")
        
        self.methods[method.type] = method
        logger.info(f"Registered method: {method.name} (type: {method.type})")
    
    def get_method(self, method_type: str) -> ExperimentMethod:
        """Get method by type."""
        if method_type not in self.methods:
            raise ValueError(f"Unknown method type: {method_type}")
        return self.methods[method_type]
    
    def list_available_methods(self) -> List[str]:
        """List all available method types."""
        return list(self.methods.keys())
    
    def get_method_names(self) -> Dict[str, str]:
        """Get mapping of method types to human-readable names."""
        return {method_type: method.name for method_type, method in self.methods.items()}
    
    def validate_method_config(self, method_type: str, config: DictConfig) -> bool:
        """Validate configuration for a specific method."""
        method = self.get_method(method_type)
        
        # Check if checkpoint is required but not provided
        if method.requires_checkpoint:
            checkpoint_path = getattr(config, 'policy_checkpoint_path', None)
            if not checkpoint_path:
                logger.error(f"Method {method_type} requires checkpoint but none provided")
                return False
            
            if not Path(checkpoint_path).exists():
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                return False
        
        return True
    
    def run_method(self, method_type: str, scm: pyr.PMap, config, run_idx: int, scm_idx: int) -> MethodResult:
        """Run a specific method and return standardized result."""
        try:
            method = self.get_method(method_type)
            
            # Validate configuration
            if hasattr(config, 'method_type') and not self.validate_method_config(method_type, config):
                return MethodResult(
                    method_name=method.name,
                    method_type=method_type,
                    final_target_value=0.0,
                    target_improvement=0.0,
                    structure_accuracy=0.0,
                    sample_efficiency=0.0,
                    intervention_count=0,
                    convergence_steps=0,
                    detailed_results={},
                    metadata={'run_idx': run_idx, 'scm_idx': scm_idx},
                    success=False,
                    error_message="Configuration validation failed"
                )
            
            # Run the method
            result = method.run_function(scm, config, run_idx, scm_idx)
            
            if not result:
                return MethodResult(
                    method_name=method.name,
                    method_type=method_type,
                    final_target_value=0.0,
                    target_improvement=0.0,
                    structure_accuracy=0.0,
                    sample_efficiency=0.0,
                    intervention_count=0,
                    convergence_steps=0,
                    detailed_results={},
                    metadata={'run_idx': run_idx, 'scm_idx': scm_idx},
                    success=False,
                    error_message="Method returned empty result"
                )
            
            # Convert to standardized format
            return self._convert_to_standard_result(result, method, run_idx, scm_idx, scm)
            
        except Exception as e:
            logger.error(f"Method {method_type} failed: {e}")
            return MethodResult(
                method_name=method.name if method_type in self.methods else method_type,
                method_type=method_type,
                final_target_value=0.0,
                target_improvement=0.0,
                structure_accuracy=0.0,
                sample_efficiency=0.0,
                intervention_count=0,
                convergence_steps=0,
                detailed_results={},
                metadata={'run_idx': run_idx, 'scm_idx': scm_idx},
                success=False,
                error_message=str(e)
            )
    
    def _convert_to_standard_result(self, raw_result: Dict[str, Any], method: ExperimentMethod, 
                                  run_idx: int, scm_idx: int, scm: pyr.PMap = None) -> MethodResult:
        """Convert method-specific result to standardized format."""
        from causal_bayes_opt.data_structures.scm import get_target, get_parents
        
        # Extract standard metrics with fallbacks
        final_target_value = raw_result.get('final_target_value', 
                                           raw_result.get('final_best', 0.0))
        # Handle both old 'improvement' and new 'reduction' keys
        target_reduction = raw_result.get('target_reduction',
                                        raw_result.get('reduction', 
                                        raw_result.get('target_improvement',
                                        raw_result.get('improvement', 0.0))))
        
        # Compute structure accuracy if SCM is provided and final_marginal_probs exist
        structure_accuracy = raw_result.get('structure_accuracy', 0.0)
        if scm is not None and structure_accuracy == 0.0 and 'final_marginal_probs' in raw_result:
            try:
                from scripts.core.acbo_wandb_experiment import compute_structure_accuracy_from_result
                structure_accuracy = compute_structure_accuracy_from_result(raw_result, scm)
            except Exception as e:
                logger.warning(f"Failed to compute structure accuracy: {e}")
                structure_accuracy = 0.0
        sample_efficiency = raw_result.get('sample_efficiency', 0.0)
        intervention_count = raw_result.get('intervention_count', 
                                          len(raw_result.get('learning_history', [])))
        convergence_steps = raw_result.get('convergence_steps', 
                                         len(raw_result.get('target_progress', [])))
        
        # Preserve detailed results including all trajectory data
        detailed_results = raw_result.get('detailed_results', {})
        
        # Preserve all trajectory data with standardized keys
        trajectory_keys = [
            'learning_history',
            'target_progress',
            'uncertainty_progress', 
            'marginal_prob_progress',
            'data_likelihood_progress',
            'f1_scores',
            'shd_values',
            'true_parent_likelihood',
            'uncertainty_bits',
            'target_values',
            'steps'
        ]
        
        # First check detailed_results for these keys
        for key in trajectory_keys:
            if key in detailed_results:
                continue  # Already in detailed_results
            elif key in raw_result:
                detailed_results[key] = raw_result[key]
        
        # Also check for trajectory suffixed keys
        for key in list(raw_result.keys()):
            if key.endswith('_trajectory') or key.endswith('_progress'):
                if key not in detailed_results:
                    detailed_results[key] = raw_result[key]
        
        # Create metadata
        metadata = {
            'method': raw_result.get('method', method.type),
            'total_samples': raw_result.get('total_samples', 0),
            'final_uncertainty': raw_result.get('final_uncertainty', 0.0),
            'converged_to_truth': raw_result.get('converged_to_truth', False),
            'run_idx': run_idx,
            'scm_idx': scm_idx
        }
        
        # Add method-specific metadata
        if method.requires_checkpoint and 'policy_checkpoint_used' in raw_result:
            metadata['policy_checkpoint_used'] = raw_result['policy_checkpoint_used']
        
        return MethodResult(
            method_name=method.name,
            method_type=method.type,
            final_target_value=final_target_value,
            target_improvement=target_reduction,
            structure_accuracy=structure_accuracy,
            sample_efficiency=sample_efficiency,
            intervention_count=intervention_count,
            convergence_steps=convergence_steps,
            detailed_results=detailed_results,
            metadata=metadata,
            success=True
        )
    
    def _initialize_default_methods(self) -> None:
        """Initialize default ACBO methods."""
        # Import method implementations
        try:
            # Import method implementations
            from scripts.core.acbo_wandb_experiment import (
                run_random_untrained_demo,
                run_learned_enriched_policy_demo
            )
            from examples.complete_workflow_demo import (
                run_progressive_learning_demo_with_scm,
                run_progressive_learning_demo_with_oracle_interventions
            )
            
            # Register default methods
            self.register_method(ExperimentMethod(
                name="Random Policy + Untrained Model",
                type="random_untrained",
                description="Random interventions with untrained surrogate model",
                run_function=self._wrap_random_untrained,
                config={}
            ))
            
            self.register_method(ExperimentMethod(
                name="Random Policy + Learning Model",
                type="random_learning", 
                description="Random interventions with learning surrogate model",
                run_function=self._wrap_random_learning,
                config={}
            ))
            
            self.register_method(ExperimentMethod(
                name="Oracle Policy + Learning Model",
                type="oracle_learning",
                description="Oracle interventions with learning surrogate model", 
                run_function=self._wrap_oracle_learning,
                config={}
            ))
            
            self.register_method(ExperimentMethod(
                name="Learned Enriched Policy + Learning Model",
                type="learned_enriched_policy",
                description="Trained enriched policy with learning surrogate model",
                run_function=self._wrap_enriched_policy,
                config={},
                requires_checkpoint=True
            ))
            
            logger.info("Initialized default ACBO methods")
            
        except ImportError as e:
            logger.error(f"Failed to import method implementations: {e}")
    
    def _wrap_random_untrained(self, scm: pyr.PMap, config, run_idx: int, scm_idx: int) -> Dict[str, Any]:
        """Wrapper for random untrained method."""
        from acbo_wandb_experiment import run_random_untrained_demo
        from examples.demo_learning import DemoConfig
        
        acbo_config = self._create_acbo_config(config, run_idx, scm_idx)
        return run_random_untrained_demo(scm, acbo_config)
    
    def _wrap_random_learning(self, scm: pyr.PMap, config, run_idx: int, scm_idx: int) -> Dict[str, Any]:
        """Wrapper for random learning method."""
        from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
        from examples.demo_learning import DemoConfig
        
        acbo_config = self._create_acbo_config(config, run_idx, scm_idx)
        return run_progressive_learning_demo_with_scm(scm, acbo_config)
    
    def _wrap_oracle_learning(self, scm: pyr.PMap, config, run_idx: int, scm_idx: int) -> Dict[str, Any]:
        """Wrapper for oracle learning method."""
        from examples.complete_workflow_demo import run_progressive_learning_demo_with_oracle_interventions
        from examples.demo_learning import DemoConfig
        
        acbo_config = self._create_acbo_config(config, run_idx, scm_idx)
        return run_progressive_learning_demo_with_oracle_interventions(scm, acbo_config)
    
    def _wrap_enriched_policy(self, scm: pyr.PMap, config, run_idx: int, scm_idx: int) -> Dict[str, Any]:
        """Wrapper for enriched policy method."""
        from examples.complete_workflow_demo import run_progressive_learning_demo_with_custom_policy
        from examples.demo_learning import DemoConfig
        from causal_bayes_opt.acquisition.grpo_enriched_integration import (
            create_enriched_policy_intervention_function
        )
        
        acbo_config = self._create_acbo_config(config, run_idx, scm_idx)
        checkpoint_path = getattr(config, 'policy_checkpoint_path', None)
        
        if not checkpoint_path:
            raise ValueError("No policy checkpoint path configured for learned enriched policy method")
        
        # Create enriched policy intervention function
        try:
            policy_fn = create_enriched_policy_intervention_function(
                checkpoint_path=checkpoint_path,
                intervention_value_range=acbo_config.intervention_value_range
            )
            
            # Run progressive learning with the enriched policy
            result = run_progressive_learning_demo_with_custom_policy(scm, acbo_config, policy_fn)
            
            # Update method name in result
            result['method'] = 'learned_enriched_policy'
            result['policy_checkpoint_used'] = checkpoint_path
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run enriched policy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # If enriched policy fails, raise the error to be handled by the caller
            raise
    
    def _create_acbo_config(self, config, run_idx: int, scm_idx: int):
        """Create ACBO configuration from experiment config."""
        from examples.demo_learning import DemoConfig
        
        # Extract ACBO config if available, otherwise use defaults
        acbo_cfg = getattr(config, 'acbo', {})
        
        return DemoConfig(
            n_observational_samples=acbo_cfg.get('n_observational_samples', 10),
            n_intervention_steps=acbo_cfg.get('n_intervention_steps', 15),
            learning_rate=acbo_cfg.get('learning_rate', 1e-3),
            scoring_method=acbo_cfg.get('scoring_method', 'bic'),
            intervention_value_range=acbo_cfg.get('intervention_value_range', (-2.0, 2.0)),
            random_seed=acbo_cfg.get('random_seed', 42) + run_idx * 100 + scm_idx
        )