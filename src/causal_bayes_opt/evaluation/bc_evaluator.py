"""
BC Evaluator Adapter

Adapts Behavioral Cloning models to the unified evaluation interface.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp

from .base_evaluator import BaseEvaluator
from .result_types import ExperimentResult, StepResult
from ..data_structures.scm import (
    get_target, get_variables, get_parents
)
from ..acquisition import create_acquisition_state, update_state_with_intervention
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)

logger = logging.getLogger(__name__)


class BCEvaluator(BaseEvaluator):
    """
    Evaluator for BC (Behavioral Cloning) methods.
    
    This adapter loads trained BC surrogate and acquisition models and
    evaluates them following the unified evaluation interface.
    """
    
    def __init__(
        self,
        surrogate_checkpoint: Optional[Path] = None,
        acquisition_checkpoint: Optional[Path] = None,
        name: Optional[str] = None
    ):
        """
        Initialize BC evaluator.
        
        Args:
            surrogate_checkpoint: Path to BC surrogate checkpoint
            acquisition_checkpoint: Path to BC acquisition checkpoint
            name: Optional custom name
        """
        # Determine configuration
        checkpoints = {}
        if surrogate_checkpoint:
            checkpoints['surrogate'] = Path(surrogate_checkpoint)
        if acquisition_checkpoint:
            checkpoints['acquisition'] = Path(acquisition_checkpoint)
            
        # Generate name based on configuration
        if name is None:
            if surrogate_checkpoint and acquisition_checkpoint:
                name = "BC_Both"
            elif surrogate_checkpoint:
                name = "BC_Surrogate_Random"
            elif acquisition_checkpoint:
                name = "BC_Acquisition_Learning"
            else:
                name = "BC_Empty"
                
        super().__init__(name=name, checkpoint_paths=checkpoints)
        
        self.use_surrogate = surrogate_checkpoint is not None
        self.use_acquisition = acquisition_checkpoint is not None
        
        # Model components
        self.surrogate_model = None
        self.surrogate_params = None
        self.acquisition_model = None
        self.acquisition_params = None
        
    def initialize(self) -> None:
        """Load BC checkpoints and initialize models."""
        logger.info(f"Initializing BC evaluator: {self.name}")
        
        try:
            # Load surrogate if provided
            if self.use_surrogate:
                self._load_surrogate()
                
            # Load acquisition if provided
            if self.use_acquisition:
                self._load_acquisition()
                
            self._initialized = True
            logger.info(f"BC evaluator initialized: surrogate={self.use_surrogate}, "
                       f"acquisition={self.use_acquisition}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BC evaluator: {e}")
            raise
            
    def _load_surrogate(self) -> None:
        """Load BC surrogate model."""
        from ..training.bc_model_loader import load_bc_surrogate_model
        
        checkpoint_path = self.checkpoint_paths['surrogate']
        logger.info(f"Loading BC surrogate from {checkpoint_path}")
        
        # Load model using new loader
        try:
            (init_fn, apply_fn, encoder_init, encoder_apply, params) = load_bc_surrogate_model(
                str(checkpoint_path)
            )
            
            # Store model components
            self.surrogate_params = params
            self.surrogate_model = {
                'init_fn': init_fn,
                'apply_fn': apply_fn,
                'encoder_init': encoder_init,
                'encoder_apply': encoder_apply
            }
            logger.info("✅ Loaded BC surrogate model")
            
        except Exception as e:
            logger.error(f"Failed to load BC surrogate: {e}")
            raise
            
    def _load_acquisition(self) -> None:
        """Load BC acquisition model."""
        from ..training.bc_model_loader import load_bc_acquisition_model
        
        checkpoint_path = self.checkpoint_paths['acquisition']
        logger.info(f"Loading BC acquisition from {checkpoint_path}")
        
        # Load model using new loader
        try:
            self.acquisition_model = load_bc_acquisition_model(str(checkpoint_path))
            
            if callable(self.acquisition_model):
                logger.info("✅ Loaded BC acquisition function")
            else:
                raise ValueError(f"Expected callable acquisition function, got {type(self.acquisition_model)}")
                
        except Exception as e:
            logger.error(f"Failed to load BC acquisition: {e}")
            raise
            
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run BC evaluation on a single SCM.
        
        Uses the BC wrappers from bc_method_wrappers.py to run actual
        CBO experiments with performance tracking.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with learning history
        """
        if not self._initialized:
            self.initialize()
            
        start_time = time.time()
        
        # Import BC runner from evaluation package
        from .bc_runner import run_bc_experiment, EvalConfig
        
        # Extract configuration
        max_interventions = config.get('experiment', {}).get('target', {}).get(
            'max_interventions', 10
        )
        n_observational_samples = config.get('experiment', {}).get('target', {}).get(
            'n_observational_samples', 100
        )
        
        # Create evaluation config
        eval_config = EvalConfig(
            n_observational_samples=n_observational_samples,
            n_intervention_steps=max_interventions,
            intervention_value_range=config.get('experiment', {}).get('target', {}).get(
                'intervention_value_range', (-2.0, 2.0)
            ),
            random_seed=seed,
            learning_rate=0.0 if self.use_surrogate else 1e-3  # No learning for BC surrogate
        )
        
        # Prepare checkpoint paths for BC models
        surrogate_checkpoint = None
        acquisition_checkpoint = None
        
        if self.use_surrogate and 'surrogate' in self.checkpoint_paths:
            surrogate_checkpoint = str(self.checkpoint_paths['surrogate'])
            
        if self.use_acquisition and 'acquisition' in self.checkpoint_paths:
            acquisition_checkpoint = str(self.checkpoint_paths['acquisition'])
            
        # Run BC experiment with tracking
        try:
            result = run_bc_experiment(
                scm=scm,
                config=eval_config,
                surrogate_checkpoint=surrogate_checkpoint,
                acquisition_checkpoint=acquisition_checkpoint,
                track_performance=True
            )
            
            # Convert BC result format to our standardized format
            return self._convert_bc_result(result, scm, seed, run_idx, start_time)
            
        except Exception as e:
            logger.error(f"BC evaluation failed: {e}")
            # Return failed result
            return ExperimentResult(
                learning_history=[],
                final_metrics={'error': str(e)},
                metadata={
                    'method': self.name,
                    'run_idx': run_idx,
                    'seed': seed
                },
                success=False,
                error_message=str(e),
                total_time=time.time() - start_time
            )
            
    def _convert_bc_result(
        self,
        bc_result: Dict[str, Any],
        scm: Any,
        seed: int,
        run_idx: int,
        start_time: float
    ) -> ExperimentResult:
        """Convert BC result format to standardized ExperimentResult."""
        
        # Get SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        
        # Extract performance trajectory if available
        learning_history = []
        
        if 'performance_trajectory' in bc_result and bc_result['performance_trajectory']:
            trajectory = bc_result['performance_trajectory']
            
            # Convert PerformanceMetrics to our format
            for i, metrics in enumerate(trajectory):
                # Extract marginals if available
                marginals = metrics.marginals if metrics.marginals else {}
                
                step_result = StepResult(
                    step=metrics.step,
                    intervention={metrics.intervention_variable: metrics.intervention_value}
                        if metrics.intervention_variable else {},
                    outcome_value=metrics.target_value,
                    marginals=marginals,
                    uncertainty=metrics.uncertainty,
                    reward=metrics.target_value - trajectory[0].target_value if i > 0 else 0.0,
                    computation_time=0.0  # Not tracked in BC
                )
                learning_history.append(step_result)
                
        elif 'learning_history' in bc_result:
            # Alternative format from run_progressive_learning_demo_with_scm
            initial_value = bc_result.get('initial_best', 0.0)
            
            # Add initial state
            learning_history.append(StepResult(
                step=0,
                intervention={},
                outcome_value=initial_value,
                marginals={},
                uncertainty=self._extract_uncertainty_from_step(0, bc_result),
                reward=0.0,
                computation_time=0.0
            ))
            
            # Add intervention steps
            for i, step_info in enumerate(bc_result['learning_history']):
                intervention_dict = {}
                if 'intervention' in step_info:
                    int_vars = step_info['intervention'].get('intervention_variables', frozenset())
                    int_vals = step_info['intervention'].get('intervention_values', ())
                    if int_vars and int_vals:
                        # Take first intervention
                        intervention_dict = {list(int_vars)[0]: float(int_vals[0])}
                        
                # Extract marginals if available
                marginals = {}
                if 'marginals' in step_info:
                    marginals = dict(step_info['marginals'])
                    
                step_result = StepResult(
                    step=i + 1,
                    intervention=intervention_dict,
                    outcome_value=step_info.get('outcome_value', step_info.get('current_best', 0.0)),
                    marginals=marginals,
                    uncertainty=step_info.get('uncertainty', 0.0),
                    reward=step_info.get('reward', 0.0),
                    computation_time=0.0
                )
                learning_history.append(step_result)
                
        # Compute final metrics
        if learning_history:
            initial_value = learning_history[0].outcome_value
            final_value = learning_history[-1].outcome_value
            final_marginals = learning_history[-1].marginals
        else:
            initial_value = bc_result.get('initial_best', 0.0)
            final_value = bc_result.get('final_best', initial_value)
            final_marginals = {}
            
        # Get performance metrics if available
        perf_metrics = bc_result.get('performance_metrics', {})
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': bc_result.get('final_best', final_value),
            'improvement': final_value - initial_value,
            'n_interventions': len(learning_history) - 1 if learning_history else 0,
            'final_f1': perf_metrics.get('final_f1', 
                        compute_f1_score_from_marginals(final_marginals, true_parents)),
            'final_shd': perf_metrics.get('final_shd',
                         compute_shd_from_marginals(final_marginals, true_parents))
        }
        
        # Create result
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
                'surrogate_checkpoint': str(self.checkpoint_paths.get('surrogate', 'none')),
                'acquisition_checkpoint': str(self.checkpoint_paths.get('acquisition', 'none')),
                'scm_info': {
                    'n_variables': len(variables),
                    'target': target_var,
                    'true_parents': true_parents
                },
                'run_idx': run_idx,
                'seed': seed
            },
            success=True,
            total_time=time.time() - start_time
        )
        
    def _extract_uncertainty_from_metrics(self, metrics: Any, bc_result: Dict[str, Any]) -> float:
        """Extract uncertainty from performance metrics or result."""
        # Check if metrics has uncertainty directly
        if hasattr(metrics, 'uncertainty'):
            return float(metrics.uncertainty)
            
        # Check if metrics has acquisition state with posterior
        if hasattr(metrics, 'acquisition_state'):
            state = metrics.acquisition_state
            if hasattr(state, 'posterior') and hasattr(state.posterior, 'uncertainty'):
                return float(state.posterior.uncertainty)
                
        # Check if there's a posterior in the metrics
        if hasattr(metrics, 'posterior') and hasattr(metrics.posterior, 'uncertainty'):
            return float(metrics.posterior.uncertainty)
            
        # If BC surrogate was used, uncertainty should be available
        # but if not found, return 0.0 as fallback
        return 0.0
        
    def _extract_uncertainty_from_step(self, step_idx: int, bc_result: Dict[str, Any]) -> float:
        """Extract uncertainty for a specific step from BC result."""
        # Check if there's acquisition state history
        if 'acquisition_states' in bc_result and step_idx < len(bc_result['acquisition_states']):
            state = bc_result['acquisition_states'][step_idx]
            if hasattr(state, 'posterior') and hasattr(state.posterior, 'uncertainty'):
                return float(state.posterior.uncertainty)
                
        # Check if there's uncertainty history
        if 'uncertainty_history' in bc_result and step_idx < len(bc_result['uncertainty_history']):
            return float(bc_result['uncertainty_history'][step_idx])
            
        # Default to 0.0 if not available
        return 0.0