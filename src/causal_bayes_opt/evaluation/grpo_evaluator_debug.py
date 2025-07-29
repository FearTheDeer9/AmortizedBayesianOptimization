"""
Debug version of GRPO Evaluator with enhanced logging
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pickle
import json
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp

from .grpo_evaluator import GRPOEvaluator
from .result_types import ExperimentResult, StepResult

logger = logging.getLogger(__name__)


class GRPOEvaluatorDebug(GRPOEvaluator):
    """Debug version of GRPO evaluator with extensive logging."""
    
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """Debug version with extensive logging."""
        logger.info(f"DEBUG: Starting GRPO evaluation for run {run_idx}")
        
        try:
            # Call parent method but wrap key parts
            result = super().evaluate_single_run(scm, config, seed, run_idx)
            
            # Log result summary
            logger.info(f"DEBUG: Evaluation completed")
            logger.info(f"  Initial value: {result.final_metrics['initial_value']}")
            logger.info(f"  Final value: {result.final_metrics['final_value']}")
            logger.info(f"  Improvement: {result.final_metrics['improvement']}")
            logger.info(f"  Steps: {len(result.learning_history)}")
            
            # Check for suspicious 0.0 improvement
            if abs(result.final_metrics['improvement']) < 1e-10:
                logger.warning("DEBUG: Zero improvement detected!")
                logger.warning(f"  Initial: {result.final_metrics['initial_value']}")
                logger.warning(f"  Final: {result.final_metrics['final_value']}")
                
                # Log intervention history
                for step in result.learning_history:
                    if step.intervention:
                        logger.warning(f"  Step {step.step}: {step.intervention} -> {step.outcome_value}")
            
            return result
            
        except Exception as e:
            logger.error(f"DEBUG: Exception in evaluate_single_run: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a failed result instead of crashing
            return ExperimentResult(
                learning_history=[],
                final_metrics={
                    'initial_value': 0.0,
                    'final_value': 0.0,
                    'improvement': 0.0,
                    'error': str(e)
                },
                metadata={
                    'method': self.name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                },
                success=False,
                total_time=0.0
            )
    
    def _policy_output_to_action(self,
                                policy_output: Dict[str, jnp.ndarray],
                                variables: List[str],
                                target: str) -> Tuple[int, float]:
        """Debug version with logging."""
        logger.info("DEBUG: Converting policy output to action")
        
        # Log policy output
        logger.info(f"  Variables: {variables}")
        logger.info(f"  Target: {target}")
        
        if 'variable_logits' in policy_output:
            logits = policy_output['variable_logits']
            logger.info(f"  Variable logits shape: {logits.shape}")
            logger.info(f"  Variable logits: {logits}")
            
            # Check for collapsed logits
            logit_variance = float(jnp.var(logits))
            if logit_variance < 1e-6:
                logger.warning(f"  WARNING: Low logit variance: {logit_variance}")
                logger.warning("  This indicates posterior collapse!")
        
        # Call parent method
        selected_var_idx, intervention_value = super()._policy_output_to_action(
            policy_output, variables, target
        )
        
        logger.info(f"  Selected variable: {variables[selected_var_idx]} (idx {selected_var_idx})")
        logger.info(f"  Intervention value: {intervention_value}")
        
        return selected_var_idx, intervention_value