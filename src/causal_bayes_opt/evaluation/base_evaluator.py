"""
Base Evaluator Interface

Abstract base class defining the interface that all evaluation methods must implement.
This ensures consistency across GRPO, BC, and future evaluation approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .result_types import ExperimentResult

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation methods.
    
    This class defines the interface that must be implemented by all evaluation
    methods (GRPO, BC, etc.) to ensure consistent behavior and data format.
    """
    
    def __init__(self, name: str, checkpoint_paths: Optional[Dict[str, Path]] = None):
        """
        Initialize evaluator.
        
        Args:
            name: Human-readable name for this evaluation method
            checkpoint_paths: Optional dict mapping component names to checkpoint paths
                             e.g., {'policy': Path(...), 'surrogate': Path(...)}
        """
        self.name = name
        self.checkpoint_paths = checkpoint_paths or {}
        self._initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the evaluator by loading checkpoints and setting up models.
        
        This method should be called before running any evaluations.
        Raises an exception if initialization fails.
        """
        pass
        
    @abstractmethod
    def evaluate_single_run(
        self, 
        scm: Any, 
        config: Dict[str, Any], 
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run a single evaluation on one SCM.
        
        Args:
            scm: Structural Causal Model to evaluate on
            config: Configuration dict with evaluation parameters
            seed: Random seed for reproducibility
            run_idx: Index of this run (for logging)
            
        Returns:
            ExperimentResult with standardized format
        """
        pass
    
    def evaluate_multiple_runs(
        self,
        scm: Any,
        config: Dict[str, Any], 
        n_runs: int = 3,
        base_seed: int = 42
    ) -> List[ExperimentResult]:
        """
        Run multiple evaluations on the same SCM with different seeds.
        
        Args:
            scm: Structural Causal Model to evaluate on
            config: Configuration dict with evaluation parameters
            n_runs: Number of runs to perform
            base_seed: Base random seed (will be incremented for each run)
            
        Returns:
            List of ExperimentResult objects
        """
        if not self._initialized:
            self.initialize()
            
        results = []
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            logger.info(f"Running {self.name} evaluation {run_idx + 1}/{n_runs} with seed {seed}")
            
            try:
                result = self.evaluate_single_run(scm, config, seed, run_idx)
                results.append(result)
            except Exception as e:
                logger.error(f"Run {run_idx} failed: {e}")
                # Create failed result
                from .result_types import ExperimentResult, StepResult
                failed_result = ExperimentResult(
                    learning_history=[],
                    final_metrics={'error': str(e)},
                    metadata={'run_idx': run_idx, 'seed': seed},
                    success=False
                )
                results.append(failed_result)
                
        return results
    
    def get_method_name(self) -> str:
        """Return human-readable method name."""
        return self.name
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Return information about loaded checkpoints."""
        return {
            'method': self.name,
            'checkpoints': {
                name: str(path) for name, path in self.checkpoint_paths.items()
            },
            'initialized': self._initialized
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self._initialized})"