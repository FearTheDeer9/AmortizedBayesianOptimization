"""
Standardized Result Types for Unified Evaluation

Defines the common data structures used by all evaluation methods to ensure
consistency in data format and facilitate comparison across methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class StepResult:
    """
    Result from a single intervention step in the learning process.
    
    This captures all relevant information at each step to enable
    trajectory analysis and metric computation.
    """
    step: int
    intervention: Dict[str, float]  # variable -> value
    outcome_value: float  # target variable value after intervention
    marginals: Dict[str, float]  # variable -> P(is_parent)
    uncertainty: float = 0.0  # model uncertainty in bits
    reward: Optional[float] = None  # reward signal if available
    computation_time: float = 0.0  # time taken for this step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step': self.step,
            'intervention': self.intervention,
            'outcome_value': self.outcome_value,
            'marginals': self.marginals,
            'uncertainty': self.uncertainty,
            'reward': self.reward,
            'computation_time': self.computation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """
    Complete result from running an evaluation on a single SCM.
    
    This is the standard output format for all evaluation methods.
    """
    learning_history: List[StepResult]
    final_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    total_time: float = 0.0
    
    @property
    def n_steps(self) -> int:
        """Number of intervention steps."""
        return len(self.learning_history)
    
    @property
    def final_value(self) -> float:
        """Final target value achieved."""
        if self.learning_history:
            return self.learning_history[-1].outcome_value
        return self.final_metrics.get('final_value', 0.0)
    
    @property
    def initial_value(self) -> float:
        """Initial target value."""
        if self.learning_history:
            return self.learning_history[0].outcome_value
        return self.final_metrics.get('initial_value', 0.0)
    
    @property
    def improvement(self) -> float:
        """Improvement in target value."""
        # Check if improvement was explicitly calculated (e.g., with optimization direction)
        if 'improvement' in self.final_metrics:
            return self.final_metrics['improvement']
        # Otherwise, use default calculation (assumes maximization)
        return self.final_value - self.initial_value
    
    def get_trajectory(self, metric: str, true_parents: Optional[List[str]] = None) -> List[float]:
        """Extract trajectory for a specific metric."""
        if metric == 'outcome_value':
            return [step.outcome_value for step in self.learning_history]
        elif metric == 'uncertainty':
            return [step.uncertainty for step in self.learning_history]
        elif metric == 'reward':
            return [step.reward or 0.0 for step in self.learning_history]
        elif metric == 'f1_score' and true_parents is not None:
            from ..analysis.trajectory_metrics import compute_f1_score_from_marginals
            return [compute_f1_score_from_marginals(step.marginals, true_parents) 
                   for step in self.learning_history]
        elif metric == 'shd' and true_parents is not None:
            from ..analysis.trajectory_metrics import compute_shd_from_marginals
            return [compute_shd_from_marginals(step.marginals, true_parents)
                   for step in self.learning_history]
        else:
            raise ValueError(f"Unknown metric: {metric} or missing true_parents")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'learning_history': [step.to_dict() for step in self.learning_history],
            'final_metrics': self.final_metrics,
            'metadata': self.metadata,
            'success': self.success,
            'error_message': self.error_message,
            'total_time': self.total_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        data = data.copy()
        data['learning_history'] = [
            StepResult.from_dict(step) for step in data['learning_history']
        ]
        return cls(**data)


@dataclass
class MethodMetrics:
    """
    Aggregated metrics for a method across multiple runs.
    """
    method_name: str
    n_runs: int
    n_successful: int
    
    # Aggregated metrics
    mean_final_value: float
    std_final_value: float
    mean_improvement: float
    std_improvement: float
    mean_steps: float
    mean_time: float
    
    # Trajectories (mean and std across runs)
    outcome_trajectory_mean: List[float]
    outcome_trajectory_std: List[float]
    uncertainty_trajectory_mean: List[float]
    uncertainty_trajectory_std: List[float]
    f1_trajectory_mean: List[float] = field(default_factory=list)
    f1_trajectory_std: List[float] = field(default_factory=list)
    shd_trajectory_mean: List[float] = field(default_factory=list)
    shd_trajectory_std: List[float] = field(default_factory=list)
    
    # Additional metrics
    success_rate: float = field(init=False)
    
    def __post_init__(self):
        self.success_rate = self.n_successful / self.n_runs if self.n_runs > 0 else 0.0
    
    @classmethod
    def from_results(cls, method_name: str, results: List[ExperimentResult]) -> 'MethodMetrics':
        """Compute metrics from a list of experiment results."""
        if not results:
            return cls(
                method_name=method_name,
                n_runs=0,
                n_successful=0,
                mean_final_value=0.0,
                std_final_value=0.0,
                mean_improvement=0.0,
                std_improvement=0.0,
                mean_steps=0.0,
                mean_time=0.0,
                outcome_trajectory_mean=[],
                outcome_trajectory_std=[],
                uncertainty_trajectory_mean=[],
                uncertainty_trajectory_std=[],
                f1_trajectory_mean=[],
                f1_trajectory_std=[],
                shd_trajectory_mean=[],
                shd_trajectory_std=[]
            )
        
        # Filter successful runs
        successful_results = [r for r in results if r.success]
        
        # Extract metrics
        final_values = [r.final_value for r in successful_results]
        improvements = [r.improvement for r in successful_results]
        steps = [r.n_steps for r in successful_results]
        times = [r.total_time for r in successful_results]
        
        # Extract trajectories and pad to same length
        outcome_trajectories = [r.get_trajectory('outcome_value') for r in successful_results]
        uncertainty_trajectories = [r.get_trajectory('uncertainty') for r in successful_results]
        
        # Extract F1 and SHD trajectories if true_parents are available
        f1_trajectories = []
        shd_trajectories = []
        
        for result in successful_results:
            # Check if result has true_parents in metadata (under scm_info)
            scm_info = result.metadata.get('scm_info', {})
            true_parents = scm_info.get('true_parents', result.metadata.get('true_parents', None))
            if true_parents is not None and result.learning_history:
                # Check if first step has marginals (indicates structure learning method)
                if result.learning_history[0].marginals:
                    f1_traj = result.get_trajectory('f1_score', true_parents)
                    shd_traj = result.get_trajectory('shd', true_parents)
                    f1_trajectories.append(f1_traj)
                    shd_trajectories.append(shd_traj)
                else:
                    # No marginals means no structure learning (e.g., Random baseline)
                    # Use worst-case values
                    n_steps = len(result.learning_history)
                    f1_trajectories.append([0.0] * n_steps)
                    shd_trajectories.append([len(true_parents)] * n_steps)
        
        # Pad trajectories
        max_length = max(len(t) for t in outcome_trajectories) if outcome_trajectories else 0
        
        def pad_trajectory(traj: List[float], max_len: int) -> List[float]:
            """Pad trajectory with last value."""
            if len(traj) < max_len:
                return traj + [traj[-1]] * (max_len - len(traj))
            return traj
        
        padded_outcomes = [pad_trajectory(t, max_length) for t in outcome_trajectories]
        padded_uncertainties = [pad_trajectory(t, max_length) for t in uncertainty_trajectories]
        padded_f1s = [pad_trajectory(t, max_length) for t in f1_trajectories] if f1_trajectories else []
        padded_shds = [pad_trajectory(t, max_length) for t in shd_trajectories] if shd_trajectories else []
        
        # Compute statistics
        if padded_outcomes:
            outcome_array = np.array(padded_outcomes)
            uncertainty_array = np.array(padded_uncertainties)
            
            outcome_mean = np.mean(outcome_array, axis=0).tolist()
            outcome_std = np.std(outcome_array, axis=0).tolist()
            uncertainty_mean = np.mean(uncertainty_array, axis=0).tolist()
            uncertainty_std = np.std(uncertainty_array, axis=0).tolist()
        else:
            outcome_mean = []
            outcome_std = []
            uncertainty_mean = []
            uncertainty_std = []
        
        # Compute F1/SHD statistics if available
        if padded_f1s:
            f1_array = np.array(padded_f1s)
            shd_array = np.array(padded_shds)
            
            f1_mean = np.mean(f1_array, axis=0).tolist()
            f1_std = np.std(f1_array, axis=0).tolist()
            shd_mean = np.mean(shd_array, axis=0).tolist()
            shd_std = np.std(shd_array, axis=0).tolist()
        else:
            f1_mean = []
            f1_std = []
            shd_mean = []
            shd_std = []
        
        return cls(
            method_name=method_name,
            n_runs=len(results),
            n_successful=len(successful_results),
            mean_final_value=np.mean(final_values) if final_values else 0.0,
            std_final_value=np.std(final_values) if final_values else 0.0,
            mean_improvement=np.mean(improvements) if improvements else 0.0,
            std_improvement=np.std(improvements) if improvements else 0.0,
            mean_steps=np.mean(steps) if steps else 0.0,
            mean_time=np.mean(times) if times else 0.0,
            outcome_trajectory_mean=outcome_mean,
            outcome_trajectory_std=outcome_std,
            uncertainty_trajectory_mean=uncertainty_mean,
            uncertainty_trajectory_std=uncertainty_std,
            f1_trajectory_mean=f1_mean,
            f1_trajectory_std=f1_std,
            shd_trajectory_mean=shd_mean,
            shd_trajectory_std=shd_std
        )


@dataclass
class ComparisonResults:
    """
    Results from comparing multiple methods.
    """
    method_metrics: Dict[str, MethodMetrics]
    scm_metadata: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_results: Optional[Dict[str, List[ExperimentResult]]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    
    def get_method_names(self) -> List[str]:
        """Get list of method names."""
        return list(self.method_metrics.keys())
    
    def get_best_method(self, metric: str = 'mean_final_value') -> str:
        """Get best performing method by a specific metric."""
        if not self.method_metrics:
            return ""
        
        if metric == 'mean_final_value':
            # For optimization, higher is usually better
            return max(self.method_metrics.items(), 
                      key=lambda x: x[1].mean_final_value)[0]
        elif metric == 'success_rate':
            return max(self.method_metrics.items(), 
                      key=lambda x: x[1].success_rate)[0]
        else:
            raise ValueError(f"Unknown metric for comparison: {metric}")
    
    def get_learning_curves(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get learning curves in format expected by visualization functions.
        
        Returns dict mapping method names to trajectory data.
        """
        curves = {}
        
        for method_name, metrics in self.method_metrics.items():
            if metrics.outcome_trajectory_mean:
                curves[method_name] = {
                    'steps': list(range(len(metrics.outcome_trajectory_mean))),
                    'target_mean': metrics.outcome_trajectory_mean,
                    'target_std': metrics.outcome_trajectory_std,
                    'n_runs': metrics.n_runs
                }
                
                # Add F1 and SHD trajectories if available
                if metrics.f1_trajectory_mean:
                    curves[method_name]['f1_mean'] = metrics.f1_trajectory_mean
                    curves[method_name]['f1_std'] = metrics.f1_trajectory_std
                    
                if metrics.shd_trajectory_mean:
                    curves[method_name]['shd_mean'] = metrics.shd_trajectory_mean
                    curves[method_name]['shd_std'] = metrics.shd_trajectory_std
        
        return curves
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_metrics': {
                name: {
                    'n_runs': m.n_runs,
                    'n_successful': m.n_successful,
                    'success_rate': m.success_rate,
                    'mean_final_value': m.mean_final_value,
                    'std_final_value': m.std_final_value,
                    'mean_improvement': m.mean_improvement,
                    'std_improvement': m.std_improvement,
                    'mean_steps': m.mean_steps,
                    'mean_time': m.mean_time
                }
                for name, m in self.method_metrics.items()
            },
            'scm_metadata': self.scm_metadata,
            'config': self.config,
            'timestamp': self.timestamp
        }
    
    def summary_dataframe(self) -> Any:
        """Create pandas DataFrame summary (if pandas available)."""
        try:
            import pandas as pd
            
            data = []
            for name, metrics in self.method_metrics.items():
                data.append({
                    'Method': name,
                    'Success Rate': f"{metrics.success_rate:.1%}",
                    'Mean Final Value': f"{metrics.mean_final_value:.3f}",
                    'Std Dev': f"{metrics.std_final_value:.3f}",
                    'Mean Improvement': f"{metrics.mean_improvement:.3f}",
                    'Mean Steps': f"{metrics.mean_steps:.1f}",
                    'Mean Time (s)': f"{metrics.mean_time:.1f}"
                })
            
            return pd.DataFrame(data)
        except ImportError:
            # Return dict if pandas not available
            return self.to_dict()['method_metrics']