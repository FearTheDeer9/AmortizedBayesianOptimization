#!/usr/bin/env python3
"""
Trajectory Validator for Behavioral Cloning

Validates behavioral cloning models by comparing their predictions with
expert trajectories. Provides comprehensive metrics for both surrogate
and acquisition model validation.

Key Features:
1. Trajectory-level validation on held-out expert demonstrations
2. Surrogate model posterior prediction validation
3. Acquisition policy intervention choice validation
4. End-to-end trajectory similarity metrics
5. Statistical significance testing

Design Principles (Rich Hickey Approved):
- Pure functions for all validation metrics
- Immutable result structures
- Clear separation of validation concerns
- Composable metric computation
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from statistics import mean, stdev
from scipy import stats

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from .expert_collection.data_structures import ExpertDemonstration
from .trajectory_processor import TrajectoryStep, DifficultyLevel
from .behavioral_cloning_adapter import (
    extract_posterior_history,
    extract_intervention_sequence,
    create_acquisition_state,
    extract_avici_samples
)
from ..acquisition.state import AcquisitionState
from ..avici_integration.parent_set.posterior import ParentSetPosterior

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PosteriorSimilarityMetrics:
    """Metrics for comparing predicted vs expert posteriors."""
    kl_divergence: float  # KL divergence between distributions
    js_divergence: float  # Jensen-Shannon divergence
    correlation: float    # Correlation between probabilities
    top_k_overlap: float  # Overlap in top-k parent sets
    entropy_difference: float  # Difference in posterior entropy


@dataclass(frozen=True)
class InterventionAccuracyMetrics:
    """Metrics for comparing predicted vs expert intervention choices."""
    variable_selection_accuracy: float  # Fraction of correct variable choices
    exact_match_accuracy: float         # Fraction of exact action matches
    intervention_value_mse: float       # MSE for intervention values
    strategy_consistency: float         # Consistency in intervention strategy
    
    # Per-step breakdown
    step_accuracies: List[float]
    step_difficulties: List[float]


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Comprehensive trajectory-level metrics."""
    trajectory_id: str
    difficulty_level: DifficultyLevel
    
    # Individual model metrics
    surrogate_metrics: PosteriorSimilarityMetrics
    acquisition_metrics: InterventionAccuracyMetrics
    
    # End-to-end metrics
    final_accuracy_match: bool       # Did both models reach same accuracy?
    convergence_speed_ratio: float   # How much faster/slower than expert
    trajectory_length_ratio: float   # Length ratio vs expert
    
    # Quality scores
    overall_quality_score: float     # Combined quality metric [0, 1]
    behavioral_fidelity: float       # How well does it mimic expert behavior


@dataclass(frozen=True)
class ValidationResults:
    """Complete validation results across all test trajectories."""
    individual_trajectories: List[TrajectoryMetrics]
    
    # Aggregate metrics
    mean_surrogate_accuracy: float
    mean_acquisition_accuracy: float
    mean_trajectory_quality: float
    
    # Difficulty-stratified results
    results_by_difficulty: Dict[DifficultyLevel, Dict[str, float]]
    
    # Statistical tests
    significance_tests: Dict[str, Dict[str, float]]
    
    # Summary statistics
    summary_stats: Dict[str, Any]


class TrajectoryValidator:
    """Validator for behavioral cloning models on expert trajectories."""
    
    def __init__(
        self,
        surrogate_model: Optional[Any] = None,
        acquisition_policy: Optional[Any] = None,
        top_k: int = 5,
        significance_level: float = 0.05
    ):
        """
        Initialize trajectory validator.
        
        Args:
            surrogate_model: Trained surrogate model for posterior prediction
            acquisition_policy: Trained acquisition policy for intervention selection
            top_k: Number of top parent sets to compare
            significance_level: Level for statistical significance tests
        """
        self.surrogate_model = surrogate_model
        self.acquisition_policy = acquisition_policy
        self.top_k = top_k
        self.significance_level = significance_level
    
    def validate_surrogate_predictions(
        self,
        model: Any,
        expert_trajectory: ExpertDemonstration,
        trajectory_id: str
    ) -> PosteriorSimilarityMetrics:
        """
        Pure function: Validate surrogate model predictions against expert posteriors.
        
        Args:
            model: Trained surrogate model
            expert_trajectory: Expert demonstration to validate against
            trajectory_id: Unique identifier for trajectory
            
        Returns:
            PosteriorSimilarityMetrics comparing model vs expert predictions
        """
        try:
            # Extract expert posterior history
            expert_posteriors = extract_posterior_history(expert_trajectory)
            
            # Extract AVICI data for model input
            avici_data = extract_avici_samples(expert_trajectory)
            
            # Get model predictions for each step
            model_posteriors = []
            for step in range(len(expert_posteriors)):
                # Create input for model at this step
                model_input = self._prepare_model_input(avici_data, expert_trajectory, step)
                
                # Get model prediction (placeholder for now)
                predicted_posterior = self._predict_posterior(model, model_input)
                model_posteriors.append(predicted_posterior)
            
            # Compute similarity metrics
            return self._compute_posterior_similarity(
                model_posteriors=model_posteriors,
                expert_posteriors=expert_posteriors
            )
            
        except Exception as e:
            logger.warning(f"Failed to validate surrogate for {trajectory_id}: {e}")
            return self._create_default_posterior_metrics()
    
    def validate_acquisition_choices(
        self,
        policy: Any,
        expert_trajectory: ExpertDemonstration,
        trajectory_id: str
    ) -> InterventionAccuracyMetrics:
        """
        Pure function: Validate acquisition policy choices against expert actions.
        
        Args:
            policy: Trained acquisition policy
            expert_trajectory: Expert demonstration to validate against
            trajectory_id: Unique identifier for trajectory
            
        Returns:
            InterventionAccuracyMetrics comparing policy vs expert choices
        """
        try:
            # Extract expert intervention sequence
            expert_interventions = extract_intervention_sequence(expert_trajectory)
            
            # Extract AVICI data
            avici_data = extract_avici_samples(expert_trajectory)
            
            # Simulate policy choices for each step
            policy_choices = []
            step_accuracies = []
            step_difficulties = []
            
            for step in range(len(expert_interventions)):
                # Create acquisition state for this step
                acq_state = create_acquisition_state(
                    demo=expert_trajectory,
                    step=step,
                    avici_data=avici_data,
                    intervention_history=expert_interventions
                )
                
                # Get policy prediction
                predicted_action = self._predict_action(policy, acq_state)
                policy_choices.append(predicted_action)
                
                # Compare with expert action
                expert_action = expert_interventions[step]
                step_accuracy = self._compute_action_accuracy(predicted_action, expert_action)
                step_accuracies.append(step_accuracy)
                
                # Assess step difficulty
                step_difficulty = self._assess_step_difficulty(acq_state, expert_action)
                step_difficulties.append(step_difficulty)
            
            # Compute overall accuracy metrics
            return self._compute_intervention_accuracy(
                policy_choices=policy_choices,
                expert_interventions=expert_interventions,
                step_accuracies=step_accuracies,
                step_difficulties=step_difficulties
            )
            
        except Exception as e:
            logger.warning(f"Failed to validate acquisition for {trajectory_id}: {e}")
            return self._create_default_intervention_metrics()
    
    def compute_trajectory_similarity(
        self,
        predicted_trajectory: List[Dict[str, Any]],
        expert_trajectory: ExpertDemonstration,
        trajectory_id: str
    ) -> TrajectoryMetrics:
        """
        Pure function: Compute end-to-end trajectory similarity metrics.
        
        Args:
            predicted_trajectory: Model-generated trajectory
            expert_trajectory: Expert demonstration
            trajectory_id: Unique identifier
            
        Returns:
            TrajectoryMetrics with comprehensive similarity assessment
        """
        try:
            # Validate individual components
            surrogate_metrics = self.validate_surrogate_predictions(
                model=self.surrogate_model,
                expert_trajectory=expert_trajectory,
                trajectory_id=trajectory_id
            )
            
            acquisition_metrics = self.validate_acquisition_choices(
                policy=self.acquisition_policy,
                expert_trajectory=expert_trajectory,
                trajectory_id=trajectory_id
            )
            
            # Compute end-to-end metrics
            final_accuracy_match = self._compare_final_accuracy(
                predicted_trajectory, expert_trajectory
            )
            
            convergence_speed_ratio = self._compute_convergence_speed_ratio(
                predicted_trajectory, expert_trajectory
            )
            
            trajectory_length_ratio = self._compute_length_ratio(
                predicted_trajectory, expert_trajectory
            )
            
            # Compute quality scores
            overall_quality_score = self._compute_overall_quality(
                surrogate_metrics, acquisition_metrics
            )
            
            behavioral_fidelity = self._compute_behavioral_fidelity(
                acquisition_metrics, surrogate_metrics
            )
            
            # Classify difficulty
            from .behavioral_cloning_adapter import compute_demonstration_complexity
            from .trajectory_processor import DifficultyLevel
            complexity = compute_demonstration_complexity(expert_trajectory)
            
            if complexity < 0.25:
                difficulty_level = DifficultyLevel.EASY
            elif complexity < 0.5:
                difficulty_level = DifficultyLevel.MEDIUM
            elif complexity < 0.75:
                difficulty_level = DifficultyLevel.HARD
            else:
                difficulty_level = DifficultyLevel.EXPERT
            
            return TrajectoryMetrics(
                trajectory_id=trajectory_id,
                difficulty_level=difficulty_level,
                surrogate_metrics=surrogate_metrics,
                acquisition_metrics=acquisition_metrics,
                final_accuracy_match=final_accuracy_match,
                convergence_speed_ratio=convergence_speed_ratio,
                trajectory_length_ratio=trajectory_length_ratio,
                overall_quality_score=overall_quality_score,
                behavioral_fidelity=behavioral_fidelity
            )
            
        except Exception as e:
            logger.warning(f"Failed to compute trajectory similarity for {trajectory_id}: {e}")
            return self._create_default_trajectory_metrics(trajectory_id)
    
    def validate_on_test_set(
        self,
        test_demonstrations: List[ExpertDemonstration],
        max_trajectories: Optional[int] = None
    ) -> ValidationResults:
        """
        Validate models on complete test set of expert demonstrations.
        
        Args:
            test_demonstrations: List of expert demonstrations for testing
            max_trajectories: Maximum number of trajectories to validate
            
        Returns:
            ValidationResults with comprehensive validation metrics
        """
        if max_trajectories is not None:
            test_demonstrations = test_demonstrations[:max_trajectories]
        
        logger.info(f"Validating on {len(test_demonstrations)} test trajectories")
        
        # Validate each trajectory
        trajectory_metrics = []
        for i, demo in enumerate(test_demonstrations):
            trajectory_id = f"test_trajectory_{i:04d}"
            
            # For now, use empty predicted trajectory (placeholder)
            predicted_trajectory = []
            
            metrics = self.compute_trajectory_similarity(
                predicted_trajectory=predicted_trajectory,
                expert_trajectory=demo,
                trajectory_id=trajectory_id
            )
            trajectory_metrics.append(metrics)
        
        # Compute aggregate metrics
        mean_surrogate_accuracy = mean([
            m.surrogate_metrics.correlation for m in trajectory_metrics
        ])
        
        mean_acquisition_accuracy = mean([
            m.acquisition_metrics.variable_selection_accuracy for m in trajectory_metrics
        ])
        
        mean_trajectory_quality = mean([
            m.overall_quality_score for m in trajectory_metrics
        ])
        
        # Stratify results by difficulty
        results_by_difficulty = self._stratify_by_difficulty(trajectory_metrics)
        
        # Run statistical significance tests
        significance_tests = self._run_significance_tests(trajectory_metrics)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(trajectory_metrics)
        
        return ValidationResults(
            individual_trajectories=trajectory_metrics,
            mean_surrogate_accuracy=mean_surrogate_accuracy,
            mean_acquisition_accuracy=mean_acquisition_accuracy,
            mean_trajectory_quality=mean_trajectory_quality,
            results_by_difficulty=results_by_difficulty,
            significance_tests=significance_tests,
            summary_stats=summary_stats
        )
    
    def _prepare_model_input(
        self,
        avici_data: jnp.ndarray,
        demo: ExpertDemonstration,
        step: int
    ) -> Dict[str, Any]:
        """Prepare input for surrogate model at specific step."""
        return {
            'observational_data': avici_data,
            'target_variable': demo.target_variable,
            'step': step
        }
    
    def _predict_posterior(
        self,
        model: Any,
        model_input: Dict[str, Any]
    ) -> Dict[frozenset, float]:
        """Get posterior prediction from surrogate model."""
        # Placeholder implementation
        # In reality, would call the actual trained model
        return {
            frozenset(['X1']): 0.6,
            frozenset(['X2']): 0.3,
            frozenset(): 0.1
        }
    
    def _predict_action(
        self,
        policy: Any,
        state: AcquisitionState
    ) -> Tuple[frozenset, Tuple[float, ...]]:
        """Get action prediction from acquisition policy."""
        # Placeholder implementation
        # In reality, would call the actual trained policy
        variables = list(state.scm_info.get('variables', frozenset()))
        if variables:
            selected_var = variables[0]  # Simplified selection
            return frozenset([selected_var]), (0.5,)
        else:
            return frozenset(), tuple()
    
    def _compute_posterior_similarity(
        self,
        model_posteriors: List[Dict[frozenset, float]],
        expert_posteriors: List[Dict[frozenset, float]]
    ) -> PosteriorSimilarityMetrics:
        """Compute similarity metrics between model and expert posteriors."""
        if not model_posteriors or not expert_posteriors:
            return self._create_default_posterior_metrics()
        
        # Compute metrics for each step and average
        kl_divergences = []
        js_divergences = []
        correlations = []
        top_k_overlaps = []
        entropy_diffs = []
        
        min_length = min(len(model_posteriors), len(expert_posteriors))
        
        for i in range(min_length):
            model_post = model_posteriors[i]
            expert_post = expert_posteriors[i]
            
            # KL divergence
            kl_div = self._compute_kl_divergence(model_post, expert_post)
            kl_divergences.append(kl_div)
            
            # JS divergence
            js_div = self._compute_js_divergence(model_post, expert_post)
            js_divergences.append(js_div)
            
            # Correlation
            corr = self._compute_posterior_correlation(model_post, expert_post)
            correlations.append(corr)
            
            # Top-k overlap
            overlap = self._compute_top_k_overlap(model_post, expert_post, self.top_k)
            top_k_overlaps.append(overlap)
            
            # Entropy difference
            entropy_diff = self._compute_entropy_difference(model_post, expert_post)
            entropy_diffs.append(entropy_diff)
        
        return PosteriorSimilarityMetrics(
            kl_divergence=mean(kl_divergences) if kl_divergences else float('inf'),
            js_divergence=mean(js_divergences) if js_divergences else 1.0,
            correlation=mean(correlations) if correlations else 0.0,
            top_k_overlap=mean(top_k_overlaps) if top_k_overlaps else 0.0,
            entropy_difference=mean(entropy_diffs) if entropy_diffs else 1.0
        )
    
    def _compute_intervention_accuracy(
        self,
        policy_choices: List[Tuple[frozenset, Tuple[float, ...]]],
        expert_interventions: List[Tuple[frozenset, Tuple[float, ...]]],
        step_accuracies: List[float],
        step_difficulties: List[float]
    ) -> InterventionAccuracyMetrics:
        """Compute intervention accuracy metrics."""
        if not policy_choices or not expert_interventions:
            return self._create_default_intervention_metrics()
        
        # Variable selection accuracy
        variable_matches = 0
        exact_matches = 0
        value_mses = []
        
        min_length = min(len(policy_choices), len(expert_interventions))
        
        for i in range(min_length):
            policy_vars, policy_vals = policy_choices[i]
            expert_vars, expert_vals = expert_interventions[i]
            
            # Variable selection accuracy
            if policy_vars == expert_vars:
                variable_matches += 1
                
                # Exact match (variables and values)
                if policy_vals == expert_vals:
                    exact_matches += 1
                
                # Value MSE (only if variables match)
                if len(policy_vals) == len(expert_vals):
                    mse = sum(
                        (p - e) ** 2 for p, e in zip(policy_vals, expert_vals)
                    ) / len(policy_vals)
                    value_mses.append(mse)
        
        variable_selection_accuracy = variable_matches / min_length if min_length > 0 else 0.0
        exact_match_accuracy = exact_matches / min_length if min_length > 0 else 0.0
        intervention_value_mse = mean(value_mses) if value_mses else float('inf')
        
        # Strategy consistency (variance in step accuracies)
        strategy_consistency = 1.0 - (stdev(step_accuracies) if len(step_accuracies) > 1 else 0.0)
        
        return InterventionAccuracyMetrics(
            variable_selection_accuracy=variable_selection_accuracy,
            exact_match_accuracy=exact_match_accuracy,
            intervention_value_mse=intervention_value_mse,
            strategy_consistency=max(0.0, strategy_consistency),
            step_accuracies=step_accuracies,
            step_difficulties=step_difficulties
        )
    
    def _compute_action_accuracy(
        self,
        predicted_action: Tuple[frozenset, Tuple[float, ...]],
        expert_action: Tuple[frozenset, Tuple[float, ...]]
    ) -> float:
        """Compute accuracy for single action prediction."""
        pred_vars, pred_vals = predicted_action
        expert_vars, expert_vals = expert_action
        
        # Variable match score
        var_score = 1.0 if pred_vars == expert_vars else 0.0
        
        # Value similarity score (if variables match)
        if pred_vars == expert_vars and len(pred_vals) == len(expert_vals):
            if len(pred_vals) == 0:
                val_score = 1.0
            else:
                # Normalized MSE
                mse = sum((p - e) ** 2 for p, e in zip(pred_vals, expert_vals)) / len(pred_vals)
                val_score = 1.0 / (1.0 + mse)  # Convert MSE to similarity score
        else:
            val_score = 0.0
        
        # Combined score
        return 0.7 * var_score + 0.3 * val_score
    
    def _assess_step_difficulty(
        self,
        state: AcquisitionState,
        expert_action: Tuple[frozenset, Tuple[float, ...]]
    ) -> float:
        """Assess difficulty of prediction step."""
        # Simplified difficulty assessment
        n_variables = len(state.scm_info.get('variables', frozenset()))
        step_in_trajectory = len(state.intervention_history)
        
        # Higher difficulty for more variables and later steps
        difficulty = (
            (n_variables / 20.0) * 0.5 +  # Normalized by typical max variables
            (step_in_trajectory / 10.0) * 0.5  # Normalized by typical max steps
        )
        
        return min(difficulty, 1.0)
    
    def _create_default_posterior_metrics(self) -> PosteriorSimilarityMetrics:
        """Create default posterior metrics for error cases."""
        return PosteriorSimilarityMetrics(
            kl_divergence=float('inf'),
            js_divergence=1.0,
            correlation=0.0,
            top_k_overlap=0.0,
            entropy_difference=1.0
        )
    
    def _create_default_intervention_metrics(self) -> InterventionAccuracyMetrics:
        """Create default intervention metrics for error cases."""
        return InterventionAccuracyMetrics(
            variable_selection_accuracy=0.0,
            exact_match_accuracy=0.0,
            intervention_value_mse=float('inf'),
            strategy_consistency=0.0,
            step_accuracies=[],
            step_difficulties=[]
        )
    
    def _create_default_trajectory_metrics(self, trajectory_id: str) -> TrajectoryMetrics:
        """Create default trajectory metrics for error cases."""
        return TrajectoryMetrics(
            trajectory_id=trajectory_id,
            difficulty_level=DifficultyLevel.MEDIUM,
            surrogate_metrics=self._create_default_posterior_metrics(),
            acquisition_metrics=self._create_default_intervention_metrics(),
            final_accuracy_match=False,
            convergence_speed_ratio=1.0,
            trajectory_length_ratio=1.0,
            overall_quality_score=0.0,
            behavioral_fidelity=0.0
        )
    
    def _compare_final_accuracy(
        self,
        predicted_trajectory: List[Dict[str, Any]],
        expert_trajectory: ExpertDemonstration
    ) -> bool:
        """Compare final accuracy achieved by model vs expert."""
        # Simplified comparison
        return abs(1.0 - expert_trajectory.accuracy) < 0.1  # Within 10% of expert
    
    def _compute_convergence_speed_ratio(
        self,
        predicted_trajectory: List[Dict[str, Any]],
        expert_trajectory: ExpertDemonstration
    ) -> float:
        """Compute ratio of convergence speeds."""
        # Simplified implementation
        expert_iterations = expert_trajectory.parent_posterior.get('trajectory', {}).get('iterations', 10)
        predicted_iterations = len(predicted_trajectory) if predicted_trajectory else expert_iterations
        
        return predicted_iterations / expert_iterations if expert_iterations > 0 else 1.0
    
    def _compute_length_ratio(
        self,
        predicted_trajectory: List[Dict[str, Any]],
        expert_trajectory: ExpertDemonstration
    ) -> float:
        """Compute ratio of trajectory lengths."""
        expert_length = len(expert_trajectory.parent_posterior.get('posterior_history', []))
        predicted_length = len(predicted_trajectory) if predicted_trajectory else expert_length
        
        return predicted_length / expert_length if expert_length > 0 else 1.0
    
    def _compute_overall_quality(
        self,
        surrogate_metrics: PosteriorSimilarityMetrics,
        acquisition_metrics: InterventionAccuracyMetrics
    ) -> float:
        """Compute overall quality score [0, 1]."""
        surrogate_score = max(0.0, surrogate_metrics.correlation)
        acquisition_score = acquisition_metrics.variable_selection_accuracy
        
        return 0.5 * surrogate_score + 0.5 * acquisition_score
    
    def _compute_behavioral_fidelity(
        self,
        acquisition_metrics: InterventionAccuracyMetrics,
        surrogate_metrics: PosteriorSimilarityMetrics
    ) -> float:
        """Compute behavioral fidelity score [0, 1]."""
        return min(1.0, acquisition_metrics.strategy_consistency + 0.3 * surrogate_metrics.correlation)
    
    def _stratify_by_difficulty(
        self,
        trajectory_metrics: List[TrajectoryMetrics]
    ) -> Dict[DifficultyLevel, Dict[str, float]]:
        """Stratify results by difficulty level."""
        results = {}
        
        for level in DifficultyLevel:
            level_metrics = [m for m in trajectory_metrics if m.difficulty_level == level]
            
            if level_metrics:
                results[level] = {
                    'count': len(level_metrics),
                    'mean_surrogate_accuracy': mean([m.surrogate_metrics.correlation for m in level_metrics]),
                    'mean_acquisition_accuracy': mean([m.acquisition_metrics.variable_selection_accuracy for m in level_metrics]),
                    'mean_quality_score': mean([m.overall_quality_score for m in level_metrics])
                }
        
        return results
    
    def _run_significance_tests(
        self,
        trajectory_metrics: List[TrajectoryMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Run statistical significance tests."""
        # Simplified significance testing
        surrogate_scores = [m.surrogate_metrics.correlation for m in trajectory_metrics]
        acquisition_scores = [m.acquisition_metrics.variable_selection_accuracy for m in trajectory_metrics]
        
        # Test if scores are significantly different from random (0.5)
        if len(surrogate_scores) > 1:
            surrogate_t_stat, surrogate_p_value = stats.ttest_1samp(surrogate_scores, 0.5)
        else:
            surrogate_t_stat, surrogate_p_value = 0.0, 1.0
        
        if len(acquisition_scores) > 1:
            acquisition_t_stat, acquisition_p_value = stats.ttest_1samp(acquisition_scores, 0.5)
        else:
            acquisition_t_stat, acquisition_p_value = 0.0, 1.0
        
        return {
            'surrogate_vs_random': {
                't_statistic': surrogate_t_stat,
                'p_value': surrogate_p_value,
                'significant': surrogate_p_value < self.significance_level
            },
            'acquisition_vs_random': {
                't_statistic': acquisition_t_stat,
                'p_value': acquisition_p_value,
                'significant': acquisition_p_value < self.significance_level
            }
        }
    
    def _compute_summary_statistics(
        self,
        trajectory_metrics: List[TrajectoryMetrics]
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not trajectory_metrics:
            return {}
        
        surrogate_scores = [m.surrogate_metrics.correlation for m in trajectory_metrics]
        acquisition_scores = [m.acquisition_metrics.variable_selection_accuracy for m in trajectory_metrics]
        quality_scores = [m.overall_quality_score for m in trajectory_metrics]
        
        return {
            'n_trajectories': len(trajectory_metrics),
            'surrogate_stats': {
                'mean': mean(surrogate_scores),
                'std': stdev(surrogate_scores) if len(surrogate_scores) > 1 else 0.0,
                'min': min(surrogate_scores),
                'max': max(surrogate_scores)
            },
            'acquisition_stats': {
                'mean': mean(acquisition_scores),
                'std': stdev(acquisition_scores) if len(acquisition_scores) > 1 else 0.0,
                'min': min(acquisition_scores),
                'max': max(acquisition_scores)
            },
            'quality_stats': {
                'mean': mean(quality_scores),
                'std': stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                'min': min(quality_scores),
                'max': max(quality_scores)
            }
        }
    
    def _compute_kl_divergence(
        self,
        p: Dict[frozenset, float],
        q: Dict[frozenset, float]
    ) -> float:
        """Compute KL divergence between two posterior distributions."""
        # Get all parent sets
        all_sets = set(p.keys()) | set(q.keys())
        
        if not all_sets:
            return 0.0
        
        # Normalize distributions
        p_sum = sum(p.values())
        q_sum = sum(q.values())
        
        kl_div = 0.0
        for parent_set in all_sets:
            p_prob = p.get(parent_set, 1e-10) / p_sum if p_sum > 0 else 1e-10
            q_prob = q.get(parent_set, 1e-10) / q_sum if q_sum > 0 else 1e-10
            
            kl_div += p_prob * onp.log(p_prob / q_prob)
        
        return float(kl_div)
    
    def _compute_js_divergence(
        self,
        p: Dict[frozenset, float],
        q: Dict[frozenset, float]
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        # Create average distribution
        all_sets = set(p.keys()) | set(q.keys())
        p_sum = sum(p.values())
        q_sum = sum(q.values())
        
        m = {}
        for parent_set in all_sets:
            p_prob = p.get(parent_set, 0.0) / p_sum if p_sum > 0 else 0.0
            q_prob = q.get(parent_set, 0.0) / q_sum if q_sum > 0 else 0.0
            m[parent_set] = 0.5 * (p_prob + q_prob)
        
        # Compute JS divergence
        kl_pm = self._compute_kl_divergence(p, m)
        kl_qm = self._compute_kl_divergence(q, m)
        
        return 0.5 * kl_pm + 0.5 * kl_qm
    
    def _compute_posterior_correlation(
        self,
        p: Dict[frozenset, float],
        q: Dict[frozenset, float]
    ) -> float:
        """Compute correlation between posterior probabilities."""
        all_sets = sorted(set(p.keys()) | set(q.keys()))
        
        if len(all_sets) < 2:
            return 1.0 if len(all_sets) == 1 and p.get(all_sets[0], 0) > 0 and q.get(all_sets[0], 0) > 0 else 0.0
        
        p_sum = sum(p.values())
        q_sum = sum(q.values())
        
        p_probs = [p.get(ps, 0.0) / p_sum if p_sum > 0 else 0.0 for ps in all_sets]
        q_probs = [q.get(ps, 0.0) / q_sum if q_sum > 0 else 0.0 for ps in all_sets]
        
        try:
            correlation = stats.pearsonr(p_probs, q_probs)[0]
            return correlation if not onp.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _compute_top_k_overlap(
        self,
        p: Dict[frozenset, float],
        q: Dict[frozenset, float],
        k: int
    ) -> float:
        """Compute overlap in top-k parent sets."""
        # Get top-k from each distribution
        p_sorted = sorted(p.items(), key=lambda x: x[1], reverse=True)[:k]
        q_sorted = sorted(q.items(), key=lambda x: x[1], reverse=True)[:k]
        
        p_top_k = set(item[0] for item in p_sorted)
        q_top_k = set(item[0] for item in q_sorted)
        
        if not p_top_k and not q_top_k:
            return 1.0
        
        intersection = len(p_top_k & q_top_k)
        union = len(p_top_k | q_top_k)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_entropy_difference(
        self,
        p: Dict[frozenset, float],
        q: Dict[frozenset, float]
    ) -> float:
        """Compute difference in posterior entropies."""
        def entropy(dist):
            probs = list(dist.values())
            total = sum(probs)
            if total == 0:
                return 0.0
            normalized = [prob / total for prob in probs if prob > 0]
            return -sum(prob * onp.log(prob) for prob in normalized)
        
        p_entropy = entropy(p)
        q_entropy = entropy(q)
        
        return abs(p_entropy - q_entropy)


def create_trajectory_validator(
    surrogate_model: Optional[Any] = None,
    acquisition_policy: Optional[Any] = None,
    top_k: int = 5
) -> TrajectoryValidator:
    """
    Factory function to create trajectory validator.
    
    Args:
        surrogate_model: Trained surrogate model
        acquisition_policy: Trained acquisition policy
        top_k: Number of top parent sets to compare
        
    Returns:
        Configured TrajectoryValidator
    """
    return TrajectoryValidator(
        surrogate_model=surrogate_model,
        acquisition_policy=acquisition_policy,
        top_k=top_k
    )