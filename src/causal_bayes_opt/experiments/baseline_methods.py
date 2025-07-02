"""
Baseline Methods for ACBO Validation

This module implements various baseline methods for comparison with the
neural network-based ACBO approach, following functional programming principles.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, FrozenSet
from dataclasses import dataclass
from enum import Enum
import warnings

# Standard numerical libraries
import jax.numpy as jnp
import jax.random as random
import numpy as onp  # For I/O only
import pyrsistent as pyr

# Local imports
from ..data_structures.scm import get_variables, get_target, get_parents
from ..data_structures.sample import create_sample, get_values
from ..data_structures.buffer import ExperienceBuffer, create_empty_buffer
from ..mechanisms.linear import sample_from_linear_scm

logger = logging.getLogger(__name__)


class BaselineType(Enum):
    """Types of baseline methods for comparison."""
    RANDOM_POLICY = "random_policy"
    GREEDY_POLICY = "greedy_policy"
    ORACLE_POLICY = "oracle_policy"
    UNIFORM_STRUCTURE = "uniform_structure"
    TRUE_STRUCTURE = "true_structure"
    MAJORITY_CLASS = "majority_class"


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline methods."""
    baseline_type: BaselineType
    intervention_budget: int = 20
    intervention_strength: float = 2.0
    exploration_rate: float = 0.1
    random_seed: int = 42


@dataclass(frozen=True)
class BaselineResult:
    """Results from running a baseline method."""
    baseline_type: BaselineType
    final_target_value: float
    target_improvement: float
    structure_accuracy: float
    sample_efficiency: float
    intervention_count: int
    convergence_steps: int
    intervention_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class RandomPolicyBaseline:
    """
    Random intervention policy baseline.
    
    Selects interventions uniformly at random from available variables,
    providing a lower bound for policy performance.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize random policy baseline.
        
        Args:
            config: Baseline configuration
        """
        self.config = config
        self.key = random.PRNGKey(config.random_seed)
    
    def select_intervention(
        self,
        available_variables: List[str],
        current_state: Optional[Dict[str, Any]] = None
    ) -> pyr.PMap:
        """
        Select a random intervention.
        
        Args:
            available_variables: Variables that can be intervened on
            current_state: Current state (ignored for random policy)
            
        Returns:
            Intervention PMap
        """
        if not available_variables:
            return pyr.pmap({})
        
        self.key, subkey = random.split(self.key)
        
        # Select random variable
        var_idx = random.randint(subkey, (), 0, len(available_variables))
        selected_var = available_variables[int(var_idx)]
        
        # Select random intervention value
        _, value_key = random.split(subkey)
        intervention_value = random.normal(value_key) * self.config.intervention_strength
        
        return pyr.pmap({
            'type': 'perfect',
            'targets': frozenset([selected_var]),
            'values': pyr.pmap({selected_var: float(intervention_value)}),
            'selected_by': 'random_policy'
        })
    
    def update(self, intervention: Dict[str, Any], outcome: pyr.PMap) -> None:
        """
        Update policy (no-op for random policy).
        
        Args:
            intervention: The intervention that was applied
            outcome: The resulting outcome
        """
        pass  # Random policy doesn't learn


class GreedyPolicyBaseline:
    """
    Greedy intervention policy baseline.
    
    Uses simple heuristics to select interventions, such as:
    - Intervening on variables with highest observed variance
    - Alternating between different variables
    - Focusing on variables that seem to affect the target
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize greedy policy baseline.
        
        Args:
            config: Baseline configuration
        """
        self.config = config
        self.variable_scores: Dict[str, float] = {}
        self.intervention_counts: Dict[str, int] = {}
        self.step_count = 0
    
    def select_intervention(
        self,
        available_variables: List[str],
        current_state: Optional[Dict[str, Any]] = None
    ) -> pyr.PMap:
        """
        Select intervention using greedy heuristics.
        
        Args:
            available_variables: Variables that can be intervened on
            current_state: Current state including buffer and target info
            
        Returns:
            Intervention PMap
        """
        if not available_variables:
            return pyr.pmap({})
        
        # Initialize scores if needed
        for var in available_variables:
            if var not in self.variable_scores:
                self.variable_scores[var] = 0.0
                self.intervention_counts[var] = 0
        
        # Use exploration vs exploitation
        rng = onp.random.RandomState(self.config.random_seed + self.step_count)
        
        if rng.random() < self.config.exploration_rate:
            # Exploration: select least tried variable
            min_count = min(self.intervention_counts[var] for var in available_variables)
            candidates = [var for var in available_variables 
                         if self.intervention_counts[var] == min_count]
            selected_var = rng.choice(candidates)
        else:
            # Exploitation: select highest scoring variable
            scores = [(var, self.variable_scores[var]) for var in available_variables]
            selected_var = max(scores, key=lambda x: x[1])[0]
        
        # Generate intervention value
        intervention_value = rng.normal(0, self.config.intervention_strength)
        
        self.step_count += 1
        
        return pyr.pmap({
            'type': 'perfect',
            'targets': frozenset([selected_var]),
            'values': pyr.pmap({selected_var: float(intervention_value)}),
            'selected_by': 'greedy_policy'
        })
    
    def update(self, intervention: Dict[str, Any], outcome: pyr.PMap) -> None:
        """
        Update variable scores based on intervention outcome.
        
        Args:
            intervention: The intervention that was applied
            outcome: The resulting outcome
        """
        if 'values' not in intervention:
            return
        
        outcome_values = get_values(outcome)
        target_value = outcome_values.get('target', 0.0)  # Simplified
        
        # Update scores for intervened variables
        for var in intervention['values']:
            if var in self.variable_scores:
                # Simple scoring: higher target values → higher scores
                self.variable_scores[var] = 0.9 * self.variable_scores[var] + 0.1 * target_value
                self.intervention_counts[var] += 1


class OraclePolicyBaseline:
    """
    Oracle intervention policy baseline.
    
    Has perfect knowledge of the true causal structure and mechanisms,
    providing an upper bound for policy performance.
    """
    
    def __init__(self, config: BaselineConfig, true_scm: pyr.PMap):
        """
        Initialize oracle policy baseline.
        
        Args:
            config: Baseline configuration
            true_scm: True structural causal model
        """
        self.config = config
        self.true_scm = true_scm
        self.target = get_target(true_scm)
        
        # Compute true parent relationships
        if self.target:
            self.true_parents = get_parents(true_scm, self.target)
        else:
            self.true_parents = frozenset()
    
    def select_intervention(
        self,
        available_variables: List[str],
        current_state: Optional[Dict[str, Any]] = None
    ) -> pyr.PMap:
        """
        Select intervention using perfect knowledge.
        
        Args:
            available_variables: Variables that can be intervened on
            current_state: Current state (optional)
            
        Returns:
            Intervention PMap
        """
        if not available_variables:
            return pyr.pmap({})
        
        # Prefer intervening on true parents of the target
        parent_candidates = [var for var in available_variables if var in self.true_parents]
        
        if parent_candidates:
            # Select from true parents
            rng = onp.random.RandomState(self.config.random_seed)
            selected_var = rng.choice(parent_candidates)
            
            # Use optimal intervention value (simplified: maximize expected target)
            intervention_value = self.config.intervention_strength
            
        else:
            # Fall back to random selection if no parents available
            rng = onp.random.RandomState(self.config.random_seed)
            selected_var = rng.choice(available_variables)
            intervention_value = rng.normal(0, self.config.intervention_strength)
        
        return pyr.pmap({
            'type': 'perfect',
            'targets': frozenset([selected_var]),
            'values': pyr.pmap({selected_var: float(intervention_value)}),
            'selected_by': 'oracle_policy'
        })
    
    def update(self, intervention: Dict[str, Any], outcome: pyr.PMap) -> None:
        """
        Update policy (oracle doesn't need to learn).
        
        Args:
            intervention: The intervention that was applied
            outcome: The resulting outcome
        """
        pass  # Oracle has perfect knowledge


class UniformStructureBaseline:
    """
    Uniform structure learning baseline.
    
    Predicts uniform probability distributions over all possible parent sets,
    representing the worst-case scenario for structure learning.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize uniform structure baseline.
        
        Args:
            config: Baseline configuration
        """
        self.config = config
    
    def predict_parents(
        self,
        variables: List[str],
        target_variable: str,
        data: Optional[onp.ndarray] = None
    ) -> Dict[str, float]:
        """
        Predict uniform parent probabilities.
        
        Args:
            variables: All variables in the system
            target_variable: Variable to predict parents for
            data: Observational data (ignored)
            
        Returns:
            Dictionary mapping variables to uniform parent probabilities
        """
        potential_parents = [var for var in variables if var != target_variable]
        uniform_prob = 1.0 / len(potential_parents) if potential_parents else 0.0
        
        return {var: uniform_prob for var in potential_parents}


class TrueStructureBaseline:
    """
    True structure baseline (oracle).
    
    Returns the true parent relationships, providing an upper bound
    for structure learning performance.
    """
    
    def __init__(self, config: BaselineConfig, true_scm: pyr.PMap):
        """
        Initialize true structure baseline.
        
        Args:
            config: Baseline configuration
            true_scm: True structural causal model
        """
        self.config = config
        self.true_scm = true_scm
    
    def predict_parents(
        self,
        variables: List[str],
        target_variable: str,
        data: Optional[onp.ndarray] = None
    ) -> Dict[str, float]:
        """
        Return true parent relationships.
        
        Args:
            variables: All variables in the system
            target_variable: Variable to predict parents for
            data: Observational data (ignored)
            
        Returns:
            Dictionary with 1.0 for true parents, 0.0 for non-parents
        """
        true_parents = get_parents(self.true_scm, target_variable)
        
        result = {}
        for var in variables:
            if var != target_variable:
                result[var] = 1.0 if var in true_parents else 0.0
        
        return result


class MajorityClassBaseline:
    """
    Majority class baseline for structure learning.
    
    Always predicts the most common parent relationship (typically "no parent")
    across all variables, representing a naive baseline.
    """
    
    def __init__(self, config: BaselineConfig, assume_sparse: bool = True):
        """
        Initialize majority class baseline.
        
        Args:
            config: Baseline configuration
            assume_sparse: If True, assumes most variables are not parents (sparse graphs)
        """
        self.config = config
        self.assume_sparse = assume_sparse
    
    def predict_parents(
        self,
        variables: List[str],
        target_variable: str,
        data: Optional[onp.ndarray] = None
    ) -> Dict[str, float]:
        """
        Predict majority class (usually "no parent").
        
        Args:
            variables: All variables in the system
            target_variable: Variable to predict parents for
            data: Observational data (ignored)
            
        Returns:
            Dictionary with majority class probabilities
        """
        potential_parents = [var for var in variables if var != target_variable]
        
        # For sparse graphs, majority class is "not a parent"
        majority_prob = 0.0 if self.assume_sparse else 1.0
        
        return {var: majority_prob for var in potential_parents}


def run_baseline_experiment(
    baseline_type: BaselineType,
    scm: pyr.PMap,
    config: BaselineConfig
) -> BaselineResult:
    """
    Run a complete baseline experiment.
    
    Args:
        baseline_type: Type of baseline to run
        scm: True structural causal model
        config: Baseline configuration
        
    Returns:
        BaselineResult with performance metrics
    """
    logger.info(f"Running baseline experiment: {baseline_type.value}")
    
    variables = list(get_variables(scm))
    target = get_target(scm)
    
    if not target:
        raise ValueError("SCM must have a target variable for baseline experiments")
    
    # Initialize baseline method
    if baseline_type == BaselineType.RANDOM_POLICY:
        policy = RandomPolicyBaseline(config)
    elif baseline_type == BaselineType.GREEDY_POLICY:
        policy = GreedyPolicyBaseline(config)
    elif baseline_type == BaselineType.ORACLE_POLICY:
        policy = OraclePolicyBaseline(config, scm)
    else:
        raise ValueError(f"Policy baseline type {baseline_type.value} not implemented")
    
    # Run experiment
    buffer = create_empty_buffer()
    intervention_history = []
    target_values = []
    
    # Generate initial observational data
    initial_samples = sample_from_linear_scm(scm, n_samples=10, seed=config.random_seed)
    for sample in initial_samples:
        buffer.add_observation(sample)
        target_values.append(get_values(sample)[target])
    
    initial_target = max(target_values) if target_values else 0.0
    
    # Run intervention loop
    for step in range(config.intervention_budget):
        # Select intervention
        available_vars = [var for var in variables if var != target]
        intervention = policy.select_intervention(available_vars)
        
        if not intervention or 'values' not in intervention:
            # Skip if no valid intervention
            continue
        
        # Apply intervention and sample outcome
        outcome_samples = sample_with_intervention_simple(scm, intervention, n_samples=1, seed=config.random_seed + step)
        if outcome_samples:
            outcome = outcome_samples[0]
            buffer.add_intervention(intervention, outcome)
            
            # Track target value
            outcome_value = get_values(outcome)[target]
            target_values.append(outcome_value)
            
            # Update policy
            policy.update(intervention, outcome)
            
            # Record intervention
            intervention_history.append({
                'step': step,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'target_improvement': outcome_value - initial_target
            })
    
    # Compute final metrics
    final_target = max(target_values) if target_values else initial_target
    target_improvement = final_target - initial_target
    sample_efficiency = target_improvement / len(target_values) if target_values else 0.0
    
    # Structure accuracy (only for structure learning baselines)
    structure_accuracy = 0.0
    if baseline_type in [BaselineType.UNIFORM_STRUCTURE, BaselineType.TRUE_STRUCTURE, BaselineType.MAJORITY_CLASS]:
        structure_accuracy = _compute_structure_accuracy_baseline(baseline_type, scm, config)
    
    return BaselineResult(
        baseline_type=baseline_type,
        final_target_value=final_target,
        target_improvement=target_improvement,
        structure_accuracy=structure_accuracy,
        sample_efficiency=sample_efficiency,
        intervention_count=len(intervention_history),
        convergence_steps=len(target_values),
        intervention_history=intervention_history,
        metadata={
            'initial_target': initial_target,
            'config': config.__dict__,
            'total_samples': len(target_values)
        }
    )


def _compute_structure_accuracy_baseline(
    baseline_type: BaselineType,
    scm: pyr.PMap,
    config: BaselineConfig
) -> float:
    """
    Compute structure learning accuracy for structure learning baselines.
    
    Args:
        baseline_type: Type of baseline
        scm: True structural causal model
        config: Baseline configuration
        
    Returns:
        Structure accuracy score
    """
    variables = list(get_variables(scm))
    target = get_target(scm)
    
    if not target:
        return 0.0
    
    # Initialize baseline
    if baseline_type == BaselineType.UNIFORM_STRUCTURE:
        baseline = UniformStructureBaseline(config)
    elif baseline_type == BaselineType.TRUE_STRUCTURE:
        baseline = TrueStructureBaseline(config, scm)
    elif baseline_type == BaselineType.MAJORITY_CLASS:
        baseline = MajorityClassBaseline(config)
    else:
        return 0.0
    
    # Get predictions
    predicted_probs = baseline.predict_parents(variables, target)
    
    # Get true parents
    true_parents = get_parents(scm, target)
    
    # Compute accuracy
    correct = 0
    total = 0
    
    for var in variables:
        if var != target:
            is_true_parent = var in true_parents
            predicted_prob = predicted_probs.get(var, 0.0)
            predicted_parent = predicted_prob > 0.5
            
            if is_true_parent == predicted_parent:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def sample_with_intervention_simple(
    scm: pyr.PMap,
    intervention: Dict[str, Any],
    n_samples: int = 1,
    seed: int = 42
) -> List[pyr.PMap]:
    """
    Simple intervention sampling for baseline experiments.
    
    Args:
        scm: Structural causal model
        intervention: Intervention specification
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        List of samples under intervention
    """
    # This is a simplified implementation
    # In practice, would use the full intervention mechanism
    try:
        from ..environments.sampling import sample_with_intervention
        return sample_with_intervention(scm, intervention, n_samples, seed)
    except ImportError:
        # Fallback: sample from original SCM (not ideal but functional for testing)
        logger.warning("Using fallback intervention sampling")
        return sample_from_linear_scm(scm, n_samples, seed)


def compare_baselines(
    scm: pyr.PMap,
    baseline_types: List[BaselineType] = None,
    config: BaselineConfig = None
) -> Dict[str, BaselineResult]:
    """
    Compare multiple baseline methods on the same SCM.
    
    Args:
        scm: Structural causal model to test on
        baseline_types: List of baseline types to compare
        config: Configuration for baselines
        
    Returns:
        Dictionary mapping baseline names to results
    """
    if baseline_types is None:
        baseline_types = [
            BaselineType.RANDOM_POLICY,
            BaselineType.GREEDY_POLICY,
            BaselineType.ORACLE_POLICY
        ]
    
    if config is None:
        config = BaselineConfig()
    
    results = {}
    
    for baseline_type in baseline_types:
        try:
            result = run_baseline_experiment(baseline_type, scm, config)
            results[baseline_type.value] = result
            logger.info(f"Baseline {baseline_type.value}: improvement={result.target_improvement:.3f}")
        except Exception as e:
            logger.error(f"Failed to run baseline {baseline_type.value}: {e}")
            # Create error result
            results[baseline_type.value] = BaselineResult(
                baseline_type=baseline_type,
                final_target_value=0.0,
                target_improvement=0.0,
                structure_accuracy=0.0,
                sample_efficiency=0.0,
                intervention_count=0,
                convergence_steps=0,
                intervention_history=[],
                metadata={'error': str(e)}
            )
    
    return results


# Export public interface
__all__ = [
    'BaselineType',
    'BaselineConfig',
    'BaselineResult',
    'RandomPolicyBaseline',
    'GreedyPolicyBaseline',
    'OraclePolicyBaseline',
    'UniformStructureBaseline',
    'TrueStructureBaseline',
    'MajorityClassBaseline',
    'run_baseline_experiment',
    'compare_baselines'
]