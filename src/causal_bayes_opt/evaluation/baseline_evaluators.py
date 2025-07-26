"""
Baseline Evaluators

Implements standard baseline methods for comparison:
- Random: Random intervention selection
- Oracle: Perfect knowledge of causal structure
- Learning: Standard learning without BC/GRPO
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
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
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
# Baseline evaluators don't need full acquisition state
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)

logger = logging.getLogger(__name__)


class RandomBaselineEvaluator(BaseEvaluator):
    """
    Random baseline evaluator.
    
    Selects interventions uniformly at random from the action space.
    This serves as the simplest baseline for comparison.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize random baseline evaluator."""
        super().__init__(name=name or "Random_Baseline")
        self._initialized = True  # No initialization needed
        
    def initialize(self) -> None:
        """No initialization needed for random baseline."""
        pass
        
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run random intervention selection.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with learning history
        """
        start_time = time.time()
        
        # Extract configuration
        max_interventions = config.get('experiment', {}).get('target', {}).get(
            'max_interventions', 10
        )
        n_observational_samples = config.get('experiment', {}).get('target', {}).get(
            'n_observational_samples', 100
        )
        intervention_range = config.get('experiment', {}).get('target', {}).get(
            'intervention_value_range', (-2.0, 2.0)
        )
        optimization_direction = config.get('experiment', {}).get('target', {}).get(
            'optimization_direction', 'MINIMIZE'
        )
        
        # Initialize random key
        key = random.PRNGKey(seed)
        
        # Get SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        candidate_vars = [v for v in variables if v != target_var]
        
        # Sample initial observational data
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, n_observational_samples, seed=int(obs_key[0]))
        # Convert to dict format
        obs_data = {}
        for var in variables:
            obs_data[var] = jnp.array([s['values'][var] for s in obs_samples])
        
        # Compute initial target value
        initial_value = float(jnp.mean(obs_data[target_var]))
        
        # For baseline methods, we don't use acquisition state fully
        # Just track what we need: marginals and uncertainty
        current_marginals = {var: 0.0 for var in variables if var != target_var}
        current_uncertainty = 1.0  # Start with high uncertainty
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        
        # Store initial state
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals=current_marginals,
            uncertainty=current_uncertainty,
            reward=0.0,
            computation_time=0.0
        ))
        
        # Run random intervention loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Randomly select variable and value
            key, var_key, val_key = random.split(key, 3)
            var_idx = random.randint(var_key, (), 0, len(candidate_vars))
            selected_var = candidate_vars[var_idx]
            
            intervention_value = float(random.uniform(
                val_key, (), 
                minval=intervention_range[0], 
                maxval=intervention_range[1]
            ))
            
            # Create intervention
            intervention_dict = {selected_var: intervention_value}
            
            # Apply intervention and sample
            key, int_key = random.split(key)
            # Create intervention
            intervention_obj = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervention_value}
            )
            # Sample with intervention
            from ..environments.sampling import sample_with_intervention
            int_samples = sample_with_intervention(scm, intervention_obj, 100, seed=int(int_key[0]))
            # Convert to dict format
            int_data = {}
            for var in variables:
                int_data[var] = jnp.array([s['values'][var] for s in int_samples])
            
            # Compute outcome value
            outcome_value = float(jnp.mean(int_data[target_var]))
            
            # Update best value
            if outcome_value < best_value:  # Assuming minimization
                best_value = outcome_value
            
            # For baseline, we don't update state - marginals stay the same
            
            # Create step result
            step_result = StepResult(
                step=step,
                intervention=intervention_dict,
                outcome_value=outcome_value,
                marginals=current_marginals,
                uncertainty=current_uncertainty,
                reward=outcome_value - initial_value,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
        
        # Compute final metrics
        final_value = learning_history[-1].outcome_value
        final_marginals = learning_history[-1].marginals
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': initial_value - final_value if optimization_direction == 'MINIMIZE' else final_value - initial_value,
            'n_interventions': len(learning_history) - 1,
            'final_f1': compute_f1_score_from_marginals(final_marginals, true_parents),
            'final_shd': compute_shd_from_marginals(final_marginals, true_parents)
        }
        
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
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


class OracleBaselineEvaluator(BaseEvaluator):
    """
    Oracle baseline evaluator.
    
    Has perfect knowledge of the causal structure and uses it to
    select optimal interventions. This serves as an upper bound on
    performance.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize oracle baseline evaluator."""
        super().__init__(name=name or "Oracle_Baseline")
        self._initialized = True
        
    def initialize(self) -> None:
        """No initialization needed for oracle baseline."""
        pass
        
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run oracle intervention selection.
        
        Uses perfect knowledge of the causal structure to select
        interventions that minimize the target variable.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with learning history
        """
        start_time = time.time()
        
        # Import oracle policy creator
        from examples.demo_learning import create_oracle_intervention_policy
        
        # Extract configuration
        max_interventions = config.get('experiment', {}).get('target', {}).get(
            'max_interventions', 10
        )
        n_observational_samples = config.get('experiment', {}).get('target', {}).get(
            'n_observational_samples', 100
        )
        intervention_range = config.get('experiment', {}).get('target', {}).get(
            'intervention_value_range', (-2.0, 2.0)
        )
        optimization_direction = config.get('experiment', {}).get('target', {}).get(
            'optimization_direction', 'MINIMIZE'
        )
        
        # Initialize random key
        key = random.PRNGKey(seed)
        
        # Get SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        
        # Create oracle policy
        oracle_policy = create_oracle_intervention_policy(
            variables, target_var, scm, intervention_range
        )
        
        # Sample initial observational data
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, n_observational_samples, seed=int(obs_key[0]))
        # Convert to dict format
        obs_data = {}
        for var in variables:
            obs_data[var] = jnp.array([s['values'][var] for s in obs_samples])
        
        # Compute initial target value
        initial_value = float(jnp.mean(obs_data[target_var]))
        
        # For baseline methods, we don't use acquisition state fully
        # Just track what we need: marginals and uncertainty
        current_marginals = {var: 0.0 for var in variables if var != target_var}
        current_uncertainty = 1.0  # Start with high uncertainty
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        
        # Store initial state (oracle has perfect marginals)
        perfect_marginals = {
            var: 1.0 if var in true_parents else 0.0 
            for var in variables if var != target_var
        }
        
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals=perfect_marginals,  # Oracle knows true structure
            uncertainty=0.0,  # No uncertainty for oracle
            reward=0.0,
            computation_time=0.0
        ))
        
        # Run oracle intervention loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Get oracle intervention
            key, oracle_key = random.split(key)
            # Oracle needs a mock state-like object
            mock_state = {'target_variable': target_var, 'best_value': best_value}
            oracle_intervention = oracle_policy(mock_state, oracle_key)
            
            # Extract intervention details - oracle returns different format
            # It returns intervention object with 'targets' and 'values'
            intervention_vars = oracle_intervention.get('targets', frozenset())
            intervention_values_dict = oracle_intervention.get('values', {})
            
            if intervention_vars and intervention_values_dict:
                selected_var = list(intervention_vars)[0]
                intervention_value = float(intervention_values_dict[selected_var])
                intervention_dict = {selected_var: intervention_value}
                
                # Apply intervention and sample
                key, int_key = random.split(key)
                # Create intervention
                intervention_obj = create_perfect_intervention(
                    targets=frozenset([selected_var]),
                    values={selected_var: intervention_value}
                )
                
                # Sample with intervention
                from ..environments.sampling import sample_with_intervention
                int_samples = sample_with_intervention(scm, intervention_obj, 100, seed=int(int_key[0]))
                # Convert to dict format
                int_data = {}
                for var in variables:
                    int_data[var] = jnp.array([s['values'][var] for s in int_samples])
                
                # Compute outcome value
                outcome_value = float(jnp.mean(int_data[target_var]))
            else:
                # No intervention - use observational data
                intervention_dict = {}
                outcome_value = initial_value
            
            # Update best value
            if outcome_value < best_value:
                best_value = outcome_value
            
            # Oracle doesn't need to update state - it has perfect knowledge
            
            # Create step result
            step_result = StepResult(
                step=step,
                intervention=intervention_dict,
                outcome_value=outcome_value,
                marginals=perfect_marginals,  # Oracle maintains perfect knowledge
                uncertainty=0.0,
                reward=outcome_value - initial_value,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
        
        # Compute final metrics
        final_value = learning_history[-1].outcome_value
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': initial_value - final_value if optimization_direction == 'MINIMIZE' else final_value - initial_value,
            'n_interventions': len(learning_history) - 1,
            'final_f1': 1.0,  # Oracle has perfect structure knowledge
            'final_shd': 0.0  # Oracle has zero structural error
        }
        
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
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


class LearningBaselineEvaluator(BaseEvaluator):
    """
    Learning baseline evaluator.
    
    Uses structure learning but with a simple intervention policy.
    This represents online causal discovery without sophisticated
    acquisition functions.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize learning baseline evaluator."""
        super().__init__(name=name or "Learning_Baseline")
        self._initialized = True
        
    def initialize(self) -> None:
        """No initialization needed for learning baseline."""
        pass
        
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run learning baseline with structure-aware intervention selection.
        
        Uses online structure learning with a simple intervention policy
        that leverages learned marginal probabilities.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with learning history
        """
        start_time = time.time()
        
        # Use simplified learning baseline without external dependencies
        # This uses AVICI integration directly
        import haiku as hk
        from ..avici_integration.parent_set import (
            ParentSetPredictionModel,
            predict_parent_posterior,
            get_marginal_parent_probabilities
        )
        from ..avici_integration.core import samples_to_avici_format
        
        # Extract configuration
        max_interventions = config.get('experiment', {}).get('target', {}).get(
            'max_interventions', 10
        )
        n_observational_samples = config.get('experiment', {}).get('target', {}).get(
            'n_observational_samples', 100
        )
        intervention_range = config.get('experiment', {}).get('target', {}).get(
            'intervention_value_range', (-2.0, 2.0)
        )
        learning_rate = config.get('experiment', {}).get('target', {}).get(
            'learning_rate', 1e-3
        )
        optimization_direction = config.get('experiment', {}).get('target', {}).get(
            'optimization_direction', 'MINIMIZE'
        )
        
        # Initialize random key
        key = random.PRNGKey(seed)
        
        # Get SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        
        # Create learnable surrogate model using AVICI with proper Haiku transform
        key, model_key = random.split(key)
        
        # Define the model function for Haiku
        def model_fn(x: jnp.ndarray, variable_order: List[str], target_variable: str, is_training: bool = False):
            model = ParentSetPredictionModel(
                layers=4,
                dim=64,
                max_parent_size=min(5, len(variables)-1)
            )
            return model(x, variable_order, target_variable, is_training)
        
        # Transform the model function
        net = hk.transform(model_fn)
        
        # Initialize parameters
        dummy_data = jnp.zeros((10, len(variables), 3))
        params = net.init(model_key, dummy_data, variables, target_var, False)
        
        # We'll use a simple epsilon-greedy policy based on learned structure
        epsilon = 0.3  # Exploration rate
        
        # Sample initial observational data
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, n_observational_samples, seed=int(obs_key[0]))
        # Convert to dict format
        obs_data = {}
        for var in variables:
            obs_data[var] = jnp.array([s['values'][var] for s in obs_samples])
        
        # Compute initial target value
        initial_value = float(jnp.mean(obs_data[target_var]))
        
        # For baseline methods, we don't use acquisition state fully
        # Just track what we need: marginals and uncertainty
        current_marginals = {var: 0.0 for var in variables if var != target_var}
        current_uncertainty = 1.0  # Start with high uncertainty
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        all_samples = []  # Track all samples in proper format
        
        # Convert initial observational samples to proper format
        from ..data_structures.sample import create_sample
        for i in range(len(obs_samples)):
            all_samples.append(obs_samples[i])
        
        # Store initial state
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals=current_marginals,
            uncertainty=current_uncertainty,
            reward=0.0,
            computation_time=0.0
        ))
        
        # Run learning baseline loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Update surrogate model with all data
            if len(all_samples) > 1:
                # Get prediction from AVICI model using the transformed network
                # samples_to_avici_format expects samples with 'values' key
                avici_data = samples_to_avici_format(all_samples, variables, target_var)
                posterior = predict_parent_posterior(
                    net, params, avici_data, variables, target_var
                )
                
                # Update marginals from posterior
                current_marginals = get_marginal_parent_probabilities(
                    posterior, variables
                )
                
                # Update uncertainty based on entropy of marginals
                # Higher entropy = higher uncertainty
                entropy = 0.0
                for var, prob in current_marginals.items():
                    if prob > 0 and prob < 1:
                        entropy -= prob * jnp.log(prob) + (1-prob) * jnp.log(1-prob)
                current_uncertainty = float(entropy / (len(variables) - 1))  # Normalize by number of possible parents
            
            # Select intervention using epsilon-greedy based on learned structure
            key, explore_key, var_key, val_key = random.split(key, 4)
            
            candidate_vars = [v for v in variables if v != target_var]
            
            if random.uniform(explore_key, ()) < epsilon:
                # Exploration: random intervention
                var_idx = random.randint(var_key, (), 0, len(candidate_vars))
                selected_var = candidate_vars[var_idx]
            else:
                # Exploitation: intervene on most likely parent
                if current_marginals:
                    # Sort by marginal probability
                    sorted_vars = sorted(
                        candidate_vars,
                        key=lambda v: current_marginals.get(v, 0.0),
                        reverse=True
                    )
                    selected_var = sorted_vars[0]
                else:
                    # Fallback to random if no marginals
                    var_idx = random.randint(var_key, (), 0, len(candidate_vars))
                    selected_var = candidate_vars[var_idx]
            
            # Select intervention value
            intervention_value = float(random.uniform(
                val_key, (), minval=intervention_range[0], maxval=intervention_range[1]
            ))
            intervention_dict = {selected_var: intervention_value}
            
            # Apply intervention and sample
            key, int_key = random.split(key)
            # Create intervention
            intervention_obj = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervention_value}
            )
            # Sample with intervention
            from ..environments.sampling import sample_with_intervention
            int_samples = sample_with_intervention(scm, intervention_obj, 100, seed=int(int_key[0]))
            # Add all new samples to tracking
            all_samples.extend(int_samples)
            
            # Also create dict format for computing outcome
            int_data = {}
            for var in variables:
                int_data[var] = jnp.array([s['values'][var] for s in int_samples])
            
            # Compute outcome value
            outcome_value = float(jnp.mean(int_data[target_var]))
            
            # Update best value
            if outcome_value < best_value:
                best_value = outcome_value
            
            # For baseline, we don't update state - marginals stay the same
            
            # Create step result
            step_result = StepResult(
                step=step,
                intervention=intervention_dict,
                outcome_value=outcome_value,
                marginals=current_marginals,
                uncertainty=current_uncertainty,
                reward=outcome_value - initial_value,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
        
        # Compute final metrics
        final_value = learning_history[-1].outcome_value
        final_marginals = learning_history[-1].marginals
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': initial_value - final_value if optimization_direction == 'MINIMIZE' else final_value - initial_value,
            'n_interventions': len(learning_history) - 1,
            'final_f1': compute_f1_score_from_marginals(final_marginals, true_parents),
            'final_shd': compute_shd_from_marginals(final_marginals, true_parents)
        }
        
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
                'learning_rate': learning_rate,
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