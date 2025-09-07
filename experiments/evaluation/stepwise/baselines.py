#!/usr/bin/env python3
"""
Baseline policies for evaluation.

Provides random and oracle baselines that work with the stepwise evaluation framework
and properly handle variable-specific ranges from the SCM metadata.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Dict, Any, List, Optional, Tuple


class RandomBaseline:
    """Random baseline that selects variables and values uniformly at random."""
    
    def __init__(self, seed: int = 42):
        """Initialize random baseline.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.jax_key = random.PRNGKey(seed)
    
    def select_intervention(
        self,
        buffer: Any,
        target_var: str,
        variables: List[str],
        variable_ranges: Dict[str, Tuple[float, float]],
        **kwargs
    ) -> Tuple[str, float]:
        """Select random intervention.
        
        Args:
            buffer: Experience buffer (unused)
            target_var: Target variable name
            variables: List of all variables
            variable_ranges: Dict mapping variables to (min, max) ranges
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (selected_variable, intervention_value)
        """
        # Select random variable (excluding target)
        candidates = [v for v in variables if v != target_var]
        selected_var = self.rng.choice(candidates)
        
        # Select random value within the variable's range
        var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
        intervention_value = self.rng.uniform(var_range[0], var_range[1])
        
        return selected_var, intervention_value


class OracleBaseline:
    """Oracle baseline with perfect knowledge of SCM structure and coefficients."""
    
    def __init__(self, scm: Any, optimization_direction: str = 'MINIMIZE'):
        """Initialize oracle baseline.
        
        Args:
            scm: The structural causal model with full information
            optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        """
        self.scm = scm
        self.optimization_direction = optimization_direction
        self._extract_scm_info()
    
    def _extract_scm_info(self):
        """Extract and cache SCM structure and coefficients."""
        from src.causal_bayes_opt.data_structures.scm import (
            get_target, get_parents, get_mechanisms
        )
        
        self.target = get_target(self.scm)
        self.true_parents = set(get_parents(self.scm, self.target))
        
        # Extract coefficients from mechanisms
        mechanisms = get_mechanisms(self.scm)
        self.coefficients = {}
        
        if self.target in mechanisms:
            target_mechanism = mechanisms[self.target]
            
            # Try multiple ways to extract coefficients
            if hasattr(target_mechanism, 'coefficients'):
                # Direct attribute access
                self.coefficients = target_mechanism.coefficients
            elif hasattr(target_mechanism, 'descriptor'):
                # Through descriptor (for serializable mechanisms)
                descriptor = target_mechanism.descriptor
                if hasattr(descriptor, 'coefficients'):
                    self.coefficients = descriptor.coefficients
            elif callable(target_mechanism):
                # Try to extract from the function's closure or attributes
                # This handles wrapped functions from create_linear_mechanism
                import inspect
                closure_vars = inspect.getclosurevars(target_mechanism)
                for var_name, var_value in closure_vars.nonlocals.items():
                    if 'coeff' in var_name.lower() and isinstance(var_value, dict):
                        self.coefficients = var_value
                        break
        
        # Debug logging with more detail
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Oracle initialized:")
        logger.info(f"  Target: {self.target}")
        logger.info(f"  True parents: {self.true_parents}")
        logger.info(f"  Coefficients extracted: {self.coefficients}")
        logger.info(f"  Number of parents: {len(self.true_parents)}")
        logger.info(f"  Number of coefficients: {len(self.coefficients)}")
        
        # Validation
        if self.true_parents and not self.coefficients:
            logger.warning(f"WARNING: Oracle has parents {self.true_parents} but no coefficients!")
    
    def select_intervention(
        self,
        buffer: Any,
        target_var: str,
        variables: List[str],
        variable_ranges: Dict[str, Tuple[float, float]],
        **kwargs
    ) -> Tuple[str, float]:
        """Select optimal intervention using perfect knowledge.
        
        Args:
            buffer: Experience buffer (unused by oracle)
            target_var: Target variable name (should match self.target)
            variables: List of all variables
            variable_ranges: Dict mapping variables to (min, max) ranges
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (selected_variable, intervention_value)
        """
        if not self.true_parents:
            # No parents means root node as target - this shouldn't happen!
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"ERROR: Oracle called with root node as target! Target={target_var}")
            logger.error(f"  This indicates a bug in SCM generation - targets should have parents")
            
            # Return a random non-target variable to avoid crashes
            # But this will result in 0% parent selection which is correct for this error case
            candidates = [v for v in variables if v != target_var]
            if candidates:
                # Select randomly instead of always first
                import random
                selected = random.choice(candidates)
                var_range = variable_ranges.get(selected, (-2.0, 2.0))
                # Use extreme value for minimization
                value = var_range[0] if self.optimization_direction == 'MINIMIZE' else var_range[1]
                return selected, value
            return variables[0], 0.0
        
        # Find the parent with the maximum possible effect
        best_parent = None
        best_value = None
        best_effect_magnitude = 0.0
        
        for parent in self.true_parents:
            # Get coefficient - default to 1.0 if not found (conservative assumption)
            if parent not in self.coefficients:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"No coefficient found for parent {parent}, using default 1.0")
                coeff = 1.0
            else:
                coeff = self.coefficients[parent]
            var_range = variable_ranges.get(parent, (-2.0, 2.0))
            min_val, max_val = var_range
            
            # Calculate the effect at each extreme
            effect_at_min = coeff * min_val
            effect_at_max = coeff * max_val
            
            if self.optimization_direction == 'MINIMIZE':
                # We want the most negative contribution to the target
                if effect_at_min < effect_at_max:
                    # min gives more negative contribution
                    optimal_value = min_val
                    optimal_effect = effect_at_min
                else:
                    # max gives more negative contribution
                    optimal_value = max_val
                    optimal_effect = effect_at_max
            else:  # MAXIMIZE
                # We want the most positive contribution to the target
                if effect_at_min > effect_at_max:
                    # min gives more positive contribution
                    optimal_value = min_val
                    optimal_effect = effect_at_min
                else:
                    # max gives more positive contribution
                    optimal_value = max_val
                    optimal_effect = effect_at_max
            
            # Track the parent with the best (largest magnitude) effect
            effect_magnitude = abs(optimal_effect)
            if effect_magnitude > best_effect_magnitude:
                best_effect_magnitude = effect_magnitude
                best_parent = parent
                best_value = optimal_value
        
        # If no valid parent found, fall back to first parent
        if best_parent is None:
            best_parent = list(self.true_parents)[0]
            var_range = variable_ranges.get(best_parent, (-2.0, 2.0))
            best_value = (var_range[0] + var_range[1]) / 2.0
        
        # Validation: ensure we're returning a parent
        assert best_parent in self.true_parents, f"Oracle selected non-parent {best_parent}! Parents: {self.true_parents}"
        
        return best_parent, best_value


class GreedyBaseline:
    """Greedy baseline that uses surrogate predictions to select parents."""
    
    def __init__(self, seed: int = 42):
        """Initialize greedy baseline.
        
        Args:
            seed: Random seed for tie-breaking
        """
        self.rng = np.random.RandomState(seed)
    
    def select_intervention(
        self,
        buffer: Any,
        target_var: str,
        variables: List[str],
        variable_ranges: Dict[str, Tuple[float, float]],
        parent_probs: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """Select intervention greedily based on parent probabilities.
        
        Args:
            buffer: Experience buffer (unused)
            target_var: Target variable name
            variables: List of all variables
            variable_ranges: Dict mapping variables to (min, max) ranges
            parent_probs: Optional dict of parent probabilities from surrogate
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (selected_variable, intervention_value)
        """
        if parent_probs is None:
            # No surrogate predictions - fall back to random
            candidates = [v for v in variables if v != target_var]
            selected_var = self.rng.choice(candidates)
        else:
            # Select variable with highest parent probability
            valid_probs = {v: p for v, p in parent_probs.items() if v != target_var}
            
            if not valid_probs:
                # No valid probabilities - select randomly
                candidates = [v for v in variables if v != target_var]
                selected_var = self.rng.choice(candidates)
            else:
                # Select variable with highest probability
                max_prob = max(valid_probs.values())
                best_vars = [v for v, p in valid_probs.items() if p == max_prob]
                
                # Break ties randomly
                selected_var = self.rng.choice(best_vars)
        
        # Select middle value (could be optimized based on estimated coefficients)
        var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
        intervention_value = (var_range[0] + var_range[1]) / 2.0
        
        return selected_var, intervention_value


class OracleSurrogateBaseline:
    """Oracle surrogate that provides perfect parent predictions.
    
    This is NOT an intervention selector but a surrogate replacement that
    gives perfect structural information for testing policy behavior.
    """
    
    def __init__(self, scm: Any):
        """Initialize oracle surrogate.
        
        Args:
            scm: The structural causal model with full information
        """
        self.scm = scm
        self._extract_scm_info()
    
    def _extract_scm_info(self):
        """Extract and cache SCM structure."""
        from src.causal_bayes_opt.data_structures.scm import (
            get_target, get_parents
        )
        
        self.target = get_target(self.scm)
        self.true_parents = set(get_parents(self.scm, self.target))
    
    def __call__(self, tensor_3ch, target_var_name, variable_list):
        """Oracle surrogate function interface.
        
        Args:
            tensor_3ch: 3-channel tensor (unused - oracle knows ground truth)
            target_var_name: Name of target variable
            variable_list: List of all variables in order
            
        Returns:
            Dict with 'parent_probs' key containing perfect predictions
        """
        import jax.numpy as jnp
        
        # Return perfect parent probabilities
        parent_probs = []
        for var in variable_list:
            if var == target_var_name:
                parent_probs.append(0.0)  # Target itself
            elif var in self.true_parents:
                parent_probs.append(1.0)  # True parent
            else:
                parent_probs.append(0.0)  # Non-parent
        
        return {'parent_probs': jnp.array(parent_probs)}


def create_baseline(baseline_type: str, scm: Optional[Any] = None, seed: int = 42) -> Any:
    """Factory function to create baseline policies.
    
    Args:
        baseline_type: Type of baseline ('random', 'oracle', 'greedy')
        scm: SCM (required for oracle)
        seed: Random seed
        
    Returns:
        Baseline policy instance
    """
    if baseline_type == 'random':
        return RandomBaseline(seed=seed)
    elif baseline_type == 'oracle':
        if scm is None:
            raise ValueError("Oracle baseline requires SCM")
        return OracleBaseline(scm=scm)
    elif baseline_type == 'greedy':
        return GreedyBaseline(seed=seed)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")