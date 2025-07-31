"""
Posterior validation utilities for ACBO.

This module provides utilities to validate and normalize posterior predictions
from different surrogate models to ensure consistent format across the system.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import jax.numpy as jnp

logger = logging.getLogger(__name__)


class PosteriorValidator:
    """Validator for posterior predictions from surrogate models."""
    
    @staticmethod
    def validate_posterior(
        posterior: Any,
        variable_order: List[str],
        target_variable: str
    ) -> Tuple[bool, List[str], Optional[Dict[str, float]]]:
        """
        Validate and extract marginal probabilities from posterior.
        
        Args:
            posterior: Posterior prediction from surrogate
            variable_order: List of all variables
            target_variable: Target variable name
            
        Returns:
            Tuple of:
            - is_valid: Whether posterior is valid
            - issues: List of validation issues
            - marginal_probs: Extracted marginal probabilities (if valid)
        """
        issues = []
        
        if posterior is None:
            issues.append("Posterior is None")
            return False, issues, None
        
        # Try to extract marginal probabilities
        marginal_probs = PosteriorValidator._extract_marginal_probs(
            posterior, variable_order, target_variable
        )
        
        if marginal_probs is None:
            issues.append("Could not extract marginal probabilities from posterior")
            return False, issues, None
        
        # Validate probabilities
        for var in variable_order:
            if var == target_variable:
                # Target shouldn't have parent probability
                if var in marginal_probs and marginal_probs[var] != 0.0:
                    issues.append(f"Target variable {var} has non-zero parent probability")
            else:
                # Non-target variables should have probabilities
                if var not in marginal_probs:
                    issues.append(f"Missing probability for variable {var}")
                else:
                    prob = marginal_probs[var]
                    if not (0.0 <= prob <= 1.0):
                        issues.append(f"Invalid probability {prob} for variable {var}")
        
        # Check for meaningful signal
        non_target_probs = [p for v, p in marginal_probs.items() if v != target_variable]
        if non_target_probs:
            if all(p == 0.0 for p in non_target_probs):
                issues.append("All parent probabilities are zero")
            elif len(set(non_target_probs)) == 1:
                issues.append(f"All parent probabilities are identical: {non_target_probs[0]}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Posterior validation failed: {', '.join(issues)}")
        
        return is_valid, issues, marginal_probs
    
    @staticmethod
    def _extract_marginal_probs(
        posterior: Any,
        variable_order: List[str],
        target_variable: str
    ) -> Optional[Dict[str, float]]:
        """Extract marginal probabilities from various posterior formats."""
        marginal_probs = {}
        
        # Try different extraction methods
        extracted = False
        
        # Method 1: Direct dictionary
        if isinstance(posterior, dict):
            if 'marginal_parent_probs' in posterior:
                raw_probs = posterior['marginal_parent_probs']
                extracted = True
            elif 'metadata' in posterior and isinstance(posterior['metadata'], dict):
                if 'marginal_parent_probs' in posterior['metadata']:
                    raw_probs = posterior['metadata']['marginal_parent_probs']
                    extracted = True
        
        # Method 2: Object attributes
        if not extracted and hasattr(posterior, 'marginal_parent_probs'):
            raw_probs = posterior.marginal_parent_probs
            extracted = True
        
        # Method 3: Check metadata attribute (handles pyrsistent PMap)
        if not extracted and hasattr(posterior, 'metadata'):
            try:
                # Works for both dict and pyrsistent PMap
                if 'marginal_parent_probs' in posterior.metadata:
                    raw_probs = posterior.metadata['marginal_parent_probs']
                    extracted = True
            except Exception as e:
                logger.debug(f"Failed to extract from metadata: {e}")
        
        # Method 4: ParentSetPosterior object
        if not extracted and hasattr(posterior, 'parent_sets') and hasattr(posterior, 'probabilities'):
            # Try to compute marginals from parent sets
            try:
                marginals = PosteriorValidator._compute_marginals_from_sets(
                    posterior, variable_order, target_variable
                )
                if marginals:
                    raw_probs = marginals
                    extracted = True
            except Exception as e:
                logger.debug(f"Failed to compute marginals from parent sets: {e}")
        
        if not extracted:
            return None
        
        # Normalize to ensure all variables have entries
        for var in variable_order:
            if var == target_variable:
                marginal_probs[var] = 0.0
            else:
                if isinstance(raw_probs, dict):
                    marginal_probs[var] = float(raw_probs.get(var, 0.0))
                else:
                    # Assume it's array-like indexed by variable position
                    try:
                        var_idx = variable_order.index(var)
                        marginal_probs[var] = float(raw_probs[var_idx])
                    except:
                        marginal_probs[var] = 0.0
        
        return marginal_probs
    
    @staticmethod
    def _compute_marginals_from_sets(
        posterior: Any,
        variable_order: List[str],
        target_variable: str
    ) -> Optional[Dict[str, float]]:
        """Compute marginal probabilities from parent set representation."""
        try:
            marginals = {}
            
            # Initialize all to zero
            for var in variable_order:
                marginals[var] = 0.0
            
            # Sum probabilities for sets containing each variable
            parent_sets = posterior.parent_sets
            probabilities = posterior.probabilities
            
            for parent_set, prob in zip(parent_sets, probabilities):
                for parent in parent_set:
                    if parent in marginals:
                        marginals[parent] += float(prob)
            
            # Ensure probabilities are in [0, 1]
            for var in marginals:
                marginals[var] = min(1.0, marginals[var])
            
            return marginals
            
        except Exception as e:
            logger.debug(f"Failed to compute marginals: {e}")
            return None
    
    @staticmethod
    def create_mock_posterior(
        variable_order: List[str],
        target_variable: str,
        parent_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create a mock posterior for testing.
        
        Args:
            variable_order: List of all variables
            target_variable: Target variable name
            parent_probs: Optional specific probabilities to use
            
        Returns:
            Mock posterior in standard format
        """
        if parent_probs is None:
            # Create some reasonable default probabilities
            parent_probs = {}
            for i, var in enumerate(variable_order):
                if var != target_variable:
                    # Decreasing probability by index
                    parent_probs[var] = 0.8 * (0.5 ** i)
        
        # Ensure target has zero probability
        parent_probs[target_variable] = 0.0
        
        return {
            'marginal_parent_probs': parent_probs,
            'target_variable': target_variable,
            'entropy': -sum(p * jnp.log(p + 1e-8) for p in parent_probs.values() if p > 0)
        }
    
    @staticmethod
    def log_posterior_summary(
        posterior: Any,
        variable_order: List[str],
        target_variable: str,
        prefix: str = ""
    ) -> None:
        """
        Log a summary of posterior predictions.
        
        Args:
            posterior: Posterior to summarize
            variable_order: List of variables
            target_variable: Target variable
            prefix: Optional prefix for log messages
        """
        is_valid, issues, marginal_probs = PosteriorValidator.validate_posterior(
            posterior, variable_order, target_variable
        )
        
        if prefix:
            prefix = f"{prefix} "
        
        if not is_valid:
            logger.warning(f"{prefix}Invalid posterior: {', '.join(issues)}")
            return
        
        # Log summary statistics
        non_target_probs = [p for v, p in marginal_probs.items() if v != target_variable]
        
        if non_target_probs:
            mean_prob = sum(non_target_probs) / len(non_target_probs)
            max_prob = max(non_target_probs)
            min_prob = min(non_target_probs)
            
            # Find most likely parents
            sorted_probs = sorted(
                [(v, p) for v, p in marginal_probs.items() if v != target_variable],
                key=lambda x: x[1],
                reverse=True
            )
            
            top_parents = sorted_probs[:3]  # Top 3
            
            logger.info(
                f"{prefix}Posterior summary: "
                f"mean_prob={mean_prob:.3f}, "
                f"range=[{min_prob:.3f}, {max_prob:.3f}], "
                f"top_parents={[(v, f'{p:.3f}') for v, p in top_parents]}"
            )
        else:
            logger.warning(f"{prefix}No parent probabilities found in posterior")


def validate_tensor_posterior_compatibility(
    tensor: jnp.ndarray,
    posterior: Any,
    variable_order: List[str],
    target_variable: str
) -> Tuple[bool, List[str]]:
    """
    Validate that tensor and posterior are compatible.
    
    Args:
        tensor: Input tensor (3 or 5 channel)
        posterior: Posterior predictions
        variable_order: Variable ordering
        target_variable: Target variable
        
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check tensor shape
    if len(tensor.shape) != 3:
        issues.append(f"Expected 3D tensor, got shape {tensor.shape}")
    elif tensor.shape[1] != len(variable_order):
        issues.append(
            f"Tensor has {tensor.shape[1]} variables but "
            f"variable_order has {len(variable_order)}"
        )
    
    # Validate posterior
    is_valid, posterior_issues, marginal_probs = PosteriorValidator.validate_posterior(
        posterior, variable_order, target_variable
    )
    
    if not is_valid:
        issues.extend([f"Posterior: {issue}" for issue in posterior_issues])
    
    # Check if tensor already has surrogate predictions (5-channel)
    if tensor.shape[2] >= 5 and marginal_probs is not None:
        # Compare with existing surrogate channel
        existing_probs = tensor[-1, :, 3]  # Last timestep, surrogate channel
        
        for i, var in enumerate(variable_order):
            expected = marginal_probs.get(var, 0.0)
            actual = float(existing_probs[i])
            
            if abs(expected - actual) > 1e-3:
                issues.append(
                    f"Mismatch for {var}: tensor has {actual:.3f}, "
                    f"posterior has {expected:.3f}"
                )
    
    return len(issues) == 0, issues