#!/usr/bin/env python3
"""
Utilities for creating proper ParentSetPosterior objects for baseline methods.
"""

import jax.numpy as jnp
from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior


def create_uniform_posterior(variables, target):
    """Create a uniform posterior for baseline methods."""
    # For baseline, just use empty parent set (no parents)
    parent_sets = [frozenset()]
    probabilities = jnp.array([1.0])
    
    return create_parent_set_posterior(
        target_variable=target,
        parent_sets=parent_sets,
        probabilities=probabilities,
        metadata={'type': 'uniform_baseline'}
    )


def create_oracle_posterior(variables, target, true_parents):
    """Create an oracle posterior that knows the true parents."""
    # Oracle knows the true parent set with certainty
    parent_sets = [frozenset(true_parents)]
    probabilities = jnp.array([1.0])
    
    return create_parent_set_posterior(
        target_variable=target,
        parent_sets=parent_sets,
        probabilities=probabilities,
        metadata={'type': 'oracle', 'true_parents': list(true_parents)}
    )


def dict_to_posterior(posterior_dict, variables):
    """Convert dict-based posteriors to proper ParentSetPosterior objects."""
    result = {}
    
    for var in variables:
        if var in posterior_dict:
            # Get parent sets and their probabilities
            parent_set_probs = posterior_dict[var]
            parent_sets = list(parent_set_probs.keys())
            probabilities = jnp.array(list(parent_set_probs.values()))
            
            # Normalize probabilities
            prob_sum = jnp.sum(probabilities)
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
            else:
                # Fallback to uniform
                probabilities = jnp.ones_like(probabilities) / len(probabilities)
            
            # Create proper posterior
            result[var] = create_parent_set_posterior(
                target_variable=var,
                parent_sets=parent_sets,
                probabilities=probabilities,
                metadata={'converted_from': 'dict'}
            )
        else:
            # Default to empty parent set
            result[var] = create_uniform_posterior(variables, var)
    
    return result