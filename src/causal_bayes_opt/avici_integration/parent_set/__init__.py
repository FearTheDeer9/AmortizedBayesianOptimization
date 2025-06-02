"""
Parent set prediction module.

This module provides a clean, modular API for parent set prediction models.
"""

from .model import ParentSetPredictionModel, create_parent_set_model
from .inference import predict_parent_posterior, compute_loss, create_train_step
from .enumeration import enumerate_possible_parent_sets, compute_adaptive_k
from .encoding import encode_parent_set, create_parent_set_indicators
from .posterior import (
    ParentSetPosterior,
    create_parent_set_posterior,
    get_most_likely_parents,
    get_parent_set_probability,
    get_marginal_parent_probabilities,
    compute_posterior_entropy,
    compute_posterior_concentration,
    filter_parent_sets_by_probability,
    compare_posteriors,
    summarize_posterior
)

__all__ = [
    # Core model
    'ParentSetPredictionModel',
    'create_parent_set_model',
    
    # Inference utilities  
    'predict_parent_posterior',
    'compute_loss',
    'create_train_step',
    
    # Posterior data structure and utilities
    'ParentSetPosterior',
    'create_parent_set_posterior',
    'get_most_likely_parents',
    'get_parent_set_probability',
    'get_marginal_parent_probabilities',
    'compute_posterior_entropy',
    'compute_posterior_concentration',
    'filter_parent_sets_by_probability',
    'compare_posteriors',
    'summarize_posterior',
    
    # Enumeration utilities
    'enumerate_possible_parent_sets', 
    'compute_adaptive_k',
    
    # Encoding utilities
    'encode_parent_set',
    'create_parent_set_indicators',
]
