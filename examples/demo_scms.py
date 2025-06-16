#!/usr/bin/env python3
"""
SCM creation functions for ACBO demo experiments.

Provides SCMs of increasing complexity for testing causal discovery algorithms.
"""

import pyrsistent as pyr
from causal_bayes_opt.data_structures import create_scm
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism


def create_easy_scm() -> pyr.PMap:
    """Level 1: Easy - Direct parents, strong signal, low noise."""
    # Structure: A → B → D ← C ← E (target is D with parents B and C)
    variables = frozenset(['A', 'B', 'C', 'D', 'E'])
    edges = frozenset([('A', 'B'), ('B', 'D'), ('E', 'C'), ('C', 'D')])
    
    mechanisms = {
        'A': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'E': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'B': create_linear_mechanism(
            parents=['A'],
            coefficients={'A': 1.5},
            intercept=0.0,
            noise_scale=0.8
        ),
        'C': create_linear_mechanism(
            parents=['E'],
            coefficients={'E': 2.0},
            intercept=0.0,
            noise_scale=0.8
        ),
        'D': create_linear_mechanism(
            parents=['B', 'C'],
            coefficients={'B': 1.2, 'C': 0.8},  # Strong coefficients
            intercept=0.0,
            noise_scale=0.5  # Low noise
        )
    }
    
    return create_scm(
        variables=variables,
        edges=edges, 
        mechanisms=mechanisms,
        target='D'  # Target D has true parents B and C
    )


def create_easy_scm_with_disconnected_var() -> pyr.PMap:
    """Easy SCM with truly disconnected variable X for negative control testing."""
    # Structure: A → B → D ← C ← E, X (disconnected from everything)
    variables = frozenset(['A', 'B', 'C', 'D', 'E', 'X'])
    edges = frozenset([('A', 'B'), ('B', 'D'), ('E', 'C'), ('C', 'D')])  # X has no edges
    
    mechanisms = {
        'A': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'E': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'X': create_root_mechanism(mean=0.0, noise_scale=1.0),  # Completely disconnected
        'B': create_linear_mechanism(
            parents=['A'],
            coefficients={'A': 1.5},
            intercept=0.0,
            noise_scale=0.8
        ),
        'C': create_linear_mechanism(
            parents=['E'],
            coefficients={'E': 2.0},
            intercept=0.0,
            noise_scale=0.8
        ),
        'D': create_linear_mechanism(
            parents=['B', 'C'],
            coefficients={'B': 1.2, 'C': 0.8},  # Strong coefficients
            intercept=0.0,
            noise_scale=0.5  # Low noise
        )
    }
    
    return create_scm(
        variables=variables,
        edges=edges, 
        mechanisms=mechanisms,
        target='D'  # Target D has true parents B and C, X is completely disconnected
    )


def create_medium_scm() -> pyr.PMap:
    """Level 2: Medium - More variables, moderate signal, ancestral correlation."""
    # Structure: A → B → D ← C ← E, A → C (ancestral correlation)
    # Target F has parents D and G: A → B → D → F ← G ← H
    variables = frozenset(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    edges = frozenset([
        ('A', 'B'), ('B', 'D'), ('E', 'C'), ('C', 'D'),  # Original structure
        ('A', 'C'),  # Ancestral correlation (confounding)
        ('H', 'G'), ('G', 'F'), ('D', 'F')  # Extension for target F
    ])
    
    mechanisms = {
        'A': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'E': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'H': create_root_mechanism(mean=0.0, noise_scale=1.0),
        'B': create_linear_mechanism(
            parents=['A'],
            coefficients={'A': 1.2},
            intercept=0.0,
            noise_scale=1.0
        ),
        'C': create_linear_mechanism(
            parents=['E', 'A'],  # Confounded by A
            coefficients={'E': 1.8, 'A': 0.5},
            intercept=0.0,
            noise_scale=1.0
        ),
        'D': create_linear_mechanism(
            parents=['B', 'C'],
            coefficients={'B': 0.7, 'C': 0.6},  # Moderate coefficients
            intercept=0.0,
            noise_scale=1.2
        ),
        'G': create_linear_mechanism(
            parents=['H'],
            coefficients={'H': 1.4},
            intercept=0.0,
            noise_scale=1.0
        ),
        'F': create_linear_mechanism(
            parents=['D', 'G'],
            coefficients={'D': 0.8, 'G': 0.9},  # Moderate coefficients
            intercept=0.0,
            noise_scale=1.0
        )
    }
    
    return create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target='F'  # Target F has true parents D and G
    )


def create_hard_scm() -> pyr.PMap:
    """Level 3: Hard - Many variables, weak signal, high noise, complex structure."""
    # Structure: Complex DAG with 10 variables
    # A → B → E → H → J ← I ← F ← C ← D (long chain)
    # A → C (shortcut), D → G (branch)
    variables = frozenset(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    edges = frozenset([
        ('A', 'B'), ('B', 'E'), ('E', 'H'), ('H', 'J'),  # Main chain
        ('A', 'C'), ('C', 'F'), ('F', 'I'), ('I', 'J'),  # Parallel path
        ('D', 'C'), ('D', 'G')  # Additional complexity
    ])
    
    mechanisms = {
        'A': create_root_mechanism(mean=0.0, noise_scale=1.5),
        'D': create_root_mechanism(mean=0.0, noise_scale=1.5),
        'B': create_linear_mechanism(
            parents=['A'],
            coefficients={'A': 0.4},  # Weak coefficient
            intercept=0.0,
            noise_scale=2.0  # High noise
        ),
        'C': create_linear_mechanism(
            parents=['A', 'D'],
            coefficients={'A': 0.3, 'D': 0.4},  # Weak coefficients
            intercept=0.0,
            noise_scale=2.0
        ),
        'E': create_linear_mechanism(
            parents=['B'],
            coefficients={'B': 0.5},
            intercept=0.0,
            noise_scale=2.0
        ),
        'F': create_linear_mechanism(
            parents=['C'],
            coefficients={'C': 0.4},
            intercept=0.0,
            noise_scale=2.0
        ),
        'G': create_linear_mechanism(
            parents=['D'],
            coefficients={'D': 0.3},
            intercept=0.0,
            noise_scale=2.0
        ),
        'H': create_linear_mechanism(
            parents=['E'],
            coefficients={'E': 0.4},
            intercept=0.0,
            noise_scale=2.0
        ),
        'I': create_linear_mechanism(
            parents=['F'],
            coefficients={'F': 0.3},
            intercept=0.0,
            noise_scale=2.0
        ),
        'J': create_linear_mechanism(
            parents=['H', 'I'],
            coefficients={'H': 0.3, 'I': 0.4},  # Weak coefficients
            intercept=0.0,
            noise_scale=2.0  # High noise
        )
    }
    
    return create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target='J'  # Target J has true parents H and I
    )