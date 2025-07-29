"""
Tests for bootstrap surrogate functionality and integration.

This module tests the bootstrap surrogate's structural heuristics and
its interaction with the global standardization fix.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import List, Dict, Tuple

from src.causal_bayes_opt.surrogate.bootstrap import (
    create_bootstrap_surrogate_features,
    BootstrapSurrogateOutputs,
    BootstrapConfig
)
from src.causal_bayes_opt.surrogate.phase_manager import PhaseConfig
from src.causal_bayes_opt.surrogate.structure_encoding import (
    encode_causal_structure,
    compute_structural_parent_probabilities
)
from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_edges


class TestBootstrapSurrogate:
    """Test suite for bootstrap surrogate functionality."""
    
    def test_structural_heuristics_provide_diversity(self):
        """Test that structural heuristics create diverse parent probabilities."""
        # Create SCM with clear structure
        variables = ['X0', 'X1', 'X2', 'X3', 'X4']
        edges = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4')]
        scm = create_simple_linear_scm(
            variables=variables,
            edges=edges,
            coefficients={e: 1.5 for e in edges},
            noise_scales={v: 0.1 for v in variables},
            target='X2'
        )
        
        # Compute structural parent probabilities
        probs = compute_structural_parent_probabilities(
            variables=variables,
            edges=edges,
            target='X2'
        )
        
        # Check diversity
        prob_values = probs.numpy() if hasattr(probs, 'numpy') else np.array(probs)
        diversity = np.std(prob_values)
        
        # Should have meaningful diversity
        assert diversity > 0.1, f"Parent probability diversity too low: {diversity}"
        
        # Check specific expectations
        # X1 should have highest probability (direct parent)
        # X0 should have lower probability (grandparent)
        # X3, X4 should have very low probability (descendants)
        assert prob_values[1] > prob_values[0]  # X1 > X0
        assert prob_values[0] > prob_values[3]  # X0 > X3
        assert prob_values[0] > prob_values[4]  # X0 > X4
    
    def test_bootstrap_features_evolution(self):
        """Test that bootstrap features evolve with training step."""
        variables = ['X0', 'X1', 'X2']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2')],
            coefficients={('X0', 'X1'): 1.5, ('X1', 'X2'): 1.5},
            noise_scales={v: 0.1 for v in variables},
            target='X1'
        )
        
        phase_config = PhaseConfig(
            bootstrap_steps=100,
            transition_steps=50,
            exploration_noise_start=0.5,
            exploration_noise_end=0.1
        )
        
        bootstrap_config = BootstrapConfig(
            structure_encoding_dim=128,
            use_graph_distance=True,
            use_structural_priors=True
        )
        
        # Test at different steps
        steps = [0, 50, 100]
        exploration_factors = []
        
        for step in steps:
            features = create_bootstrap_surrogate_features(
                scm=scm,
                step=step,
                config=phase_config,
                bootstrap_config=bootstrap_config,
                rng_key=jax.random.PRNGKey(42 + step)
            )
            
            exploration_factors.append(features.metadata['exploration_factor'])
        
        # Exploration should decrease over time
        assert exploration_factors[0] > exploration_factors[1]
        assert exploration_factors[1] > exploration_factors[2]
        
        # At step 100, exploration should be at minimum
        assert exploration_factors[2] < 0.2
    
    def test_bootstrap_initialization_strategies(self):
        """Test different bootstrap initialization strategies."""
        variables = ['X0', 'X1', 'X2', 'X3']
        edges = [('X0', 'X1'), ('X0', 'X2'), ('X1', 'X3'), ('X2', 'X3')]
        scm = create_simple_linear_scm(
            variables=variables,
            edges=edges,
            coefficients={e: 1.5 for e in edges},
            noise_scales={v: 0.1 for v in variables},
            target='X3'
        )
        
        # Test structural heuristics
        structural_probs = compute_structural_parent_probabilities(
            variables=variables,
            edges=edges,
            target='X3'
        )
        
        # X1 and X2 should have high probability (direct parents)
        # X0 should have lower probability (grandparent)
        assert structural_probs[1] > structural_probs[0]  # X1 > X0
        assert structural_probs[2] > structural_probs[0]  # X2 > X0
        
        # Test that probabilities sum to 1
        assert abs(jnp.sum(structural_probs) - 1.0) < 1e-6
    
    def test_bootstrap_output_validation(self):
        """Test that bootstrap outputs are properly validated."""
        variables = ['X0', 'X1']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1')],
            coefficients={('X0', 'X1'): 1.5},
            noise_scales={v: 0.1 for v in variables},
            target='X1'
        )
        
        phase_config = PhaseConfig()
        bootstrap_config = BootstrapConfig()
        
        features = create_bootstrap_surrogate_features(
            scm=scm,
            step=0,
            config=phase_config,
            bootstrap_config=bootstrap_config
        )
        
        # Check output structure
        assert features.node_embeddings.shape == (2, 128)
        assert features.parent_probabilities.shape == (2,)
        assert features.uncertainties.shape == (2,)
        
        # Check value ranges
        assert jnp.all(features.parent_probabilities >= 0)
        assert jnp.all(features.parent_probabilities <= 1)
        assert jnp.all(features.uncertainties >= 0)
        assert jnp.all(features.uncertainties <= 1)
        
        # Check metadata
        assert 'bootstrap' in features.metadata
        assert features.metadata['bootstrap'] == True
        assert 'exploration_factor' in features.metadata


class TestBootstrapIntegration:
    """Test bootstrap surrogate integration with other components."""
    
    def test_bootstrap_with_enriched_history(self):
        """Test bootstrap integration with enriched history building."""
        from src.causal_bayes_opt.jax_native.state import create_tensor_backed_state_from_scm
        from src.causal_bayes_opt.acquisition.enriched.state_enrichment import (
            create_enriched_history_tensor
        )
        
        variables = ['X0', 'X1', 'X2']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2')],
            coefficients={('X0', 'X1'): 2.0, ('X1', 'X2'): 1.5},
            noise_scales={v: 0.1 for v in variables},
            target='X1'
        )
        
        # Create state with bootstrap
        state = create_tensor_backed_state_from_scm(
            scm, step=0, use_bootstrap_surrogate=True
        )
        
        # Create enriched history
        enriched_history, _ = create_enriched_history_tensor(state)
        
        # Check all channels are properly populated
        assert enriched_history.shape[2] >= 5  # At least 5 channels
        
        # Value channel (0) should have diversity with global standardization
        value_diversity = float(jnp.std(enriched_history[:, :, 0]))
        assert value_diversity > 0.01, f"Value channel lacks diversity: {value_diversity}"
        
        # Parent prob channel (3) should have bootstrap-provided diversity
        parent_prob_diversity = float(jnp.std(enriched_history[:, :, 3]))
        assert parent_prob_diversity > 0.01, f"Parent prob channel lacks diversity: {parent_prob_diversity}"
    
    def test_bootstrap_to_trained_transition(self):
        """Test transition from bootstrap to trained surrogate."""
        # This would test the phase transition logic
        # For now, just verify the phase config works
        phase_config = PhaseConfig(
            bootstrap_steps=100,
            transition_steps=50,
            exploration_noise_start=0.5,
            exploration_noise_end=0.1,
            transition_schedule="linear"
        )
        
        # Verify exploration factor computation at different phases
        from src.causal_bayes_opt.surrogate.phase_manager import compute_exploration_factor
        
        bootstrap_config = BootstrapConfig()
        
        # During bootstrap phase (step < 100)
        factor_early = compute_exploration_factor(50, phase_config, bootstrap_config)
        assert 0.2 < factor_early < 0.4
        
        # During transition phase (100 <= step < 150)
        factor_transition = compute_exploration_factor(125, phase_config, bootstrap_config)
        assert factor_transition < factor_early
        
        # After transition (step >= 150)
        factor_late = compute_exploration_factor(200, phase_config, bootstrap_config)
        assert factor_late <= bootstrap_config.min_noise_factor


def test_alternative_initialization_strategies():
    """Test alternative initialization strategies discussed in investigation."""
    variables = ['X0', 'X1', 'X2', 'X3']
    edges = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3')]
    
    # Uniform initialization (baseline)
    uniform_probs = jnp.ones(len(variables)) / len(variables)
    uniform_probs = uniform_probs.at[variables.index('X2')].set(0.0)  # Target can't be parent
    uniform_probs = uniform_probs / jnp.sum(uniform_probs)
    uniform_diversity = float(jnp.std(uniform_probs))
    
    # Structural initialization (current)
    structural_probs = compute_structural_parent_probabilities(
        variables=variables,
        edges=edges,
        target='X2'
    )
    structural_diversity = float(jnp.std(structural_probs))
    
    # Structural should provide more diversity than uniform
    assert structural_diversity > uniform_diversity
    
    # Correlation-based initialization (simulated)
    # Would use actual data correlations in practice
    correlations = {'X0': 0.3, 'X1': 0.8, 'X2': 0.0, 'X3': 0.2}
    corr_probs = jnp.array([correlations[v] for v in variables])
    corr_probs = corr_probs / jnp.sum(corr_probs)
    corr_diversity = float(jnp.std(corr_probs))
    
    # Correlation-based should also provide good diversity
    assert corr_diversity > 0.2


if __name__ == "__main__":
    # Run tests manually for debugging
    test_suite = TestBootstrapSurrogate()
    test_suite.test_structural_heuristics_provide_diversity()
    test_suite.test_bootstrap_features_evolution()
    test_suite.test_bootstrap_initialization_strategies()
    test_suite.test_bootstrap_output_validation()
    
    integration_suite = TestBootstrapIntegration()
    integration_suite.test_bootstrap_with_enriched_history()
    integration_suite.test_bootstrap_to_trained_transition()
    
    test_alternative_initialization_strategies()
    
    print("✅ All bootstrap tests passed!")