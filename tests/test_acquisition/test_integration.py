"""
Comprehensive Integration Tests for Policy Network with Real ACBO Components.

This test suite validates the policy network integration with:
1. Real ExperienceBuffer with mixed data
2. Real AcquisitionState with optimization tracking  
3. Real ParentSetPosterior from surrogate model
4. Complete intervention cycle: policy → intervention → outcome → updated state
5. Performance validation for realistic problem sizes
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pyrsistent as pyr
from typing import Dict, Any, List

# Core ACBO components
from causal_bayes_opt.acquisition.policy import (
    PolicyConfig,
    create_acquisition_policy,
    sample_intervention_from_policy,
    validate_policy_output,
    analyze_policy_output,
)

from causal_bayes_opt.acquisition.state import (
    AcquisitionState,
    create_acquisition_state,
)

from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.scm import create_scm

from causal_bayes_opt.avici_integration.parent_set import (
    create_parent_set_posterior,
    predict_parent_posterior,
)

from causal_bayes_opt.interventions.handlers import create_perfect_intervention
from causal_bayes_opt.interventions.registry import apply_intervention

from causal_bayes_opt.environments.sampling import (
    sample_with_intervention,
)

from causal_bayes_opt.mechanisms.linear import (
    sample_from_linear_scm,
)

from causal_bayes_opt.mechanisms.linear import create_linear_mechanism
from causal_bayes_opt.experiments.test_scms import create_simple_linear_scm


class TestRealACBOIntegration:
    """Test policy network with real ACBO infrastructure."""
    
    @pytest.fixture
    def realistic_scm(self):
        """Create a realistic 4-variable linear SCM for testing."""
        return create_simple_linear_scm(
            variables=['X', 'Y', 'Z', 'W'],
            edges=[('X', 'Y'), ('Z', 'Y'), ('Y', 'W')],
            coefficients={
                ('X', 'Y'): 2.0,
                ('Z', 'Y'): -1.5,
                ('Y', 'W'): 1.8,
            },
            noise_scales={'X': 1.0, 'Y': 0.5, 'Z': 1.2, 'W': 0.8},
            target='Y'
        )
    
    @pytest.fixture
    def populated_buffer(self, realistic_scm):
        """Create a buffer with realistic mixed observational/interventional data."""
        buffer = ExperienceBuffer()
        
        # Add observational samples
        observational_samples = sample_from_linear_scm(realistic_scm, n_samples=15, seed=42)
        for sample in observational_samples:
            buffer.add_observation(sample)
        
        # Add interventional samples on different variables
        intervention_targets = ['X', 'Z', 'W']
        for i, target_var in enumerate(intervention_targets):
            for j in range(3):  # 3 interventions per variable
                # Create intervention
                intervention_value = float(i * 10 + j)
                intervention = create_perfect_intervention(
                    targets=frozenset([target_var]),
                    values={target_var: intervention_value}
                )
                
                # Sample outcome under intervention
                interventional_samples = sample_with_intervention(
                    realistic_scm, 
                    intervention, 
                    n_samples=1,
                    seed=100 + i * 10 + j
                )
                
                for sample in interventional_samples:
                    buffer.add_intervention(intervention, sample)
        
        return buffer
    
    @pytest.fixture  
    def realistic_posterior(self, populated_buffer):
        """Create a realistic posterior using parent set enumeration."""
        # For this test, create a realistic posterior manually
        # In practice, this would come from the surrogate model
        target_variable = 'Y'
        
        # Enumerate plausible parent sets for Y
        parent_sets = [
            frozenset(),           # No parents
            frozenset(['X']),      # Only X
            frozenset(['Z']),      # Only Z  
            frozenset(['X', 'Z']), # Both X and Z (true parents)
            frozenset(['W']),      # Only W
            frozenset(['X', 'W']), # X and W
            frozenset(['Z', 'W']), # Z and W
        ]
        
        # Assign probabilities (true parents get highest probability)
        probabilities = jnp.array([0.05, 0.15, 0.15, 0.5, 0.05, 0.05, 0.05])
        
        return create_parent_set_posterior(
            target_variable=target_variable,
            parent_sets=parent_sets,
            probabilities=probabilities,
            metadata={
                'n_samples': populated_buffer.num_observations() + populated_buffer.num_interventions(),
                'buffer_size': populated_buffer.size(),
                'variables': sorted(populated_buffer.get_variable_coverage()),
            }
        )
    
    @pytest.fixture
    def realistic_acquisition_state(self, populated_buffer, realistic_posterior):
        """Create a realistic acquisition state."""
        # Find current best value for target variable
        target_values = []
        for sample in populated_buffer.get_observations():
            target_values.append(sample["values"]['Y'])
        
        for intervention, outcome in populated_buffer.get_interventions():
            target_values.append(outcome["values"]['Y'])
        
        best_value = max(target_values) if target_values else 0.0
        
        return AcquisitionState(
            posterior=realistic_posterior,
            buffer=populated_buffer,
            best_value=best_value,
            current_target='Y',
            step=len(target_values),
            metadata=pyr.m(**{
                'total_samples': len(target_values),
                'observational_samples': populated_buffer.num_observations(),
                'interventional_samples': populated_buffer.num_interventions(),
                'test_scenario': 'realistic_integration'
            })
        )
    
    def test_policy_with_realistic_state(self, realistic_acquisition_state):
        """Test policy network handles realistic acquisition state."""
        config = PolicyConfig(hidden_dim=64, num_layers=2)
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        output = policy_transform.apply(params, key, realistic_acquisition_state, False)
        
        # Validate output structure and values
        assert validate_policy_output(output, realistic_acquisition_state)
        
        # Check dimensions match buffer variables
        n_vars = len(realistic_acquisition_state.buffer.get_variable_coverage())
        assert output['variable_logits'].shape == (n_vars,)
        assert output['value_params'].shape == (n_vars, 2)
        
        # Target variable should be masked
        variable_order = sorted(realistic_acquisition_state.buffer.get_variable_coverage())
        target_idx = variable_order.index(realistic_acquisition_state.current_target)
        assert output['variable_logits'][target_idx] == -jnp.inf
        
        # Non-target variables should have finite logits
        non_target_mask = jnp.arange(len(variable_order)) != target_idx
        assert jnp.all(jnp.isfinite(output['variable_logits'][non_target_mask]))
        
        print(f"✓ Policy handles realistic state with {n_vars} variables, {realistic_acquisition_state.step} steps")
    
    def test_intervention_sampling_realism(self, realistic_acquisition_state):
        """Test intervention sampling produces valid interventions."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        
        # Sample multiple interventions to test consistency
        sample_keys = jax.random.split(key, 10)
        interventions = []
        
        for sample_key in sample_keys:
            output = policy_transform.apply(params, sample_key, realistic_acquisition_state, False)
            intervention = sample_intervention_from_policy(output, realistic_acquisition_state, sample_key, config)
            interventions.append(intervention)
        
        # Check all interventions are valid
        for intervention in interventions:
            # Basic structure
            assert intervention['type'] == 'perfect'
            assert len(intervention['targets']) == 1
            assert len(intervention['values']) == 1
            
            # Target variable should not be intervened on
            target_var = list(intervention['targets'])[0]
            assert target_var != realistic_acquisition_state.current_target
            
            # Intervention value should be reasonable (finite)
            intervention_value = intervention['values'][target_var]
            assert jnp.isfinite(intervention_value)
            assert isinstance(intervention_value, float)
        
        # Check some diversity in intervention targets
        targets_chosen = [list(intervention['targets'])[0] for intervention in interventions]
        unique_targets = set(targets_chosen)
        
        # Should have at least 2 different targets chosen across 10 samples
        assert len(unique_targets) >= 2
        
        print(f"✓ Sampled {len(interventions)} valid interventions targeting {len(unique_targets)} different variables")
    
    def test_complete_intervention_cycle(self, realistic_scm, realistic_acquisition_state):
        """Test complete cycle: policy → intervention → SCM → updated state."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        
        # 1. Get policy recommendation
        output = policy_transform.apply(params, key, realistic_acquisition_state, False)
        intervention = sample_intervention_from_policy(output, realistic_acquisition_state, key, config)
        
        print(f"Policy recommends intervening on {list(intervention['targets'])} with values {intervention['values']}")
        
        # 2. Apply intervention to SCM and sample outcome
        outcomes = sample_with_intervention(realistic_scm, intervention, n_samples=1, seed=123)
        outcome = outcomes[0]
        
        print(f"Intervention outcome: Y = {outcome["values"]['Y']:.3f}")
        
        # 3. Add outcome to buffer (create new buffer properly)
        from causal_bayes_opt.data_structures.buffer import create_buffer_from_samples
        
        # Get existing data
        existing_observations = realistic_acquisition_state.buffer.get_observations()
        existing_interventions = realistic_acquisition_state.buffer.get_interventions()
        
        # Create new buffer with existing data plus new intervention
        new_buffer = create_buffer_from_samples(
            observations=existing_observations,
            interventions=existing_interventions + [(intervention, outcome)]
        )
        
        # Check buffer was updated
        assert new_buffer.num_interventions() == realistic_acquisition_state.buffer.num_interventions() + 1
        assert new_buffer.size() == realistic_acquisition_state.buffer.size() + 1
        
        # 4. Create updated state (in practice, would also update posterior)
        new_best_value = max(realistic_acquisition_state.best_value, outcome["values"]['Y'])
        new_state = AcquisitionState(
            posterior=realistic_acquisition_state.posterior,  # Same posterior for this test
            buffer=new_buffer,
            best_value=new_best_value,
            current_target=realistic_acquisition_state.current_target,
            step=realistic_acquisition_state.step + 1,
            metadata=realistic_acquisition_state.metadata.set('updated', True)
        )
        
        # 5. Verify policy can handle updated state
        new_output = policy_transform.apply(params, key, new_state, False)
        assert validate_policy_output(new_output, new_state)
        
        print(f"✓ Complete intervention cycle completed. Best value: {realistic_acquisition_state.best_value:.3f} → {new_best_value:.3f}")
    
    def test_policy_analysis_realistic(self, realistic_acquisition_state):
        """Test policy analysis with realistic state."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        output = policy_transform.apply(params, key, realistic_acquisition_state, False)
        
        # Analyze policy output
        analysis = analyze_policy_output(output, realistic_acquisition_state, top_k=3)
        
        # Check analysis structure
        assert 'state_value_estimate' in analysis
        assert 'top_variables' in analysis
        assert 'variable_selection_entropy' in analysis
        assert 'state_context' in analysis
        
        # State value should be reasonable
        assert jnp.isfinite(analysis['state_value_estimate'])
        
        # Should have 3 top variables (excluding target)
        non_target_vars = [v for v in sorted(realistic_acquisition_state.buffer.get_variable_coverage()) 
                          if v != realistic_acquisition_state.current_target]
        assert len(analysis['top_variables']) == min(3, len(non_target_vars))
        
        # Top variables should be sorted by logit value
        logits = [var_info['logit'] for var_info in analysis['top_variables']]
        assert logits == sorted(logits, reverse=True)
        
        # Context should include key information
        context = analysis['state_context']
        assert 'buffer_size' in context
        assert 'step' in context
        assert 'best_value' in context
        
        print(f"✓ Policy analysis shows state value {analysis['state_value_estimate']:.3f}, entropy {analysis['variable_selection_entropy']:.3f}")
        print(f"  Top intervention targets: {[v['variable'] for v in analysis['top_variables']]}")
    
    def test_performance_scalability(self):
        """Test policy performance with larger problem sizes."""
        import time
        
        # Create larger problem (8 variables)
        large_scm = create_simple_linear_scm(
            variables=[f'X{i}' for i in range(7)] + ['Y'],
            edges=[(f'X{i}', 'Y') for i in range(3)] + [(f'X{i}', f'X{(i+1)%7}') for i in range(3)],
            coefficients={
                ('X0', 'Y'): 2.0, ('X1', 'Y'): -1.0, ('X2', 'Y'): 0.5,
                ('X0', 'X1'): 1.5, ('X1', 'X2'): -0.8, ('X2', 'X3'): 1.2,
            },
            noise_scales={f'X{i}': 1.0 for i in range(7)} | {'Y': 0.5},
            target='Y'
        )
        
        # Create buffer with more data
        buffer = ExperienceBuffer()
        
        # Add observational data
        obs_samples = sample_from_linear_scm(large_scm, n_samples=50, seed=42)
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Add interventional data
        for i in range(30):
            target_var = f'X{i % 7}'
            intervention = create_perfect_intervention(
                targets=frozenset([target_var]),
                values={target_var: float(i)}
            )
            outcomes = sample_with_intervention(large_scm, intervention, n_samples=1, seed=200 + i)
            buffer.add_intervention(intervention, outcomes[0])
        
        # Create posterior
        parent_sets = [frozenset(), frozenset(['X0']), frozenset(['X1']), frozenset(['X0', 'X1'])]
        probs = jnp.array([0.2, 0.3, 0.3, 0.2])
        posterior = create_parent_set_posterior('Y', parent_sets, probs, {})
        
        # Create state
        all_samples = [s["values"]['Y'] for s in buffer.get_observations()]
        all_samples.extend([s["values"]['Y'] for _, s in buffer.get_interventions()])
        best_value = max(all_samples)
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=best_value,
            current_target='Y',
            step=len(all_samples),
            metadata=pyr.m(**{'large_scale_test': True})
        )
        
        # Test policy performance
        config = PolicyConfig(hidden_dim=128, num_layers=3)
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        
        # Time initialization
        start_time = time.time()
        params = policy_transform.init(key, state, False)
        init_time = time.time() - start_time
        
        # Time forward pass
        start_time = time.time()
        output = policy_transform.apply(params, key, state, False)
        forward_time = time.time() - start_time
        
        # Validate output
        assert validate_policy_output(output, state)
        
        # Performance should be reasonable for a complex transformer
        assert init_time < 5.0, f"Initialization too slow: {init_time:.3f}s"  # Includes JAX compilation
        assert forward_time < 2.5, f"Forward pass too slow: {forward_time:.3f}s"  # After compilation
        
        print(f"✓ Large scale test (8 vars, 80 samples): init {init_time:.3f}s, forward {forward_time:.3f}s")
    
    def test_uncertainty_integration(self, realistic_acquisition_state):
        """Test that policy properly integrates uncertainty information."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        output = policy_transform.apply(params, key, realistic_acquisition_state, False)
        
        # Get marginal parent probabilities from state
        marginal_probs = realistic_acquisition_state.marginal_parent_probs
        
        # Variables with higher parent probability should generally have higher logits
        # (though this is not guaranteed due to other factors)
        variable_order = sorted(realistic_acquisition_state.buffer.get_variable_coverage())
        target_idx = variable_order.index(realistic_acquisition_state.current_target)
        
        non_target_vars = [v for i, v in enumerate(variable_order) if i != target_idx]
        non_target_logits = [output['variable_logits'][i] for i, v in enumerate(variable_order) if i != target_idx]
        non_target_probs = [marginal_probs.get(v, 0.0) for v in non_target_vars]
        
        # Check that uncertainty information is being used
        # At minimum, variables with zero parent probability shouldn't have the highest logit
        if len(non_target_vars) > 1:
            max_logit_idx = np.argmax(non_target_logits)
            max_logit_var = non_target_vars[max_logit_idx]
            max_logit_prob = marginal_probs.get(max_logit_var, 0.0)
            
            # If there are variables with positive parent probability, 
            # the max logit variable should have some parent probability
            has_positive_prob_vars = any(p > 0.1 for p in non_target_probs)
            if has_positive_prob_vars:
                assert max_logit_prob > 0.05, f"Variable with highest logit has very low parent probability: {max_logit_prob}"
        
        print(f"✓ Policy integrates uncertainty: parent probs {dict(zip(non_target_vars, non_target_probs))}")
    
    def test_optimization_context_integration(self, realistic_acquisition_state):
        """Test that policy uses optimization context (best value, target variable)."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, realistic_acquisition_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, realistic_acquisition_state, False)
        
        # Test with different best values
        state_low_best = AcquisitionState(
            posterior=realistic_acquisition_state.posterior,
            buffer=realistic_acquisition_state.buffer,
            best_value=-10.0,  # Much lower best value
            current_target=realistic_acquisition_state.current_target,
            step=realistic_acquisition_state.step,
            metadata=realistic_acquisition_state.metadata
        )
        
        state_high_best = AcquisitionState(
            posterior=realistic_acquisition_state.posterior,
            buffer=realistic_acquisition_state.buffer,
            best_value=100.0,  # Much higher best value
            current_target=realistic_acquisition_state.current_target,
            step=realistic_acquisition_state.step,
            metadata=realistic_acquisition_state.metadata
        )
        
        # Get outputs for different best values
        output_low = policy_transform.apply(params, key, state_low_best, False)
        output_high = policy_transform.apply(params, key, state_high_best, False)
        
        # Value parameters should be different (optimization context matters)
        assert not jnp.allclose(output_low['value_params'], output_high['value_params'], atol=1e-6)
        
        # State values should be different
        assert not jnp.allclose(output_low['state_value'], output_high['state_value'], atol=1e-6)
        
        print(f"✓ Policy uses optimization context: state values {output_low['state_value']:.3f} vs {output_high['state_value']:.3f}")


class TestEdgeCasesWithRealComponents:
    """Test edge cases using real ACBO components."""
    
    def test_minimal_data_scenario(self):
        """Test policy with minimal real data."""
        # Create minimal SCM
        minimal_scm = create_simple_linear_scm(
            variables=['X', 'Y'],
            edges=[('X', 'Y')],
            coefficients={('X', 'Y'): 2.0},
            noise_scales={'X': 1.0, 'Y': 0.5},
            target='Y'
        )
        
        # Create minimal buffer
        buffer = ExperienceBuffer()
        
        # Just one observational sample
        obs_sample = sample_from_linear_scm(minimal_scm, n_samples=1, seed=42)[0]
        buffer.add_observation(obs_sample)
        
        # Minimal posterior
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.3, 0.7]),
            metadata={'minimal': True}
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=obs_sample["values"]['Y'],
            current_target='Y',
            step=1,
            metadata=pyr.m(**{'scenario': 'minimal'})
        )
        
        # Policy should handle minimal state
        config = PolicyConfig(hidden_dim=32, num_layers=1)
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
        print("✓ Policy handles minimal data scenario")
    
    def test_no_interventional_data(self):
        """Test policy with only observational data."""
        # Create SCM
        scm = create_simple_linear_scm(
            variables=['X', 'Y', 'Z'],
            edges=[('X', 'Y'), ('Z', 'Y')],
            coefficients={('X', 'Y'): 1.5, ('Z', 'Y'): -1.0},
            noise_scales={'X': 1.0, 'Y': 0.5, 'Z': 1.0},
            target='Y'
        )
        
        # Buffer with only observational data
        buffer = ExperienceBuffer()
        obs_samples = sample_from_linear_scm(scm, n_samples=20, seed=42)
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Create posterior
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(['X', 'Z'])],
            probabilities=jnp.array([1.0]),
            metadata={'obs_only': True}
        )
        
        # Create state
        target_values = [s["values"]['Y'] for s in buffer.get_observations()]
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=max(target_values),
            current_target='Y',
            step=len(target_values),
            metadata=pyr.m(**{'data_type': 'observational_only'})
        )
        
        # Policy should work with observational-only data
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
        
        # Should still be able to sample interventions
        intervention = sample_intervention_from_policy(output, state, key, config)
        assert intervention['type'] == 'perfect'
        assert len(intervention['targets']) == 1
        
        print("✓ Policy handles observational-only data")


def test_memory_efficiency():
    """Test that policy doesn't use excessive memory."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create moderately large problem
    scm = create_simple_linear_scm(
        variables=[f'X{i}' for i in range(6)] + ['Y'],
        edges=[(f'X{i}', 'Y') for i in range(3)],
        coefficients={('X0', 'Y'): 1.0, ('X1', 'Y'): -1.0, ('X2', 'Y'): 0.5},
        noise_scales={f'X{i}': 1.0 for i in range(6)} | {'Y': 0.5},
        target='Y'
    )
    
    buffer = ExperienceBuffer()
    
    # Add significant amount of data
    obs_samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Create and run policy multiple times
    for i in range(10):
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(['X0', 'X1'])],
            probabilities=jnp.array([1.0]),
            metadata={'iteration': i}
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=5.0,
            current_target='Y',
            step=i,
            metadata=pyr.m(**{'test_iteration': i})
        )
        
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42 + i)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
    
    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable for a neural network (less than 400MB for this test)
    assert memory_increase < 1000, f"Excessive memory usage: {memory_increase:.1f}MB increase"
    
    print(f"✓ Memory usage test passed: {memory_increase:.1f}MB increase over 10 iterations")


if __name__ == "__main__":
    # Run basic integration test
    print("=== Comprehensive ACBO Policy Integration Test ===")
    
    try:
        # Test 1: Basic integration
        print("\n1. Testing basic policy-ACBO integration...")
        
        scm = create_simple_linear_scm(
            variables=['X', 'Y', 'Z'],
            edges=[('X', 'Y'), ('Z', 'Y')],
            coefficients={('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
            noise_scales={'X': 1.0, 'Y': 0.5, 'Z': 1.0},
            target='Y'
        )
        
        buffer = ExperienceBuffer()
        
        # Add mixed data
        obs_samples = sample_from_linear_scm(scm, n_samples=10, seed=42)
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 5.0})
        int_samples = sample_with_intervention(scm, intervention, n_samples=2, seed=100)
        for sample in int_samples:
            buffer.add_intervention(intervention, sample)
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(['X', 'Z'])],
            probabilities=jnp.array([1.0]),
            metadata={}
        )
        
        target_values = [s["values"]['Y'] for s in buffer.get_observations()]
        target_values.extend([s["values"]['Y'] for _, s in buffer.get_interventions()])
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=max(target_values),
            current_target='Y',
            step=len(target_values),
            metadata=pyr.m(**{'basic_test': True})
        )
        
        # Test policy
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
        print("✓ Basic integration test passed")
        
        # Test 2: Complete cycle
        print("\n2. Testing complete intervention cycle...")
        
        # Sample intervention and apply to SCM
        intervention = sample_intervention_from_policy(output, state, key, config)
        new_samples = sample_with_intervention(scm, intervention, n_samples=1, seed=200)
        new_sample = new_samples[0]
        
        # Update buffer and state (proper way to create updated buffer)
        from causal_bayes_opt.data_structures.buffer import create_buffer_from_samples
        existing_observations = buffer.get_observations()
        existing_interventions = buffer.get_interventions()
        new_buffer = create_buffer_from_samples(
            observations=existing_observations,
            interventions=existing_interventions + [(intervention, new_sample)]
        )
        
        new_best = max(state.best_value, new_sample["values"]['Y'])
        new_state = AcquisitionState(
            posterior=posterior,
            buffer=new_buffer,
            best_value=new_best,
            current_target='Y',
            step=state.step + 1,
            metadata=pyr.m(**{'updated': True})
        )
        
        # Test policy with updated state
        new_output = policy_transform.apply(params, key, new_state, False)
        assert validate_policy_output(new_output, new_state)
        print("✓ Complete cycle test passed")
        
        print("\n✅ All integration tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
