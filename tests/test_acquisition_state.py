"""
Comprehensive tests for the AcquisitionState implementation.

These tests validate that the state representation correctly integrates with:
- Phase 1: ExperienceBuffer, SCM, Sample data structures
- Phase 2: ParentSetPosterior and AVICI integration
- Proper error handling and validation
- Derived property computation
"""

import pytest
import time
import jax.numpy as jnp
import pyrsistent as pyr
from typing import FrozenSet

# Test imports - adjust paths based on actual structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_bayes_opt.acquisition import (
    AcquisitionState,
    create_acquisition_state,
    update_state_with_intervention,
    get_state_uncertainty_bits,
    get_state_optimization_progress,
    get_state_marginal_probabilities,
)

from causal_bayes_opt.data_structures import (
    ExperienceBuffer,
    create_empty_buffer,
    create_observational_sample,
    create_interventional_sample,
    create_scm,
)

from causal_bayes_opt.avici_integration.parent_set import (
    create_parent_set_posterior,
    ParentSetPosterior,
)

from causal_bayes_opt.interventions import create_perfect_intervention


class TestAcquisitionStateBasics:
    """Test basic functionality of AcquisitionState."""
    
    def test_acquisition_state_creation_and_properties(self):
        """Test that AcquisitionState can be created and has correct derived properties."""
        # Create test data
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer()
        
        # Create state
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.5,
            current_target='Y',
            step=10,
            metadata=pyr.m(**{'test': True})
        )
        
        # Test basic properties
        assert state.current_target == 'Y'
        assert state.best_value == 2.5
        assert state.step == 10
        assert state.metadata['test'] is True
        
        # Test derived properties
        assert isinstance(state.uncertainty_bits, float)
        assert state.uncertainty_bits > 0
        assert isinstance(state.buffer_statistics.total_samples, int)
        assert isinstance(state.marginal_parent_probs, dict)
        
        # Test uncertainty conversion (nats to bits)
        expected_bits = posterior.uncertainty / jnp.log(2.0)
        assert abs(state.uncertainty_bits - expected_bits) < 1e-6
    
    def test_state_validation(self):
        """Test that state validation catches inconsistencies."""
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer()
        
        # Test target variable mismatch
        with pytest.raises(ValueError, match="Target mismatch"):
            AcquisitionState(
                posterior=posterior,
                buffer=buffer,
                best_value=2.0,
                current_target='Z',  # Different from posterior target 'Y'
                step=0
            )
        
        # Test negative step
        with pytest.raises(ValueError, match="Step must be non-negative"):
            AcquisitionState(
                posterior=posterior,
                buffer=buffer,
                best_value=2.0,
                current_target='Y',
                step=-1
            )
        
        # Test infinite best value
        with pytest.raises(ValueError, match="Best value must be finite"):
            AcquisitionState(
                posterior=posterior,
                buffer=buffer,
                best_value=float('inf'),
                current_target='Y',
                step=0
            )
    
    def test_optimization_progress_metrics(self):
        """Test computation of optimization progress metrics."""
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer_with_progression()
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=3.0,
            current_target='Y',
            step=5
        )
        
        progress = state.get_optimization_progress()
        
        # Check that progress metrics are computed
        assert 'improvement_from_start' in progress
        assert 'recent_improvement' in progress
        assert 'optimization_rate' in progress
        assert 'stagnation_steps' in progress
        
        # Basic sanity checks
        assert isinstance(progress['improvement_from_start'], float)
        assert isinstance(progress['optimization_rate'], float)
        assert progress['stagnation_steps'] >= 0
    
    def test_exploration_coverage_metrics(self):
        """Test computation of exploration coverage metrics."""
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer_with_interventions()
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=3
        )
        
        coverage = state.get_exploration_coverage()
        
        # Check that coverage metrics are computed
        assert 'target_coverage_rate' in coverage
        assert 'intervention_diversity' in coverage
        assert 'unexplored_variables' in coverage
        
        # Basic sanity checks
        assert 0.0 <= coverage['target_coverage_rate'] <= 1.0
        assert 0.0 <= coverage['intervention_diversity'] <= 1.0
        assert 0.0 <= coverage['unexplored_variables'] <= 1.0
    
    def test_state_summary(self):
        """Test that state summary provides comprehensive information."""
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer()
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=7
        )
        
        summary = state.summary()
        
        # Check essential summary fields
        required_fields = [
            'step', 'target_variable', 'best_value',
            'uncertainty_bits', 'uncertainty_nats', 'posterior_concentration',
            'total_samples', 'observational_samples', 'interventional_samples',
            'most_likely_parents', 'most_likely_probability',
            'optimization_progress', 'exploration_coverage'
        ]
        
        for field in required_fields:
            assert field in summary, f"Missing summary field: {field}"
        
        # Check summary content
        assert summary['step'] == 7
        assert summary['target_variable'] == 'Y'
        assert summary['best_value'] == 2.0
        assert isinstance(summary['most_likely_parents'], list)
    
    def test_state_repr(self):
        """Test string representation."""
        posterior = self._create_test_posterior()
        buffer = self._create_test_buffer()
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.5,
            current_target='Y',
            step=15
        )
        
        repr_str = repr(state)
        
        # Check that key information appears in representation
        assert 'step=15' in repr_str
        assert "target='Y'" in repr_str
        assert 'best_value=2.5' in repr_str
        assert 'uncertainty=' in repr_str
        assert 'samples=' in repr_str
    
    # Helper methods for creating test data
    def _create_test_posterior(self) -> ParentSetPosterior:
        """Create a test posterior for Y with X and Z as potential parents."""
        parent_sets = [
            frozenset(),  # Empty set
            frozenset(['X']),  # X only
            frozenset(['Z']),  # Z only  
            frozenset(['X', 'Z'])  # Both X and Z
        ]
        probabilities = jnp.array([0.1, 0.3, 0.2, 0.4])
        
        return create_parent_set_posterior(
            target_variable='Y',
            parent_sets=parent_sets,
            probabilities=probabilities,
            metadata={'test_posterior': True}
        )
    
    def _create_test_buffer(self) -> ExperienceBuffer:
        """Create a simple test buffer with observational data."""
        buffer = create_empty_buffer()
        
        # Add some observational samples
        samples = [
            create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}),
            create_observational_sample({'X': 2.0, 'Y': 2.5, 'Z': 1.0}),
            create_observational_sample({'X': 0.5, 'Y': 1.5, 'Z': 2.0}),
        ]
        
        for sample in samples:
            buffer.add_observation(sample)
        
        return buffer
    
    def _create_test_buffer_with_progression(self) -> ExperienceBuffer:
        """Create buffer with clear progression in target variable."""
        buffer = create_empty_buffer()
        
        # Add samples with increasing Y values (clear progression)
        progression_data = [
            {'X': 1.0, 'Y': 1.0, 'Z': 1.0},
            {'X': 1.5, 'Y': 1.5, 'Z': 1.0},
            {'X': 2.0, 'Y': 2.0, 'Z': 1.0},
            {'X': 2.5, 'Y': 2.5, 'Z': 1.0},
            {'X': 3.0, 'Y': 3.0, 'Z': 1.0},
        ]
        
        for values in progression_data:
            sample = create_observational_sample(values)
            buffer.add_observation(sample)
        
        return buffer
    
    def _create_test_buffer_with_interventions(self) -> ExperienceBuffer:
        """Create buffer with both observational and interventional data."""
        buffer = create_empty_buffer()
        
        # Add observational samples
        obs_sample = create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5})
        buffer.add_observation(obs_sample)
        
        # Add interventional samples
        intervention_X = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome_X = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.5},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        buffer.add_intervention(intervention_X, outcome_X)
        
        intervention_Z = create_perfect_intervention(frozenset(['Z']), {'Z': 0.5})
        outcome_Z = create_interventional_sample(
            values={'X': 1.0, 'Y': 1.8, 'Z': 0.5},
            intervention_type='perfect',
            targets=frozenset(['Z'])
        )
        buffer.add_intervention(intervention_Z, outcome_Z)
        
        return buffer


class TestAcquisitionStateFactory:
    """Test factory functions for creating and updating states."""
    
    def test_create_acquisition_state_basic(self):
        """Test basic state creation with mock surrogate model."""
        # Create test SCM
        scm = create_scm(
            variables=frozenset(['X', 'Y', 'Z']),
            edges=frozenset([('X', 'Y'), ('Z', 'Y')]),
            mechanisms=pyr.m(**{
                'X': lambda noise: noise,
                'Y': lambda values, noise: values.get('X', 0) + values.get('Z', 0) + noise,
                'Z': lambda noise: noise
            }),
            target='Y'
        )
        
        # Create buffer with data
        buffer = self._create_test_buffer()
        
        # Create state (will use fallback uniform posterior)
        state = create_acquisition_state(
            scm=scm,
            buffer=buffer,
            surrogate_model=None,  # Will trigger fallback
            surrogate_params=None,
            target_variable='Y',
            step=5,
            metadata={'test_creation': True}
        )
        
        # Validate created state
        assert state.current_target == 'Y'
        assert state.step == 5
        assert state.best_value >= 1.5  # Should be max Y value from buffer
        assert state.uncertainty_bits > 0
        assert 'test_creation' in state.metadata
        assert state.metadata['test_creation'] is True
        
        # Check fallback posterior was created
        assert 'fallback_uniform' in state.posterior.metadata
        assert state.posterior.metadata['fallback_uniform'] is True
    
    def test_create_acquisition_state_validation(self):
        """Test input validation for state creation."""
        scm = self._create_test_scm()
        buffer = self._create_test_buffer()
        
        # Test empty target variable
        with pytest.raises(ValueError, match="Target variable cannot be empty"):
            create_acquisition_state(scm, buffer, None, None, target_variable='', step=0)
        
        # Test negative step
        with pytest.raises(ValueError, match="Step must be non-negative"):
            create_acquisition_state(scm, buffer, None, None, target_variable='Y', step=-1)
        
        # Test empty buffer
        empty_buffer = create_empty_buffer()
        with pytest.raises(ValueError, match="Buffer must contain at least one sample"):
            create_acquisition_state(scm, empty_buffer, None, None, target_variable='Y', step=0)
        
        # Test target not in buffer
        with pytest.raises(ValueError, match="not in buffer variables"):
            create_acquisition_state(scm, buffer, None, None, target_variable='W', step=0)
    
    def test_update_state_with_intervention(self):
        """Test state update after intervention."""
        # Create initial state
        initial_state = self._create_test_state()
        
        # Create intervention and outcome
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 5.0})
        outcome = create_interventional_sample(
            values={'X': 5.0, 'Y': 4.0, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        # Create new posterior (simplified for testing)
        new_posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(['X']), frozenset(['X', 'Z'])],
            probabilities=jnp.array([0.7, 0.3]),
            metadata={'updated': True}
        )
        
        # Update state
        updated_state = update_state_with_intervention(
            initial_state, intervention, outcome, new_posterior
        )
        
        # Validate updated state
        assert updated_state.step == initial_state.step + 1
        assert updated_state.current_target == initial_state.current_target
        assert updated_state.best_value >= initial_state.best_value  # Should be at least as good
        assert updated_state.posterior.metadata.get('updated') is True
        
        # Check that intervention was added to buffer
        new_interventions = updated_state.buffer.get_interventions()
        old_interventions = initial_state.buffer.get_interventions()
        assert len(new_interventions) == len(old_interventions) + 1
        
        # Check metadata update
        assert 'last_intervention_step' in updated_state.metadata
        assert updated_state.metadata['last_intervention_step'] == initial_state.step
    
    def test_update_state_validation(self):
        """Test validation for state updates."""
        initial_state = self._create_test_state()
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 5.0})
        outcome = create_interventional_sample(
            values={'X': 5.0, 'Y': 4.0, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        # Create posterior with wrong target
        wrong_posterior = create_parent_set_posterior(
            target_variable='Z',  # Wrong target!
            parent_sets=[frozenset(['X'])],
            probabilities=jnp.array([1.0])
        )
        
        # Should raise error for target mismatch
        with pytest.raises(ValueError, match="doesn't match state target"):
            update_state_with_intervention(initial_state, intervention, outcome, wrong_posterior)
    
    # Helper methods
    def _create_test_scm(self):
        """Create a simple test SCM."""
        return create_scm(
            variables=frozenset(['X', 'Y', 'Z']),
            edges=frozenset([('X', 'Y'), ('Z', 'Y')]),
            mechanisms=pyr.m(**{
                'X': lambda noise: noise,
                'Y': lambda values, noise: values.get('X', 0) + values.get('Z', 0) + noise,
                'Z': lambda noise: noise
            }),
            target='Y'
        )
    
    def _create_test_buffer(self):
        """Create test buffer with sample data."""
        buffer = create_empty_buffer()
        
        samples = [
            create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}),
            create_observational_sample({'X': 2.0, 'Y': 2.5, 'Z': 1.0}),
        ]
        
        for sample in samples:
            buffer.add_observation(sample)
        
        return buffer
    
    def _create_test_state(self):
        """Create a test acquisition state."""
        scm = self._create_test_scm()
        buffer = self._create_test_buffer()
        
        return create_acquisition_state(
            scm=scm,
            buffer=buffer,
            surrogate_model=None,
            surrogate_params=None,
            target_variable='Y',
            step=10
        )


class TestAcquisitionStateUtilities:
    """Test utility functions for accessing state information."""
    
    def test_get_state_uncertainty_bits(self):
        """Test uncertainty accessor function."""
        state = self._create_test_state()
        uncertainty = get_state_uncertainty_bits(state)
        
        assert isinstance(uncertainty, float)
        assert uncertainty > 0
        assert uncertainty == state.uncertainty_bits
    
    def test_get_state_optimization_progress(self):
        """Test optimization progress accessor."""
        state = self._create_test_state()
        progress = get_state_optimization_progress(state)
        
        assert isinstance(progress, dict)
        required_keys = ['improvement_from_start', 'recent_improvement', 'optimization_rate', 'stagnation_steps']
        for key in required_keys:
            assert key in progress
    
    def test_get_state_marginal_probabilities(self):
        """Test marginal probabilities accessor."""
        state = self._create_test_state()
        marginals = get_state_marginal_probabilities(state)
        
        assert isinstance(marginals, dict)
        assert len(marginals) >= 0  # Could be empty if no variables
        
        # All values should be probabilities
        for var, prob in marginals.items():
            assert isinstance(var, str)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0
    
    def _create_test_state(self):
        """Create a test acquisition state."""
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        scm = create_scm(
            variables=frozenset(['X', 'Y', 'Z']),
            edges=frozenset([('X', 'Y')]),
            mechanisms=pyr.m(**{'X': lambda n: n, 'Y': lambda v, n: v.get('X', 0) + n, 'Z': lambda n: n}),
            target='Y'
        )
        
        return create_acquisition_state(scm, buffer, None, None, target_variable='Y', step=5)


if __name__ == '__main__':
    # Run basic smoke tests if executed directly
    print("Running basic smoke tests for AcquisitionState...")
    
    # Test 1: Basic state creation
    try:
        from causal_bayes_opt.data_structures import create_empty_buffer, create_observational_sample
        from causal_bayes_opt.avici_integration.parent_set import create_parent_set_posterior
        
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.4, 0.6])
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=1
        )
        
        print(f"✓ State created successfully: {state}")
        print(f"✓ Uncertainty: {state.uncertainty_bits:.2f} bits")
        print(f"✓ Buffer size: {state.buffer_statistics.total_samples}")
        print(f"✓ Marginal probabilities: {state.marginal_parent_probs}")
        
    except Exception as e:
        print(f"✗ Basic state creation failed: {e}")
        raise
    
    # Test 2: Factory function
    try:
        from causal_bayes_opt.data_structures import create_scm
        import pyrsistent as pyr
        
        scm = create_scm(
            variables=frozenset(['X', 'Y']),
            edges=frozenset([('X', 'Y')]),
            mechanisms=pyr.m(**{'X': lambda n: n, 'Y': lambda v, n: v.get('X', 0) + n}),
            target='Y'
        )
        
        factory_state = create_acquisition_state(
            scm=scm,
            buffer=buffer,
            surrogate_model=None,
            surrogate_params=None,
            target_variable='Y',
            step=0
        )
        
        print(f"✓ Factory state created: {factory_state}")
        print(f"✓ Using fallback posterior: {'fallback_uniform' in factory_state.posterior.metadata}")
        
    except Exception as e:
        print(f"✗ Factory function failed: {e}")
        raise
    
    print("\n🎉 All smoke tests passed! AcquisitionState implementation looks good.")
    print("\nNow run full test suite with: pytest tests/test_acquisition_state.py -v")
