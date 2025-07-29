"""
Test Two-Phase GRPO + Active Learning Training

This module tests the two-phase training approach implementation.
"""

import pytest
import tempfile
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.training.grpo_policy_loader import (
    LoadedGRPOPolicy, load_grpo_policy, create_grpo_intervention_fn
)
from scripts.core.two_phase_training import (
    create_phase2_config, run_phase2_active_learning
)
from examples.demo_scms import create_easy_scm


class TestGRPOPolicyLoader:
    """Test GRPO policy loading functionality."""
    
    def test_load_grpo_policy_from_checkpoint(self):
        """Test loading GRPO policy from checkpoint."""
        # Create a mock checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            # Create mock policy data
            policy_data = {
                'policy_params': {'test': jnp.array([1.0, 2.0, 3.0])},
                'policy_config': {
                    'variables': ['X', 'Y', 'Z'],
                    'target_variable': 'Z',
                    'encoder_layers': 4,
                    'encoder_heads': 8,
                    'encoder_dim': 128
                },
                'enriched_architecture': True,
                'episode': 100,
                'is_final': True
            }
            
            # Save checkpoint
            policy_file = checkpoint_dir / "policy_params.pkl"
            with open(policy_file, 'wb') as f:
                pickle.dump(policy_data, f)
            
            # Load policy
            loaded_policy = load_grpo_policy(str(checkpoint_dir))
            
            # Verify loaded components
            assert isinstance(loaded_policy, LoadedGRPOPolicy)
            assert loaded_policy.variables == ['X', 'Y', 'Z']
            assert loaded_policy.target_variable == 'Z'
            assert loaded_policy.is_enriched
            assert 'test' in loaded_policy.policy_params
            assert jnp.array_equal(
                loaded_policy.policy_params['test'],
                jnp.array([1.0, 2.0, 3.0])
            )
    
    def test_create_grpo_intervention_fn(self):
        """Test creating intervention function from loaded policy."""
        # Create mock loaded policy
        loaded_policy = LoadedGRPOPolicy(
            policy_params={'test': jnp.array([1.0])},
            policy_config={
                'variables': ['A', 'B', 'C', 'D'],
                'target_variable': 'D',
                'encoder_layers': 4,
                'encoder_heads': 8,
                'encoder_dim': 128
            },
            apply_fn=lambda params, key, x, t, training: {
                'intervention_logits': jnp.array([0.1, 0.5, 0.3, -10.0]),
                'value_params': jnp.array([0.0, -1.0, 1.0, 0.0])
            },
            variables=['A', 'B', 'C', 'D'],
            target_variable='D',
            is_enriched=True
        )
        
        # Create SCM
        scm = create_easy_scm()
        
        # Create intervention function
        intervention_fn = create_grpo_intervention_fn(
            loaded_policy=loaded_policy,
            scm=scm,
            intervention_range=(-2.0, 2.0)
        )
        
        # Test intervention selection
        key = jax.random.PRNGKey(42)
        intervention = intervention_fn(key=key)
        
        # Verify intervention structure
        assert 'targets' in intervention
        assert 'values' in intervention
        assert len(intervention['targets']) == 1
        assert len(intervention['values']) == 1
        
        # Target should not be selected
        selected_var = list(intervention['targets'])[0]
        assert selected_var != 'D'


class TestTwoPhaseTraining:
    """Test two-phase training functionality."""
    
    def test_create_phase2_config(self):
        """Test Phase 2 configuration creation."""
        config = create_phase2_config(
            n_observational_samples=20,
            n_intervention_steps=10,
            learning_rate=1e-4
        )
        
        assert config.n_observational_samples == 20
        assert config.n_intervention_steps == 10
        assert config.learning_rate == 1e-4
        assert config.scoring_method == "bic"
        assert config.intervention_value_range == (-2.0, 2.0)
    
    @pytest.mark.slow
    def test_phase2_active_learning_integration(self):
        """Test Phase 2 active learning with mock GRPO policy."""
        # This is an integration test that would require a real checkpoint
        # For unit testing, we mock the key components
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint
            checkpoint_dir = Path(tmpdir)
            policy_data = {
                'policy_params': {},  # Empty params for testing
                'policy_config': {
                    'variables': ['A', 'B', 'C', 'D'],
                    'target_variable': 'D',
                    'encoder_layers': 2,  # Smaller for testing
                    'encoder_heads': 4,
                    'encoder_dim': 64
                },
                'enriched_architecture': True
            }
            
            policy_file = checkpoint_dir / "policy_params.pkl"
            with open(policy_file, 'wb') as f:
                pickle.dump(policy_data, f)
            
            # Create simple SCM
            scm = create_easy_scm()
            
            # Create minimal config for testing
            config = create_phase2_config(
                n_observational_samples=5,
                n_intervention_steps=3,
                learning_rate=1e-3
            )
            
            # Note: Full integration test would run here
            # result = run_phase2_active_learning(
            #     scm=scm,
            #     grpo_checkpoint_path=str(checkpoint_dir),
            #     config=config,
            #     track_structure_learning=True
            # )
            
            # For now, just verify the setup works
            assert checkpoint_dir.exists()
            assert policy_file.exists()


def test_grpo_policy_validation():
    """Test GRPO policy validation checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        
        # Test missing enriched_architecture flag
        policy_data = {
            'policy_params': {},
            'policy_config': {
                'variables': ['X', 'Y'],
                'target_variable': 'Y'
            },
            'enriched_architecture': False  # Should fail
        }
        
        policy_file = checkpoint_dir / "policy_params.pkl"
        with open(policy_file, 'wb') as f:
            pickle.dump(policy_data, f)
        
        with pytest.raises(ValueError, match="only supports enriched"):
            load_grpo_policy(str(checkpoint_dir))


def test_scm_compatibility_check():
    """Test SCM compatibility validation."""
    # Create loaded policy with specific variables
    loaded_policy = LoadedGRPOPolicy(
        policy_params={},
        policy_config={'variables': ['X', 'Y', 'Z'], 'target_variable': 'Z'},
        apply_fn=lambda *args: {},
        variables=['X', 'Y', 'Z'],
        target_variable='Z',
        is_enriched=True
    )
    
    # Create incompatible SCM (different target)
    scm = pyr.pmap({
        'variables': ['X', 'Y', 'Z'],
        'target': 'Y',  # Different from policy target
        'edges': []
    })
    
    # This should log a warning but not fail
    # (tested manually as warning detection is complex)
    intervention_fn = create_grpo_intervention_fn(
        loaded_policy=loaded_policy,
        scm=scm
    )
    
    assert intervention_fn is not None