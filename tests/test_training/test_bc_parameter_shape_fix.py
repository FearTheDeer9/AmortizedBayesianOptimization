#!/usr/bin/env python3
"""
Test BC Parameter Shape Fix

Tests for fixing the Haiku parameter shape mismatch issue in BC acquisition training.
This test suite verifies that the temporal aggregation works correctly with variable
sequence lengths.

Key test cases:
1. Parameter shape mismatch reproduction
2. Variable sequence length handling
3. Aggregation output consistency
4. Learning capability preservation
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

from causal_bayes_opt.acquisition.enriched.enriched_policy import EnrichedAttentionEncoder
from causal_bayes_opt.acquisition.enhanced_policy_network import EnhancedPolicyNetwork
from causal_bayes_opt.training.bc_acquisition_trainer import BCAcquisitionTrainer, BCPolicyConfig
from causal_bayes_opt.acquisition.policy import PolicyConfig


class TestBCParameterShapeFix:
    """Test suite for BC parameter shape mismatch fix."""
    
    def test_variable_sequence_length_aggregation(self):
        """Test that aggregation works with different sequence lengths."""
        key = random.PRNGKey(42)
        
        def forward_fn(seq_len: int):
            """Create encoder and process sequence of given length."""
            encoder = EnrichedAttentionEncoder(
                num_layers=2,
                num_heads=4,
                hidden_dim=64,
                key_size=16
            )
            
            # Create input with variable sequence length
            n_vars = 5
            num_channels = 10
            enriched_history = jnp.ones((seq_len, n_vars, num_channels))
            
            return encoder(enriched_history, is_training=False)
        
        # Transform the function
        init_fn = hk.transform(lambda: forward_fn(4))
        apply_fn = hk.transform(lambda seq_len: forward_fn(seq_len))
        
        # Initialize with one sequence length
        params = init_fn.init(key)
        
        # Apply with different sequence lengths
        for seq_len in [2, 3, 4, 5, 8]:
            output = apply_fn.apply(params, key, seq_len)
            
            # Check output shape is consistent
            assert output.shape == (5, 64), f"Wrong output shape for seq_len={seq_len}: {output.shape}"
    
    def test_enhanced_policy_network_variable_history(self):
        """Test that EnhancedPolicyNetwork handles variable history lengths."""
        key = random.PRNGKey(42)
        
        def forward_with_history(history_len: int):
            """Forward pass with given history length."""
            # Create network inside transform
            network = EnhancedPolicyNetwork(
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                num_variables=5
            )
            
            state_tensor = jnp.ones((5, 10))
            history_tensor = jnp.ones((history_len, 5, 10)) if history_len > 0 else None
            
            return network(
                state_tensor=state_tensor,
                target_variable_idx=0,
                history_tensor=history_tensor,
                is_training=False
            )
        
        # Transform and initialize
        init_fn = hk.transform(lambda: forward_with_history(3))
        apply_fn = hk.transform(lambda hist_len: forward_with_history(hist_len))
        
        params = init_fn.init(key)
        
        # Test with different history lengths
        for history_len in [0, 1, 2, 3, 5]:
            outputs = apply_fn.apply(params, key, history_len)
            
            # Verify all expected outputs are present
            assert 'variable_logits' in outputs
            assert 'value_params' in outputs
            assert 'state_value' in outputs
            
            # Verify shapes
            assert outputs['variable_logits'].shape == (5,)
            assert outputs['value_params'].shape == (5, 2)
    
    def test_aggregation_consistency_manual(self):
        """Test aggregation output shape is consistent for different sequence lengths."""
        key = random.PRNGKey(42)
        
        # Test with different configurations
        test_configs = [
            ([2, 3, 4, 5], 5, 64),  # seq_lengths, n_vars, hidden_dim
            ([1, 4, 8], 3, 32),
            ([3, 5, 7], 8, 128),
        ]
        
        for seq_lengths, n_vars, hidden_dim in test_configs:
            def create_and_apply(seq_len):
                """Create encoder and apply to sequence."""
                encoder = EnrichedAttentionEncoder(
                    num_layers=1,
                    num_heads=2,
                    hidden_dim=hidden_dim,
                    key_size=16
                )
                
                # Assume 10 input channels
                enriched_history = jnp.ones((seq_len, n_vars, 10))
                return encoder(enriched_history, is_training=False)
            
            # Initialize with first sequence length
            init_fn = hk.transform(lambda: create_and_apply(seq_lengths[0]))
            params = init_fn.init(key)
            
            # Apply to all sequence lengths and collect outputs
            outputs = []
            for seq_len in seq_lengths:
                apply_fn = hk.transform(lambda: create_and_apply(seq_len))
                output = apply_fn.apply(params, key)
                outputs.append(output)
            
            # All outputs should have the same shape
            expected_shape = (n_vars, hidden_dim)
            for i, output in enumerate(outputs):
                assert output.shape == expected_shape, \
                    f"Sequence length {seq_lengths[i]} produced wrong shape: {output.shape}"
    
    def test_bc_trainer_initialization_and_training_consistency(self):
        """Test that BC trainer can initialize and train without shape mismatches."""
        # Create trainer config
        policy_config = PolicyConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            use_enhanced_policy=True  # Use the enhanced policy
        )
        
        bc_config = BCPolicyConfig(
            policy_config=policy_config,
            learning_rate=1e-3,
            batch_size=8,
            use_jax_compilation=True
        )
        
        # Create trainer
        trainer = BCAcquisitionTrainer(bc_config)
        
        # Create dummy acquisition states with different history lengths
        from causal_bayes_opt.acquisition.state import AcquisitionState
        from causal_bayes_opt.data_structures import ExperienceBuffer
        
        states = []
        for i in range(4):
            # Create state with variable history length
            state = AcquisitionState(
                buffer=ExperienceBuffer(capacity=100),
                scm_info={'variables': ['A', 'B', 'C', 'D', 'E']},
                intervention_history=[{'variables': set(), 'values': {}}] * (i + 1),
                best_value=0.0,
                step=i
            )
            states.append(state)
        
        # Initialize trainer
        key = random.PRNGKey(42)
        trainer._create_jax_functions(key)
        
        # This should not raise any parameter shape errors
        assert trainer._policy_network is not None
        assert trainer.jax_train_step is not None
    
    def test_loss_decreases_with_training(self):
        """Test that the model can still learn after the fix."""
        key = random.PRNGKey(42)
        
        # Simple training setup
        def create_training_step():
            """Create a simple training step."""
            network = EnhancedPolicyNetwork(
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                num_variables=3
            )
            
            def loss_fn(params, batch):
                """Compute loss for a batch."""
                outputs = network(
                    state_tensor=batch['state'],
                    history_tensor=batch['history'],
                    is_training=True
                )
                
                # Simple MSE loss on logits
                target_logits = batch['target_logits']
                pred_logits = outputs['variable_logits']
                return jnp.mean((pred_logits - target_logits) ** 2)
            
            return hk.transform(loss_fn)
        
        # Initialize
        loss_transform = create_training_step()
        
        # Create dummy batch with consistent shapes
        batch = {
            'state': jnp.ones((3, 10)),  # 3 vars, 10 features
            'history': jnp.ones((3, 3, 10)),  # history_len=3
            'target_logits': jnp.array([1.0, 0.0, 0.0])
        }
        
        # Initialize parameters
        params = loss_transform.init(key, batch)
        
        # Compute initial loss
        initial_loss = loss_transform.apply(params, key, batch)
        
        # Simple gradient update
        grads = jax.grad(lambda p: loss_transform.apply(p, key, batch))(params)
        
        # Update parameters (simplified - just subtract small fraction of gradients)
        import jax.tree_util as tree
        updated_params = tree.tree_map(
            lambda p, g: p - 0.1 * g,
            params, grads
        )
        
        # Compute loss after update
        updated_loss = loss_transform.apply(updated_params, key, batch)
        
        # Loss should decrease
        assert updated_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss} -> {updated_loss}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])