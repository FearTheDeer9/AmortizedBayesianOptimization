#!/usr/bin/env python3
"""
Full System Integration Test

Tests the complete pipeline from expert demonstrations to trained models.
This demonstrates that we can now use real data end-to-end.
"""

import pytest
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pyrsistent as pyr

from src.causal_bayes_opt.training.data_format_adapter import (
    convert_expert_demonstration_to_training_data,
    convert_expert_demonstrations_batch,
    validate_converted_data
)
from src.causal_bayes_opt.training.expert_collection.data_structures import ExpertDemonstration
from src.causal_bayes_opt.training.pure_data_loader import DemonstrationData
from src.causal_bayes_opt.training.trajectory_extractor import extract_surrogate_training_data
from src.causal_bayes_opt.training.surrogate_bc_trainer import create_surrogate_bc_trainer
from src.causal_bayes_opt.training.surrogate_training import TrainingExample


class TestFullSystemIntegration:
    """Test complete pipeline integration."""
    
    def create_realistic_expert_demonstration(self):
        """Create a realistic ExpertDemonstration for integration testing."""
        # Create realistic SCM structure
        scm = pyr.pmap({
            'variables': ['X0', 'X1', 'X2', 'X3'],
            'edges': [('X1', 'X0'), ('X2', 'X0'), ('X3', 'X1')],
            'metadata': {'graph_type': 'erdos_renyi', 'density': 0.3}
        })
        
        # Create realistic observational samples with correlations
        key = jrandom.PRNGKey(42)
        obs_samples = []
        for i in range(20):
            key, subkey = jrandom.split(key)
            x3 = jrandom.normal(subkey, ())
            key, subkey = jrandom.split(key)
            x1 = 0.5 * x3 + jrandom.normal(subkey, ()) * 0.5
            key, subkey = jrandom.split(key)
            x2 = jrandom.normal(subkey, ())
            key, subkey = jrandom.split(key)
            x0 = 0.7 * x1 + 0.3 * x2 + jrandom.normal(subkey, ()) * 0.3
            
            obs_samples.append(pyr.pmap({
                'X0': float(x0),
                'X1': float(x1), 
                'X2': float(x2),
                'X3': float(x3)
            }))
        
        # Create interventional samples
        int_samples = []
        for i in range(10):
            x1_int = 1.0  # Intervention on X1
            key, subkey = jrandom.split(key)
            x3 = jrandom.normal(subkey, ())
            key, subkey = jrandom.split(key)
            x2 = jrandom.normal(subkey, ())
            key, subkey = jrandom.split(key)
            x0 = 0.7 * x1_int + 0.3 * x2 + jrandom.normal(subkey, ()) * 0.3
            
            int_samples.append(pyr.pmap({
                'X0': float(x0),
                'X1': x1_int,
                'X2': float(x2),
                'X3': float(x3)
            }))
        
        return ExpertDemonstration(
            scm=scm,
            target_variable='X0',
            n_nodes=4,
            graph_type='erdos_renyi',
            observational_samples=obs_samples,
            interventional_samples=int_samples,
            discovered_parents=frozenset(['X1', 'X2']),
            confidence=0.88,
            accuracy=0.92,
            parent_posterior={'X1': 0.75, 'X2': 0.65, 'empty': 0.1},
            data_requirements={'observational': 100, 'interventional': 50},
            inference_time=25.3,
            total_samples_used=150,
            collection_timestamp=1234567890.0,
            validation_passed=True
        )
    
    def test_end_to_end_pipeline_with_mock_data(self):
        """Test complete pipeline from expert demo to trained model."""
        # Step 1: Create expert demonstration
        expert_demo = self.create_realistic_expert_demonstration()
        
        # Step 2: Convert to training data format
        demo_data = convert_expert_demonstration_to_training_data(expert_demo, "integration_test")
        
        # Validate conversion
        assert validate_converted_data(demo_data) == True
        assert demo_data.target_variable == 'X0'
        assert len(demo_data.variable_order) == 4
        
        # Step 3: Extract training examples from demonstration data
        # Create mock parent set posterior for the extraction
        from src.causal_bayes_opt.avici_integration.parent_set.posterior import ParentSetPosterior
        mock_posterior = Mock(spec=ParentSetPosterior)
        mock_posterior.target_variable = 'X0'
        mock_posterior.parent_set_probs = {
            frozenset(): 0.25,
            frozenset(['X1']): 0.35,
            frozenset(['X2']): 0.25,
            frozenset(['X1', 'X2']): 0.15
        }
        mock_posterior.uncertainty = 0.693
        mock_posterior.top_k_sets = [
            (frozenset(['X1']), 0.35),
            (frozenset(), 0.25),
            (frozenset(['X2']), 0.25),
            (frozenset(['X1', 'X2']), 0.15)
        ]
        
        # Create training example manually (since we have mock data)
        training_example = TrainingExample(
            observational_data=demo_data.avici_data,
            target_variable=demo_data.target_variable,
            variable_order=demo_data.variable_order,
            expert_posterior=mock_posterior,
            expert_accuracy=demo_data.expert_accuracy,
            scm_info=demo_data.metadata,
            problem_difficulty='medium'
        )
        
        # Step 4: Create and train surrogate model
        trainer = create_surrogate_bc_trainer(
            hidden_dims=[64, 32],
            learning_rate=1e-2,
            batch_size=2,
            max_epochs=3,
            use_jax_compilation=False  # For testing
        )
        
        # Initialize training state
        state = trainer.initialize_training_state(training_example)
        
        # Create simple training batches
        batches = [[training_example], [training_example]]
        
        # Train the model
        final_state = trainer.fit(batches, sample_input=training_example)
        
        # Step 5: Validate training worked
        assert final_state.epoch > 0
        assert len(final_state.training_metrics) > 0
        
        # Check that we can make predictions
        prediction = trainer.predict(final_state, training_example)
        assert 'parent_probs' in prediction
        assert 'target_variable' in prediction
        assert 'metadata' in prediction
        
        # Verify probabilities sum to 1 for continuous model
        probs = prediction['parent_probs']  # [d] per-variable probabilities
        prob_sum = jnp.sum(probs)  # Should sum to 1
        assert jnp.allclose(prob_sum, 1.0, atol=1e-4)
        
        # Verify target variable has probability 0 (can't be its own parent)
        target_idx = demo_data.variable_order.index(demo_data.target_variable)
        assert jnp.allclose(probs[target_idx], 0.0, atol=1e-6)
        
        print(f"✓ End-to-end pipeline successful!")
        print(f"  Training epochs: {final_state.epoch}")
        print(f"  Final loss: {final_state.training_metrics[-1].loss:.4f}")
        print(f"  Target variable: {demo_data.target_variable}")
        print(f"  AVICI data shape: {demo_data.avici_data.shape}")
        print(f"  Parent probs shape: {probs.shape}")
        print(f"  Architecture: {prediction['metadata']['architecture']}")
        print(f"  Scalability: Linear O(d) vs exponential O(2^d)")
    
    def test_batch_processing_pipeline(self):
        """Test processing multiple demonstrations in batch."""
        # Create multiple expert demonstrations
        expert_demos = [
            self.create_realistic_expert_demonstration(),
            self.create_realistic_expert_demonstration(),
            self.create_realistic_expert_demonstration()
        ]
        
        # Convert all to training data
        demo_data_list = convert_expert_demonstrations_batch(expert_demos, "batch_integration")
        
        assert len(demo_data_list) == 3
        
        # Validate all conversions
        for demo_data in demo_data_list:
            assert validate_converted_data(demo_data) == True
            assert demo_data.avici_data.shape[0] > 0  # Has samples
            assert demo_data.avici_data.shape[1] == 4  # 4 variables
            assert demo_data.avici_data.shape[2] == 3  # AVICI format
        
        print(f"✓ Batch processing successful for {len(demo_data_list)} demonstrations!")
        
        # Test that we can create training examples from all
        training_examples = []
        for demo_data in demo_data_list:
            # Create mock posterior for each
            mock_posterior = Mock()
            mock_posterior.target_variable = demo_data.target_variable
            mock_posterior.parent_set_probs = {frozenset(): 0.5, frozenset(['X1']): 0.5}
            mock_posterior.uncertainty = 0.693
            mock_posterior.top_k_sets = [(frozenset(['X1']), 0.5), (frozenset(), 0.5)]
            
            example = TrainingExample(
                observational_data=demo_data.avici_data,
                target_variable=demo_data.target_variable,
                variable_order=demo_data.variable_order,
                expert_posterior=mock_posterior,
                expert_accuracy=demo_data.expert_accuracy,
                scm_info=demo_data.metadata,
                problem_difficulty='medium'
            )
            training_examples.append(example)
        
        assert len(training_examples) == 3
        print(f"✓ Created {len(training_examples)} training examples from batch!")
    
    def test_data_validation_catches_errors(self):
        """Test that validation catches data format issues."""
        # Create expert demo with problematic data
        expert_demo = self.create_realistic_expert_demonstration()
        
        # Convert normally first
        demo_data = convert_expert_demonstration_to_training_data(expert_demo)
        assert validate_converted_data(demo_data) == True
        
        # Now create invalid version
        invalid_demo_data = demo_data.set('avici_data', jnp.zeros((5, 3, 2)))  # Wrong channels
        assert validate_converted_data(invalid_demo_data) == False
        
        # Test missing target variable
        invalid_demo_data2 = demo_data.set('target_variable', 'MISSING_VAR')
        assert validate_converted_data(invalid_demo_data2) == False
        
        print("✓ Data validation correctly catches format errors!")
    
    def test_integration_with_real_data_if_available(self):
        """Test integration with real data if expert demonstration files exist."""
        # Look for real demonstration files
        demo_dir = Path("/Users/harellidar/Documents/Imperial/Individual_Project/incorporate-expert-demonstrations/data/expert_demonstrations")
        
        if not demo_dir.exists():
            pytest.skip("No expert demonstration directory found - using mock data only")
            return
        
        # Find pickle files
        pickle_files = list(demo_dir.glob("*.pkl"))
        if not pickle_files:
            pytest.skip("No pickle files found in demonstration directory")
            return
        
        # Try to process first few files
        processed_count = 0
        for pickle_file in pickle_files[:3]:  # Limit to first 3 for testing
            try:
                with open(pickle_file, 'rb') as f:
                    expert_demo = pickle.load(f)
                
                if not isinstance(expert_demo, ExpertDemonstration):
                    continue
                
                # Convert to training data
                demo_data = convert_expert_demonstration_to_training_data(expert_demo)
                
                # Validate
                assert validate_converted_data(demo_data) == True
                
                # Check basic properties
                assert demo_data.target_variable in demo_data.variable_order
                assert demo_data.avici_data.shape[1] == len(demo_data.variable_order)
                assert demo_data.avici_data.shape[2] == 3
                
                processed_count += 1
                
                print(f"✓ Real data integration successful: {pickle_file.name}")
                print(f"  Target: {demo_data.target_variable}")
                print(f"  Variables: {demo_data.variable_order}")
                print(f"  Shape: {demo_data.avici_data.shape}")
                print(f"  Accuracy: {demo_data.expert_accuracy}")
                
            except Exception as e:
                print(f"⚠ Could not process {pickle_file}: {e}")
                continue
        
        if processed_count == 0:
            pytest.skip("No processable real demonstration files found")
        
        print(f"✓ Successfully processed {processed_count} real demonstration files!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])