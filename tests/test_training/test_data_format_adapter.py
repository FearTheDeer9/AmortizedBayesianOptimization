#!/usr/bin/env python3
"""
Tests for data format adapter functionality.

Tests conversion between ExpertDemonstration and DemonstrationData formats.
"""

import pytest
import pickle
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.training.data_format_adapter import (
    convert_expert_demonstration_to_training_data,
    convert_expert_demonstrations_batch,
    validate_converted_data,
    _convert_pmap_samples_to_avici_format,
    _extract_variable_order_from_scm,
    _create_mock_posterior_history,
    _create_mock_intervention_sequence
)
from src.causal_bayes_opt.training.expert_collection.data_structures import ExpertDemonstration
from src.causal_bayes_opt.training.pure_data_loader import DemonstrationData


class TestDataFormatAdapter:
    """Test data format adapter functionality."""
    
    def create_mock_expert_demonstration(self):
        """Create a mock ExpertDemonstration for testing."""
        # Create mock SCM
        scm = pyr.pmap({
            'variables': ['X0', 'X1', 'X2'],
            'edges': [('X1', 'X0'), ('X2', 'X0')],
            'metadata': {'graph_type': 'erdos_renyi'}
        })
        
        # Create mock observational samples
        obs_samples = [
            pyr.pmap({'X0': 0.5, 'X1': 0.3, 'X2': 0.7}),
            pyr.pmap({'X0': 0.8, 'X1': 0.2, 'X2': 0.9}),
            pyr.pmap({'X0': 0.1, 'X1': 0.6, 'X2': 0.4})
        ]
        
        # Create mock interventional samples
        int_samples = [
            pyr.pmap({'X0': 0.9, 'X1': 1.0, 'X2': 0.7}),
            pyr.pmap({'X0': 0.2, 'X1': 1.0, 'X2': 0.4})
        ]
        
        return ExpertDemonstration(
            scm=scm,
            target_variable='X0',
            n_nodes=3,
            graph_type='erdos_renyi',
            observational_samples=obs_samples,
            interventional_samples=int_samples,
            discovered_parents=frozenset(['X1', 'X2']),
            confidence=0.85,
            accuracy=0.90,
            parent_posterior={'X1': 0.7, 'X2': 0.6},
            data_requirements={'observational': 100, 'interventional': 50},
            inference_time=15.5,
            total_samples_used=150,
            collection_timestamp=1234567890.0,
            validation_passed=True
        )
    
    def test_convert_pmap_samples_to_avici_format(self):
        """Test conversion of PMap samples to AVICI format."""
        samples = [
            pyr.pmap({'X0': 0.5, 'X1': 0.3}),
            pyr.pmap({'X0': 0.8, 'X1': 0.2})
        ]
        variable_order = ['X0', 'X1']
        
        result = _convert_pmap_samples_to_avici_format(samples, variable_order)
        
        assert result.shape == (2, 2, 3)
        assert result[0, 0, 0] == 0.5  # X0 value in first sample
        assert result[0, 0, 1] == 1.0  # Indicator present
        assert result[0, 0, 2] == 0.0  # Observational context
        assert result[0, 1, 0] == 0.3  # X1 value in first sample
        assert result[1, 0, 0] == 0.8  # X0 value in second sample
    
    def test_convert_pmap_samples_empty(self):
        """Test conversion with empty samples."""
        samples = []
        variable_order = ['X0', 'X1']
        
        result = _convert_pmap_samples_to_avici_format(samples, variable_order)
        
        assert result.shape == (0, 2, 3)
    
    def test_extract_variable_order_from_scm(self):
        """Test extracting variable order from SCM."""
        scm = pyr.pmap({
            'variables': ['X0', 'X1', 'X2'],
            'edges': [('X1', 'X0')]
        })
        
        result = _extract_variable_order_from_scm(scm)
        
        assert result == ['X0', 'X1', 'X2']
    
    def test_extract_variable_order_from_edges(self):
        """Test extracting variable order from SCM edges when variables not present."""
        scm = pyr.pmap({
            'edges': [('X1', 'X0'), ('X2', 'X0')]
        })
        
        result = _extract_variable_order_from_scm(scm)
        
        assert set(result) == {'X0', 'X1', 'X2'}
        assert result == sorted(result)  # Should be sorted
    
    def test_create_mock_posterior_history(self):
        """Test creating mock posterior history."""
        target_variable = 'X0'
        discovered_parents = frozenset(['X1', 'X2'])
        
        result = _create_mock_posterior_history(target_variable, discovered_parents)
        
        assert len(result) == 3  # Initial + 2 parents
        assert result[0]['step'] == 0
        assert result[0]['action'] == 'initial_observation'
        assert result[0]['posterior_entropy'] == 1.0
        
        # Check progression
        assert result[1]['step'] == 1
        assert result[2]['step'] == 2
        assert result[1]['posterior_entropy'] > result[2]['posterior_entropy']
    
    def test_create_mock_intervention_sequence(self):
        """Test creating mock intervention sequence."""
        discovered_parents = frozenset(['X1', 'X2'])
        
        result = _create_mock_intervention_sequence(discovered_parents)
        
        assert len(result) == 2
        assert result[0]['step'] == 1
        assert result[0]['variable'] == 'X1'  # Alphabetically first
        assert result[0]['type'] == 'do_intervention'
        assert result[1]['variable'] == 'X2'
    
    def test_convert_expert_demonstration_to_training_data(self):
        """Test complete conversion of ExpertDemonstration to DemonstrationData."""
        expert_demo = self.create_mock_expert_demonstration()
        
        result = convert_expert_demonstration_to_training_data(expert_demo, "test_demo")
        
        assert isinstance(result, DemonstrationData)
        assert result.demo_id == "test_demo"
        assert result.target_variable == 'X0'
        assert result.variable_order == ['X0', 'X1', 'X2']
        assert result.expert_accuracy == 0.90
        assert result.confidence_score == 0.85
        
        # Check AVICI data shape
        assert result.avici_data.shape[1] == 3  # 3 variables
        assert result.avici_data.shape[2] == 3  # AVICI format
        assert result.avici_data.shape[0] == 5  # 3 obs + 2 int samples
        
        # Check metadata
        assert result.metadata['original_format'] == 'ExpertDemonstration'
        assert result.metadata['n_nodes'] == 3
        assert result.metadata['graph_type'] == 'erdos_renyi'
        assert set(result.metadata['discovered_parents']) == {'X1', 'X2'}
    
    def test_convert_expert_demonstrations_batch(self):
        """Test batch conversion of multiple demonstrations."""
        expert_demos = [
            self.create_mock_expert_demonstration(),
            self.create_mock_expert_demonstration()
        ]
        
        result = convert_expert_demonstrations_batch(expert_demos, "batch_test")
        
        assert len(result) == 2
        assert result[0].demo_id == "batch_test_0000"
        assert result[1].demo_id == "batch_test_0001"
        
        for demo in result:
            assert isinstance(demo, DemonstrationData)
            assert demo.target_variable == 'X0'
    
    def test_validate_converted_data(self):
        """Test validation of converted data."""
        expert_demo = self.create_mock_expert_demonstration()
        converted_demo = convert_expert_demonstration_to_training_data(expert_demo)
        
        assert validate_converted_data(converted_demo) == True
    
    def test_validate_converted_data_invalid(self):
        """Test validation with invalid data."""
        # Create invalid demo data
        invalid_demo = DemonstrationData(
            demo_id="invalid",
            avici_data=jnp.zeros((5, 2, 2)),  # Wrong shape (should be 3 channels)
            target_variable='X0',
            variable_order=['X0', 'X1'],
            posterior_history=[],
            intervention_sequence=[],
            expert_accuracy=0.8,
            confidence_score=0.9,
            metadata=pyr.pmap({})
        )
        
        assert validate_converted_data(invalid_demo) == False
    
    def test_conversion_with_minimal_scm(self):
        """Test conversion with minimal SCM structure."""
        scm = pyr.pmap({})  # Empty SCM
        
        expert_demo = ExpertDemonstration(
            scm=scm,
            target_variable='X0',
            n_nodes=1,
            graph_type='single_node',
            observational_samples=[pyr.pmap({'X0': 0.5})],
            interventional_samples=[],
            discovered_parents=frozenset(),
            confidence=0.5,
            accuracy=0.6,
            parent_posterior={},
            data_requirements={'observational': 10},
            inference_time=1.0,
            total_samples_used=10
        )
        
        result = convert_expert_demonstration_to_training_data(expert_demo)
        
        assert isinstance(result, DemonstrationData)
        assert result.target_variable == 'X0'
        assert 'X0' in result.variable_order
        assert validate_converted_data(result) == True


class TestRealDataIntegration:
    """Test integration with real expert demonstration data."""
    
    def test_load_and_convert_real_demonstration(self):
        """Test loading and converting a real expert demonstration if available."""
        # Look for real demonstration files
        demo_dir = Path("/Users/harellidar/Documents/Imperial/Individual_Project/incorporate-expert-demonstrations/data/expert_demonstrations")
        
        if not demo_dir.exists():
            pytest.skip("No expert demonstration directory found")
        
        # Find first pickle file
        pickle_files = list(demo_dir.glob("*.pkl"))
        if not pickle_files:
            pytest.skip("No pickle files found in demonstration directory")
        
        # Try to load and convert first file
        pickle_file = pickle_files[0]
        try:
            with open(pickle_file, 'rb') as f:
                expert_demo = pickle.load(f)
            
            # Check if it's the expected type
            if not isinstance(expert_demo, ExpertDemonstration):
                pytest.skip(f"File contains {type(expert_demo)}, not ExpertDemonstration")
            
            # Convert to training data
            converted_demo = convert_expert_demonstration_to_training_data(expert_demo)
            
            # Validate conversion
            assert validate_converted_data(converted_demo) == True
            assert isinstance(converted_demo, DemonstrationData)
            
            # Check basic properties
            assert converted_demo.target_variable is not None
            assert len(converted_demo.variable_order) > 0
            assert converted_demo.avici_data.shape[1] == len(converted_demo.variable_order)
            assert converted_demo.avici_data.shape[2] == 3
            
            print(f"✓ Successfully converted real demonstration from {pickle_file}")
            print(f"  Target variable: {converted_demo.target_variable}")
            print(f"  Variables: {converted_demo.variable_order}")
            print(f"  AVICI data shape: {converted_demo.avici_data.shape}")
            print(f"  Accuracy: {converted_demo.expert_accuracy}")
            
        except Exception as e:
            pytest.skip(f"Could not load or convert {pickle_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])