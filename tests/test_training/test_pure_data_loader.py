#!/usr/bin/env python3
"""
Comprehensive tests for pure_data_loader.py

Following TDD approach with complete test coverage for all functions.
"""

import pytest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from src.causal_bayes_opt.training.pure_data_loader import (
    load_demonstration_batch,
    process_demonstration,
    load_demonstrations_from_directory,
    validate_demonstration_data,
    _extract_avici_data,
    _extract_posterior_history,
    _extract_intervention_sequence,
    _compute_demonstration_complexity,
    _compute_entropy,
    _compute_complexity_score,
    AVICIData,
    PosteriorStep,
    InterventionStep,
    DemonstrationData,
    LoadError,
    ValidationError
)
from src.causal_bayes_opt.training.expert_collection.data_structures import (
    ExpertDemonstration,
    DemonstrationBatch
)


class TestLoadDemonstrationBatch:
    """Test load_demonstration_batch function"""
    
    @patch('src.causal_bayes_opt.training.pure_data_loader.pickle.load')
    def test_load_valid_batch(self, mock_pickle_load, tmp_path):
        """Test loading valid demonstration batch"""
        # Create test file
        batch_file = tmp_path / "test_batch.pkl"
        batch_file.write_bytes(b"dummy_pickle_data")
        
        # Mock pickle.load to return valid batch
        mock_demo = Mock(spec=ExpertDemonstration)
        mock_demo.accuracy = 0.8
        mock_demo.graph_type = "erdos_renyi" 
        mock_demo.n_nodes = 5
        mock_batch = DemonstrationBatch(
            demonstrations=[mock_demo],
            batch_id="test_batch",
            collection_config={}
        )
        mock_pickle_load.return_value = mock_batch
        
        # Load and verify
        loaded_batch = load_demonstration_batch(str(batch_file))
        assert isinstance(loaded_batch, DemonstrationBatch)
        assert len(loaded_batch.demonstrations) == 1
        mock_pickle_load.assert_called_once()
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises LoadError"""
        with pytest.raises(LoadError, match="Demonstration file not found"):
            load_demonstration_batch("nonexistent_file.pkl")
    
    def test_load_directory_instead_of_file(self, tmp_path):
        """Test loading directory instead of file raises LoadError"""
        with pytest.raises(LoadError, match="Path is not a file"):
            load_demonstration_batch(str(tmp_path))
    
    def test_load_corrupted_pickle(self, tmp_path):
        """Test loading corrupted pickle raises LoadError"""
        # Create corrupted file
        corrupted_file = tmp_path / "corrupted.pkl"
        with open(corrupted_file, 'wb') as f:
            f.write(b"not a pickle")
        
        with pytest.raises(LoadError, match="Failed to unpickle"):
            load_demonstration_batch(str(corrupted_file))
    
    def test_load_wrong_type(self, tmp_path):
        """Test loading wrong object type raises LoadError"""
        # Save wrong type
        wrong_file = tmp_path / "wrong_type.pkl"
        with open(wrong_file, 'wb') as f:
            pickle.dump({"not": "a batch"}, f)
        
        with pytest.raises(LoadError, match="Expected DemonstrationBatch"):
            load_demonstration_batch(str(wrong_file))
    
    def test_load_empty_batch(self, tmp_path):
        """Test loading empty batch raises LoadError"""
        empty_batch = DemonstrationBatch(
            demonstrations=[],
            batch_id="empty_batch",
            collection_config={}
        )
        
        batch_file = tmp_path / "empty_batch.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(empty_batch, f)
        
        with pytest.raises(LoadError, match="Batch contains no demonstrations"):
            load_demonstration_batch(str(batch_file))


class TestComputeEntropy:
    """Test _compute_entropy function"""
    
    def test_compute_entropy_uniform(self):
        """Test entropy computation for uniform distribution"""
        # Uniform distribution should have maximum entropy
        probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        entropy = _compute_entropy(probs)
        
        # Entropy of uniform distribution with 4 elements = log(4)
        expected = jnp.log(4.0)
        assert jnp.isclose(entropy, expected, atol=1e-6)
    
    def test_compute_entropy_certain(self):
        """Test entropy computation for certain distribution"""
        # Certain distribution should have zero entropy
        probs = jnp.array([1.0, 0.0, 0.0, 0.0])
        entropy = _compute_entropy(probs)
        
        assert jnp.isclose(entropy, 0.0, atol=1e-6)
    
    def test_compute_entropy_with_zeros(self):
        """Test entropy computation handles zero probabilities"""
        probs = jnp.array([0.5, 0.5, 0.0, 0.0])
        entropy = _compute_entropy(probs)
        
        # Should not raise error and should return finite value
        assert jnp.isfinite(entropy)
        assert entropy > 0


class TestValidateDemonstration:
    """Test _validate_demonstration function"""
    
    def create_mock_demo(self, **kwargs):
        """Create mock demonstration with default values"""
        defaults = {
            'n_nodes': 5,
            'target_variable': 'X0',
            'scm': Mock(),
            'observational_samples': [Mock()],
            'interventional_samples': [Mock()],
            'accuracy': 0.8
        }
        defaults.update(kwargs)
        
        demo = Mock(spec=ExpertDemonstration)
        for key, value in defaults.items():
            setattr(demo, key, value)
        
        return demo
    
    def test_validate_valid_demo(self):
        """Test validation passes for valid demonstration"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo()
        # Should not raise exception
        _validate_demonstration(demo)
    
    def test_validate_wrong_type(self):
        """Test validation fails for wrong type"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        with pytest.raises(ValidationError, match="Expected ExpertDemonstration"):
            _validate_demonstration("not a demo")
    
    def test_validate_invalid_node_count(self):
        """Test validation fails for invalid node count"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo(n_nodes=0)
        with pytest.raises(ValidationError, match="Invalid node count"):
            _validate_demonstration(demo)
    
    def test_validate_missing_target_variable(self):
        """Test validation fails for missing target variable"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo(target_variable="")
        with pytest.raises(ValidationError, match="Missing target variable"):
            _validate_demonstration(demo)
    
    def test_validate_missing_scm(self):
        """Test validation fails for missing SCM"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo(scm=None)
        with pytest.raises(ValidationError, match="Missing SCM"):
            _validate_demonstration(demo)
    
    def test_validate_no_samples(self):
        """Test validation fails for no samples"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo(
            observational_samples=[],
            interventional_samples=[]
        )
        with pytest.raises(ValidationError, match="No samples available"):
            _validate_demonstration(demo)
    
    def test_validate_invalid_accuracy(self):
        """Test validation fails for invalid accuracy"""
        from src.causal_bayes_opt.training.pure_data_loader import _validate_demonstration
        
        demo = self.create_mock_demo(accuracy=1.5)
        with pytest.raises(ValidationError, match="Invalid accuracy"):
            _validate_demonstration(demo)


class TestComputeComplexityScore:
    """Test _compute_complexity_score function"""
    
    def test_compute_complexity_basic(self):
        """Test basic complexity computation"""
        complexity = _compute_complexity_score(
            n_nodes=5,
            edge_count=6,
            n_iterations=10,
            graph_type_weight=1.0
        )
        
        # Should be positive and finite
        assert complexity > 0
        assert jnp.isfinite(complexity)
    
    def test_compute_complexity_single_node(self):
        """Test complexity computation for single node"""
        complexity = _compute_complexity_score(
            n_nodes=1,
            edge_count=0,
            n_iterations=1,
            graph_type_weight=1.0
        )
        
        # Should handle single node case
        assert complexity > 0
        assert jnp.isfinite(complexity)
    
    def test_compute_complexity_increases_with_nodes(self):
        """Test complexity increases with node count"""
        complexity_small = _compute_complexity_score(3, 2, 5, 1.0)
        complexity_large = _compute_complexity_score(10, 15, 5, 1.0)  # More edges for larger graph
        
        assert complexity_large > complexity_small


class TestProcessDemonstration:
    """Test process_demonstration function"""
    
    @patch('src.causal_bayes_opt.training.pure_data_loader._validate_demonstration')
    @patch('src.causal_bayes_opt.training.pure_data_loader._extract_avici_data')
    @patch('src.causal_bayes_opt.training.pure_data_loader._extract_posterior_history')
    @patch('src.causal_bayes_opt.training.pure_data_loader._extract_intervention_sequence')
    @patch('src.causal_bayes_opt.training.pure_data_loader._compute_demonstration_complexity')
    def test_process_demonstration_success(self, mock_complexity, mock_interventions, 
                                         mock_posterior, mock_avici, mock_validate):
        """Test successful demonstration processing"""
        # Setup mocks
        mock_validate.return_value = None
        mock_avici.return_value = AVICIData(
            samples=jnp.zeros((10, 3, 3)),
            variables=('X0', 'X1', 'X2'),
            target_variable='X0',
            sample_count=10
        )
        mock_posterior.return_value = pyr.pvector([
            PosteriorStep(0, {frozenset(): 0.5}, 0.693)
        ])
        mock_interventions.return_value = pyr.pvector([
            InterventionStep(0, frozenset({'X1'}), (1.0,))
        ])
        mock_complexity.return_value = 5.0
        
        # Create mock demonstration
        demo = Mock(spec=ExpertDemonstration)
        demo.n_nodes = 3
        demo.graph_type = 'erdos_renyi'
        demo.target_variable = 'X0'
        demo.accuracy = 0.8
        
        # Process
        result = process_demonstration(demo, "test_demo")
        
        # Verify result
        assert isinstance(result, DemonstrationData)
        assert result.demo_id == "test_demo"
        assert result.n_nodes == 3
        assert result.graph_type == 'erdos_renyi'
        assert result.target_variable == 'X0'
        assert result.accuracy == 0.8
        assert result.complexity_score == 5.0
    
    @patch('src.causal_bayes_opt.training.pure_data_loader._validate_demonstration')
    def test_process_demonstration_validation_error(self, mock_validate):
        """Test processing fails when validation fails"""
        mock_validate.side_effect = ValidationError("Test error")
        
        demo = Mock(spec=ExpertDemonstration)
        
        with pytest.raises(ValidationError, match="Test error"):
            process_demonstration(demo, "test_demo")


class TestLoadDemonstrationsFromDirectory:
    """Test load_demonstrations_from_directory function"""
    
    def test_load_from_nonexistent_directory(self):
        """Test loading from nonexistent directory raises LoadError"""
        with pytest.raises(LoadError, match="Directory not found"):
            load_demonstrations_from_directory("nonexistent_directory")
    
    def test_load_from_file_instead_of_directory(self, tmp_path):
        """Test loading from file instead of directory raises LoadError"""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test")
        
        with pytest.raises(LoadError, match="Path is not a directory"):
            load_demonstrations_from_directory(str(test_file))
    
    def test_load_from_empty_directory(self, tmp_path):
        """Test loading from directory with no pickle files raises LoadError"""
        with pytest.raises(LoadError, match="No pickle files found"):
            load_demonstrations_from_directory(str(tmp_path))
    
    @patch('src.causal_bayes_opt.training.pure_data_loader.load_demonstration_batch')
    @patch('src.causal_bayes_opt.training.pure_data_loader.process_demonstration')
    def test_load_successful(self, mock_process, mock_load_batch, tmp_path):
        """Test successful loading from directory"""
        # Create test pickle files
        test_file = tmp_path / "test.pkl"
        test_file.write_bytes(b"dummy")
        
        # Setup mocks
        mock_demo = Mock(spec=ExpertDemonstration)
        mock_demo.accuracy = 0.8
        mock_demo.graph_type = "erdos_renyi"
        mock_demo.n_nodes = 5
        mock_batch = DemonstrationBatch(
            demonstrations=[mock_demo],
            batch_id="test_batch",
            collection_config={}
        )
        mock_load_batch.return_value = mock_batch
        
        mock_demo_data = Mock(spec=DemonstrationData)
        mock_process.return_value = mock_demo_data
        
        # Load
        result = load_demonstrations_from_directory(str(tmp_path))
        
        # Verify
        assert len(result) == 1
        assert result[0] == mock_demo_data
        mock_load_batch.assert_called_once()
        mock_process.assert_called_once()


class TestValidateDemonstrationData:
    """Test validate_demonstration_data function"""
    
    def create_mock_demo_data(self, **kwargs):
        """Create mock demonstration data with defaults"""
        defaults = {
            'demo_id': 'test_demo',
            'n_nodes': 5,
            'target_variable': 'X0',
            'avici_data': AVICIData(
                samples=jnp.zeros((10, 3, 3)),
                variables=('X0', 'X1', 'X2'),
                target_variable='X0',
                sample_count=10
            ),
            'posterior_history': pyr.pvector([
                PosteriorStep(0, {frozenset(): 0.5}, 0.693)
            ]),
            'complexity_score': 5.0
        }
        defaults.update(kwargs)
        
        return Mock(spec=DemonstrationData, **defaults)
    
    def test_validate_valid_data(self):
        """Test validation passes for valid data"""
        demo_data = self.create_mock_demo_data()
        
        # Should not raise exception
        validate_demonstration_data(demo_data)
    
    def test_validate_missing_demo_id(self):
        """Test validation fails for missing demo ID"""
        demo_data = self.create_mock_demo_data(demo_id="")
        
        with pytest.raises(ValidationError, match="Missing demo ID"):
            validate_demonstration_data(demo_data)
    
    def test_validate_invalid_node_count(self):
        """Test validation fails for invalid node count"""
        demo_data = self.create_mock_demo_data(n_nodes=0)
        
        with pytest.raises(ValidationError, match="Invalid node count"):
            validate_demonstration_data(demo_data)
    
    def test_validate_missing_target_variable(self):
        """Test validation fails for missing target variable"""
        demo_data = self.create_mock_demo_data(target_variable="")
        
        with pytest.raises(ValidationError, match="Missing target variable"):
            validate_demonstration_data(demo_data)
    
    def test_validate_no_samples(self):
        """Test validation fails for no samples"""
        avici_data = AVICIData(
            samples=jnp.zeros((0, 3, 3)),  # No samples
            variables=('X0', 'X1', 'X2'),
            target_variable='X0',
            sample_count=0
        )
        demo_data = self.create_mock_demo_data(avici_data=avici_data)
        
        with pytest.raises(ValidationError, match="No samples in AVICI data"):
            validate_demonstration_data(demo_data)
    
    def test_validate_no_posterior_history(self):
        """Test validation fails for no posterior history"""
        demo_data = self.create_mock_demo_data(posterior_history=pyr.pvector([]))
        
        with pytest.raises(ValidationError, match="No posterior history"):
            validate_demonstration_data(demo_data)
    
    def test_validate_invalid_complexity_score(self):
        """Test validation fails for invalid complexity score"""
        demo_data = self.create_mock_demo_data(complexity_score=-1.0)
        
        with pytest.raises(ValidationError, match="Invalid complexity score"):
            validate_demonstration_data(demo_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])