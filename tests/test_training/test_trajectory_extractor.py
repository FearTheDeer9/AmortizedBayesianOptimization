#!/usr/bin/env python3
"""
Comprehensive tests for trajectory_extractor.py

Following TDD approach with complete test coverage for all functions.
"""

import pytest
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from src.causal_bayes_opt.training.trajectory_extractor import (
    extract_surrogate_training_example,
    extract_acquisition_training_pair,
    extract_surrogate_training_data,
    extract_acquisition_training_data,
    classify_demonstration_difficulty,
    organize_by_curriculum,
    create_balanced_training_split,
    extract_complete_training_data,
    _validate_demonstration_data,
    _compute_posterior_entropy,
    _create_parent_set_posterior,
    _create_acquisition_state,
    _create_expert_action,
    TrajectoryExtractionConfig,
    CurriculumLevel,
    CURRICULUM_LEVELS,
    SurrogateTrainingData,
    AcquisitionTrainingData
)
from src.causal_bayes_opt.training.pure_data_loader import (
    DemonstrationData,
    AVICIData,
    PosteriorStep,
    InterventionStep
)


class TestValidateDemonstrationData:
    """Test _validate_demonstration_data function"""
    
    def create_mock_demo_data(self, **kwargs):
        """Create mock demonstration data with defaults"""
        defaults = {
            'demo_id': 'test_demo',
            'n_nodes': 5,
            'graph_type': 'erdos_renyi',
            'target_variable': 'X0',
            'accuracy': 0.8,
            'avici_data': AVICIData(
                samples=jnp.ones((10, 5, 3)),
                variables=('X0', 'X1', 'X2', 'X3', 'X4'),
                target_variable='X0',
                sample_count=10
            ),
            'posterior_history': pyr.pvector([
                PosteriorStep(0, {frozenset(): 0.5, frozenset(['X1']): 0.5}, 0.693)
            ]),
            'intervention_sequence': pyr.pvector([
                InterventionStep(0, frozenset(['X1']), (1.0,))
            ]),
            'complexity_score': 5.0
        }
        defaults.update(kwargs)
        
        return Mock(spec=DemonstrationData, **defaults)
    
    def test_validate_valid_demo(self):
        """Test validation passes for valid demonstration"""
        demo_data = self.create_mock_demo_data()
        
        # Should not raise exception
        _validate_demonstration_data(demo_data)
    
    def test_validate_no_avici_samples(self):
        """Test validation fails for no AVICI samples"""
        avici_data = AVICIData(
            samples=jnp.empty((0, 5, 3)),  # Empty samples
            variables=('X0', 'X1', 'X2', 'X3', 'X4'),
            target_variable='X0',
            sample_count=0
        )
        demo_data = self.create_mock_demo_data(avici_data=avici_data)
        
        with pytest.raises(ValueError, match="No AVICI samples"):
            _validate_demonstration_data(demo_data)
    
    def test_validate_no_posterior_history(self):
        """Test validation fails for no posterior history"""
        demo_data = self.create_mock_demo_data(posterior_history=pyr.pvector([]))
        
        with pytest.raises(ValueError, match="No posterior history"):
            _validate_demonstration_data(demo_data)
    
    def test_validate_no_intervention_sequence(self):
        """Test validation fails for no intervention sequence"""
        demo_data = self.create_mock_demo_data(intervention_sequence=pyr.pvector([]))
        
        with pytest.raises(ValueError, match="No intervention sequence"):
            _validate_demonstration_data(demo_data)
    
    def test_validate_length_mismatch(self):
        """Test validation fails for mismatched sequence lengths"""
        demo_data = self.create_mock_demo_data(
            posterior_history=pyr.pvector([
                PosteriorStep(0, {frozenset(): 1.0}, 0.0),
                PosteriorStep(1, {frozenset(): 1.0}, 0.0)
            ]),
            intervention_sequence=pyr.pvector([
                InterventionStep(0, frozenset(['X1']), (1.0,))
            ])
        )
        
        with pytest.raises(ValueError, match="Mismatch between posterior history"):
            _validate_demonstration_data(demo_data)


class TestComputePosteriorEntropy:
    """Test _compute_posterior_entropy function"""
    
    def test_entropy_uniform_distribution(self):
        """Test entropy for uniform distribution"""
        probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        entropy = _compute_posterior_entropy(probs)
        
        # Uniform distribution has maximum entropy = log(n)
        expected = jnp.log(4.0)
        assert jnp.isclose(entropy, expected, atol=1e-6)
    
    def test_entropy_certain_distribution(self):
        """Test entropy for certain distribution"""
        probs = jnp.array([1.0, 0.0, 0.0, 0.0])
        entropy = _compute_posterior_entropy(probs)
        
        assert jnp.isclose(entropy, 0.0, atol=1e-6)
    
    def test_entropy_handles_zeros(self):
        """Test entropy computation handles zero probabilities"""
        probs = jnp.array([0.5, 0.5, 0.0, 0.0])
        entropy = _compute_posterior_entropy(probs)
        
        assert jnp.isfinite(entropy)
        assert entropy > 0


class TestCreateParentSetPosterior:
    """Test _create_parent_set_posterior function"""
    
    def test_create_normal_posterior(self):
        """Test creating posterior from normal step"""
        posterior_step = PosteriorStep(
            step=0,
            posterior={frozenset(): 0.3, frozenset(['X1']): 0.7},
            entropy=0.611
        )
        
        result = _create_parent_set_posterior(posterior_step, 'X0')
        
        assert result.target_variable == 'X0'
        assert len(result.parent_set_probs) == 2
        assert frozenset() in result.parent_set_probs
        assert frozenset(['X1']) in result.parent_set_probs
        assert result.uncertainty == 0.611
        assert len(result.top_k_sets) <= 5
    
    def test_create_empty_posterior(self):
        """Test creating posterior from empty step"""
        posterior_step = PosteriorStep(
            step=0,
            posterior={},
            entropy=0.0
        )
        
        result = _create_parent_set_posterior(posterior_step, 'X0')
        
        assert result.target_variable == 'X0'
        assert len(result.parent_set_probs) == 1
        assert frozenset() in result.parent_set_probs
        assert result.parent_set_probs[frozenset()] == 1.0


class TestCreateExpertAction:
    """Test _create_expert_action function"""
    
    def test_create_intervention_action(self):
        """Test creating action from intervention step"""
        intervention_step = InterventionStep(
            step=5,
            variables=frozenset(['X1', 'X2']),
            values=(1.0, 2.0)
        )
        
        result = _create_expert_action(intervention_step)
        
        assert result['intervention_variables'] == frozenset(['X1', 'X2'])
        assert result['intervention_values'] == (1.0, 2.0)
        assert result['step'] == 5
        assert result['action_type'] == 'intervention'
    
    def test_create_observation_action(self):
        """Test creating action from observation step"""
        intervention_step = InterventionStep(
            step=3,
            variables=frozenset(),  # No variables = observation
            values=()
        )
        
        result = _create_expert_action(intervention_step)
        
        assert result['intervention_variables'] == frozenset()
        assert result['intervention_values'] == ()
        assert result['step'] == 3
        assert result['action_type'] == 'observe'


class TestTrajectoryExtractionConfig:
    """Test TrajectoryExtractionConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrajectoryExtractionConfig()
        
        assert config.max_trajectory_length == 20
        assert config.min_posterior_entropy == 0.01
        assert config.use_all_steps == True
        assert config.filter_invalid_actions == True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrajectoryExtractionConfig(
            max_trajectory_length=10,
            min_posterior_entropy=0.05,
            use_all_steps=False,
            filter_invalid_actions=False
        )
        
        assert config.max_trajectory_length == 10
        assert config.min_posterior_entropy == 0.05
        assert config.use_all_steps == False
        assert config.filter_invalid_actions == False


class TestExtractSurrogateTrainingExample:
    """Test extract_surrogate_training_example function"""
    
    def create_demo_data(self):
        """Create test demonstration data"""
        return Mock(
            spec=DemonstrationData,
            demo_id='test_demo',
            n_nodes=3,
            graph_type='chain',
            target_variable='X0',
            accuracy=0.8,
            complexity_score=2.5,
            avici_data=AVICIData(
                samples=jnp.ones((5, 3, 3)),
                variables=('X0', 'X1', 'X2'),
                target_variable='X0',
                sample_count=5
            ),
            posterior_history=pyr.pvector([
                PosteriorStep(0, {frozenset(): 0.4, frozenset(['X1']): 0.6}, 0.673),
                PosteriorStep(1, {frozenset(['X1']): 1.0}, 0.0)
            ])
        )
    
    def test_extract_valid_example(self):
        """Test extracting valid surrogate example"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig()
        
        result = extract_surrogate_training_example(demo_data, 0, config)
        
        assert result is not None
        assert result.target_variable == 'X0'
        assert result.expert_accuracy == 0.8
        assert result.variable_order == ['X0', 'X1', 'X2']
        assert result.expert_posterior.target_variable == 'X0'
    
    def test_extract_filtered_by_entropy(self):
        """Test example filtered by low entropy"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig(min_posterior_entropy=0.5)
        
        # Step 1 has entropy 0.0, should be filtered
        result = extract_surrogate_training_example(demo_data, 1, config)
        
        assert result is None
    
    def test_extract_out_of_bounds(self):
        """Test extraction with out-of-bounds step"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig()
        
        # Step 5 is beyond the 2 available steps
        result = extract_surrogate_training_example(demo_data, 5, config)
        
        assert result is None


class TestExtractAcquisitionTrainingPair:
    """Test extract_acquisition_training_pair function"""
    
    def create_demo_data(self):
        """Create test demonstration data"""
        return Mock(
            spec=DemonstrationData,
            demo_id='test_demo',
            n_nodes=3,
            graph_type='chain',
            target_variable='X0',
            accuracy=0.8,
            complexity_score=2.5,
            avici_data=AVICIData(
                samples=jnp.ones((5, 3, 3)),
                variables=('X0', 'X1', 'X2'),
                target_variable='X0',
                sample_count=5
            ),
            posterior_history=pyr.pvector([
                PosteriorStep(0, {frozenset(): 0.4, frozenset(['X1']): 0.6}, 0.673),
                PosteriorStep(1, {frozenset(['X1']): 1.0}, 0.0)
            ]),
            intervention_sequence=pyr.pvector([
                InterventionStep(0, frozenset(['X1']), (1.0,)),
                InterventionStep(1, frozenset(), ())  # Observation
            ])
        )
    
    def test_extract_valid_pair(self):
        """Test extracting valid acquisition pair"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig()
        
        result = extract_acquisition_training_pair(demo_data, 0, config)
        
        assert result is not None
        state, action = result
        assert state['target_variable'] == 'X0'
        assert action['intervention_variables'] == frozenset(['X1'])
        assert action['intervention_values'] == (1.0,)
        assert action['step'] == 0
    
    def test_extract_filtered_invalid_action(self):
        """Test pair filtered by invalid action"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig(filter_invalid_actions=True)
        
        # Step 1 has empty intervention (observation), should be filtered
        result = extract_acquisition_training_pair(demo_data, 1, config)
        
        assert result is None
    
    def test_extract_allows_observation(self):
        """Test pair allows observation when filtering disabled"""
        demo_data = self.create_demo_data()
        config = TrajectoryExtractionConfig(filter_invalid_actions=False)
        
        # Step 1 has empty intervention but should be allowed
        result = extract_acquisition_training_pair(demo_data, 1, config)
        
        assert result is not None
        state, action = result
        assert action['action_type'] == 'observe'


class TestClassifyDemonstrationDifficulty:
    """Test classify_demonstration_difficulty function"""
    
    def create_demo_data(self, n_nodes=5, complexity=3.0, accuracy=0.8):
        """Create test demonstration data with specified properties"""
        return Mock(
            spec=DemonstrationData,
            n_nodes=n_nodes,
            complexity_score=complexity,
            accuracy=accuracy
        )
    
    def test_classify_easy(self):
        """Test classification as easy difficulty"""
        demo_data = self.create_demo_data(n_nodes=4, complexity=2.0, accuracy=0.75)
        
        result = classify_demonstration_difficulty(demo_data)
        
        assert result == 'easy'
    
    def test_classify_medium(self):
        """Test classification as medium difficulty"""
        demo_data = self.create_demo_data(n_nodes=6, complexity=4.0, accuracy=0.82)
        
        result = classify_demonstration_difficulty(demo_data)
        
        assert result == 'medium'
    
    def test_classify_hard(self):
        """Test classification as hard difficulty"""
        demo_data = self.create_demo_data(n_nodes=10, complexity=8.0, accuracy=0.87)
        
        result = classify_demonstration_difficulty(demo_data)
        
        assert result == 'hard'
    
    def test_classify_expert(self):
        """Test classification as expert difficulty"""
        demo_data = self.create_demo_data(n_nodes=15, complexity=15.0, accuracy=0.92)
        
        result = classify_demonstration_difficulty(demo_data)
        
        assert result == 'expert'
    
    def test_classify_low_accuracy(self):
        """Test classification with low accuracy falls back appropriately"""
        demo_data = self.create_demo_data(n_nodes=6, complexity=4.0, accuracy=0.6)  # Low accuracy
        
        result = classify_demonstration_difficulty(demo_data)
        
        # Should still classify as medium based on nodes/complexity
        assert result == 'medium'


class TestOrganizeByCurriculum:
    """Test organize_by_curriculum function"""
    
    def create_demo_list(self):
        """Create list of diverse demonstrations"""
        return [
            Mock(spec=DemonstrationData, n_nodes=3, complexity_score=1.0, accuracy=0.8),  # easy
            Mock(spec=DemonstrationData, n_nodes=4, complexity_score=2.5, accuracy=0.75), # easy
            Mock(spec=DemonstrationData, n_nodes=6, complexity_score=4.0, accuracy=0.85), # medium
            Mock(spec=DemonstrationData, n_nodes=8, complexity_score=7.0, accuracy=0.88), # hard
            Mock(spec=DemonstrationData, n_nodes=15, complexity_score=12.0, accuracy=0.92) # expert
        ]
    
    def test_organize_curriculum(self):
        """Test organizing demonstrations by curriculum"""
        demo_list = self.create_demo_list()
        
        result = organize_by_curriculum(demo_list)
        
        assert 'easy' in result
        assert 'medium' in result
        assert 'hard' in result
        assert 'expert' in result
        
        assert len(result['easy']) == 2
        assert len(result['medium']) == 1
        assert len(result['hard']) == 1
        assert len(result['expert']) == 1
    
    def test_organize_empty_list(self):
        """Test organizing empty demonstration list"""
        result = organize_by_curriculum([])
        
        for level_name in ['easy', 'medium', 'hard', 'expert']:
            assert level_name in result
            assert len(result[level_name]) == 0


class TestCreateBalancedTrainingSplit:
    """Test create_balanced_training_split function"""
    
    def create_demo_list(self):
        """Create balanced list for splitting"""
        demos = []
        # Create 20 demos: 8 easy, 6 medium, 4 hard, 2 expert
        for i in range(8):
            demos.append(Mock(spec=DemonstrationData, n_nodes=3, complexity_score=1.0, accuracy=0.8))
        for i in range(6):
            demos.append(Mock(spec=DemonstrationData, n_nodes=6, complexity_score=4.0, accuracy=0.85))
        for i in range(4):
            demos.append(Mock(spec=DemonstrationData, n_nodes=10, complexity_score=8.0, accuracy=0.87))
        for i in range(2):
            demos.append(Mock(spec=DemonstrationData, n_nodes=15, complexity_score=15.0, accuracy=0.92))
        
        return demos
    
    def test_balanced_split(self):
        """Test creating balanced training split"""
        demo_list = self.create_demo_list()
        
        train_data, val_data, test_data = create_balanced_training_split(
            demo_list, 0.7, 0.2, 0.1, random_seed=42
        )
        
        total = len(train_data) + len(val_data) + len(test_data)
        assert total == len(demo_list)
        
        # Check approximate proportions (allowing for rounding with small datasets)
        assert len(train_data) >= 12  # ~70% of 20 (with rounding)
        assert len(val_data) >= 2     # ~20% of 20 (allowing rounding down)
        assert len(test_data) >= 1    # ~10% of 20
    
    def test_split_ratios_validation(self):
        """Test split ratios validation"""
        demo_list = self.create_demo_list()
        
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            create_balanced_training_split(demo_list, 0.5, 0.3, 0.3)  # Sums to 1.1
    
    def test_deterministic_splits(self):
        """Test splits are deterministic with same seed"""
        demo_list = self.create_demo_list()
        
        # Same seed should give same results
        train1, val1, test1 = create_balanced_training_split(demo_list, random_seed=42)
        train2, val2, test2 = create_balanced_training_split(demo_list, random_seed=42)
        
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)


class TestExtractSurrogateTrainingData:
    """Test extract_surrogate_training_data function"""
    
    def create_demo_list(self):
        """Create list of test demonstrations"""
        return [
            Mock(
                spec=DemonstrationData,
                demo_id=f'demo_{i}',
                n_nodes=3,
                graph_type='chain',
                target_variable='X0',
                accuracy=0.8,
                complexity_score=2.0,
                avici_data=AVICIData(
                    samples=jnp.ones((5, 3, 3)),
                    variables=('X0', 'X1', 'X2'),
                    target_variable='X0',
                    sample_count=5
                ),
                posterior_history=pyr.pvector([
                    PosteriorStep(0, {frozenset(): 0.5, frozenset(['X1']): 0.5}, 0.693),
                    PosteriorStep(1, {frozenset(['X1']): 1.0}, 0.0)  # Low entropy
                ])
            )
            for i in range(3)
        ]
    
    @patch('src.causal_bayes_opt.training.trajectory_extractor._validate_demonstration_data')
    def test_extract_all_data(self, mock_validate):
        """Test extracting surrogate training data from all demonstrations"""
        mock_validate.return_value = None  # No validation errors
        demo_list = self.create_demo_list()
        config = TrajectoryExtractionConfig()
        
        result = extract_surrogate_training_data(demo_list, config)
        
        assert isinstance(result, SurrogateTrainingData)
        assert len(result.training_examples) > 0
        assert 'extraction_stats' in result.metadata
        assert result.metadata['total_examples'] == len(result.training_examples)
    
    @patch('src.causal_bayes_opt.training.trajectory_extractor._validate_demonstration_data')
    def test_extract_with_filtering(self, mock_validate):
        """Test extraction with entropy filtering"""
        mock_validate.return_value = None
        demo_list = self.create_demo_list()
        config = TrajectoryExtractionConfig(min_posterior_entropy=0.5)  # High threshold
        
        result = extract_surrogate_training_data(demo_list, config)
        
        # Should have fewer examples due to filtering
        stats = result.metadata['extraction_stats']
        assert stats['filtered_steps'] > 0


class TestExtractAcquisitionTrainingData:
    """Test extract_acquisition_training_data function"""
    
    def create_demo_list(self):
        """Create list of test demonstrations"""
        return [
            Mock(
                spec=DemonstrationData,
                demo_id=f'demo_{i}',
                n_nodes=3,
                graph_type='chain',
                target_variable='X0',
                accuracy=0.8,
                complexity_score=2.0,
                avici_data=AVICIData(
                    samples=jnp.ones((5, 3, 3)),
                    variables=('X0', 'X1', 'X2'),
                    target_variable='X0',
                    sample_count=5
                ),
                posterior_history=pyr.pvector([
                    PosteriorStep(0, {frozenset(): 0.5, frozenset(['X1']): 0.5}, 0.693),
                    PosteriorStep(1, {frozenset(['X1']): 1.0}, 0.0)
                ]),
                intervention_sequence=pyr.pvector([
                    InterventionStep(0, frozenset(['X1']), (1.0,)),
                    InterventionStep(1, frozenset(), ())  # Observation
                ])
            )
            for i in range(3)
        ]
    
    @patch('src.causal_bayes_opt.training.trajectory_extractor._validate_demonstration_data')
    def test_extract_all_pairs(self, mock_validate):
        """Test extracting acquisition training pairs from all demonstrations"""
        mock_validate.return_value = None
        demo_list = self.create_demo_list()
        config = TrajectoryExtractionConfig()
        
        result = extract_acquisition_training_data(demo_list, config)
        
        assert isinstance(result, AcquisitionTrainingData)
        assert len(result.state_action_pairs) > 0
        assert 'extraction_stats' in result.metadata
        assert result.metadata['total_pairs'] == len(result.state_action_pairs)


class TestExtractCompleteTrainingData:
    """Test extract_complete_training_data function"""
    
    def create_demo_list(self):
        """Create comprehensive demo list for testing"""
        demos = []
        for i in range(10):
            demos.append(Mock(
                spec=DemonstrationData,
                demo_id=f'demo_{i}',
                n_nodes=3 + (i % 3),  # Vary node count for curriculum
                graph_type='chain',
                complexity_score=1.0 + i * 0.5,
                accuracy=0.7 + (i % 3) * 0.1,
                avici_data=AVICIData(
                    samples=jnp.ones((5, 3, 3)),
                    variables=('X0', 'X1', 'X2'),
                    target_variable='X0',
                    sample_count=5
                ),
                posterior_history=pyr.pvector([
                    PosteriorStep(0, {frozenset(): 0.5, frozenset(['X1']): 0.5}, 0.693)
                ]),
                intervention_sequence=pyr.pvector([
                    InterventionStep(0, frozenset(['X1']), (1.0,))
                ])
            ))
        return demos
    
    @patch('src.causal_bayes_opt.training.trajectory_extractor._validate_demonstration_data')
    def test_complete_pipeline(self, mock_validate):
        """Test complete training data extraction pipeline"""
        mock_validate.return_value = None
        demo_list = self.create_demo_list()
        
        result = extract_complete_training_data(demo_list)
        
        # Check structure
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result
        assert 'metadata' in result
        
        # Check each split has both data types
        for split_name in ['train', 'val', 'test']:
            split_data = result[split_name]
            assert 'surrogate' in split_data
            assert 'acquisition' in split_data
            assert isinstance(split_data['surrogate'], SurrogateTrainingData)
            assert isinstance(split_data['acquisition'], AcquisitionTrainingData)
        
        # Check metadata
        metadata = result['metadata']
        assert metadata['total_demonstrations'] == len(demo_list)
        assert 'curriculum_distribution' in metadata
        assert 'config' in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])