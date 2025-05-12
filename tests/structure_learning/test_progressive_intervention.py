"""
Tests for progressive intervention loop functionality.

This module contains tests for the progressive intervention loop, which
iteratively selects interventions to improve causal graph structure learning.
"""

import pytest
import numpy as np
import torch
from torch.optim import Adam

from causal_meta.structure_learning import (
    RandomDAGGenerator,
    LinearSCMGenerator,
    SimpleGraphLearner,
    SimpleGraphLearnerTrainer,
    generate_observational_data,
    generate_interventional_data,
    create_intervention_mask,
    convert_to_tensor,
    normalize_data
)
from causal_meta.structure_learning.progressive_intervention import (
    ProgressiveInterventionConfig,
    ProgressiveInterventionLoop,
    GraphStructureAcquisition
)
from causal_meta.structure_learning.training import evaluate_graph


class TestProgressiveInterventionConfig:
    """Tests for the progressive intervention configuration."""
    
    def test_config_validation(self):
        """Test that configuration validation works correctly."""
        # Valid configuration
        valid_config = ProgressiveInterventionConfig(
            num_nodes=5,
            num_iterations=5,
            num_obs_samples=100,
            num_int_samples=20,
            acquisition_strategy="uncertainty",
            int_budget=1.0
        )
        assert valid_config.validate() is None
        
        # Invalid configuration (negative iterations)
        invalid_config = ProgressiveInterventionConfig(
            num_nodes=5,
            num_iterations=-1,
            num_obs_samples=100,
            num_int_samples=20,
            acquisition_strategy="uncertainty",
            int_budget=1.0
        )
        assert invalid_config.validate() is not None  # Should return an error message, not raise an exception
        
        # Invalid acquisition strategy
        invalid_config = ProgressiveInterventionConfig(
            num_nodes=5,
            num_iterations=5,
            num_obs_samples=100,
            num_int_samples=20,
            acquisition_strategy="invalid_strategy",
            int_budget=1.0
        )
        assert invalid_config.validate() is not None  # Should return an error message, not raise an exception


class TestGraphStructureAcquisition:
    """Tests for the graph structure acquisition strategy."""
    
    def test_acquisition_initialization(self):
        """Test that the acquisition strategy initializes correctly."""
        # Create acquisition strategy
        acquisition = GraphStructureAcquisition(strategy_type="uncertainty")
        assert acquisition.strategy_type == "uncertainty"
        
        # Test with invalid strategy type
        with pytest.raises(ValueError):
            GraphStructureAcquisition(strategy_type="invalid_strategy")
    
    def test_intervention_selection(self):
        """Test that intervention selection works correctly."""
        # Create a simple model and graph
        model = SimpleGraphLearner(input_dim=3, hidden_dim=32)
        acquisition = GraphStructureAcquisition(strategy_type="uncertainty")
        
        # Generate a random adjacency matrix
        adj_matrix = RandomDAGGenerator.generate_random_dag(
            num_nodes=3, 
            edge_probability=0.3,
            as_adjacency_matrix=True
        )
        
        # Generate fake data
        data = torch.randn(10, 3)
        
        # Select an intervention
        intervention = acquisition.select_intervention(
            model=model,
            data=data,
            budget=1.0
        )
        
        # Check that intervention has the expected format
        assert 'target_node' in intervention
        assert 'value' in intervention
        assert 0 <= intervention['target_node'] < 3
        
        # Ensure we can select a batch of interventions
        batch = acquisition.select_batch(
            model=model,
            data=data,
            budget=1.0,
            batch_size=2
        )
        
        assert len(batch) == 2
        assert all('target_node' in item for item in batch)
        assert all('value' in item for item in batch)


class TestProgressiveInterventionLoop:
    """Tests for the progressive intervention loop."""
    
    @pytest.fixture
    def setup_experiment(self):
        """Set up a basic experiment for testing."""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate a random DAG
        n_nodes = 3
        edge_prob = 0.3
        adj_matrix = RandomDAGGenerator.generate_random_dag(
            num_nodes=n_nodes,
            edge_probability=edge_prob,
            as_adjacency_matrix=True,
            seed=42
        )
        
        # Create a linear SCM
        scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=adj_matrix,
            noise_scale=0.1,
            seed=42
        )
        
        # Generate observational data
        n_obs_samples = 50
        obs_data = generate_observational_data(
            scm=scm,
            n_samples=n_obs_samples,
            as_tensor=False
        )
        
        # Create configuration
        config = ProgressiveInterventionConfig(
            num_nodes=n_nodes,
            num_iterations=2,
            num_obs_samples=n_obs_samples,
            num_int_samples=10,
            acquisition_strategy="uncertainty",
            int_budget=1.0
        )
        
        return {
            'config': config,
            'scm': scm,
            'adj_matrix': adj_matrix,
            'obs_data': obs_data
        }
    
    def test_loop_initialization(self, setup_experiment):
        """Test that the loop initializes correctly."""
        setup = setup_experiment
        
        # Initialize the loop
        loop = ProgressiveInterventionLoop(
            config=setup['config'],
            scm=setup['scm'],
            obs_data=setup['obs_data'],
            true_adj_matrix=setup['adj_matrix']
        )
        
        assert loop.iteration == 0
        assert loop.config == setup['config']
        assert loop.obs_data.shape == setup['obs_data'].shape
        
    def test_single_iteration(self, setup_experiment):
        """Test a single iteration of the loop."""
        setup = setup_experiment
        
        # Initialize the loop
        loop = ProgressiveInterventionLoop(
            config=setup['config'],
            scm=setup['scm'],
            obs_data=setup['obs_data'],
            true_adj_matrix=setup['adj_matrix']
        )
        
        # Run a single iteration
        result = loop.run_iteration()
        
        # Check that we have the expected data
        assert loop.iteration == 1
        assert 'model' in result
        assert 'metrics' in result
        assert isinstance(result['model'], SimpleGraphLearner)
        assert isinstance(result['metrics'], dict)
        assert 'accuracy' in result['metrics']
        assert 'shd' in result['metrics']
        
    def test_full_experiment(self, setup_experiment):
        """Test running the full experiment."""
        setup = setup_experiment
        
        # Initialize the loop
        loop = ProgressiveInterventionLoop(
            config=setup['config'],
            scm=setup['scm'],
            obs_data=setup['obs_data'],
            true_adj_matrix=setup['adj_matrix']
        )
        
        # Run the experiment
        results = loop.run_experiment()
        
        # Check results
        assert len(results) == setup['config'].num_iterations + 1  # +1 for initial
        assert all('model' in result for result in results)
        assert all('metrics' in result for result in results)
        assert all('accuracy' in result['metrics'] for result in results)
        assert all('shd' in result['metrics'] for result in results)
        
        # Ensure metrics are tracked over iterations
        assert all('iteration' in result for result in results)
        
        # Check that we have interventions recorded
        assert all('intervention' in result for result in results[1:])
        
    def test_comparison_with_random_interventions(self, setup_experiment):
        """Test comparing with random intervention baseline."""
        setup = setup_experiment
        
        # Initialize the loop with strategic interventions
        strategic_loop = ProgressiveInterventionLoop(
            config=setup['config'],
            scm=setup['scm'],
            obs_data=setup['obs_data'],
            true_adj_matrix=setup['adj_matrix']
        )
        
        # Initialize the loop with random interventions
        random_config = setup['config'].copy()
        random_config.acquisition_strategy = "random"
        random_loop = ProgressiveInterventionLoop(
            config=random_config,
            scm=setup['scm'],
            obs_data=setup['obs_data'],
            true_adj_matrix=setup['adj_matrix']
        )
        
        # Run both experiments
        strategic_results = strategic_loop.run_experiment()
        random_results = random_loop.run_experiment()
        
        # Ensure both have completed the same number of iterations
        assert len(strategic_results) == len(random_results) 