"""
Tests for the AmortizedCBO class.

This module contains tests for the Amortized Causal Bayesian Optimization (AmortizedCBO)
class, which implements efficient intervention selection for causal discovery using
neural network-based surrogate models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from causal_meta.meta_learning.amortized_cbo import AmortizedCBO
from causal_meta.meta_learning.meta_learning import TaskEmbedding
from causal_meta.graph.causal_graph import CausalGraph


@pytest.fixture
def simple_model():
    """Create a simple mock model for testing."""
    mock_model = MagicMock()
    
    # Set up predict_intervention_outcomes method to return mock predictions and uncertainty
    def mock_predict(x, intervention_targets=None, intervention_values=None, return_uncertainty=False, **kwargs):
        batch_size = x.size(0)
        n_variables = x.size(2)
        
        # Create some mock predictions
        predictions = torch.randn(batch_size, n_variables)
        
        # If return_uncertainty is True, also return mock uncertainty
        if return_uncertainty:
            uncertainty = torch.abs(torch.randn(batch_size, n_variables))
            return predictions, uncertainty
        else:
            return predictions
    
    mock_model.predict_intervention_outcomes.side_effect = mock_predict
    
    # Set up meta-learning methods
    mock_model.meta_adapt.return_value = mock_model
    
    # Mock parameters method to return an iterator
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    
    # Mock train_epoch method
    mock_model.train_epoch.return_value = {"loss": 0.1}
    
    return mock_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a simple dataset with 2 variables and 10 samples
    batch_size = 10
    n_variables = 3
    seq_length = 5
    
    # Time series data: [batch_size, seq_length, n_variables]
    x = torch.randn(batch_size, seq_length, n_variables)
    
    # True values after intervention: [batch_size, n_variables]
    y = torch.randn(batch_size, n_variables)
    
    # Edge index for the graph: [2, num_edges]
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    # Node features: [batch_size * n_variables, input_dim]
    node_features = torch.randn(batch_size * n_variables, 3)
    
    # Batch assignment for nodes: [batch_size * n_variables]
    batch = torch.repeat_interleave(torch.arange(batch_size), n_variables)
    
    # Adjacency matrix: [n_variables, n_variables]
    adj_matrix = torch.zeros(n_variables, n_variables)
    adj_matrix[0, 1] = 0.8  # Strong edge from 0 to 1
    adj_matrix[1, 0] = 0.2  # Weak edge from 1 to 0
    adj_matrix[0, 2] = 0.6  # Medium edge from 0 to 2
    adj_matrix[2, 0] = 0.3  # Weak edge from 2 to 0
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'node_features': node_features,
        'batch': batch,
        'adj_matrix': adj_matrix,
        'n_variables': n_variables,
        'batch_size': batch_size
    }


@pytest.fixture
def mock_causal_graph():
    """Create a mock causal graph for testing."""
    # Create a simple causal graph X0 -> X1 -> X2
    graph = CausalGraph()
    graph.add_node('X0')
    graph.add_node('X1')
    graph.add_node('X2')
    graph.add_edge('X0', 'X1')
    graph.add_edge('X1', 'X2')
    
    return graph


@pytest.fixture
def mock_task_embedding():
    """Create a mock task embedding for testing."""
    mock_embedding = MagicMock(spec=TaskEmbedding)
    mock_embedding.encode_graph.return_value = torch.randn(32)  # Mock embedding vector
    mock_embedding.compute_similarity.return_value = torch.tensor(0.8)  # Mock similarity score
    
    return mock_embedding


class TestAcquisitionFunctions:
    """Tests for the acquisition functions in AmortizedCBO."""
    
    def test_expected_improvement(self, simple_model, sample_data):
        """Test the expected improvement acquisition function."""
        # Initialize AmortizedCBO with EI acquisition function
        cbo = AmortizedCBO(model=simple_model, acquisition_type='ei')
        
        # Create mock predictions and uncertainty
        mean_predictions = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])
        uncertainty = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
        
        # Set a current best value
        best_value = torch.tensor(0.6)
        
        # Calculate EI scores
        ei_scores = cbo._expected_improvement(mean_predictions, uncertainty, best_value)
        
        # EI should be higher for predictions with higher mean and/or higher uncertainty
        assert ei_scores.shape == mean_predictions.shape
        assert torch.all(ei_scores >= 0)  # EI should be non-negative
    
    def test_upper_confidence_bound(self, simple_model, sample_data):
        """Test the upper confidence bound acquisition function."""
        # Initialize AmortizedCBO with UCB acquisition function
        cbo = AmortizedCBO(model=simple_model, acquisition_type='ucb', exploration_weight=2.0)
        
        # Create mock predictions and uncertainty
        mean_predictions = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])
        uncertainty = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
        
        # Calculate UCB scores
        ucb_scores = cbo._upper_confidence_bound(mean_predictions, uncertainty)
        
        # UCB should be mean + exploration_weight * uncertainty
        expected_ucb = mean_predictions + 2.0 * uncertainty
        assert torch.allclose(ucb_scores, expected_ucb)
    
    def test_probability_of_improvement(self, simple_model, sample_data):
        """Test the probability of improvement acquisition function."""
        # Initialize AmortizedCBO with PI acquisition function
        cbo = AmortizedCBO(model=simple_model, acquisition_type='pi')
        
        # Create mock predictions and uncertainty
        mean_predictions = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])
        uncertainty = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
        
        # Set a current best value
        best_value = torch.tensor(0.6)
        
        # Calculate PI scores
        pi_scores = cbo._probability_of_improvement(mean_predictions, uncertainty, best_value)
        
        # PI should be between 0 and 1
        assert torch.all(pi_scores >= 0) and torch.all(pi_scores <= 1)
        assert pi_scores.shape == mean_predictions.shape
    
    def test_thompson_sampling(self, simple_model, sample_data):
        """Test the Thompson sampling acquisition function."""
        # Initialize AmortizedCBO with Thompson sampling acquisition function
        cbo = AmortizedCBO(model=simple_model, acquisition_type='thompson')
        
        # Create mock predictions and uncertainty
        mean_predictions = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])
        uncertainty = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Calculate Thompson samples
        thompson_samples = cbo._thompson_sampling(mean_predictions, uncertainty)
        
        # Thompson samples should be different from the mean predictions
        assert not torch.allclose(thompson_samples, mean_predictions)
        assert thompson_samples.shape == mean_predictions.shape
        
        # Verify that different calls give different samples
        torch.manual_seed(24)
        thompson_samples_2 = cbo._thompson_sampling(mean_predictions, uncertainty)
        assert not torch.allclose(thompson_samples, thompson_samples_2)


class TestInterventionSelection:
    """Tests for the intervention selection mechanism in AmortizedCBO."""
    
    def test_select_intervention(self, simple_model, sample_data):
        """Test that the intervention selection mechanism chooses the correct intervention."""
        # Initialize AmortizedCBO
        cbo = AmortizedCBO(model=simple_model, acquisition_type='ucb')
        
        # Mock the _evaluate_interventions method to return predetermined scores
        intervention_scores = torch.tensor([0.2, 0.5, 0.3])
        cbo._evaluate_interventions = MagicMock(return_value=intervention_scores)
        
        # Select the best intervention
        best_target, best_value = cbo.select_intervention(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch'],
            intervention_values=torch.tensor([0.5, 1.0, 1.5])
        )
        
        # The best intervention should be the one with the highest score (index 1)
        assert best_target == 1
        assert best_value == 1.0
    
    def test_evaluate_interventions(self, simple_model, sample_data):
        """Test that intervention evaluation correctly computes scores for each candidate."""
        # Initialize AmortizedCBO
        cbo = AmortizedCBO(model=simple_model, acquisition_type='ucb', exploration_weight=1.0)
        
        # Mock the acquisition function to return predetermined values
        mock_scores = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.1, 0.7]])
        cbo._compute_acquisition = MagicMock(return_value=mock_scores)
        
        # Evaluate interventions
        scores = cbo._evaluate_interventions(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch'],
            intervention_targets=torch.tensor([0, 1]),
            intervention_values=torch.tensor([0.5, 1.0])
        )
        
        # Check that the scores match the expected values
        assert scores.shape == torch.Size([2])
        assert torch.allclose(scores, torch.tensor([0.5, 0.7]))
    
    def test_intervention_budget_constraint(self, simple_model, sample_data):
        """Test that the intervention selection respects the budget constraint."""
        # Initialize AmortizedCBO with a budget constraint
        cbo = AmortizedCBO(
            model=simple_model, 
            acquisition_type='ucb',
            intervention_cost=torch.tensor([1.0, 2.0, 3.0]),
            budget=2.5
        )
        
        # Mock the _evaluate_interventions method to return predetermined scores
        intervention_scores = torch.tensor([0.2, 0.5, 0.8])
        cbo._evaluate_interventions = MagicMock(return_value=intervention_scores)
        
        # Select the best intervention
        best_target, best_value = cbo.select_intervention(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch'],
            intervention_values=torch.tensor([0.5, 1.0, 1.5])
        )
        
        # The best intervention should be index 1 (highest score that fits budget)
        # Index 2 has highest score but costs 3.0, which exceeds budget of 2.5
        assert best_target == 1
        assert best_value == 1.0


class TestModelUpdate:
    """Tests for the model update mechanism in AmortizedCBO."""
    
    def test_update_model(self, simple_model, sample_data):
        """Test that the model correctly updates after receiving new data."""
        # Initialize AmortizedCBO
        cbo = AmortizedCBO(model=simple_model)
        
        # Create mock new data
        intervention_target = 1
        intervention_value = 0.5
        observed_outcome = torch.randn(sample_data['batch_size'], sample_data['n_variables'])
        
        # Update the model
        cbo.update_model(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch'],
            intervention_target=intervention_target,
            intervention_value=intervention_value,
            observed_outcome=observed_outcome
        )
        
        # Verify that the model's train_epoch method was called
        simple_model.train_epoch.assert_called_once()
    
    def test_meta_update(self, simple_model, sample_data, mock_causal_graph, mock_task_embedding):
        """Test that the model correctly uses meta-learning for updates."""
        # Initialize AmortizedCBO with meta-learning
        cbo = AmortizedCBO(
            model=simple_model,
            use_meta_learning=True,
            task_embedding=mock_task_embedding
        )
        
        # Create mock new data
        intervention_target = 1
        intervention_value = 0.5
        observed_outcome = torch.randn(sample_data['batch_size'], sample_data['n_variables'])
        
        # Update the model with meta-learning
        cbo.update_model(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch'],
            intervention_target=intervention_target,
            intervention_value=intervention_value,
            observed_outcome=observed_outcome,
            causal_graph=mock_causal_graph
        )
        
        # Verify that the model's meta_adapt method was called
        simple_model.meta_adapt.assert_called_once()
        mock_task_embedding.encode_graph.assert_called_once_with(mock_causal_graph)


class TestOptimizationLoop:
    """Tests for the full optimization loop in AmortizedCBO."""
    
    def test_optimize_loop(self, simple_model, sample_data):
        """Test the full optimization loop."""
        # Initialize AmortizedCBO
        cbo = AmortizedCBO(
            model=simple_model,
            acquisition_type='ucb',
            max_iterations=3
        )
        
        # Mock the select_intervention and update_model methods
        cbo.select_intervention = MagicMock(return_value=(0, 0.5))
        cbo.update_model = MagicMock()
        
        # Mock the observed outcomes for each iteration
        mock_outcomes = [
            torch.randn(sample_data['batch_size'], sample_data['n_variables']),
            torch.randn(sample_data['batch_size'], sample_data['n_variables']),
            torch.randn(sample_data['batch_size'], sample_data['n_variables'])
        ]
        simple_model.predict_intervention_outcomes.side_effect = mock_outcomes
        
        # Run the optimization loop
        results = cbo.optimize(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch']
        )
        
        # Verify that select_intervention and update_model were called the right number of times
        assert cbo.select_intervention.call_count == 3
        assert cbo.update_model.call_count == 3
        
        # Check that results have the expected structure
        assert 'best_target' in results
        assert 'best_value' in results
        assert 'best_outcome' in results
        assert 'intervention_history' in results
        assert len(results['intervention_history']) == 3
    
    def test_early_stopping(self, simple_model, sample_data):
        """Test that optimization stops early when improvement threshold is reached."""
        # Initialize AmortizedCBO with early stopping
        cbo = AmortizedCBO(
            model=simple_model,
            acquisition_type='ucb',
            max_iterations=10,
            improvement_threshold=0.01
        )
        
        # Mock the select_intervention method
        cbo.select_intervention = MagicMock(side_effect=[
            (0, 0.5),  # First intervention
            (1, 1.0),  # Second intervention
            (2, 1.5)   # Third intervention
        ])
        
        # Mock the update_model method
        cbo.update_model = MagicMock()
        
        # Mock the observed outcomes for each iteration with decreasing improvement
        mock_outcomes = [
            torch.ones(sample_data['batch_size'], sample_data['n_variables']) * 0.8,  # Initial outcome
            torch.ones(sample_data['batch_size'], sample_data['n_variables']) * 0.85,  # Improvement of 0.05
            torch.ones(sample_data['batch_size'], sample_data['n_variables']) * 0.855  # Improvement of 0.005 (less than threshold)
        ]
        
        # Mock the predict_intervention_outcomes method to return these outcomes
        def mock_predict(x, intervention_targets=None, intervention_values=None, return_uncertainty=False, **kwargs):
            # Return the next outcome from the list
            if len(mock_outcomes) > 0:
                outcome = mock_outcomes.pop(0)
                if return_uncertainty:
                    uncertainty = torch.ones_like(outcome) * 0.1
                    return outcome, uncertainty
                else:
                    return outcome
            return torch.zeros_like(x[:, -1, :])
        
        simple_model.predict_intervention_outcomes.side_effect = mock_predict
        
        # Run the optimization loop
        results = cbo.optimize(
            sample_data['x'],
            node_features=sample_data['node_features'],
            edge_index=sample_data['edge_index'],
            batch=sample_data['batch']
        )
        
        # Verify that the optimization stopped after 3 iterations due to small improvement
        assert cbo.select_intervention.call_count == 3
        assert cbo.update_model.call_count == 3
        assert len(results['intervention_history']) == 3


class TestMetaLearningIntegration:
    """Tests for the meta-learning integration in AmortizedCBO."""
    
    def test_task_embedding_usage(self, simple_model, sample_data, mock_causal_graph, mock_task_embedding):
        """Test that task embeddings are properly used to adapt to new causal structures."""
        # Initialize AmortizedCBO with meta-learning
        cbo = AmortizedCBO(
            model=simple_model,
            use_meta_learning=True,
            task_embedding=mock_task_embedding
        )
        
        # Mock model adaptation
        adapted_model = MagicMock()
        simple_model.meta_adapt.return_value = adapted_model
        
        # Mock the evaluate_interventions method to avoid issues
        cbo._evaluate_interventions = MagicMock(return_value=torch.tensor([0.5, 0.6, 0.7]))
        
        # Test that the adapted model is used in the optimization loop
        with patch.object(cbo, 'update_model') as mock_update:
            cbo.optimize(
                sample_data['x'],
                node_features=sample_data['node_features'],
                edge_index=sample_data['edge_index'],
                batch=sample_data['batch'],
                causal_graph=mock_causal_graph
            )
            
            # Verify that meta_adapt was called with the task embedding
            simple_model.meta_adapt.assert_called_once()
            mock_task_embedding.encode_graph.assert_called_once_with(mock_causal_graph)
    
    def test_similar_task_adaptation(self, simple_model, sample_data, mock_causal_graph, mock_task_embedding):
        """Test adaptation to similar tasks using meta-learning."""
        # Initialize AmortizedCBO with meta-learning
        cbo = AmortizedCBO(
            model=simple_model,
            use_meta_learning=True,
            task_embedding=mock_task_embedding,
            adaptation_steps=5
        )
        
        # Create a similar task
        similar_graph = CausalGraph()
        similar_graph.add_node('X0')
        similar_graph.add_node('X1')
        similar_graph.add_node('X2')
        similar_graph.add_edge('X0', 'X1')
        similar_graph.add_edge('X1', 'X2')
        similar_graph.add_edge('X0', 'X2')  # Additional edge
        
        # Mock that the similar task has high similarity
        mock_task_embedding.compute_similarity.return_value = torch.tensor(0.9)
        
        # Mock model adaptation
        simple_model.meta_adapt.return_value = simple_model
        
        # Set previous task embedding
        cbo.previous_task_embedding = torch.randn(32)
        
        # Run adaptation
        adapted_model = cbo._adapt_to_task(mock_causal_graph)
        
        # Verify that meta_adapt was called with fewer steps due to high similarity
        args, kwargs = simple_model.meta_adapt.call_args
        assert 'num_steps' in kwargs
        assert kwargs['num_steps'] <= 5  # Default is 5, should be lower for similar tasks 