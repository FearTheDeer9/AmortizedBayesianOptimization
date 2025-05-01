import torch
import pytest
import numpy as np

from causal_meta.meta_learning.data_generation import (
    SyntheticDataGenerator, GraphDataset, GraphDataLoader
)
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.environments.scm import StructuralCausalModel


class TestSyntheticDataGeneration:
    @pytest.fixture
    def sample_scm(self):
        """Create a sample SCM for testing."""
        # Create a small DAG
        graph_factory = GraphFactory()
        graph = graph_factory.create_random_dag(num_nodes=5, edge_probability=0.3)
        
        # Define structural equations
        def linear_function(values, coefs, noise):
            return np.sum(values * coefs) + noise
        
        # Create structural equations and noise distributions
        structural_equations = {}
        noise_distributions = {}
        
        for node in graph.nodes():
            parents = list(graph.parents(node))
            num_parents = len(parents)
            
            # Create random coefficients for linear functions
            if num_parents > 0:
                coefs = np.random.uniform(-1, 1, size=num_parents)
                structural_equations[node] = lambda values, node=node, coefs=coefs, lin_func=linear_function: lin_func(values, coefs, np.random.normal(0, 0.1))
            else:
                # For root nodes, just use noise
                structural_equations[node] = lambda values, node=node: np.random.normal(0, 1)
            
            # Set noise distribution
            noise_distributions[node] = np.random.normal
        
        # Create SCM
        scm = StructuralCausalModel(graph, structural_equations, noise_distributions)
        
        return scm
    
    def test_synthetic_data_generator_init(self, sample_scm):
        """Test initialization of SyntheticDataGenerator."""
        generator = SyntheticDataGenerator(sample_scm)
        
        assert hasattr(generator, 'scm')
        assert hasattr(generator, 'graph')
        assert generator.scm == sample_scm
        assert generator.graph == sample_scm.get_causal_graph()
    
    def test_generate_observational_data(self, sample_scm):
        """Test generation of observational data."""
        generator = SyntheticDataGenerator(sample_scm)
        
        # Generate data
        n_samples = 100
        data = generator.generate_observational_data(n_samples=n_samples)
        
        # Check shape
        assert isinstance(data, torch.Tensor)
        assert data.shape == (n_samples, len(sample_scm.get_causal_graph().nodes()))
        
        # Check data properties
        assert not torch.isnan(data).any()
        assert not torch.isinf(data).any()
    
    def test_generate_interventional_data(self, sample_scm):
        """Test generation of interventional data."""
        generator = SyntheticDataGenerator(sample_scm)
        
        # Get a node to intervene on
        node = list(sample_scm.get_causal_graph().nodes())[0]
        
        # Generate data with intervention
        n_samples = 100
        intervention_value = 5.0
        data = generator.generate_interventional_data(
            target_node=node,
            intervention_value=intervention_value,
            n_samples=n_samples
        )
        
        # Check shape
        assert isinstance(data, torch.Tensor)
        assert data.shape == (n_samples, len(sample_scm.get_causal_graph().nodes()))
        
        # Check intervention effect
        node_idx = list(sample_scm.get_causal_graph().nodes()).index(node)
        assert torch.allclose(data[:, node_idx], torch.tensor(intervention_value), atol=1e-5)
    
    def test_generate_multiple_interventions(self, sample_scm):
        """Test generation of data with multiple interventions."""
        generator = SyntheticDataGenerator(sample_scm)
        
        # Get nodes to intervene on
        nodes = list(sample_scm.get_causal_graph().nodes())
        interventions = {
            nodes[0]: 5.0,
            nodes[1]: -2.0
        }
        
        # Generate data with interventions
        n_samples = 100
        data = generator.generate_multiple_interventions(
            interventions=interventions,
            n_samples=n_samples
        )
        
        # Check shape
        assert isinstance(data, torch.Tensor)
        assert data.shape == (n_samples, len(sample_scm.get_causal_graph().nodes()))
        
        # Check intervention effects
        for node, value in interventions.items():
            node_idx = nodes.index(node)
            assert torch.allclose(data[:, node_idx], torch.tensor(value), atol=1e-5)
    
    def test_add_noise(self, sample_scm):
        """Test adding noise to data."""
        generator = SyntheticDataGenerator(sample_scm)
        
        # Generate clean data
        n_samples = 100
        clean_data = generator.generate_observational_data(n_samples=n_samples, add_noise=False)
        
        # Add different types of noise
        gaussian_noise = generator.add_noise(clean_data, noise_type='gaussian', noise_scale=0.1)
        uniform_noise = generator.add_noise(clean_data, noise_type='uniform', noise_scale=0.1)
        
        # Check shapes
        assert gaussian_noise.shape == clean_data.shape
        assert uniform_noise.shape == clean_data.shape
        
        # Check that noise was added (data should be different)
        assert not torch.allclose(gaussian_noise, clean_data)
        assert not torch.allclose(uniform_noise, clean_data)
    
    def test_generate_data_batch(self, sample_scm):
        """Test generation of a batch of time series data."""
        generator = SyntheticDataGenerator(sample_scm)
        
        # Generate batch
        batch_size = 8
        seq_length = 10
        batch = generator.generate_batch(batch_size=batch_size, seq_length=seq_length)
        
        # Check shape [batch_size, seq_length, n_variables]
        assert batch.shape == (batch_size, seq_length, len(sample_scm.get_causal_graph().nodes()))
        
        # Check data properties
        assert not torch.isnan(batch).any()
        assert not torch.isinf(batch).any()
    
    def test_graph_dataset(self, sample_scm):
        """Test the GraphDataset class."""
        n_samples = 100
        seq_length = 10
        
        # Create dataset
        dataset = GraphDataset(
            scm=sample_scm,
            n_samples=n_samples,
            seq_length=seq_length
        )
        
        # Check length
        assert len(dataset) == n_samples
        
        # Get an item
        item = dataset[0]
        X, adj_matrix = item
        
        # Check shapes
        assert X.shape == (seq_length, len(sample_scm.get_causal_graph().nodes()))
        assert adj_matrix.shape == (len(sample_scm.get_causal_graph().nodes()), len(sample_scm.get_causal_graph().nodes()))
        
        # Check adjacency matrix is binary
        assert torch.all((adj_matrix == 0) | (adj_matrix == 1))
    
    def test_graph_data_loader(self, sample_scm):
        """Test the GraphDataLoader class."""
        n_samples = 16
        seq_length = 10
        batch_size = 4
        
        # Create dataset
        dataset = GraphDataset(
            scm=sample_scm,
            n_samples=n_samples,
            seq_length=seq_length
        )
        
        # Create data loader
        data_loader = GraphDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Check number of batches
        assert len(data_loader) == n_samples // batch_size
        
        # Iterate over batches
        for batch_data in data_loader:
            X_batch, adj_matrices_batch = batch_data
            
            # Check shapes
            assert X_batch.shape == (batch_size, seq_length, len(sample_scm.get_causal_graph().nodes()))
            assert adj_matrices_batch.shape == (batch_size, len(sample_scm.get_causal_graph().nodes()), len(sample_scm.get_causal_graph().nodes()))
            
            # Just check one batch
            break 