"""
Example usage of acquisition strategies for causal Bayesian optimization.

This example demonstrates how to use the AcquisitionStrategy interface and its
implementations to select interventions in a causal system.
"""

import numpy as np
import matplotlib.pyplot as plt

from causal_meta.optimization import ExpectedImprovement, UpperConfidenceBound
from causal_meta.graph.causal_graph import CausalGraph


class SimpleDynamicsModel:
    """
    Simple dynamics model for demonstration purposes.
    
    This model simulates a simple causal system where interventions on nodes
    affect the outcome in a predefined way.
    """
    
    def __init__(self, noise_level=0.1):
        """Initialize the model with a given noise level."""
        self.noise_level = noise_level
    
    def predict_intervention_outcome(self, graph, intervention, data, return_uncertainty=False):
        """
        Predict the outcome of an intervention.
        
        For this simple example, we'll use a quadratic function where the
        optimal intervention is at value 1.5 for node 0 and 0.5 for node 1.
        
        Args:
            graph: Causal graph structure
            intervention: Intervention specification
            data: Observational data
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predicted outcome, optionally with uncertainty
        """
        node = intervention.get('target_node', 0)
        value = intervention.get('value', 0.0)
        
        # Simple quadratic function with optima at (node 0, value 1.5) and (node 1, value 0.5)
        if node == 0:
            # Optimal value is 1.5
            mean = -((value - 1.5) ** 2) + 2.25  # Max value is 2.25
        elif node == 1:
            # Optimal value is 0.5
            mean = -((value - 0.5) ** 2) + 0.25  # Max value is 0.25
        else:
            # All other nodes have flat response
            mean = 0.0
        
        # Add some randomness based on the value
        mean = np.array([mean])
        
        if return_uncertainty:
            # Higher uncertainty farther from observed values
            std = np.array([self.noise_level * (1.0 + abs(value))])
            return mean, {'prediction_std': std}
        
        return mean
    
    def estimate_uncertainty(self):
        """Return a fixed uncertainty estimate."""
        return {'prediction_std': np.array([self.noise_level])}
    
    def update_model(self, data):
        """Update the model with new data (no-op for this simple example)."""
        pass


def run_example():
    """Run the acquisition strategy example."""
    # Create a simple causal graph
    graph = CausalGraph()
    # Add nodes individually
    graph.add_node('X0')
    graph.add_node('X1')
    graph.add_node('X2')
    graph.add_edge('X0', 'X1')
    graph.add_edge('X1', 'X2')
    
    # Create a simple dynamics model
    model = SimpleDynamicsModel(noise_level=0.2)
    
    # Create data
    data = {'observations': np.random.rand(10, 3)}
    
    # Create acquisition strategies
    ei = ExpectedImprovement(exploration_weight=0.1, maximize=True)
    ucb = UpperConfidenceBound(beta=2.0, maximize=True)
    
    # Set current best value for EI
    ei.set_best_value(1.0)
    
    # Generate candidate interventions for visualization
    candidates = []
    for node in range(2):  # Only use nodes 0 and 1
        for value in np.linspace(-3.0, 3.0, 50):
            candidates.append({
                'target_node': node,
                'value': float(value)
            })
    
    # Override the default candidate generation for visualization
    ei.intervention_candidates = candidates
    ucb.intervention_candidates = candidates
    
    # Compute acquisition values
    ei_values = ei.compute_acquisition(model, graph, data)
    
    # Select best interventions
    best_ei = ei.select_intervention(model, graph, data, budget=1.0)
    
    # Select batch of interventions
    batch_ei = ei.select_batch(model, graph, data, budget=1.0, batch_size=3)
    
    print(f"Best intervention (EI): {best_ei}")
    print(f"Batch interventions (EI): {batch_ei}")
    
    # Visualize the acquisition functions and outcomes
    plot_results(model, graph, candidates, ei_values)


def plot_results(model, graph, candidates, ei_values):
    """Plot the results of the acquisition function evaluation."""
    # Extract values for node 0 and node 1
    node0_values = []
    node0_acq = []
    node1_values = []
    node1_acq = []
    
    for candidate in candidates:
        node = candidate['target_node']
        value = candidate['value']
        key = f"node_{node}_value_{value}"
        acq = ei_values.get(key, 0.0)
        
        if node == 0:
            node0_values.append(value)
            node0_acq.append(acq)
        elif node == 1:
            node1_values.append(value)
            node1_acq.append(acq)
    
    # Compute true outcomes for comparison
    node0_outcomes = []
    node1_outcomes = []
    
    for value in node0_values:
        outcome = model.predict_intervention_outcome(
            graph, {'target_node': 0, 'value': value}, {}
        )
        node0_outcomes.append(float(outcome))
    
    for value in node1_values:
        outcome = model.predict_intervention_outcome(
            graph, {'target_node': 1, 'value': value}, {}
        )
        node1_outcomes.append(float(outcome))
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot true function for node 0
    axes[0, 0].plot(node0_values, node0_outcomes, 'b-', label='True function')
    axes[0, 0].set_title('True Function (Node 0)')
    axes[0, 0].set_xlabel('Intervention Value')
    axes[0, 0].set_ylabel('Outcome')
    axes[0, 0].grid(True)
    
    # Plot acquisition function for node 0
    axes[0, 1].plot(node0_values, node0_acq, 'r-', label='EI')
    axes[0, 1].set_title('Expected Improvement (Node 0)')
    axes[0, 1].set_xlabel('Intervention Value')
    axes[0, 1].set_ylabel('Acquisition Value')
    axes[0, 1].grid(True)
    
    # Plot true function for node 1
    axes[1, 0].plot(node1_values, node1_outcomes, 'b-', label='True function')
    axes[1, 0].set_title('True Function (Node 1)')
    axes[1, 0].set_xlabel('Intervention Value')
    axes[1, 0].set_ylabel('Outcome')
    axes[1, 0].grid(True)
    
    # Plot acquisition function for node 1
    axes[1, 1].plot(node1_values, node1_acq, 'r-', label='EI')
    axes[1, 1].set_title('Expected Improvement (Node 1)')
    axes[1, 1].set_xlabel('Intervention Value')
    axes[1, 1].set_ylabel('Acquisition Value')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('acquisition_example.png')
    print("Plot saved as 'acquisition_example.png'")


if __name__ == '__main__':
    run_example() 