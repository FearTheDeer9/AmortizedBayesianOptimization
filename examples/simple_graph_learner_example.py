"""
Simple example of using SimpleGraphLearner for causal graph structure learning.

This script demonstrates how to:
1. Generate a random DAG and linear SCM
2. Create observational and interventional data
3. Initialize and train a SimpleGraphLearner
4. Evaluate the learned graph against the ground truth
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam

from causal_meta.structure_learning import (
    RandomDAGGenerator,
    LinearSCMGenerator,
    SimpleGraphLearner,
    generate_observational_data,
    generate_interventional_data,
    create_intervention_mask,
    convert_to_tensor,
    normalize_data
)


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate a random DAG
    n_nodes = 5
    edge_prob = 0.3
    print(f"Generating random DAG with {n_nodes} nodes...")
    adj_matrix = RandomDAGGenerator.generate_random_dag(
        num_nodes=n_nodes,
        edge_probability=edge_prob,
        as_adjacency_matrix=True,
        seed=42
    )
    print("Adjacency matrix:")
    print(adj_matrix)
    
    # 2. Create a linear SCM
    print("\nCreating linear SCM...")
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=0.1,
        seed=42
    )
    
    # 3. Generate observational data
    n_obs_samples = 200
    print(f"\nGenerating {n_obs_samples} observational samples...")
    obs_data = generate_observational_data(
        scm=scm,
        num_samples=n_obs_samples,
        as_tensor=False,
        seed=42
    )
    print(f"Observational data shape: {obs_data.shape}")
    
    # 4. Generate interventional data
    n_int_samples = 50
    intervention_targets = [1, 3]  # Intervene on x1 and x3
    intervention_values = [2.0, -1.0]
    print(f"\nGenerating {n_int_samples} interventional samples...")
    int_data = generate_interventional_data(
        scm=scm,
        num_samples=n_int_samples,
        intervened_nodes=intervention_targets,
        intervention_values=intervention_values,
        as_tensor=False,
        seed=42
    )
    print(f"Interventional data shape: {int_data.shape}")
    
    # 5. Create intervention mask
    print("\nCreating intervention mask...")
    int_mask = np.zeros((n_int_samples, n_nodes))
    for node_idx in intervention_targets:
        int_mask[:, node_idx] = 1
    print(f"Intervention mask shape: {int_mask.shape}")
    
    # 6. Normalize data
    print("\nNormalizing data...")
    obs_data_norm, scaler = normalize_data(obs_data)
    int_data_norm = normalize_data(int_data, scaler=scaler)[0]
    
    # 7. Convert to tensors
    obs_tensor = convert_to_tensor(obs_data_norm)
    int_tensor = convert_to_tensor(int_data_norm)
    int_mask_tensor = torch.tensor(int_mask, dtype=torch.float32)
    
    # 8. Initialize and train a SimpleGraphLearner
    print("\nInitializing SimpleGraphLearner...")
    model = SimpleGraphLearner(
        input_dim=n_nodes,
        hidden_dim=64,
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Convert adjacency matrix to tensor for training
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # Set up optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    n_epochs = 100
    print(f"\nTraining for {n_epochs} epochs...")
    
    train_losses = []
    
    for epoch in range(n_epochs):
        # Train on observational data
        optimizer.zero_grad()
        edge_probs_obs = model(obs_tensor)
        loss_obs, loss_components_obs = model.calculate_loss(edge_probs_obs, adj_matrix_tensor)
        
        # Train on interventional data
        edge_probs_int = model(int_tensor, int_mask_tensor)
        loss_int, loss_components_int = model.calculate_loss(edge_probs_int, adj_matrix_tensor)
        
        # Combined loss
        total_loss = loss_obs + loss_int
        train_losses.append(total_loss.item())
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4f}")
    
    # 9. Evaluate the learned graph
    print("\nEvaluating learned graph...")
    with torch.no_grad():
        edge_probs_final = model(torch.cat([obs_tensor, int_tensor], dim=0))
    
    # Convert to binary adjacency matrix
    learned_adj_matrix = model.threshold_edge_probabilities(edge_probs_final).numpy()
    
    # Calculate accuracy
    correct_edges = (learned_adj_matrix == adj_matrix).sum()
    total_edges = adj_matrix.size
    accuracy = correct_edges / total_edges
    
    print(f"True adjacency matrix:\n{adj_matrix}")
    print(f"Learned adjacency matrix:\n{learned_adj_matrix}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 10. Convert to CausalGraph
    learned_graph = model.to_causal_graph(edge_probs_final)
    print("\nLearned graph edges:")
    for source, target in learned_graph.get_edges():
        print(f"{source} -> {target}")
    
    # 11. Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # 12. Compare true and learned graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True graph
    ax = axes[0]
    RandomDAGGenerator.visualize_dag(
        adj_matrix,
        title="True Graph",
        figsize=(6, 5)
    )
    
    # Learned graph
    ax = axes[1]
    RandomDAGGenerator.visualize_dag(
        learned_adj_matrix,
        title="Learned Graph",
        figsize=(6, 5)
    )
    
    plt.tight_layout()
    plt.savefig('graph_comparison.png')
    
    print("\nTraining loss and graph comparison plots saved.")


if __name__ == "__main__":
    main() 