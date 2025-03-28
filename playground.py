import os
from copy import deepcopy
# set the directory to the root
from algorithms.CEO_algorithm import CEO
from algorithms.CEO_ACD_algorithm import CEO_ACD  # Our new implementation
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from graphs.toy_graph import ToyGraph
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_10_nodes import Graph10Nodes
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.data_setup import setup_observational_interventional
from utils.acd_models import GraphEncoder, DynamicsDecoder  # Our model definitions
from utils.sem_sampling import sample_model

# Parameters
n_obs = 200
n_int = 2
n_anchor_points = 35
seeds_int_data = 7
noiseless = False
filename = "ErdosRenyiACD"

# Number of nodes for the Erdos-Renyi graphs
n_nodes = 8
n_training_graphs = 5

# Create training graphs (multiple random Erdos-Renyi graphs)
print(f"Creating {n_training_graphs} random Erdos-Renyi graphs for training")
training_graphs = []
for i in range(n_training_graphs):
    # Use different seeds for each graph to ensure diversity
    graph = ErdosRenyiGraph(num_nodes=n_nodes, seed=i+100, exp_edges=1)
    # Set a reasonable target node (let's use node 1 for all training graphs)
    graph.set_target("1")
    training_graphs.append(graph)
    print(f"Training graph {i+1} created with {len(graph.edges)} edges")

# Create a separate test graph with a different seed
test_seed = 42  # Different from training seeds
test_graph = ErdosRenyiGraph(num_nodes=n_nodes, seed=test_seed, exp_edges=1)
# Set the same target for consistency
test_graph.set_target("1")
print(f"Test graph created with {len(test_graph.edges)} edges")

# Setup data for test graph
print("Setting up observational and interventional data for test graph")
D_O, D_I, _ = setup_observational_interventional(
    graph_type="ErdosRenyi",  # Not actually used since we're passing the graph
    graph=test_graph,
    n_obs=n_obs,
    n_int=n_int,
    noiseless=noiseless,
    seed=seeds_int_data
)

# We'll let the model determine the exploration set dynamically
exploration_set = None


def pretrain_models(training_graphs, n_obs=200):
    """Pre-train the ACD models using multiple training graphs."""
    # Create models
    encoder = GraphEncoder()
    decoder = DynamicsDecoder()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

    # Train on each graph
    for graph_idx, graph in enumerate(training_graphs):
        print(f"Training on graph {graph_idx + 1}/{len(training_graphs)}")
        n_variables = len(graph.variables)
        print(f"Graph has {n_variables} variables: {graph.variables}")
        print(f"Graph edges: {graph.edges}")

        # Generate observational data for this graph using the sample_model function
        D_O_train = sample_model(
            static_sem=graph.SEM,
            sample_count=n_obs,
            graph=graph,
            noiseless=False
        )

        # Convert data to tensor - shape will be [n_obs, n_variables]
        obs_data = np.hstack([D_O_train[var].reshape(-1, 1)
                             for var in graph.variables])
        print(f"Observational data shape: {obs_data.shape}")
        obs_tensor = torch.tensor(obs_data, dtype=torch.float32)

        # Get the true graph as a tensor - shape will be [n_variables, n_variables]
        true_edges = torch.zeros((n_variables, n_variables))
        for from_var, to_var in graph.edges:
            from_idx = graph.variables.index(from_var)
            to_idx = graph.variables.index(to_var)
            true_edges[from_idx, to_idx] = 1.0

        print(f"True edges shape: {true_edges.shape}")

        # Train for a reasonable number of epochs
        for epoch in range(20):  # 20 epochs per graph
            # Process in mini-batches to avoid memory issues
            batch_size = 32
            n_batches = n_obs // batch_size

            total_loss = 0.0
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = start_idx + batch_size
                batch_data = obs_tensor[start_idx:end_idx]

                # Predict edges
                pred_edges = encoder(batch_data)

                # Ensure pred_edges has the same shape as true_edges
                if pred_edges.shape != true_edges.shape:
                    print(
                        f"Shape mismatch: pred_edges {pred_edges.shape}, true_edges {true_edges.shape}")
                    raise ValueError("Shapes don't match")

                # Edge prediction loss
                edge_loss = torch.nn.functional.binary_cross_entropy(
                    pred_edges, true_edges)

                # Dynamics prediction loss
                pred_next = decoder(batch_data, true_edges)
                target_idx = graph.variables.index(graph.target)
                dynamics_loss = torch.nn.functional.mse_loss(
                    pred_next[:, target_idx], batch_data[:, target_idx])

                # Total loss
                loss = edge_loss + dynamics_loss
                total_loss += loss.item()

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:  # Report progress every 5 epochs
                print(
                    f"Epoch {epoch}: Average Loss = {total_loss / n_batches:.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    print("Saving models...")
    torch.save(encoder.state_dict(), "models/encoder.pt")
    torch.save(decoder.state_dict(), "models/decoder.pt")

    print("Model training complete.")
    return encoder, decoder


# Pretrain models (or load them if they exist)
# Force retraining with our new architecture and training data
if os.path.exists("models/encoder.pt"):
    os.remove("models/encoder.pt")
if os.path.exists("models/decoder.pt"):
    os.remove("models/decoder.pt")

encoder, decoder = pretrain_models(training_graphs)

# Initialize the ACD models (trained and untrained)
model_acd_trained = CEO_ACD(
    graph_type="ErdosRenyi",  # Using Erdos-Renyi graphs
    graph=test_graph,  # For reference/comparison
    encoder_path="models/encoder.pt",
    decoder_path="models/decoder.pt",
    n_anchor_points=n_anchor_points,
    noiseless=noiseless,
    allow_fine_tuning=False  # Disable fine-tuning
)

model_acd_untrained = CEO_ACD(
    graph_type="ErdosRenyi",  # Using Erdos-Renyi graphs
    graph=test_graph,  # For reference/comparison
    encoder_path=None,  # No pre-trained models
    decoder_path=None,
    n_anchor_points=n_anchor_points,
    noiseless=noiseless,
    allow_fine_tuning=False  # Disable fine-tuning
)

# Set values and run for both ACD models
model_acd_trained.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
model_acd_untrained.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)

# Create output directory
os.makedirs(f"model_checkpoints/{filename}/", exist_ok=True)

# Run algorithms
print("Running trained model...")
results_acd_trained = model_acd_trained.run_algorithm(
    T=13, file=f"model_checkpoints/{filename}/trained_")

print("Running untrained model...")
results_acd_untrained = model_acd_untrained.run_algorithm(
    T=13, file=f"model_checkpoints/{filename}/untrained_")

# For comparison, also run the original CEO
print("Running baseline CEO model...")
model_ceo = CEO(
    graph_type="Erdos8",  # Using Erdos-Renyi graphs with 8 nodes
    all_graph_edges=[test_graph.edges],
    n_obs=n_obs,
    n_int=n_int,
    n_anchor_points=n_anchor_points,
    seed=seeds_int_data,
    noiseless=noiseless,
)
model_ceo.set_values(deepcopy(D_O), deepcopy(
    D_I), exploration_set=model_acd_trained.exploration_set)
results_ceo = model_ceo.run_algorithm(
    T=13, file=f"model_checkpoints/{filename}_ceo/")

# Compare performance
best_y_acd_trained, current_y_acd_trained, cost_array_acd_trained, intervention_set_acd_trained, intervention_values_acd_trained, _ = results_acd_trained
best_y_acd_untrained, current_y_acd_untrained, cost_array_acd_untrained, intervention_set_acd_untrained, intervention_values_acd_untrained, _ = results_acd_untrained
best_y_ceo, current_y_ceo, cost_array_ceo, intervention_set_ceo, intervention_values_ceo, _ = results_ceo

# Plot comparison of best values
plt.figure(figsize=(12, 6))
plt.plot(best_y_acd_trained, label="ACD (Trained) Best Y")
plt.plot(best_y_acd_untrained, label="ACD (Untrained) Best Y")
plt.plot(best_y_ceo, label="CEO Best Y")
plt.xlabel("Iteration")
plt.ylabel("Target Value")
plt.title("CEO vs ACD (Trained/Untrained) Performance on Erdos-Renyi Graph")
plt.legend()
plt.savefig(f"model_checkpoints/{filename}/comparison_best_y.png")

# Plot comparison of interventions at each step
plt.figure(figsize=(12, 6))
plt.plot(current_y_acd_trained, label="ACD (Trained) Current Y")
plt.plot(current_y_acd_untrained, label="ACD (Untrained) Current Y")
plt.plot(current_y_ceo, label="CEO Current Y")
plt.xlabel("Iteration")
plt.ylabel("Target Value")
plt.title("Per-iteration Intervention Outcomes on Erdos-Renyi Graph")
plt.legend()
plt.savefig(f"model_checkpoints/{filename}/comparison_current_y.png")

# Plot comparison of costs
plt.figure(figsize=(12, 6))
plt.plot(cost_array_acd_trained, label="ACD (Trained) Cost")
plt.plot(cost_array_acd_untrained, label="ACD (Untrained) Cost")
plt.plot(cost_array_ceo, label="CEO Cost")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Cost")
plt.title("Cumulative Intervention Cost on Erdos-Renyi Graph")
plt.legend()
plt.savefig(f"model_checkpoints/{filename}/comparison_cost.png")

# Print intervention choices
print("\nIntervention sets (Trained model):", intervention_set_acd_trained)
print("\nIntervention sets (Untrained model):", intervention_set_acd_untrained)
print("\nIntervention sets (CEO model):", intervention_set_ceo)

# Save the exploration sets used
with open(f"model_checkpoints/{filename}/exploration_sets.txt", "w") as f:
    f.write("Trained model exploration set:\n")
    f.write(str(model_acd_trained.exploration_set) + "\n\n")
    f.write("Untrained model exploration set:\n")
    f.write(str(model_acd_untrained.exploration_set) + "\n\n")

# Visualize final edge probabilities for both ACD models
model_acd_trained.visualize_edge_probabilities(
    save_path=f"model_checkpoints/{filename}/final_edges_trained.png")
model_acd_untrained.visualize_edge_probabilities(
    save_path=f"model_checkpoints/{filename}/final_edges_untrained.png")

# Compare with true graph
true_edges = np.zeros((len(test_graph.variables), len(test_graph.variables)))
for from_var, to_var in test_graph.edges:
    from_idx = test_graph.variables.index(from_var)
    to_idx = test_graph.variables.index(to_var)
    true_edges[from_idx, to_idx] = 1.0

# Plot true graph structure
plt.figure(figsize=(10, 8))
plt.imshow(true_edges, cmap='binary', vmin=0, vmax=1)
plt.colorbar(label='Edge Present')
plt.xticks(range(len(test_graph.variables)), test_graph.variables, rotation=90)
plt.yticks(range(len(test_graph.variables)), test_graph.variables)
plt.title('True Causal Structure of Test Erdos-Renyi Graph')
plt.tight_layout()
plt.savefig(f"model_checkpoints/{filename}/true_edges.png")

print(f"Experiment complete. Results saved to model_checkpoints/{filename}/")
plt.show()
