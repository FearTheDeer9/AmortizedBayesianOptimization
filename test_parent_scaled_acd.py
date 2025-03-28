from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.data_setup import setup_observational_interventional
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from algorithms.PARENT_SCALE_ACD import PARENT_SCALE_ACD
import logging
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import torch.nn.functional as F

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def train_and_evaluate_acd_models(
    num_training_graphs: int = 100,
    graph_size: int = 50,
    save_interval: int = 10,
    n_obs: int = 200,
    n_int: int = 2,
    n_trials: int = 10,
    noiseless: bool = True,
    nonlinear: bool = False,
    seed: int = 42
):
    """Train ACD models on multiple Erdos-Renyi graphs and evaluate their performance."""
    plt.ioff()  # Turn off interactive mode to prevent manual closing

    # Create directory for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"results/acd_models_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Training loop
    print(
        f"\nTraining ACD models on {num_training_graphs} Erdos-Renyi graphs...")
    for i in range(num_training_graphs):
        try:
            # Create new graph with expected number of edges
            graph = ErdosRenyiGraph(
                num_nodes=graph_size,
                nonlinear=nonlinear,
                exp_edges=graph_size  # Expect one edge per node on average
            )
            graph.set_target(str(graph_size-1))  # Set last node as target

            # Get data
            D_O, D_I, exploration_set = setup_observational_interventional(
                graph_type=None,
                noiseless=noiseless,
                seed=seed + i,
                n_obs=n_obs,
                n_int=n_int,
                graph=graph,
            )

            # Set population statistics based on observational data
            for var in graph.variables:
                if var != graph.target:
                    mean_val = np.mean(D_O[var])
                    std_val = np.std(D_O[var])
                    graph.population_mean_variance[var] = {
                        "mean": mean_val, "std": std_val}

            # Initialize and train ACD model
            acd_model = PARENT_SCALE_ACD(
                graph=graph,
                noiseless=noiseless,
                device="cpu"
            )
            acd_model.set_values(D_O, D_I, exploration_set)

            # Train the model
            print(f"Training model on graph {i+1}/{num_training_graphs}")
            acd_model.encoder.train()
            acd_model.decoder.train()

            # Train for a few epochs
            optimizer = torch.optim.Adam(
                list(acd_model.encoder.parameters()) + list(acd_model.decoder.parameters()))
            for epoch in range(5):  # Train for 5 epochs
                optimizer.zero_grad()

                # Forward pass
                edge_probs = acd_model.encoder(acd_model.observational_tensor)
                predictions = acd_model.decoder(
                    acd_model.observational_tensor, edge_probs, torch.zeros(graph_size, dtype=torch.bool))

                # Calculate loss
                target_idx = graph.variables.index(graph.target)
                loss = F.mse_loss(
                    predictions[:, target_idx], acd_model.observational_tensor[:, target_idx])

                # Backward pass
                loss.backward()
                optimizer.step()

                if epoch % 2 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Set models back to eval mode
            acd_model.encoder.eval()
            acd_model.decoder.eval()

            # Save model after every save_interval graphs
            if (i + 1) % save_interval == 0:
                encoder_path = os.path.join(
                    model_dir, f"encoder_graph_{i+1}.pt")
                decoder_path = os.path.join(
                    model_dir, f"decoder_graph_{i+1}.pt")
                torch.save(acd_model.encoder.state_dict(), encoder_path)
                torch.save(acd_model.decoder.state_dict(), decoder_path)
                print(f"Saved models after training on {i+1} graphs")

        except Exception as e:
            print(f"Error training on graph {i+1}: {str(e)}")
            continue

    # Evaluation on a new graph
    print("\nEvaluating models on a new Erdos-Renyi graph...")
    eval_graph = ErdosRenyiGraph(
        num_nodes=graph_size,
        nonlinear=nonlinear,
        exp_edges=graph_size  # Expect one edge per node on average
    )
    eval_graph.set_target(str(graph_size-1))

    # Get evaluation data and set reasonable population statistics
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=noiseless,
        seed=seed + num_training_graphs,
        n_obs=n_obs,
        n_int=n_int,
        graph=eval_graph,
    )

    # Set population statistics for evaluation graph
    for var in eval_graph.variables:
        if var != eval_graph.target:
            mean_val = np.mean(D_O[var])
            std_val = np.std(D_O[var])
            eval_graph.population_mean_variance[var] = {
                "mean": mean_val, "std": std_val}

    # Run baseline PARENT_SCALE with error handling
    print("\nRunning baseline PARENT_SCALE...")
    try:
        ps_model = PARENT_SCALE(
            graph=eval_graph,
            nonlinear=nonlinear,
            individual=True,
            use_doubly_robust=True,
        )
        ps_model.set_values(D_O, D_I, exploration_set)
        ps_results = ps_model.run_algorithm(T=n_trials, show_graphics=False)
    except Exception as e:
        print(f"Error running PARENT_SCALE: {str(e)}")
        ps_results = None

    # Run ACD with each saved model
    print("\nRunning ACD with each saved model...")
    acd_results = []
    model_files = sorted([f for f in os.listdir(
        model_dir) if f.startswith("encoder_graph_")])

    # Create a figure for comparing all results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    if ps_results is not None:
        plt.plot(ps_results[0], label='PARENT_SCALE', linewidth=2)

    for encoder_file in model_files:
        try:
            graph_num = int(encoder_file.split("_")[-1].split(".")[0])
            print(f"\nTesting model trained on {graph_num} graphs...")

            encoder_path = os.path.join(model_dir, encoder_file)
            decoder_path = os.path.join(
                model_dir, f"decoder_graph_{graph_num}.pt")

            acd_model = PARENT_SCALE_ACD(
                graph=eval_graph,
                noiseless=noiseless,
                device="cpu",
                encoder_path=encoder_path,
                decoder_path=decoder_path
            )
            acd_model.set_values(D_O, D_I, exploration_set)
            results = acd_model.run_algorithm(T=n_trials, show_graphics=False)
            acd_results.append((graph_num, results))

            # Plot results
            plt.plot(results[0], label=f'ACD ({graph_num} graphs)', alpha=0.7)

        except Exception as e:
            print(f"Error testing model {graph_num}: {str(e)}")
            continue

    plt.title('Best Y Value Over Trials')
    plt.xlabel('Trial')
    plt.ylabel('Best Y Value')
    plt.legend()
    plt.grid(True)

    # Plot intervention costs
    plt.subplot(2, 1, 2)
    if ps_results is not None:
        plt.plot(ps_results[2], label='PARENT_SCALE', linewidth=2)
    for graph_num, results in acd_results:
        plt.plot(results[2], label=f'ACD ({graph_num} graphs)', alpha=0.7)

    plt.title('Cumulative Cost Over Trials')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Cost')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'performance_comparison.png'))
    plt.close()

    # Visualize true graph and save it
    print("\nVisualizing true causal graph...")
    plt.figure(figsize=(10, 8))
    eval_graph.visualize_graph(
        save_path=os.path.join(model_dir, "true_graph.png"))
    plt.close()

    # Visualize and save inferred graphs from each model
    print("\nVisualizing inferred graphs...")
    for graph_num, results in acd_results:
        try:
            plt.figure(figsize=(10, 8))
            acd_model.visualize_edge_probabilities(
                save_path=os.path.join(
                    model_dir, f"inferred_graph_{graph_num}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error visualizing graph {graph_num}: {str(e)}")
            continue

    # Compare results
    print("\nResults comparison:")
    if ps_results is not None:
        print(f"PARENT_SCALE final best Y: {ps_results[0][-1]:.3f}")
    for graph_num, results in acd_results:
        print(
            f"ACD (trained on {graph_num} graphs) final best Y: {results[0][-1]:.3f}")

    # Save results
    results_dict = {
        "baseline": {
            "Best_Y": ps_results[0] if ps_results is not None else None,
            "Per_trial_Y": ps_results[1] if ps_results is not None else None,
            "Cost": ps_results[2] if ps_results is not None else None,
            "Intervention_Set": ps_results[3] if ps_results is not None else None,
            "Intervention_Value": ps_results[4] if ps_results is not None else None,
            "Uncertainty": ps_results[5] if ps_results is not None else None,
        },
        "acd_models": {
            str(graph_num): {
                "Best_Y": results[0],
                "Per_trial_Y": results[1],
                "Cost": results[2],
                "Intervention_Set": results[3],
                "Intervention_Value": results[4],
                "Uncertainty": results[5],
            }
            for graph_num, results in acd_results
        }
    }

    with open(os.path.join(model_dir, "results.pickle"), "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict


if __name__ == "__main__":
    # Train and evaluate ACD models
    train_and_evaluate_acd_models(
        num_training_graphs=1,
        graph_size=6,
        save_interval=10,
        n_obs=200,
        n_int=2,
        n_trials=10,
        noiseless=True,
        nonlinear=False,
        seed=42
    )
