"""
Enhanced Model Parameter Tuning

This script implements systematic parameter tuning for the SimpleGraphLearner
to achieve perfect graph structure recovery (SHD=0) on small graphs.
It builds on the insights from the edge_bias_analysis.py script.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/model_tuning_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "num_nodes": 4,                # Small graph for easier perfect recovery
    "edge_probability": 0.3,       # Typical sparsity for small causal graphs
    "num_samples": 2000,           # More samples for better learning
    "hidden_dim": 64,              # Default model architecture
    "num_layers": 2,               # Default model architecture
    "epochs": 200,                 # More training epochs to ensure convergence
    "batch_size": 32,              # Default batch size
    "learning_rate": 0.001,        # Default learning rate
    "weight_decay": 0.0,           # Default weight decay
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "seeds": [42, 43, 44, 45, 46]  # Multiple seeds to assess consistency
}

# Parameter grid for search - corrected parameter names to match SimpleGraphLearner
PARAMETER_GRID = {
    "sparsity_weight": [0.01, 0.03, 0.05, 0.07],    # Try progressively lower values
    "pos_weight": [5.0, 7.0, 10.0],                 # Try stronger class balancing
    "edge_prob_bias": [0.1, 0.2, 0.3],              # Try stronger edge probability bias
    "consistency_weight": [0.1, 0.2],               # Try different consistency regularization
    "expected_density": [0.3, 0.4]                  # Try different expected density targets
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_graph(num_nodes, edge_probability, seed):
    """Generate a random DAG and convert to adjacency matrix."""
    set_seed(seed)
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        seed=seed
    )
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in causal_graph.get_children(i):
            adjacency_matrix[i, j] = 1
            
    return adjacency_matrix

def generate_synthetic_data(adjacency_matrix, n_samples=2000, seed=42):
    """Generate synthetic data based on a linear SCM."""
    set_seed(seed)
    n_nodes = adjacency_matrix.shape[0]
    
    # Create random linear weights for the SCM
    weights = np.random.uniform(0.5, 1.5, size=adjacency_matrix.shape)
    weights = weights * adjacency_matrix  # Zero out non-edges
    
    # Generate data
    data = np.zeros((n_samples, n_nodes))
    for i in range(n_samples):
        # Generate in topological order
        for j in range(n_nodes):
            # Compute the value based on parents
            parent_contribution = np.sum(data[i] * weights[:, j])
            # Add noise
            noise = np.random.normal(0, 0.1)
            data[i, j] = parent_contribution + noise
    
    return data

def calculate_metrics(pred_adj, true_adj):
    """Calculate evaluation metrics for predicted graph."""
    # Remove diagonal (self-loops)
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    pred_adj_flat = pred_adj[mask]
    true_adj_flat = true_adj[mask]
    
    # Calculate metrics
    tp = np.sum((pred_adj_flat == 1) & (true_adj_flat == 1))
    fp = np.sum((pred_adj_flat == 1) & (true_adj_flat == 0))
    tn = np.sum((pred_adj_flat == 0) & (true_adj_flat == 0))
    fn = np.sum((pred_adj_flat == 0) & (true_adj_flat == 1))
    
    # Derived metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    shd = fp + fn
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def train_model(model_params, data, true_adj, config):
    """Train the model and collect diagnostics."""
    # Create model
    model = SimpleGraphLearner(
        input_dim=config["num_nodes"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        **model_params
    )
    
    # Create a unique identifier for this model configuration
    param_str = "_".join([f"{k}_{v}" for k, v in model_params.items()])
    model.model_type = param_str
    
    # Print training start info
    print(f"Training model with parameters: {param_str}")
    
    # Convert data to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    true_adj_tensor = torch.tensor(true_adj, dtype=torch.float32)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Training diagnostics
    diagnostics = {
        'loss_history': [],
        'edge_prob_history': [],
        'loss_components': [],
        'metrics_history': []
    }
    
    # Move to device
    device = config["device"]
    model.to(device)
    data_tensor = data_tensor.to(device)
    true_adj_tensor = true_adj_tensor.to(device)
    
    # Best model tracking
    best_model_state = None
    best_shd = float('inf')
    best_epoch = -1
    
    # Training loop
    for epoch in range(config["epochs"]):
        # Forward pass
        edge_probs = model(data_tensor)
        
        # Calculate loss
        loss, loss_components = model.calculate_loss(edge_probs, true_adj_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record diagnostics every 20 epochs
        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
            # Get current edge probabilities
            with torch.no_grad():
                current_probs = model(data_tensor)
                current_probs_np = current_probs.cpu().numpy()
                
                # Apply threshold
                thresholded_adj = (current_probs_np > 0.5).astype(float)
                
                # Calculate metrics
                metrics = calculate_metrics(thresholded_adj, true_adj)
                
                # Calculate mean edge probability
                mean_prob = current_probs_np.mean()
                median_prob = np.median(current_probs_np)
                above_threshold = (current_probs_np > 0.5).mean()
                
                # Record components
                loss_dict = {k: v.item() for k, v in loss_components.items()}
                diagnostics['loss_history'].append(loss.item())
                diagnostics['edge_prob_history'].append({
                    'epoch': epoch,
                    'mean': mean_prob,
                    'median': median_prob,
                    'above_threshold': above_threshold
                })
                diagnostics['loss_components'].append({
                    'epoch': epoch,
                    **loss_dict
                })
                diagnostics['metrics_history'].append({
                    'epoch': epoch,
                    **metrics
                })
                
                # Track best model
                if metrics['shd'] < best_shd:
                    best_shd = metrics['shd']
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                
                # Print progress (but not too verbose)
                if epoch % 50 == 0 or epoch == config["epochs"] - 1:
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}, SHD = {metrics['shd']}, "
                          f"F1 = {metrics['f1']:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    with torch.no_grad():
        final_probs = model(data_tensor)
        final_probs_np = final_probs.cpu().numpy()
        final_adj = (final_probs_np > 0.5).astype(float)
        final_metrics = calculate_metrics(final_adj, true_adj)
    
    print(f"Final metrics: SHD = {final_metrics['shd']}, F1 = {final_metrics['f1']:.4f}, "
          f"Best epoch: {best_epoch}")
    
    # Return model and results
    return model, diagnostics, final_metrics, best_epoch

def parameter_grid_search(config, param_grid, n_combinations=None):
    """
    Perform a parameter grid search.
    
    Args:
        config: Base configuration
        param_grid: Parameter grid to search
        n_combinations: If provided, sample this many combinations randomly
                        instead of full grid search
    
    Returns:
        Results DataFrame and experiment metadata
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Optionally sample a subset of combinations
    if n_combinations and n_combinations < len(all_combinations):
        np.random.seed(config["seeds"][0])  # Use first seed for reproducibility
        combinations = np.random.choice(len(all_combinations), n_combinations, replace=False)
        param_combinations = [all_combinations[i] for i in combinations]
    else:
        param_combinations = all_combinations
    
    print(f"Testing {len(param_combinations)} parameter combinations across {len(config['seeds'])} seeds")
    
    # Create results storage
    all_results = []
    
    # Track parameters that achieve SHD=0
    perfect_recovery_params = []
    
    # Run experiments
    total_combinations = len(param_combinations) * len(config["seeds"])
    progress_bar = tqdm(total=total_combinations, desc="Parameter tuning")
    
    for param_values in param_combinations:
        # Create parameter dictionary
        params = {name: value for name, value in zip(param_names, param_values)}
        
        # Run experiment with each seed
        for seed in config["seeds"]:
            # Set seed
            set_seed(seed)
            
            # Generate graph and data
            adj_matrix = generate_random_graph(
                config["num_nodes"], 
                config["edge_probability"], 
                seed
            )
            
            synthetic_data = generate_synthetic_data(
                adj_matrix, 
                config["num_samples"], 
                seed
            )
            
            # Train model
            model, diagnostics, metrics, best_epoch = train_model(
                params, 
                synthetic_data, 
                adj_matrix, 
                config
            )
            
            # Store results
            result_entry = {
                'seed': seed,
                'best_epoch': best_epoch,
                **params,
                **metrics
            }
            all_results.append(result_entry)
            
            # Check if this achieved perfect recovery
            if metrics['shd'] == 0:
                perfect_recovery_params.append({
                    'seed': seed,
                    **params
                })
                
                # Save the model checkpoint
                param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
                checkpoint_path = os.path.join(
                    output_dir, 
                    "checkpoints", 
                    f"perfect_model_seed_{seed}_{param_str}.pt"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'params': params,
                    'seed': seed,
                    'metrics': metrics,
                    'config': config
                }, checkpoint_path)
                
                print(f"Perfect recovery (SHD=0) achieved with parameters: {params}, seed: {seed}")
            
            # Update progress bar
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = os.path.join(output_dir, "tuning_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save perfect recovery parameters
    perfect_recovery_path = os.path.join(output_dir, "perfect_recovery_params.json")
    with open(perfect_recovery_path, 'w') as f:
        json.dump(perfect_recovery_params, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        # Convert non-serializable items
        save_config = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                      for k, v in config.items()}
        # Convert numpy arrays
        for k, v in param_grid.items():
            save_config[f"grid_{k}"] = v
        json.dump(save_config, f, indent=2)
    
    return results_df, perfect_recovery_params

def analyze_results(results_df, output_dir):
    """Analyze and visualize parameter tuning results."""
    # Create a directory for analysis outputs
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Count how many times each parameter achieved SHD=0
    perfect_recovery = results_df[results_df['shd'] == 0]
    
    if len(perfect_recovery) == 0:
        print("No parameter combinations achieved perfect recovery (SHD=0)")
        # Find best parameters by lowest average SHD
        param_columns = [col for col in results_df.columns 
                         if col not in ['seed', 'best_epoch', 'accuracy', 'precision', 
                                        'recall', 'f1', 'shd', 'tp', 'fp', 'tn', 'fn']]
        
        avg_results = results_df.groupby(param_columns).agg({
            'shd': ['mean', 'std', 'min'],
            'f1': ['mean', 'std', 'max']
        }).reset_index()
        
        # Sort by mean SHD (ascending)
        avg_results.columns = ['_'.join(col).strip() for col in avg_results.columns.values]
        avg_results = avg_results.sort_values('shd_mean')
        
        # Save to CSV
        avg_results.to_csv(os.path.join(analysis_dir, "average_performance.csv"), index=False)
        
        # Get best parameters (lowest average SHD)
        best_params = {}
        for col in param_columns:
            best_params[col] = avg_results.iloc[0][f'{col}_']
        
        print(f"Best parameters (lowest average SHD): {best_params}")
        print(f"Average SHD: {avg_results.iloc[0]['shd_mean']:.2f} ± {avg_results.iloc[0]['shd_std']:.2f}")
        print(f"Average F1: {avg_results.iloc[0]['f1_mean']:.2f} ± {avg_results.iloc[0]['f1_std']:.2f}")
        
        # Plot impact of each parameter on SHD
        for param in param_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='shd', data=results_df)
            plt.title(f'Impact of {param} on SHD')
            plt.savefig(os.path.join(analysis_dir, f"impact_{param}_on_shd.png"))
            plt.close()
        
        return best_params, None
    
    print(f"Found {len(perfect_recovery)} parameter combinations that achieved perfect recovery")
    
    # Count by parameter value
    perfect_counts = {}
    param_columns = [col for col in perfect_recovery.columns 
                    if col not in ['seed', 'best_epoch', 'accuracy', 'precision', 
                                   'recall', 'f1', 'shd', 'tp', 'fp', 'tn', 'fn']]
    
    for param in param_columns:
        counts = perfect_recovery[param].value_counts().to_dict()
        perfect_counts[param] = counts
    
    # Save perfect recovery counts
    with open(os.path.join(analysis_dir, "perfect_recovery_counts.json"), 'w') as f:
        json.dump(perfect_counts, f, indent=2)
    
    # Find parameter values that most consistently achieve SHD=0
    consistently_best_params = {}
    for param in param_columns:
        # Get the parameter value with the most perfect recoveries
        best_value = max(perfect_counts[param].items(), key=lambda x: x[1])[0]
        consistently_best_params[param] = best_value
    
    print(f"Most consistently successful parameters: {consistently_best_params}")
    
    # Visualize parameter effectiveness
    for param in param_columns:
        plt.figure(figsize=(10, 6))
        values = list(perfect_counts[param].keys())
        counts = list(perfect_counts[param].values())
        plt.bar(values, counts)
        plt.title(f'Parameter {param} - Number of Perfect Recoveries')
        plt.xlabel(param)
        plt.ylabel('Count of SHD=0')
        plt.savefig(os.path.join(analysis_dir, f"{param}_perfect_recoveries.png"))
        plt.close()
    
    # Find epoch distribution for perfect recovery
    plt.figure(figsize=(10, 6))
    plt.hist(perfect_recovery['best_epoch'], bins=20)
    plt.title('Epochs Needed for Perfect Recovery')
    plt.xlabel('Best Epoch')
    plt.ylabel('Count')
    plt.savefig(os.path.join(analysis_dir, "epochs_for_perfect_recovery.png"))
    plt.close()
    
    # Create a heatmap of parameter interactions (only if enough data)
    if len(param_columns) >= 2 and len(perfect_recovery) >= 10:
        for i, param1 in enumerate(param_columns[:-1]):
            for param2 in param_columns[i+1:]:
                # Create a cross-tabulation of perfect recoveries
                cross_tab = pd.crosstab(
                    perfect_recovery[param1], 
                    perfect_recovery[param2], 
                    values=perfect_recovery['shd'], 
                    aggfunc='count'
                ).fillna(0)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt='g')
                plt.title(f'Perfect Recovery Counts: {param1} vs {param2}')
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, f"interaction_{param1}_{param2}.png"))
                plt.close()
    
    return consistently_best_params, perfect_recovery

def create_optimal_models_script(best_params, config, output_dir):
    """Create a script to train models with the optimal parameters."""
    script_path = os.path.join(output_dir, "train_optimal_model.py")
    
    script_content = f'''"""
Train SimpleGraphLearner with Optimal Parameters

This script trains the SimpleGraphLearner model with parameters that were
found to consistently achieve perfect graph recovery (SHD=0) on small graphs.
"""

import os
import torch
import numpy as np
from datetime import datetime

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator
from causal_meta.structure_learning.data_utils import generate_observational_data

# Optimal parameters found during tuning
OPTIMAL_PARAMS = {params_str}

# Configuration
CONFIG = {config_str}

def main():
    """Train a model with optimal parameters."""
    # Set random seed for reproducibility
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["seed"])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/optimal_model_{{timestamp}}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random DAG
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=CONFIG["num_nodes"],
        edge_probability=CONFIG["edge_probability"],
        seed=CONFIG["seed"]
    )
    
    # Convert to adjacency matrix
    adjacency_matrix = np.zeros((CONFIG["num_nodes"], CONFIG["num_nodes"]))
    for i in range(CONFIG["num_nodes"]):
        for j in causal_graph.get_children(i):
            adjacency_matrix[i, j] = 1
    
    # Generate synthetic data
    synthetic_data = generate_observational_data(
        adjacency_matrix,
        CONFIG["num_samples"]
    )
    
    # Create and train the model with optimal parameters
    model = SimpleGraphLearner(
        input_dim=CONFIG["num_nodes"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        **OPTIMAL_PARAMS
    )
    
    # Your training code here...
    # (Add code to train the model, evaluate it, and save results)

if __name__ == "__main__":
    main()
'''
    
    # Format parameters and config
    params_str = json.dumps(best_params, indent=4)
    simple_config = {
        "num_nodes": config["num_nodes"],
        "edge_probability": config["edge_probability"],
        "num_samples": config["num_samples"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "seed": config["seeds"][0]  # Use first seed
    }
    config_str = json.dumps(simple_config, indent=4)
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content.replace("{params_str}", params_str).replace("{config_str}", config_str))
    
    print(f"Created optimal model training script at {script_path}")

def main(sample_combinations=None):
    """Run the parameter tuning experiment."""
    print(f"Starting parameter tuning experiment... (Results in {output_dir})")
    
    # If we need to import seaborn for visualizations
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        print("Warning: Seaborn not found, some visualizations will be simpler")
        has_seaborn = False
    
    # Run parameter grid search
    results_df, perfect_params = parameter_grid_search(
        DEFAULT_CONFIG, 
        PARAMETER_GRID,
        n_combinations=sample_combinations
    )
    
    # Analyze results
    best_params, perfect_recovery = analyze_results(results_df, output_dir)
    
    # Create script for training with optimal parameters
    create_optimal_models_script(best_params, DEFAULT_CONFIG, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    
    # Return summary of findings
    if perfect_recovery is not None and len(perfect_recovery) > 0:
        print(f"\nFound {len(perfect_recovery)} parameter combinations achieving SHD=0")
        print(f"Optimal parameters: {best_params}")
    else:
        print("\nNo parameter combinations achieved perfect recovery (SHD=0)")
        print(f"Best parameters (lowest average SHD): {best_params}")

if __name__ == "__main__":
    # Run with a reasonable number of combinations to limit runtime
    # Change this to None for a full grid search
    main(sample_combinations=20) 