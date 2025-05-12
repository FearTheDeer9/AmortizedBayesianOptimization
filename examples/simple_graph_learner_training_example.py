"""
Example demonstrating the training of a SimpleGraphLearner model using the training utilities.

This example shows how to:
1. Generate a random DAG and observational data
2. Train a SimpleGraphLearner model
3. Evaluate the learned graph against the ground truth
4. Visualize the results
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from causal_meta.structure_learning import (
    RandomDAGGenerator,
    LinearSCMGenerator,
    SimpleGraphLearner,
    train_simple_graph_learner,
    evaluate_graph,
    calculate_structural_hamming_distance,
    SimpleGraphLearnerTrainer
)
from causal_meta.structure_learning.data_utils import (
    generate_observational_data,
    generate_interventional_data,
    generate_random_intervention_data,
    create_intervention_mask,
    convert_to_tensor
)

# Create output directory for saving results
output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join(output_dir, f"training_example_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    # Configuration
    num_nodes = 10
    edge_probability = 0.3
    num_samples = 1000
    noise_scale = 0.1
    hidden_dim = 64
    num_layers = 2
    batch_size = 32
    epochs = 100
    lr = 0.001
    
    # 1. Generate a random DAG
    print("Generating random DAG...")
    adj_matrix = RandomDAGGenerator.generate_random_dag(
        num_nodes=num_nodes,
        edge_probability=edge_probability
    )
    
    # 2. Create a linear SCM
    print("Creating linear SCM...")
    linear_scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=noise_scale
    )
    
    # 3. Generate observational data
    print("Generating observational data...")
    obs_data = generate_observational_data(
        scm=linear_scm,
        n_samples=num_samples
    )
    
    # 4. Generate some interventional data for testing
    print("Generating interventional data for testing...")
    # Generate multiple interventions for testing
    int_data_list = []
    int_masks = []
    intervened_nodes = []
    
    # Generate 3 different interventions
    for i in range(3):
        # Generate a random intervention
        int_data, node, value = generate_random_intervention_data(
            scm=linear_scm,
            n_samples=100
        )
        int_data_list.append(int_data)
        intervened_nodes.append(node)
    
    # Create a mask for the interventions
    for data in int_data_list:
        mask = create_intervention_mask(data, intervened_nodes)
        int_masks.append(mask)
    
    # Combine all interventional data
    int_data_combined = pd.concat(int_data_list) if int_data_list else None
    int_masks_combined = np.vstack(int_masks) if int_masks else None
    
    # 5. Convert data to tensors
    obs_tensor = convert_to_tensor(obs_data)
    int_tensor = convert_to_tensor(int_data_combined) if int_data_combined is not None else None
    int_mask_tensor = torch.tensor(int_masks_combined, dtype=torch.float32) if int_masks_combined is not None else None
    
    # Create adjacency tensor with requires_grad=True
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).requires_grad_(True)
    
    # 6. Split data into train/validation/test sets
    train_size = int(0.7 * len(obs_tensor))
    val_size = int(0.15 * len(obs_tensor))
    test_size = len(obs_tensor) - train_size - val_size
    
    indices = torch.randperm(len(obs_tensor))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data = obs_tensor[train_indices]
    val_data = obs_tensor[val_indices]
    test_data = obs_tensor[test_indices]
    
    # 7. Train the model using the high-level function
    print("\nTraining model with high-level function...")
    model, history = train_simple_graph_learner(
        train_data=train_data,
        val_data=val_data,
        true_adj=adj_tensor,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        early_stopping_patience=10,
        verbose=True
    )
    
    # Print the keys in the history dictionary
    print("\nHistory keys:", list(history.keys()))
    
    # 8. Evaluate the model on test data
    print("\nEvaluating model on test data...")
    with torch.no_grad():
        pred_adj_probs = model(test_data)
        pred_adj = model.threshold_edge_probabilities(pred_adj_probs)
    
    metrics = evaluate_graph(pred_adj.cpu().numpy(), adj_matrix)
    
    print("\nTest Metrics:")
    print(f"SHD: {metrics['shd']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 9. Evaluate on interventional data
    if int_tensor is not None and int_mask_tensor is not None:
        print("\nEvaluating model on interventional data...")
        with torch.no_grad():
            int_pred_adj_probs = model(int_tensor, int_mask_tensor)
            int_pred_adj = model.threshold_edge_probabilities(int_pred_adj_probs)
        
        int_metrics = evaluate_graph(int_pred_adj.cpu().numpy(), adj_matrix)
        
        print("\nInterventional Data Metrics:")
        print(f"SHD: {int_metrics['shd']}")
        print(f"Accuracy: {int_metrics['accuracy']:.4f}")
        print(f"Precision: {int_metrics['precision']:.4f}")
        print(f"Recall: {int_metrics['recall']:.4f}")
        print(f"F1 Score: {int_metrics['f1']:.4f}")
    
    # 10. Visualize training history
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    # Extract accuracy from metrics
    train_accuracy = []
    val_accuracy = []
    
    for metrics in history['train_metrics']:
        if 'accuracy' in metrics:
            train_accuracy.append(metrics['accuracy'])
    
    if 'val_metrics' in history:
        for metrics in history['val_metrics']:
            if 'accuracy' in metrics:
                val_accuracy.append(metrics['accuracy'])
    
    plt.plot(train_accuracy, label='Train Accuracy')
    if val_accuracy:
        plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "training_history.png"))
    
    # 11. Visualize ground truth and predicted graphs
    print("\nPlotting ground truth and predicted graphs...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(adj_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Ground Truth Graph')
    plt.xlabel('To Node')
    plt.ylabel('From Node')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_adj.cpu().numpy(), cmap='Blues')
    plt.colorbar()
    plt.title('Predicted Graph')
    plt.xlabel('To Node')
    plt.ylabel('From Node')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "graph_comparison.png"))
    
    # 12. Save the model
    print("\nSaving model...")
    trainer = SimpleGraphLearnerTrainer(model=model, lr=lr)
    trainer.history = history
    trainer.save(os.path.join(result_dir, "simple_graph_learner.pt"))
    
    print(f"\nResults saved to: {result_dir}")

if __name__ == "__main__":
    main() 