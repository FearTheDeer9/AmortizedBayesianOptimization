import torch
from causal_meta.inference.models.gat_decoder import GATDecoder

# Test batched inputs
latent_dim = 32
hidden_dim = 64
output_dim = 16
num_nodes = 5
batch_size = 3

# Create decoder
decoder = GATDecoder(
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_nodes=num_nodes
)

# Create batched latent representation
latent = torch.randn(batch_size, latent_dim)

print("Running tests for GATDecoder with batched inputs...")

# Test node prediction
try:
    node_features = decoder.predict_node_features(latent)
    print(f'✓ Node prediction test passed - Shape: {node_features.shape}')
    assert node_features.shape == (batch_size, num_nodes, hidden_dim)
except Exception as e:
    print(f'✗ Node prediction test failed: {e}')

# Create batch assignment tensor for our nodes
batch_tensor = []
for i in range(batch_size):
    batch_tensor.extend([i] * num_nodes)
batch_tensor = torch.tensor(batch_tensor)

# Test edge prediction with batch tensor
try:
    # Reshape node features to match what the predict_edges function expects
    flattened_features = node_features.reshape(
        batch_size * num_nodes, hidden_dim)

    # Now call predict_edges with the batch tensor
    edge_index, edge_attr = decoder.predict_edges(
        flattened_features, batch=batch_tensor)

    print(f'✓ Edge prediction with batch tensor test passed')
    print(f'  Edge index shape: {edge_index.shape}')
    print(f'  Edge attr shape: {edge_attr.shape}')

    # Check if edge_index values are within the expected range
    assert edge_index.min() >= 0
    assert edge_index.max() < batch_size * num_nodes

    # Additional test: Extract individual graphs from batched results
    # To test our approach to handling batches in predict_edges
    all_graph_edges = []
    all_edge_attrs = []
    for i in range(batch_size):
        start_idx = i * num_nodes
        end_idx = (i + 1) * num_nodes

        mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & \
               (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)

        graph_edges = edge_index[:, mask] - start_idx
        graph_attrs = edge_attr[mask]

        all_graph_edges.append(graph_edges)
        all_edge_attrs.append(graph_attrs)

        print(f'  Graph {i+1} has {graph_edges.shape[1]} edges')

except Exception as e:
    print(f'✗ Edge prediction with batch tensor test failed: {e}')

# Test the forward method with batching
try:
    node_features, edge_index, edge_attr = decoder.forward(
        latent, batch=batch_tensor)

    print(f'✓ Forward pass with batching test passed')
    print(f'  Node features shape: {node_features.shape}')
    print(f'  Edge index shape: {edge_index.shape}')
    print(f'  Edge attr shape: {edge_attr.shape}')

    # Check dimensions
    assert node_features.shape[0] == batch_size * num_nodes
    assert edge_index.shape[0] == 2

except Exception as e:
    print(f'✗ Forward pass with batching test failed: {e}')

print("\nTests complete!")
