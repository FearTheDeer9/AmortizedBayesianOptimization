import torch
from torch_geometric.data import Data, Batch
from causal_meta.inference.models.wrapper import create_model

print("Testing EncoderDecoderWrapper with GATDecoder...")

# Create test data
input_dim = 4
hidden_dim = 16
latent_dim = 8
output_dim = input_dim
num_nodes = 5
batch_size = 2

# Create some sample graphs
graphs = []
for i in range(batch_size):
    # Create a simple graph
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3], [1, 0, 2, 3, 4]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)
    graphs.append(graph)

# Create a batch from the graphs
batch_data = Batch.from_data_list(graphs)

# Create the model with GAT architecture
try:
    model = create_model(
        architecture='gat',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_nodes=num_nodes,
        decoder_edge_prediction_method='inner_product'
    )
    print("✓ Model created successfully")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    exit(1)

# Test encoder function
try:
    latent = model.encoder(batch_data)
    print(f"✓ Encoder test passed - Latent shape: {latent.shape}")
    assert latent.shape == (batch_size, latent_dim)
except Exception as e:
    print(f"✗ Encoder test failed: {e}")

# Get batched node features directly
try:
    node_features = model.decoder.predict_node_features(latent)
    print(f"✓ Node prediction test passed - Shape: {node_features.shape}")

    # Create batch tensor for testing
    batch_tensor = []
    for i in range(batch_size):
        batch_tensor.extend([i] * num_nodes)
    batch_tensor = torch.tensor(batch_tensor)

    # Reshape node features as expected by the edge prediction function
    flattened_features = node_features.reshape(
        batch_size * num_nodes, hidden_dim)

    # Test edge prediction with batch tensor
    edge_index, edge_attr = model.decoder.predict_edges(
        flattened_features, batch=batch_tensor)
    print(
        f"✓ Edge prediction test passed - Edge index shape: {edge_index.shape}, Edge attr shape: {edge_attr.shape}")
except Exception as e:
    print(f"✗ Node/Edge prediction test failed: {e}")

print("\nTests complete!")
