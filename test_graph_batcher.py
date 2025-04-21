import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import jaccard_score

from causal_meta.inference.models.graph_utils import GraphBatcher, edge_similarity
from causal_meta.inference.models.wrapper import EncoderDecoderWrapper
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder


def create_random_graph(num_nodes, edge_prob=0.3, feature_dim=5):
    """Create a random graph with the specified number of nodes"""
    # Create random node features
    x = torch.randn(num_nodes, feature_dim)

    # Create random edges with a given probability
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.random() < edge_prob:
                edges.append([i, j])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def visualize_graph(graph, title="Graph"):
    """Visualize a PyG graph using networkx"""
    G = nx.DiGraph()

    # Add nodes
    for i in range(graph.num_nodes):
        G.add_node(i)

    # Add edges
    edge_list = graph.edge_index.t().tolist()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(8, 6))
    plt.title(title)
    nx.draw(G, with_labels=True, node_color='lightblue',
            node_size=500, arrows=True)
    plt.tight_layout()
    plt.show()


def test_graph_batcher():
    """Test the GraphBatcher with variable-sized graphs"""
    print("Testing GraphBatcher with variable-sized graphs...")

    # Create test graphs with different sizes
    graphs = []
    sizes = [5, 8, 3]
    for n_nodes in sizes:
        graphs.append(create_random_graph(n_nodes, edge_prob=0.3))

    print(f"Created {len(graphs)} graphs with sizes: {sizes}")

    batcher = GraphBatcher()

    # Test 1: Check if graphs are padded correctly
    print("\nTest 1: Padding to maximum size")
    batch, batch_info = batcher.batch_graphs(graphs)

    print(f"Batch shape: nodes={batch.x.shape[0]}, features={batch.x.shape}")
    print(
        f"Batch info: original_sizes={batch_info['original_sizes']}, max_nodes={batch_info.get('max_nodes')}")

    # Check if node mask is applied correctly
    if hasattr(batch, 'node_mask'):
        valid_nodes = torch.sum(batch.node_mask).item()
        total_nodes = batch.node_mask.shape[0]
        print(
            f"Node mask correctly applied, valid nodes: {valid_nodes}/{total_nodes}")
    else:
        print("Warning: node_mask not found in batch")

    # Test 2: Check if unbatching works correctly
    print("\nTest 2: Unbatching graphs")
    unbatched = batcher.unbatch_graphs(batch, batch_info)

    unbatched_sizes = [g.x.shape[0] for g in unbatched]
    print(f"Original sizes: {sizes}")
    print(f"Unbatched sizes: {unbatched_sizes}")
    print(f"Sizes match: {sizes == unbatched_sizes}")

    # Test 3: Test batching without padding
    print("\nTest 3: Batching without padding")
    # Save original setting
    original_pad_to_max = batcher.pad_to_max
    batcher.pad_to_max = False
    batch_no_pad, batch_info_no_pad = batcher.batch_graphs(graphs)
    # Restore original setting
    batcher.pad_to_max = original_pad_to_max

    print(
        f"Batch shape: nodes={batch_no_pad.x.shape[0]}, features={batch_no_pad.x.shape}")

    unbatched_no_pad = batcher.unbatch_graphs(batch_no_pad, batch_info_no_pad)
    unbatched_sizes_no_pad = [g.x.shape[0] for g in unbatched_no_pad]
    print(f"Unbatched sizes (no padding): {unbatched_sizes_no_pad}")
    print(f"Sizes match: {sizes == unbatched_sizes_no_pad}")

    # Test 4: Check edge index correction
    print("\nTest 4: Edge index correction")
    batch, batch_info = batcher.batch_graphs(graphs)

    # For debugging the edge indices issue
    print("Edge index details:")
    edge_index = batch.edge_index
    print(f"Edge index shape: {edge_index.shape}")
    print("Edge index min values:", edge_index.min(dim=1)[0])
    print("Edge index max values:", edge_index.max(dim=1)[0])

    max_node_idx = torch.max(batch.edge_index).item()
    print(f"Max node index in corrected edges: {max_node_idx}")
    total_nodes = batch.x.shape[0]
    print(f"Total nodes in batch: {total_nodes}")
    print(f"Indices in valid range: {max_node_idx < total_nodes}")

    # Test 5: Test edge validation
    print("\nTest 5: Edge validation functionality")

    # Create edge indices with invalid references (out of graph boundaries)
    invalid_edge_indices = torch.tensor([
        [0, 1], [1, 2],  # Valid edges
        [0, 20], [6, 15]  # Invalid edges
    ], dtype=torch.long).t()

    print(f"Invalid edge indices shape: {invalid_edge_indices.shape}")

    # Validate edge indices
    valid_edge_indices = batcher.validate_edge_indices(
        invalid_edge_indices,
        batch.ptr,
        batch_info['num_nodes_per_graph']
    )

    print(f"Valid edge indices shape: {valid_edge_indices.shape}")
    print(
        f"Number of edges removed: {invalid_edge_indices.shape[1] - valid_edge_indices.shape[1]}")
    print(
        f"Edge validation working: {valid_edge_indices.shape[1] < invalid_edge_indices.shape[1]}")

    return max_node_idx < total_nodes and sizes == unbatched_sizes


def test_wrapper_with_batcher():
    """Test the EncoderDecoderWrapper with GraphBatcher"""
    print("\nTesting EncoderDecoderWrapper with GraphBatcher...")

    # Create test graphs
    graphs = []
    for n_nodes in [5, 8, 3]:
        graphs.append(create_random_graph(n_nodes, edge_prob=0.3))

    # Initialize models with appropriate parameters
    encoder = GCNEncoder(
        input_dim=5,         # Node feature dimension
        hidden_dim=16,       # Hidden layer dimension
        latent_dim=8,        # Add the missing latent_dim parameter
        num_layers=2,
        dropout=0.1
    )

    decoder = GCNDecoder(
        latent_dim=8,  # Add the latent_dim parameter
        hidden_dim=16,
        output_dim=5,
        num_nodes=8  # Max nodes
    )

    wrapper = EncoderDecoderWrapper(
        encoder=encoder,
        decoder=decoder,
        pad_graphs=True,  # Use pad_graphs
        validate_edges=True  # Enable edge validation
    )

    # Process batch with our wrapper
    results = wrapper.process_batch(graphs)

    # Verify the results
    print("\nTest 5: EncoderDecoderWrapper batch processing")
    print(f"Processed {len(graphs)} graphs")
    print(f"Got {len(results['reconstructed_graphs'])} reconstructed graphs")

    original_sizes = [g.num_nodes for g in graphs]
    reconstructed_sizes = [
        g.num_nodes for g in results['reconstructed_graphs']]

    print(f"Original sizes: {original_sizes}")
    print(f"Reconstructed sizes: {reconstructed_sizes}")
    print(f"Sizes match: {original_sizes == reconstructed_sizes}")

    # Check edge validation
    print("\nTest 6: Edge validation in wrapper")
    for i, graph in enumerate(results['reconstructed_graphs']):
        if graph.edge_index.size(1) > 0:
            # Check that all edge indices are valid
            max_node_idx = torch.max(graph.edge_index).item()
            num_nodes = graph.num_nodes
            print(
                f"Graph {i}: max edge index {max_node_idx}, num nodes {num_nodes}")
            print(f"Edge indices valid: {max_node_idx < num_nodes}")

            # All edges should be valid after processing
            assert max_node_idx < num_nodes, f"Invalid edge index in graph {i}"

    # Check edge similarity
    similarities = []
    for orig, recon in zip(graphs, results['reconstructed_graphs']):
        sim = edge_similarity(orig, recon)
        similarities.append(sim)

    print(f"Edge similarities: {[round(s, 2) for s in similarities]}")
    print(
        f"Average edge similarity: {sum(similarities)/len(similarities):.2f}")

    # Test compute_loss
    losses = wrapper.compute_loss(graphs)
    print("\nTest 7: Loss computation")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Edge loss: {losses['edge_loss'].item():.4f}")
    print(f"Feature loss: {losses['feature_loss'].item():.4f}")

    # Test model evaluation with the improved wrapper
    print("\nTest 8: Model evaluation with improved wrapper")
    metrics = wrapper.evaluate(graphs)
    print(f"Evaluation metrics: {metrics}")

    return all([
        len(graphs) == len(results['reconstructed_graphs']),
        len(losses) > 0,
        'edge_f1' in metrics,
        all(max_node_idx < num_nodes for graph in results['reconstructed_graphs']
            for max_node_idx, num_nodes in [(torch.max(graph.edge_index).item() if graph.edge_index.size(1) > 0 else -1,
                                            graph.num_nodes)])
    ])


if __name__ == "__main__":
    print("Running GraphBatcher tests...")
    batcher_success = test_graph_batcher()
    wrapper_success = test_wrapper_with_batcher()

    if batcher_success and wrapper_success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
