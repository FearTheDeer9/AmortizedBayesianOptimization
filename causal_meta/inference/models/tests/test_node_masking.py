import unittest
import torch
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.graph_utils import mask_nodes


class TestNodeMasking(unittest.TestCase):
    """Test cases for node masking functionality."""

    def test_node_masking(self):
        """Test masking of nodes in a graph."""
        # Create test data
        x = torch.randn(5, 3)  # 5 nodes, 3 features each
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        # Create a graph
        graph = Data(x=x, edge_index=edge_index)

        # Create mask (mask nodes 1 and 3)
        # Fix: Use boolean tensor for mask
        mask = torch.zeros(5, dtype=torch.bool)
        mask[1] = True
        mask[3] = True

        # Apply masking
        masked_graph = mask_nodes(graph, mask)

        # Verify masking
        # Should have 3 nodes left
        self.assertEqual(masked_graph.x.shape[0], 3)
        # Node 0 should be unchanged
        self.assertTrue(torch.allclose(masked_graph.x[0], x[0]))
        # Node 1 in masked graph should be node 2 from original
        self.assertTrue(torch.allclose(masked_graph.x[1], x[2]))
        # Node 2 in masked graph should be node 4 from original
        self.assertTrue(torch.allclose(masked_graph.x[2], x[4]))

    def test_batch_node_masking(self):
        """Test masking of nodes in a batch of graphs."""
        # Create two graphs
        x1 = torch.randn(3, 2)  # 3 nodes, 2 features each
        edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        graph1 = Data(x=x1, edge_index=edge_index1)

        x2 = torch.randn(4, 2)  # 4 nodes, 2 features each
        edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        graph2 = Data(x=x2, edge_index=edge_index2)

        # Create batch
        batch = Batch.from_data_list([graph1, graph2])

        # Create mask (mask node 1 in graph 1 and nodes 0 and 2 in graph 2)
        # Fix: Use boolean tensor for mask
        mask = torch.zeros(7, dtype=torch.bool)
        mask[1] = True  # Mask node 1 in graph 1
        mask[3] = True  # Mask node 0 in graph 2 (index 3 in batch)
        mask[5] = True  # Mask node 2 in graph 2 (index 5 in batch)

        # Apply masking
        masked_batch = mask_nodes(batch, mask)

        # Verify masking
        # Should have 4 nodes left
        self.assertEqual(masked_batch.x.shape[0], 4)

        # Check which nodes remain
        batch_idx = masked_batch.batch
        self.assertEqual(sum(batch_idx == 0).item(), 2)  # 2 nodes from graph 1
        self.assertEqual(sum(batch_idx == 1).item(), 2)  # 2 nodes from graph 2


if __name__ == "__main__":
    unittest.main()
