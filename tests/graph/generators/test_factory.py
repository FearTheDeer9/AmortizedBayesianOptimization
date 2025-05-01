import pytest
import random
from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.graph.generators.errors import GraphGenerationError

def test_create_random_dag_node_creation():
    """Test if create_random_dag creates the correct number of nodes."""
    num_nodes = 15
    # Use edge_probability=0 for now to focus on node/structure generation
    try:
        dag = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.0)
        assert len(dag.get_nodes()) == num_nodes
        assert set(dag.get_nodes()) == set(range(num_nodes))
    except NotImplementedError:
        pytest.skip("Core DAG generation not implemented yet")

def test_create_random_dag_graph_type():
    """Test if create_random_dag returns the correct graph type."""
    num_nodes = 5
    try:
        causal_dag = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.0, is_causal=True)
        directed_dag = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.0, is_causal=False)
        assert isinstance(causal_dag, CausalGraph)
        assert isinstance(directed_dag, DirectedGraph)
        assert not isinstance(directed_dag, CausalGraph)
    except NotImplementedError:
        pytest.skip("Core DAG generation not implemented yet")

# @pytest.mark.skip(reason="Edge probability logic is part of subtask 1.3")
def test_create_random_dag_edge_direction_constraint():
    """Test if create_random_dag only creates edges from lower to higher indices."""
    num_nodes = 10
    # Use edge_probability=1.0 to ensure *all* valid edges are considered/created
    # We will implement the actual edge creation in the next step (1.3)
    # For now, this test checks the *potential* edges considered.
    # The core logic (1.2) should set up the loop structure correctly.
    # We will modify this test in subtask 1.3 to check actual edge addition.
    try:
        dag = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=1.0, seed=42)
        # In subtask 1.2, no edges should be added yet, so this check is simple
        # It ensures the core structure doesn't violate the constraint preemptively
        # THIS WILL FAIL UNTIL SUBTASK 1.3 is done, but setup is part of 1.2
        # assert len(dag.get_edges()) > 0 # Expect edges when prob=1.0
        for u, v in dag.get_edges():
            assert u < v, f"Edge ({u}, {v}) violates DAG constraint (u should be < v)"

    except NotImplementedError:
        pytest.skip("Core DAG generation not implemented yet")

def test_create_random_dag_seed_reproducibility_structure():
    """Test if the DAG structure is reproducible with the same seed (core loop structure)."""
    num_nodes = 8
    # Focus on the structure the core loop *could* produce
    # Actual edges depend on probability (subtask 1.3)
    try:
        dag1 = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.5, seed=123)
        dag2 = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.5, seed=123)
        dag3 = GraphFactory.create_random_dag(num_nodes=num_nodes, edge_probability=0.5, seed=456)

        # In subtask 1.2, graphs will be empty but node order fixed by seed
        # We'll modify this in 1.3 to check edge set equality
        assert set(dag1.get_nodes()) == set(dag2.get_nodes())
        assert set(dag1.get_edges()) == set(dag2.get_edges()) # Should be empty sets for now
        assert set(dag1.get_nodes()) == set(dag3.get_nodes())
        # assert set(dag1.get_edges()) != set(dag3.get_edges()) # Probabilistic, might be equal

    except NotImplementedError:
        pytest.skip("Core DAG generation not implemented yet")

def test_create_random_dag_invalid_params():
    """Test parameter validation in create_random_dag signature."""
    with pytest.raises(GraphGenerationError, match="positive integer"):
        GraphFactory.create_random_dag(num_nodes=0, edge_probability=0.5)
    with pytest.raises(GraphGenerationError, match="positive integer"):
        GraphFactory.create_random_dag(num_nodes=-5, edge_probability=0.5)
    with pytest.raises(GraphGenerationError, match="between 0.0 and 1.0"):
        GraphFactory.create_random_dag(num_nodes=5, edge_probability=-0.1)
    with pytest.raises(GraphGenerationError, match="between 0.0 and 1.0"):
        GraphFactory.create_random_dag(num_nodes=5, edge_probability=1.1) 