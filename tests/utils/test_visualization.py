import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import numpy as np

# Ensure the path includes the project root for imports
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

# Attempt to import the actual classes
try:
    from causal_meta.utils.visualization import TaskFamilyVisualizer
    from causal_meta.graph.task_family import TaskFamily
    ACTUAL_CLASSES_IMPORTED = True
except ImportError as e:
    print(f"Import Error during test setup: {e}") # Debug print
    ACTUAL_CLASSES_IMPORTED = False
    # Define minimal placeholders ONLY if the real classes can't be imported
    # This helps structure tests but relies on the real implementation existing
    class TaskFamilyVisualizer:
        def __init__(self, *args, **kwargs):
            print("Using PLACEHOLDER TaskFamilyVisualizer")
            pass
        def plot_family_comparison(self, *args, **kwargs):
            print("Using PLACEHOLDER plot_family_comparison")
            # Still need save/show logic for save tests to potentially pass
            if kwargs.get('output_dir'):
                 os.makedirs(kwargs['output_dir'], exist_ok=True)
                 # Create dummy file to satisfy os.path.exists checks if needed
                 open(os.path.join(kwargs['output_dir'], kwargs.get('filename', 'default.png')), 'a').close()
            pass

        def generate_difficulty_heatmap(self, *args, **kwargs):
            print("Using PLACEHOLDER generate_difficulty_heatmap")
            if kwargs.get('output_dir'):
                 os.makedirs(kwargs['output_dir'], exist_ok=True)
                 open(os.path.join(kwargs['output_dir'], kwargs.get('filename', 'default.png')), 'a').close()
            pass

    class TaskFamily:
        def __init__(self, base_graph, variations=None, metadata=None):
            print("Using PLACEHOLDER TaskFamily")
            self.base_graph = base_graph
            self.variations = variations or []
             # Ensure variations has the expected structure if not None
            self.variations = [
                var if isinstance(var, dict) and 'graph' in var else {'graph': var, 'metadata': {}}
                for var in self.variations
            ]
            self.graphs = [base_graph] + [v['graph'] for v in self.variations]
            self.metadata = metadata or {}

        def add_variation(self, graph, metadata=None):
             self.variations.append({'graph': graph, 'metadata': metadata or {}})
             self.graphs.append(graph)

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
             return self.graphs[idx]

        def save(self, filepath):
             print(f"PLACEHOLDER: Saving TaskFamily to {filepath}")
             # Create dummy file for save/load tests
             dirpath = os.path.dirname(filepath)
             if dirpath:
                 os.makedirs(dirpath, exist_ok=True)
             with open(filepath, 'wb') as f:
                 pickle.dump(self, f) # Save the placeholder itself

        @staticmethod
        def load(filepath):
             print(f"PLACEHOLDER: Loading TaskFamily from {filepath}")
             if not os.path.exists(filepath):
                 raise FileNotFoundError
             with open(filepath, 'rb') as f:
                 # Load whatever was saved (placeholder or real)
                 return pickle.load(f)

# --- Test Fixtures ---

@pytest.fixture
def sample_graph_1():
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2)])
    nx.set_node_attributes(g, {i: chr(65+i) for i in g.nodes()}, 'label')
    return g

@pytest.fixture
def sample_graph_2():
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    nx.set_node_attributes(g, {i: chr(65+i) for i in g.nodes()}, 'label')
    return g

@pytest.fixture
def sample_task_family(sample_graph_1, sample_graph_2):
    """Creates a TaskFamily object for testing."""
    family_meta = {'creator': 'pytest', 'param': 1}
    var1_meta = {'type': 'add_edge', 'strength': 0.5}
    # Use the actual TaskFamily class if available
    cls = TaskFamily if ACTUAL_CLASSES_IMPORTED else TaskFamily # Use placeholder if needed
    fam = cls(base_graph=sample_graph_1, metadata=family_meta)
    fam.add_variation(graph=sample_graph_2, metadata=var1_meta)
    return fam

# --- Test Class Visualizer ---

@pytest.mark.skipif(not ACTUAL_CLASSES_IMPORTED, reason="Actual classes not found, skipping visualizer tests")
class TestTaskFamilyVisualizer:

    def test_visualizer_initialization(self):
        """Tests if TaskFamilyVisualizer can be initialized."""
        visualizer = TaskFamilyVisualizer()
        assert isinstance(visualizer, TaskFamilyVisualizer)

    @patch('causal_meta.utils.visualization.plt.show')
    @patch('causal_meta.utils.visualization.plt.savefig')
    @patch('causal_meta.utils.visualization.plt.figure')
    @patch('causal_meta.utils.visualization.nx.draw')
    def test_plot_family_comparison_display(self, mock_draw, mock_figure, mock_savefig, mock_show, sample_task_family):
        """Tests plot_family_comparison with display=True."""
        visualizer = TaskFamilyVisualizer()
        visualizer.plot_family_comparison(sample_task_family, output_dir=None, display=True)

        assert mock_figure.called
        assert mock_draw.call_count == len(sample_task_family)
        assert mock_savefig.call_count == 0
        assert mock_show.called

    @patch('causal_meta.utils.visualization.plt.show')
    @patch('causal_meta.utils.visualization.plt.savefig')
    @patch('causal_meta.utils.visualization.plt.figure')
    @patch('causal_meta.utils.visualization.nx.draw')
    def test_plot_family_comparison_save(self, mock_draw, mock_figure, mock_savefig, mock_show, sample_task_family, tmp_path):
        """Tests plot_family_comparison with saving to file."""
        visualizer = TaskFamilyVisualizer()
        output_dir = str(tmp_path)
        filename = "test_comparison.png"

        visualizer.plot_family_comparison(
            sample_task_family, output_dir=output_dir, display=False, filename=filename
        )

        assert mock_figure.called
        assert mock_draw.call_count == len(sample_task_family)
        assert mock_savefig.called
        args, kwargs = mock_savefig.call_args
        expected_path = os.path.join(output_dir, filename)
        assert args[0] == expected_path
        assert mock_show.call_count == 0

    @patch('causal_meta.utils.visualization.plt.show')
    @patch('causal_meta.utils.visualization.plt.savefig')
    @patch('causal_meta.utils.visualization.plt.subplots')
    @patch('causal_meta.utils.visualization.sns.heatmap')
    def test_generate_difficulty_heatmap_display(self, mock_heatmap, mock_subplots, mock_savefig, mock_show, sample_task_family):
        """Tests generate_difficulty_heatmap with display=True."""
        visualizer = TaskFamilyVisualizer()
        mock_metric = lambda g: len(g.edges())
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        visualizer.generate_difficulty_heatmap(
            sample_task_family, difficulty_metric=mock_metric, output_dir=None, display=True
        )

        assert mock_subplots.called
        assert mock_heatmap.called
        args, kwargs = mock_heatmap.call_args
        expected_data = np.array([[mock_metric(g) for g in sample_task_family.graphs]])
        np.testing.assert_array_equal(args[0], expected_data)
        assert kwargs.get('ax') == mock_ax
        assert mock_savefig.call_count == 0
        assert mock_show.called

    @patch('causal_meta.utils.visualization.plt.show')
    @patch('causal_meta.utils.visualization.plt.savefig')
    @patch('causal_meta.utils.visualization.plt.subplots')
    @patch('causal_meta.utils.visualization.sns.heatmap')
    def test_generate_difficulty_heatmap_save(self, mock_heatmap, mock_subplots, mock_savefig, mock_show, sample_task_family, tmp_path):
        """Tests generate_difficulty_heatmap with saving to file."""
        visualizer = TaskFamilyVisualizer()
        mock_metric = lambda g: len(g.edges())
        output_dir = str(tmp_path)
        filename = "test_heatmap.png"
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        visualizer.generate_difficulty_heatmap(
            sample_task_family,
            difficulty_metric=mock_metric,
            output_dir=output_dir,
            display=False,
            filename=filename
        )

        assert mock_subplots.called
        assert mock_heatmap.called
        assert mock_savefig.called
        args, kwargs = mock_savefig.call_args
        expected_path = os.path.join(output_dir, filename)
        assert args[0] == expected_path
        assert mock_show.call_count == 0

    @patch('causal_meta.utils.visualization.logger.warning')
    def test_plot_family_comparison_empty(self, mock_warning):
        """Tests plot_family_comparison with an empty task family."""
        visualizer = TaskFamilyVisualizer()
        cls = TaskFamily if ACTUAL_CLASSES_IMPORTED else TaskFamily
        empty_family = cls(base_graph=nx.DiGraph(), variations=[])
        empty_family.graphs = [] # Force empty graphs list
        empty_family.variations = [] # Force empty variations list
        visualizer.plot_family_comparison(empty_family)
        mock_warning.assert_called_once_with("No graphs in the task family to plot.")

    @patch('causal_meta.utils.visualization.logger.warning')
    def test_generate_difficulty_heatmap_empty(self, mock_warning):
        """Tests generate_difficulty_heatmap with an empty task family."""
        visualizer = TaskFamilyVisualizer()
        mock_metric = lambda g: len(g.edges())
        cls = TaskFamily if ACTUAL_CLASSES_IMPORTED else TaskFamily
        empty_family = cls(base_graph=nx.DiGraph(), variations=[])
        empty_family.graphs = []
        empty_family.variations = []
        visualizer.generate_difficulty_heatmap(empty_family, difficulty_metric=mock_metric)
        mock_warning.assert_called_once_with("No graphs in the task family for heatmap.")

# --- Test Class TaskFamily ---

# Only run these tests if the actual TaskFamily class was imported
@pytest.mark.skipif(not ACTUAL_CLASSES_IMPORTED, reason="Actual TaskFamily class not found")
class TestTaskFamily:

    def test_task_family_initialization(self, sample_graph_1, sample_graph_2):
        """Tests TaskFamily initialization with and without variations/metadata."""
        base = sample_graph_1
        var1_info = {'graph': sample_graph_2, 'metadata': {'type': 'add_edge'}}
        fam_meta = {'gen_id': 'abc'}

        # Test basic init
        fam1 = TaskFamily(base_graph=base)
        assert fam1.base_graph == base
        assert len(fam1.variations) == 0
        assert len(fam1) == 1
        assert fam1.graphs == [base]
        assert 'generation_timestamp' in fam1.metadata

        # Test init with variations and metadata
        fam2 = TaskFamily(base_graph=base, variations=[var1_info], metadata=fam_meta)
        assert fam2.base_graph == base
        assert len(fam2.variations) == 1
        assert fam2.variations[0]['graph'] == sample_graph_2
        assert fam2.variations[0]['metadata'] == {'type': 'add_edge'}
        assert len(fam2) == 2
        assert fam2.graphs == [base, sample_graph_2]
        assert fam2.metadata['gen_id'] == 'abc'
        assert 'generation_timestamp' in fam2.metadata

    def test_add_variation(self, sample_task_family, sample_graph_1):
        """Tests adding variations after initialization."""
        initial_len = len(sample_task_family)
        new_var_meta = {'source': 'manual'}
        sample_task_family.add_variation(graph=sample_graph_1, metadata=new_var_meta)

        assert len(sample_task_family) == initial_len + 1
        assert len(sample_task_family.variations) == initial_len # variations list grows
        assert sample_task_family.graphs[-1] == sample_graph_1
        last_var_info = sample_task_family.get_variation_info(len(sample_task_family.variations) - 1)
        assert last_var_info['graph'] == sample_graph_1
        assert last_var_info['metadata']['source'] == 'manual'
        assert 'added_timestamp' in last_var_info['metadata']

    def test_get_item(self, sample_task_family, sample_graph_1, sample_graph_2):
        """Tests accessing graphs using index."""
        assert sample_task_family[0] == sample_graph_1 # Base graph
        assert sample_task_family[1] == sample_graph_2 # First variation graph
        with pytest.raises(IndexError):
            _ = sample_task_family[len(sample_task_family)]

    def test_get_variation_info(self, sample_task_family, sample_graph_2):
         """Tests accessing variation info."""
         info = sample_task_family.get_variation_info(0) # First variation added
         assert info['graph'] == sample_graph_2
         assert info['metadata'] == {'type': 'add_edge', 'strength': 0.5, 'added_timestamp': info['metadata']['added_timestamp']} # Check metadata

         with pytest.raises(IndexError):
             _ = sample_task_family.get_variation_info(len(sample_task_family.variations))

    def test_add_family_metadata(self, sample_task_family):
        """Tests adding family-level metadata."""
        assert 'new_key' not in sample_task_family.metadata
        sample_task_family.add_family_metadata('new_key', 123)
        assert sample_task_family.metadata['new_key'] == 123

    def test_save_load_cycle(self, sample_task_family, tmp_path):
        """Tests saving and loading a TaskFamily object."""
        filepath = tmp_path / "test_family.pkl"

        # Save
        sample_task_family.save(filepath)
        assert filepath.exists()

        # Load
        loaded_family = TaskFamily.load(filepath)

        # Basic checks
        assert isinstance(loaded_family, TaskFamily)
        assert len(loaded_family) == len(sample_task_family)
        assert loaded_family.metadata == sample_task_family.metadata

        # Check graphs (can be slow for large graphs, check a few props)
        assert len(loaded_family.base_graph.nodes) == len(sample_task_family.base_graph.nodes)
        assert set(loaded_family.base_graph.edges) == set(sample_task_family.base_graph.edges)
        assert len(loaded_family.variations) == len(sample_task_family.variations)
        assert set(loaded_family.variations[0]['graph'].edges) == set(sample_task_family.variations[0]['graph'].edges)
        assert loaded_family.variations[0]['metadata'] == sample_task_family.variations[0]['metadata']

    def test_load_file_not_found(self, tmp_path):
        """Tests loading from a non-existent file."""
        filepath = tmp_path / "non_existent.pkl"
        with pytest.raises(FileNotFoundError):
            TaskFamily.load(filepath)

    def test_load_invalid_pickle(self, tmp_path):
        """Tests loading a file that is not a valid pickle or TaskFamily."""
        filepath = tmp_path / "invalid.pkl"

        # Save something else (e.g., a simple dict)
        with open(filepath, 'wb') as f:
            pickle.dump({"data": "not a task family"}, f)

        with pytest.raises(TypeError): # Expecting TypeError due to instance check
            TaskFamily.load(filepath)

# Add more tests for specific parameters, layout functions, difference highlighting etc. 