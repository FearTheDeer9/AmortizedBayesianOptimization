import pytest
import logging
from unittest.mock import Mock, MagicMock

from causal_meta.graph.causal_graph import CausalGraph # Import real CausalGraph

# Assume the MetaCBO class is in this path
from causal_meta.meta_learning.meta_cbo import MetaCBO

# --- Fixtures ---

@pytest.fixture
def mock_task_repr():
    """Provides a mock TaskRepresentation model."""
    return Mock()

@pytest.fixture
def mock_maml_config():
    """Provides a mock MAML configuration dictionary."""
    return {"lr_inner": 0.01, "lr_outer": 0.001, "steps": 5}

@pytest.fixture
def mock_logger():
    """Provides a mock logger."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_maml_framework():
    """Provides a mock MAML framework object."""
    mock = Mock()
    initial_params = {"initial_meta_param": 0.0}
    updated_params = {"meta_param": 0.5} # Expected params after update
    mock.meta_parameters = initial_params # Start with initial

    # Configure return values for inner and outer loop updates
    mock.inner_loop_update.return_value = {"adapted_param": 1.0}

    # Make outer_loop_update modify the mock's state before returning metrics
    def outer_loop_side_effect(*args, **kwargs):
        mock.meta_parameters = updated_params # Modify the mock's internal state
        return {"meta_loss": 0.5} # Return metrics
    mock.outer_loop_update.side_effect = outer_loop_side_effect
    # mock.outer_loop_update.return_value = {"meta_loss": 0.5} # Original metrics return

    return mock

@pytest.fixture
def mock_valid_causal_graph():
    """Provides a simple, valid CausalGraph instance."""
    # Return an actual CausalGraph instance instead of a Mock
    graph = CausalGraph()
    graph.add_node('A') # Add at least one node
    # is_dag() should return True for an empty or single-node graph
    return graph

@pytest.fixture
def meta_cbo_instance(mock_task_repr, mock_maml_config, mock_logger, mock_maml_framework):
    """Provides a MetaCBO instance with mocked dependencies."""
    return MetaCBO(
        task_representation_model=mock_task_repr,
        maml_framework=mock_maml_framework, # Pass the mock MAML framework
        logger=mock_logger,
        random_seed=42
    )

# --- Test Cases for Subtask 19.1 ---

def test_meta_cbo_class_existence():
    """1. Test Class Existence: Verify the MetaCBO class can be imported."""
    # This test implicitly passes if the import above works
    from causal_meta.meta_learning.meta_cbo import MetaCBO
    assert MetaCBO is not None

def test_meta_cbo_initialization(meta_cbo_instance, mock_task_repr, mock_maml_config, mock_logger, mock_maml_framework):
    """2. Test Initialization: Verify attributes are set correctly."""
    assert meta_cbo_instance is not None
    assert meta_cbo_instance.task_representation_model is mock_task_repr
    assert meta_cbo_instance.maml_framework is mock_maml_framework # Check maml framework
    assert meta_cbo_instance.logger is mock_logger
    assert meta_cbo_instance.random_seed == 42
    # Check meta params are initialized from framework
    assert meta_cbo_instance._meta_parameters == mock_maml_framework.meta_parameters
    assert meta_cbo_instance._is_meta_trained is False
    mock_logger.info.assert_called_with("Set random seed to 42")

def test_meta_cbo_method_signatures(meta_cbo_instance):
    """3. Test Method Signatures: Verify core methods exist."""
    assert hasattr(meta_cbo_instance, 'meta_train')
    assert callable(meta_cbo_instance.meta_train)

    assert hasattr(meta_cbo_instance, 'adapt')
    assert callable(meta_cbo_instance.adapt)

    assert hasattr(meta_cbo_instance, 'evaluate')
    assert callable(meta_cbo_instance.evaluate)

    assert hasattr(meta_cbo_instance, 'optimize_interventions')
    assert callable(meta_cbo_instance.optimize_interventions)

def test_meta_cbo_placeholder_implementation(meta_cbo_instance, mock_valid_causal_graph):
    """4. Test Placeholder Implementation: Verify remaining methods raise NotImplementedError."""
    # Mock necessary inputs if needed for the methods
    # mock_task_family = [Mock()] # No longer testing meta_train placeholder
    mock_new_task = Mock()
    mock_new_task.graph = mock_valid_causal_graph # Assign valid graph for evaluate
    # mock_adaptation_data = Mock() # No longer testing adapt placeholder
    mock_test_data = Mock()
    mock_scm = Mock() # Example SCM

    # meta_train and adapt should no longer raise this error
    # with pytest.raises(NotImplementedError, match="Meta-training is not yet implemented."):
    #     meta_cbo_instance.meta_train(task_family=mock_task_family)

    # Need to set _is_meta_trained to True to bypass the check in adapt
    # meta_cbo_instance._is_meta_trained = True 
    # with pytest.raises(NotImplementedError, match="Adaptation is not yet implemented."):
    #     meta_cbo_instance.adapt(new_task=mock_new_task, adaptation_data=mock_adaptation_data)
    # meta_cbo_instance._is_meta_trained = False # Reset state

    # Test evaluate placeholder
    with pytest.raises(NotImplementedError, match="Evaluation is not yet implemented."):
        meta_cbo_instance.evaluate(task=mock_new_task, test_data=mock_test_data)

    # Assign graph from parameter
    mock_scm.graph = mock_valid_causal_graph

    # Test optimize_interventions placeholder
    with pytest.raises(NotImplementedError, match="Bayesian Optimization logic for intervention selection is not implemented."):
        meta_cbo_instance.optimize_interventions(task=mock_scm, budget=5, target_variable='Y')

def test_meta_cbo_logging_infrastructure(meta_cbo_instance, mock_logger):
    """5. Test Logging Infrastructure: Verify the logger attribute is set."""
    assert meta_cbo_instance.logger is mock_logger
    # Example: Check if logger is used in a method (e.g., init)
    mock_logger.info.assert_called() # Already called during init with seed

def test_meta_cbo_default_logger(mock_maml_framework):
    """Test that MetaCBO creates a default logger if none is provided."""
    instance = MetaCBO(
        task_representation_model=Mock(),
        maml_framework=mock_maml_framework, # Pass the mock MAML framework
        logger=None # Explicitly pass None
    )
    assert isinstance(instance.logger, logging.Logger)
    assert instance.logger.name == 'causal_meta.meta_learning.meta_cbo'
    assert instance.logger.level == logging.INFO
    assert len(instance.logger.handlers) > 0

# You might add more tests here as the class structure evolves 

# --- Test Cases for Subtask 19.2 / 19.3 --- # # Grouping tests that now depend on graph validation

def test_adapt_uses_task_representation(meta_cbo_instance, mock_task_repr, mock_maml_framework, mock_valid_causal_graph):
    """1. Test Task Embedding Usage in adapt."""
    # Make mock task provide a valid graph via .graph attribute
    mock_new_task = Mock(name="NewTask", graph=mock_valid_causal_graph)
    mock_adaptation_data = Mock(name="AdaptData")
    mock_task_repr.embed_task.return_value = "mock_embedding"

    meta_cbo_instance._is_meta_trained = True # Pretend it's trained

    # Call adapt
    meta_cbo_instance.adapt(new_task=mock_new_task, adaptation_data=mock_adaptation_data)

    # Verify embed_task was called
    mock_task_repr.embed_task.assert_called_once_with(mock_new_task, graph=mock_valid_causal_graph)

    # Verify embedding was passed to inner loop
    mock_maml_framework.inner_loop_update.assert_called_once()
    call_args, call_kwargs = mock_maml_framework.inner_loop_update.call_args
    assert call_kwargs.get('task_embedding') == "mock_embedding"

    # Reset state
    meta_cbo_instance._is_meta_trained = False

def test_meta_train_uses_task_representation(meta_cbo_instance, mock_task_repr, mock_maml_framework, mock_valid_causal_graph):
    """2. Test Task Embedding Usage in meta_train."""
    # Make mock tasks provide a valid graph
    mock_task1 = Mock(name="Task1", id=1, support_data="s1", query_data="q1", graph=mock_valid_causal_graph)
    mock_task2 = Mock(name="Task2", id=2, support_data="s2", query_data="q2", graph=mock_valid_causal_graph)
    mock_task_family = [mock_task1, mock_task2]
    mock_task_repr.embed_task.side_effect = ["embedding1", "embedding2"]

    # Call meta_train
    meta_cbo_instance.meta_train(task_family=mock_task_family)

    # Verify embed_task was called for each task
    assert mock_task_repr.embed_task.call_count == 2
    mock_task_repr.embed_task.assert_any_call(mock_task1, graph=mock_valid_causal_graph)
    mock_task_repr.embed_task.assert_any_call(mock_task2, graph=mock_valid_causal_graph)

    # Verify embeddings passed to outer loop
    mock_maml_framework.outer_loop_update.assert_called_once()
    call_args, call_kwargs = mock_maml_framework.outer_loop_update.call_args
    task_batch = call_kwargs.get('task_batch')
    assert len(task_batch) == 2
    assert task_batch[0]['embedding'] == "embedding1"
    assert task_batch[1]['embedding'] == "embedding2"

def test_adapt_calls_maml_inner_loop(meta_cbo_instance, mock_maml_framework, mock_valid_causal_graph):
    """3. Test MAML Inner Loop Integration in adapt."""
    # Make mock task provide a valid graph
    mock_new_task = Mock(name="NewTask", graph=mock_valid_causal_graph)
    mock_adaptation_data = Mock(name="AdaptData")
    initial_meta_params = meta_cbo_instance._meta_parameters
    mock_embedding = "mock_embedding_for_adapt"
    meta_cbo_instance.task_representation_model.embed_task.return_value = mock_embedding

    meta_cbo_instance._is_meta_trained = True # Pretend it's trained

    adapted_params = meta_cbo_instance.adapt(new_task=mock_new_task, adaptation_data=mock_adaptation_data)
    assert adapted_params == {"adapted_param": 1.0} # Check return value from mock

    # Verify inner_loop_update was called with correct args
    mock_maml_framework.inner_loop_update.assert_called_once_with(
        # meta_params=initial_meta_params, # initial_params is passed via positional
        initial_params=initial_meta_params, # Corrected kwarg name
        task_data=mock_adaptation_data,
        task_embedding=mock_embedding,
        # Check other kwargs passed if any
        # n_steps=..., inner_lr=... # Pass via **kwargs if needed
    )

    meta_cbo_instance._is_meta_trained = False # Reset state

def test_meta_train_calls_maml_outer_loop_and_updates_state(meta_cbo_instance, mock_maml_framework, mock_valid_causal_graph):
    """4 & 5. Test MAML Outer Loop Integration and State Update in meta_train."""
    # Make mock tasks provide a valid graph
    mock_task1 = Mock(name="Task1", id=1, support_data="s1", query_data="q1", graph=mock_valid_causal_graph)
    mock_task2 = Mock(name="Task2", id=2, support_data="s2", query_data="q2", graph=mock_valid_causal_graph)
    mock_task_family = [mock_task1, mock_task2]
    expected_meta_params = {"meta_param": 0.5} # From mock

    # Mock task embedding
    meta_cbo_instance.task_representation_model.embed_task.side_effect = ["emb1", "emb2"]

    assert meta_cbo_instance._is_meta_trained is False
    assert meta_cbo_instance._meta_parameters != expected_meta_params

    # Call meta_train
    meta_cbo_instance.meta_train(task_family=mock_task_family, epochs=1)

    # Verify state updated
    assert meta_cbo_instance._is_meta_trained is True
    assert meta_cbo_instance._meta_parameters == expected_meta_params

    # Verify outer_loop_update was called
    mock_maml_framework.outer_loop_update.assert_called_once()
    call_args, call_kwargs = mock_maml_framework.outer_loop_update.call_args
    task_batch = call_kwargs.get('task_batch')
    assert len(task_batch) == 2
    assert task_batch[0]['task'] is mock_task1
    assert task_batch[1]['task'] is mock_task2
    assert task_batch[0]['graph'] is mock_valid_causal_graph
    assert task_batch[1]['graph'] is mock_valid_causal_graph
    assert call_kwargs.get('epochs') == 1
    assert call_kwargs.get('meta_lr') == meta_cbo_instance.meta_lr

def test_meta_train_handles_task_batching(meta_cbo_instance, mock_maml_framework, mock_valid_causal_graph):
    """6. Test Task Batching in meta_train (basic check - MAML mock handles actual batching)."""
    # Note: The current implementation passes the full list to MAML.
    # Batching logic would typically reside *within* the MAML framework's outer loop.
    # This test mainly verifies the data preparation step.
    mock_tasks = [
        # Make mock tasks provide a valid graph
        Mock(name=f"Task{i}", id=i, support_data=f"s{i}", query_data=f"q{i}", graph=mock_valid_causal_graph)
        for i in range(5)
    ]
    # meta_cbo_instance.maml_config['meta_batch_size'] = 3 # Removed - MAML handles its config

    # Mock task embedding
    meta_cbo_instance.task_representation_model.embed_task.side_effect = [f"emb{i}" for i in range(5)]

    meta_cbo_instance.meta_train(task_family=mock_tasks, epochs=2)

    # Verify outer loop was called once with the prepared batch of all valid tasks
    mock_maml_framework.outer_loop_update.assert_called_once()
    _ , call_kwargs = mock_maml_framework.outer_loop_update.call_args
    task_batch = call_kwargs.get('task_batch')
    assert len(task_batch) == 5 # All tasks should be valid now
    assert call_kwargs.get('epochs') == 2

# TODO: Add tests specifically for the causal integration (subtask 19.3)
# - Test _validate_and_store_graph with various task inputs (valid/invalid graphs)
# - Test _get_valid_interventions returns expected nodes
# - Test _estimate_causal_effect calls estimation utility (mocked for now)
# - Test optimize_interventions uses causal methods (_get_valid_interventions, _estimate_causal_effect - mocked)
# - Test visualize_task_graph calls plotting utility (mocked)

# You might add more tests here as the class structure evolves 