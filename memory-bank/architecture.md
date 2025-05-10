# System Architecture

This document outlines the key architectural components of the Amortized Causal Meta-Learning Framework following the interface-first design pattern approach.

## Core Concepts & Abstractions

### Interface-First Design Pattern

The architecture follows an interface-first design pattern that completely decouples algorithm logic from specific model implementations. This enables greater flexibility in model implementation while maintaining a coherent architectural design.

1. **Core Interfaces:**

   - **`CausalStructureInferenceModel` (`causal_meta.inference.interfaces.CausalStructureInferenceModel`)**:
     - **Responsibility:** Interface for models that infer causal structure from data.
     - **Interface:**
       - `infer_structure(data)`: Infers causal structure from data.
       - `update_model(data)`: Updates the model with new data.
       - `estimate_uncertainty()`: Provides uncertainty estimates for inferred structures.

   - **`InterventionOutcomeModel` (`causal_meta.inference.interfaces.InterventionOutcomeModel`)**:
     - **Responsibility:** Interface for models that predict outcomes of interventions.
     - **Interface:**
       - `predict_intervention_outcome(graph, intervention, data)`: Predicts outcomes of interventions.
       - `update_model(data)`: Updates the model with new data.
       - `estimate_uncertainty()`: Provides uncertainty estimates for predictions.

   - **`AcquisitionStrategy` (`causal_meta.optimization.interfaces.AcquisitionStrategy`)**:
     - **Responsibility:** Interface for strategies that select interventions.
     - **Interface:**
       - `compute_acquisition(model, graph, data)`: Computes acquisition values for possible interventions.
       - `select_intervention(model, graph, data, budget)`: Selects the best intervention given a budget.
       - `select_batch(model, graph, data, budget, batch_size)`: Selects a batch of diverse interventions.

   - **`UncertaintyEstimator` (`causal_meta.inference.uncertainty.UncertaintyEstimator`)**:
     - **Responsibility:** Interface for estimating uncertainty in predictions.
     - **Interface:**
       - `estimate_uncertainty(model, data)`: Estimates uncertainty in model predictions.
       - `calibrate(model, validation_data)`: Calibrates uncertainty estimates using validation data.

   - **`Updatable` (`causal_meta.inference.interfaces.Updatable`)**:
     - **Responsibility:** Interface for models that can be updated with new data.
     - **Interface:**
       - `update(data)`: Updates the model with new data.
       - `reset()`: Resets the model to its initial state.

2. **Implementation Components:**

   - **Causal Graph (`causal_meta.graph.causal_graph.CausalGraph`, `causal_meta.graph.directed_graph.DirectedGraph`)**:
     - **Responsibility:** Represents the causal structure (Directed Acyclic Graph - DAG) underlying a task. Defines nodes and directed edges representing causal relationships.
     - **Interface:** Methods for adding/removing nodes/edges, getting parents/children, checking for paths, retrieving adjacency matrix.

   - **Structural Causal Model (`causal_meta.environments.scm.StructuralCausalModel`)**:
     - **Responsibility:** Encapsulates a complete causal model, combining a `CausalGraph` with functional mechanisms (structural equations) for each node. It allows simulating data under observational and interventional settings.
     - **Interface:**
       - `__init__(graph, structural_equations, noise_distributions)`: Constructor.
       - `sample_data(n_samples)`: Generates observational data.
       - `do_intervention(target_node, value)`: Modifies the SCM according to a perfect intervention (returns a *new* SCM instance).
       - `sample_interventional_data(interventions_dict, n_samples)`: Generates data under specified interventions.
       - `get_causal_graph()`: Returns the underlying `CausalGraph`.
       - `get_adjacency_matrix()`: Returns the adjacency matrix of the underlying causal graph.

   - **Intervention (`causal_meta.environments.interventions.Intervention` and subclasses)**:
     - **Responsibility:** Abstract base class representing different ways to modify an SCM. Subclasses implement specific intervention types.
     - **Interface:**
       - `__init__(target_node, **kwargs)`: Constructor.
       - `apply(scm)`: Abstract method that takes an SCM and returns a *new*, modified SCM instance representing the intervention.
     - **Subclasses:**
       - `PerfectIntervention`: Implements the do-operator (sets node to fixed value, removes parents).
       - `ImperfectIntervention`: Modifies node value based on original mechanism, intervention value, and strength.
       - `SoftIntervention`: Modifies or replaces the structural equation using a provided function.

   - **Graph Factory (`causal_meta.graph.generators.factory.GraphFactory`)**:
     - **Responsibility:** Generates different types of `CausalGraph` structures (e.g., random DAGs, specific patterns like chains).
     - **Interface:** Methods like `create_random_dag(num_nodes, edge_probability)`.

   - **Task Family Generator (`causal_meta.graph.generators.task_families`)**:
     - **Responsibility:** Creates a collection (family) of related SCMs/Tasks by introducing systematic variations (edge weights, structure) to a base SCM/graph.
     - **Interface:** `generate_task_family(base_graph, num_tasks, variation_type, variation_strength)`.

3. **Specific Model Implementations:**

   - **GNN-based Graph Encoder (`causal_meta.meta_learning.acd_models.GraphEncoder`)**:
     - **Responsibility:** Implements the `CausalStructureInferenceModel` interface using Graph Neural Networks.
     - **Interface:** Follows the `CausalStructureInferenceModel` interface, with additional GNN-specific methods.

   - **GNN-based Dynamics Decoder (`causal_meta.meta_learning.dynamics_decoder.DynamicsDecoder`)**:
     - **Responsibility:** Implements the `InterventionOutcomeModel` interface using Graph Neural Networks.
     - **Interface:** Follows the `InterventionOutcomeModel` interface, with additional GNN-specific methods.

   - **MLP-based Graph Encoder (`causal_meta.meta_learning.acd_models.MLPGraphEncoder`)**:
     - **Responsibility:** Alternative implementation of `CausalStructureInferenceModel` using MLPs.
     - **Interface:** Follows the `CausalStructureInferenceModel` interface, with MLP-specific methods.

   - **TransformerGraphEncoder (`causal_meta.meta_learning.acd_models.TransformerGraphEncoder`)**:
     - **Responsibility:** Implementation of `CausalStructureInferenceModel` using Transformer architecture.
     - **Interface:** Follows the `CausalStructureInferenceModel` interface, with Transformer-specific methods.

4. **Integration Components:**

   - **AmortizedCausalOptimizer (`causal_meta.optimization.amortized_causal_optimizer.AmortizedCausalOptimizer`)**:
     - **Responsibility:** High-level class that integrates `CausalStructureInferenceModel`, `InterventionOutcomeModel`, and `AcquisitionStrategy` to perform causal optimization.
     - **Interface:**
       - `__init__(structure_model, dynamics_model, acquisition_strategy)`: Constructor.
       - `train(data)`: Trains both structure and dynamics models.
       - `optimize_interventions(data, budget)`: Optimizes interventions given a budget.
       - `update(data)`: Updates models with new data.

   - **AmortizedCausalDiscovery (`causal_meta.meta_learning.amortized_causal_discovery.AmortizedCausalDiscovery`)**:
     - **Responsibility:** Combines structure inference and dynamics modeling.
     - **Interface:**
       - `__init__(graph_encoder, dynamics_decoder)`: Constructor.
       - `train(data)`: Trains both encoder and decoder models.
       - `infer_causal_graph(data)`: Infers causal graph from data.
       - `predict_intervention_outcomes(graph, intervention, data)`: Predicts outcomes of interventions.

5. **Uncertainty Estimation and Model Updating:**

   - **EnsembleUncertaintyEstimator (`causal_meta.inference.uncertainty.EnsembleUncertaintyEstimator`)**:
     - **Responsibility:** Estimates uncertainty using ensemble methods.
     - **Interface:** Follows the `UncertaintyEstimator` interface.

   - **DropoutUncertaintyEstimator (`causal_meta.inference.uncertainty.DropoutUncertaintyEstimator`)**:
     - **Responsibility:** Estimates uncertainty using Monte Carlo dropout.
     - **Interface:** Follows the `UncertaintyEstimator` interface.

   - **DirectUncertaintyEstimator (`causal_meta.inference.uncertainty.DirectUncertaintyEstimator`)**:
     - **Responsibility:** For models that directly predict variance.
     - **Interface:** Follows the `UncertaintyEstimator` interface.

   - **ConformalUncertaintyEstimator (`causal_meta.inference.uncertainty.ConformalUncertaintyEstimator`)**:
     - **Responsibility:** Uses conformal prediction methods for uncertainty estimation.
     - **Interface:** Follows the `UncertaintyEstimator` interface.

   - **IncrementalUpdater (`causal_meta.inference.interfaces.IncrementalUpdater`)**:
     - **Responsibility:** Implements the `Updatable` interface for efficient updates with new data.
     - **Interface:** Follows the `Updatable` interface.

   - **ExperienceReplayUpdater (`causal_meta.inference.interfaces.ExperienceReplayUpdater`)**:
     - **Responsibility:** Implements the `Updatable` interface with experience replay for balancing old/new data.
     - **Interface:** Follows the `Updatable` interface.

   - **FullRetrainingUpdater (`causal_meta.inference.interfaces.FullRetrainingUpdater`)**:
     - **Responsibility:** Implements the `Updatable` interface by retraining the model from scratch.
     - **Interface:** Follows the `Updatable` interface.

   - **UncertaintyThresholdManager (`causal_meta.inference.uncertainty.UncertaintyThresholdManager`)**:
     - **Responsibility:** Monitors uncertainty and triggers fallbacks when needed.
     - **Interface:** Methods for setting thresholds, monitoring uncertainty, and triggering fallbacks.

6. **Acquisition Strategies:**

   - **ExpectedImprovement (`causal_meta.optimization.acquisition.ExpectedImprovement`)**:
     - **Responsibility:** Implements the `AcquisitionStrategy` interface using expected improvement.
     - **Interface:** Follows the `AcquisitionStrategy` interface.

   - **UpperConfidenceBound (`causal_meta.optimization.acquisition.UpperConfidenceBound`)**:
     - **Responsibility:** Implements the `AcquisitionStrategy` interface using upper confidence bound.
     - **Interface:** Follows the `AcquisitionStrategy` interface.

   - **ThompsonSampling (`causal_meta.optimization.acquisition.ThompsonSampling`)**:
     - **Responsibility:** Implements the `AcquisitionStrategy` interface using Thompson sampling.
     - **Interface:** Follows the `AcquisitionStrategy` interface.

   - **BatchAcquisition (`causal_meta.optimization.acquisition.BatchAcquisition`)**:
     - **Responsibility:** Decorator for acquisition strategies to support batch selection with diversity.
     - **Interface:** Follows the `AcquisitionStrategy` interface, with additional batch selection methods.

7. **Configuration System:**

   - **ConfigurationManager (`causal_meta.utils.configuration.ConfigurationManager`)**:
     - **Responsibility:** Manages component configurations using YAML files.
     - **Interface:** Methods for loading, validating, and applying configurations.

   - **ConfigurableComponent (`causal_meta.utils.configuration.ConfigurableComponent`)**:
     - **Responsibility:** Base class for components that can be configured using YAML.
     - **Interface:** Methods for setting and validating configuration.

## Key Interactions

1. **Algorithm-Model Decoupling:**
   - The `AmortizedCausalOptimizer` depends only on the interfaces `CausalStructureInferenceModel`, `InterventionOutcomeModel`, and `AcquisitionStrategy`.
   - It does not make any assumptions about the implementation details of these components.
   - This allows for swapping different model implementations without changing the algorithm logic.

2. **Uncertainty Estimation Pipeline:**
   - Models that implement uncertainty estimation interfaces can be used with the `UncertaintyThresholdManager`.
   - The manager monitors uncertainty and can trigger fallbacks to more robust methods when uncertainty exceeds thresholds.
   - This creates a tiered approach to uncertainty handling.

3. **Model Updating Pipeline:**
   - Components that implement the `Updatable` interface can be updated with new data efficiently.
   - Different update strategies (incremental, experience replay, full retraining) can be swapped based on requirements.
   - This enables efficient adaptation to new data.

4. **YAML Configuration System:**
   - The `ConfigurationManager` loads and validates configurations from YAML files.
   - Components that implement the `ConfigurableComponent` interface can be configured using these files.
   - This enables easy experimentation with different configurations without changing code.

5. **Batch Acquisition Pipeline:**
   - The `BatchAcquisition` decorator enables batch selection of interventions with diversity considerations.
   - It can be applied to any `AcquisitionStrategy` implementation to enable batch selection.
   - This allows for efficient use of resources in experimental design.

## Interface Contracts

### CausalStructureInferenceModel Contract

```python
class CausalStructureInferenceModel(ABC):
    """Interface for models that infer causal structure from data."""
    
    @abstractmethod
    def infer_structure(self, data: Data) -> Graph:
        """
        Infers causal structure from data.
        
        Args:
            data: The input data for structure inference.
            
        Returns:
            A causal graph structure.
        """
        pass
    
    @abstractmethod
    def update_model(self, data: Data) -> None:
        """
        Updates the model with new data.
        
        Args:
            data: The new data to update the model with.
        """
        pass
    
    @abstractmethod
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Provides uncertainty estimates for inferred structures.
        
        Returns:
            Uncertainty estimates for the inferred causal structure.
        """
        pass
```

### InterventionOutcomeModel Contract

```python
class InterventionOutcomeModel(ABC):
    """Interface for models that predict outcomes of interventions."""
    
    @abstractmethod
    def predict_intervention_outcome(
        self, 
        graph: Graph, 
        intervention: Intervention, 
        data: Data
    ) -> Prediction:
        """
        Predicts outcomes of interventions.
        
        Args:
            graph: The causal graph structure.
            intervention: The intervention to apply.
            data: The observational data to condition on.
            
        Returns:
            Predicted outcomes of the intervention.
        """
        pass
    
    @abstractmethod
    def update_model(self, data: Data) -> None:
        """
        Updates the model with new data.
        
        Args:
            data: The new data to update the model with.
        """
        pass
    
    @abstractmethod
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Provides uncertainty estimates for predictions.
        
        Returns:
            Uncertainty estimates for the predicted outcomes.
        """
        pass
```

### AcquisitionStrategy Contract

```python
class AcquisitionStrategy(ABC):
    """Interface for strategies that select interventions."""
    
    @abstractmethod
    def compute_acquisition(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data
    ) -> Dict[Intervention, float]:
        """
        Computes acquisition values for possible interventions.
        
        Args:
            model: The model used to predict intervention outcomes.
            graph: The causal graph structure.
            data: The observational data to condition on.
            
        Returns:
            Dictionary mapping interventions to their acquisition values.
        """
        pass
    
    @abstractmethod
    def select_intervention(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float
    ) -> Intervention:
        """
        Selects the best intervention given a budget.
        
        Args:
            model: The model used to predict intervention outcomes.
            graph: The causal graph structure.
            data: The observational data to condition on.
            budget: The budget constraint for interventions.
            
        Returns:
            The selected intervention.
        """
        pass
    
    @abstractmethod
    def select_batch(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float, 
        batch_size: int
    ) -> List[Intervention]:
        """
        Selects a batch of diverse interventions.
        
        Args:
            model: The model used to predict intervention outcomes.
            graph: The causal graph structure.
            data: The observational data to condition on.
            budget: The budget constraint for interventions.
            batch_size: The number of interventions to select.
            
        Returns:
            A list of selected interventions.
        """
        pass
```

## Benchmarking Framework

The benchmarking framework provides a comprehensive system for evaluating causal discovery and causal Bayesian optimization methods, allowing for systematic comparison and performance analysis.

### Core Components

1. **Abstract Benchmark (Base Class)**
   - Provides the common interface and utilities for all benchmark types
   - Handles model management, result tracking, and serialization
   - Implements common evaluation metrics and visualization methods
   - Manages test environments and dataset generation

2. **Specialized Benchmark Classes**
   - **CausalDiscoveryBenchmark**: Evaluates methods that infer graph structure from data
     - Generates synthetic graphs and data for testing
     - Measures structural accuracy with SHD, precision, recall
     - Supports different data types (observational, interventional)
     - Handles various method interfaces for broad compatibility
   
   - **CBOBenchmark**: Evaluates methods that optimize interventions in causal systems
     - Creates optimization problems with varying difficulty
     - Measures intervention quality and sample efficiency
     - Supports budget-constrained optimization scenarios
     - Compares against random and theoretical optimal baselines

   - **ScalabilityBenchmark**: Assesses how methods scale with problem size
     - Tests performance across increasing graph sizes
     - Measures runtime, memory usage, and accuracy trends
     - Determines computational complexity class
     - Identifies practical size limits for different methods

3. **BenchmarkRunner**
   - Orchestrates multiple benchmarks for comprehensive evaluation
   - Aggregates results across different benchmark types
   - Generates summary reports and comparison visualizations
   - Enables statistical significance testing between methods

## Configuration System

The configuration system provides a unified approach to managing component configurations using YAML files. This enables easy experimentation with different configurations without changing code.

### Key Components

1. **ConfigurationManager**
   - Loads and validates configurations from YAML files
   - Applies configurations to components
   - Supports environment variable substitution
   - Provides default configurations for common scenarios

2. **ConfigurableComponent**
   - Base class for components that can be configured using YAML
   - Provides methods for setting and validating configuration
   - Enables configuration inheritance and overrides
   - Supports schema validation for configuration values

3. **Configuration Schema**
   - Defines the structure and data types for component configurations
   - Supports validation of configuration values
   - Provides clear error messages for invalid configurations
   - Enables documentation generation for configuration options

## MVP Causal Graph Structure Learning

The MVP Causal Graph Structure Learning implementation follows a simplified architecture to demonstrate how neural networks can learn causal graph structures from observational and interventional data. This section outlines the architecture for this specific MVP implementation.

### MVP Component Architecture

```
+-----------------------------------+
|        ExperimentRunner           |
|   (Manages experiment workflow)   |
+----------------+------------------+
                 |
                 | configures & runs
                 v
+-----------------------------------+
|        ExperimentConfig           |
|   (Configuration parameters)      |
+----------------+------------------+
                 |
           +-----+-----+
           |           |
           v           v
+-------------------+  +------------------------+
| RandomDAGGenerator|  |   SimpleGraphLearner   |
| (Graph generation)|  | (Neural network model) |
+--------+----------+  +-----------+------------+
         |                         |
         v                         |
+--------------------+             |
| LinearSCMGenerator |             |
| (SCM generation)   |             |
+--------+-----------+             |
         |                         |
         |  +---------------------+|
         |  |   InterventionUtils ||
         |  | (Intervention tools)||
         |  +----------+----------+|
         |             |           |
         v             v           |
+----------------------------------------+
|          Data Processing               |
| (Tensor conversion & formatting)       |
+-------------------+-------------------+
                    |
                    v
+----------------------------------------+
|          GraphMetrics                  |
| (Evaluation & visualization)           |
+----------------------------------------+
```

### Data Flow

1. **Configuration**: `ExperimentConfig` defines all parameters for the experiment

2. **Graph & SCM Generation**:
   - `RandomDAGGenerator` creates a random DAG structure
   - `LinearSCMGenerator` builds a linear SCM based on the graph

3. **Data Generation**:
   - SCM generates observational data
   - `InterventionUtils` performs interventions to generate interventional data

4. **Data Processing**:
   - Data is converted to appropriate tensor format
   - Intervention information is encoded as binary masks

5. **Model Training**:
   - `SimpleGraphLearner` processes both observational and interventional data
   - Model is progressively updated with each new intervention

6. **Evaluation**:
   - `GraphMetrics` evaluates predicted graph against true graph
   - Learning progress is tracked across interventions

7. **Visualization**:
   - Results are visualized for comparison and analysis

### Integration with Existing Architecture

The MVP components are designed to interact with the existing codebase in the following ways:

1. **Graph Generation**: Leverages `causal_meta.graph.generators` where appropriate, but provides simplified implementations for the MVP requirements.

2. **SCM Generation**: Works with `causal_meta.environments.scm.StructuralCausalModel` for data generation but offers a simplified interface for linear SCMs.

3. **Neural Network**: Provides a standalone `SimpleGraphLearner` implementation rather than using the more complex `AmortizedCausalDiscovery` from the existing codebase, focusing on the core concept demonstration.

4. **Evaluation**: Uses simplified metrics while maintaining compatibility with existing evaluation approaches.

### Architectural Decisions

1. **Simplicity Over Sophistication**: The MVP prioritizes clear demonstration of the core concept over advanced features.

2. **Modularity**: Components are designed with clear interfaces to enable easy replacement or enhancement in future iterations.

3. **Explicit Intervention Encoding**: The approach uses explicit binary masks to encode intervention information for the neural network, making the learning signal clearer.

4. **Progressive Learning**: The architecture implements an iterative process where the model improves with each new intervention, demonstrating the value of active interventions.

5. **Supervised Learning**: For simplicity, the initial implementation uses supervised learning with access to ground truth for clearer demonstration, while maintaining the flexibility to move to unsupervised approaches in future iterations.

6. **Focused Scope**: The architecture focuses solely on structure learning without incorporating dynamics prediction or meta-learning at this stage.

### Future Extensions

The MVP architecture is designed to be extended in the following directions:

1. **Advanced Neural Architectures**: Replace `SimpleGraphLearner` with more sophisticated architectures.

2. **Optimized Intervention Selection**: Implement information-theoretic or uncertainty-based intervention selection strategies.

3. **Integration with Meta-Learning**: Incorporate MAML or other meta-learning approaches for faster adaptation.

4. **Non-linear SCMs**: Extend to non-linear structural causal models.

5. **Joint Structure and Dynamics Learning**: Integrate with dynamics prediction components for joint learning. 