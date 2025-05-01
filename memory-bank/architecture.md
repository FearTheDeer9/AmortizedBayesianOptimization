# System Architecture

This document outlines the key architectural components of the Causal Bayesian Optimization (MetaCBO) framework.

## Core Concepts & Abstractions

1.  **Causal Graph (`causal_meta.graph.causal_graph.CausalGraph`, `causal_meta.graph.directed_graph.DirectedGraph`)**:
    - **Responsibility:** Represents the causal structure (Directed Acyclic Graph - DAG) underlying a task. Defines nodes and directed edges representing causal relationships.
    - **Interface:** Methods for adding/removing nodes/edges, getting parents/children, checking for paths, retrieving adjacency matrix.

2.  **Structural Causal Model (`causal_meta.environments.scm.StructuralCausalModel`)**:
    - **Responsibility:** Encapsulates a complete causal model, combining a `CausalGraph` with functional mechanisms (structural equations) for each node. It allows simulating data under observational and interventional settings.
    - **Interface:**
        - `__init__(graph, structural_equations, noise_distributions)`: Constructor.
        - `sample_data(n_samples)`: Generates observational data.
        - `do_intervention(target_node, value)`: Modifies the SCM according to a perfect intervention (returns a *new* SCM instance).
        - `sample_interventional_data(interventions_dict, n_samples)`: Generates data under specified interventions.
        - `get_causal_graph()`: Returns the underlying `CausalGraph`.
        - `get_adjacency_matrix()`: Returns the adjacency matrix of the underlying causal graph.
        - Potentially methods to get/set structural equations.

3.  **Intervention (`causal_meta.environments.interventions.Intervention` and subclasses)**:
    - **Responsibility:** Abstract base class representing different ways to modify an SCM. Subclasses implement specific intervention types.
    - **Interface:**
        - `__init__(target_node, **kwargs)`: Constructor.
        - `apply(scm)`: Abstract method that takes an SCM and returns a *new*, modified SCM instance representing the intervention.
    - **Subclasses:**
        - `PerfectIntervention`: Implements the do-operator (sets node to fixed value, removes parents).
        - `ImperfectIntervention`: Modifies node value based on original mechanism, intervention value, and strength.
        - `SoftIntervention`: Modifies or replaces the structural equation using a provided function.

4.  **Graph Factory (`causal_meta.graph.generators.factory.GraphFactory`)**:
    - **Responsibility:** Generates different types of `CausalGraph` structures (e.g., random DAGs, specific patterns like chains).
    - **Interface:** Methods like `create_random_dag(num_nodes, edge_probability)`.

5.  **Task Family Generator (`causal_meta.graph.generators.task_families`)**:
    - **Responsibility:** Creates a collection (family) of related SCMs/Tasks by introducing systematic variations (edge weights, structure) to a base SCM/graph.
    - **Interface:** `generate_task_family(base_graph, num_tasks, variation_type, variation_strength)`.

6.  **MetaCBO Orchestrator (Main Class - Likely in `algorithms` or root)**:
    - **Responsibility:** Manages the overall Meta-Causal Bayesian Optimization process. Integrates BO, meta-learning, SCM simulation, and intervention selection.
    - **Interface (Conceptual):**
        - `meta_train(task_family)`: Performs meta-training using MAML across a family of tasks.
        - `adapt(new_task_data)`: Adapts the meta-learned model to a new task.
        - `optimize_interventions(target_scm, budget)`: Runs the CBO loop on a specific SCM to find optimal interventions.
        - `evaluate()`: Assesses performance.
        - Visualization methods (`plot_meta_training_progress`, `visualize_task_graph`).

7.  **Task Representation (Module/Class - Defined in Task 5/18)**:
    - **Responsibility:** Converts a task (represented potentially by its SCM or graph) into a fixed-size numerical embedding suitable for input to the meta-learning algorithm (MAML).
    - **Interface (Conceptual):** `embed_task(task_scm_or_graph)`.

8.  **Meta-Learning Algorithm (MAML - Module/Class - Defined in Task 5/17)**:
    - **Responsibility:** Implements the Model-Agnostic Meta-Learning algorithm. Learns a model initialization that can be quickly adapted to new tasks.
    - **Interface (Conceptual):** `inner_loop_update(task_data, task_embedding)`, `outer_loop_update(batch_of_tasks)`, state management.

9.  **Bayesian Optimization Components (Leveraging BoTorch)**:
    - **Responsibility:** Handles the core BO loop components within `MetaCBO`.
    - **Components:** Gaussian Process (GP) model, Acquisition Function (e.g., custom `CausalExpectedImprovement`), Optimizer for acquisition function.

## Key Interactions

- `GraphFactory` generates a `CausalGraph`.
- `StructuralCausalModel` takes a `CausalGraph` and functional definitions.
- `TaskFamilyGenerator` uses a base `CausalGraph` (or `SCM`) to create variations.
- `MetaCBO` uses `TaskFamilyGenerator` to get tasks for meta-training.
- `TaskRepresentation` embeds tasks for `MAML`.
- `MAML` performs meta-training using embedded tasks and data sampled from their respective `SCM`s.
- During optimization on a specific task, `MetaCBO` uses its (potentially adapted) GP model and acquisition function (BoTorch) to propose an `Intervention`.
- The `Intervention`'s `apply` method creates a modified `SCM`.
- `MetaCBO` samples data from the original or intervened `SCM` to evaluate outcomes and update the GP model.
- Visualization methods use data logged by `MetaCBO` and graph structures.

*This architecture overview is based on the current understanding of the project components and tasks. It will be refined as implementation progresses.*

### Core Data Structures

*   **`causal_meta.graph.CausalGraph`**: (From `causal_graph.py`) Represents the underlying DAG structure. Uses `networkx.DiGraph`. Provides methods for graph manipulation and analysis relevant to causality (e.g., identifying parents, children, ancestors, descendants, colliders).
*   **`causal_meta.environments.StructuralCausalModel`**: (From `scm.py`) Implements an SCM based on a `CausalGraph`. Defines functional mechanisms for each node and allows sampling observational and interventional data.
*   **`causal_meta.graph.TaskFamily`**: (From `task_family.py`) Represents a collection of related causal tasks, typically generated by varying a base graph. Contains the base graph and a list of variation graphs, along with associated metadata for the family and individual variations. Provides save/load functionality.

### Key Modules & Classes

*   **`causal_meta.graph.generators`**: Contains modules for generating graphs and task families.
    *   `factory.py`: `GraphFactory` for creating specific graph structures (e.g., random DAGs, chains).
    *   `task_families.py`: `generate_task_family` function (Task 2) to create families of related tasks by varying edge weights, structure, etc.
*   **`causal_meta.utils.visualization`**: (From `visualization.py`) Contains utility classes for plotting.
    *   `TaskFamilyVisualizer`: Provides methods like `plot_family_comparison` and `generate_difficulty_heatmap` to visualize `TaskFamily` objects.
*   **`causal_meta.meta_learning`**: (Placeholder) Will contain MAML and related meta-learning algorithms.
    *   `maml.py`: (Placeholder - Task 5) `MAML` implementation.
    *   `task_representation.py`: (Placeholder - Task 5) `TaskRepresentation` for embedding tasks.
*   **`causal_meta.optimization`**: Contains CBO logic.
    *   `meta_cbo.py`: (Placeholder - Task 4) `MetaCBO` class implementing the main Bayesian Optimization loop, intervention selection, and effect estimation.
    *   `acquisition_functions.py`: (Placeholder - Task 4) Custom acquisition functions like `CausalExpectedImprovement`.

### Amortized Causal Discovery Components

*   **`causal_meta.meta_learning.amortized_causal_discovery`**: Core implementation of ACD.
    *   `AmortizedCausalDiscovery`: Main class that combines the encoder (for graph structure inference) and decoder (for dynamics modeling).
    *   `train`, `infer_causal_graph`, and `predict_intervention_outcomes` methods.
*   **`causal_meta.meta_learning.acd_models`**: Neural network models for causal structure inference.
    *   `GraphEncoder`: Neural network for causal structure learning using attention mechanisms.
    *   `AttentionBlock`: Multi-head self-attention block for processing time series data.
    *   Methods for encoding time series data into graph structures, calculating sparsity and acyclicity losses.
*   **`causal_meta.meta_learning.dynamics_decoder`**: Neural network models for dynamics prediction.
    *   `DynamicsDecoder`: Neural network for predicting outcomes under interventions.
    *   Methods for conditioning on graph structure and intervention information.
    *   Uncertainty quantification through ensemble predictions.
*   **`causal_meta.meta_learning.amortized_cbo`**: CBO implementation using ACD.
    *   `AmortizedCBO`: Replaces traditional MetaCBO by using neural networks instead of GPs.
    *   Methods for meta-training, adaptation, and intervention optimization.

### Demo Implementation

*   **`demos/parent_scale_acd_demo.py`**: Demonstrates parent-scaled ACD with neural networks.
    *   Uses GraphEncoder and DynamicsDecoder for structure learning and prediction.
    *   Implements parent-count based intervention target selection.
    *   Creates synthetic data with proper structural causal models.
    *   Shows how neural networks can be used as drop-in replacements for traditional causal discovery.
    *   Visualizes both ground truth and inferred graph structures.
    *   Uses probabilistic structural equations for realistic data generation.
*   **`demos/full_acd_pipeline_demo.py`**: Demonstrates the full amortized causal discovery pipeline.
    *   Includes task family generation and meta-learning components.
    *   Shows the full training and adaptation process for amortized causal discovery.
    *   Leverages neural networks for both structure learning and dynamics prediction.
    *   Will incorporate proper benchmarking and evaluation metrics. 