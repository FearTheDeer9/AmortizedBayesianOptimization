# Implementation Plan

This document reflects the current state of the project based on the Amortized Causal Discovery approach.

## Task Template with Sequential Thinking

Each task in this implementation plan should follow this template format, incorporating Sequential Thinking analysis:

```
- **Subtask X.Y: Task Name (Status)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - Initial assessment of what the task requires
      - Key requirements and constraints
      - Integration points with existing components
      - Expected inputs and outputs

    - **Thought 2: Component Identification**
      - Key components needed for implementation
      - Data structures and algorithms required
      - External dependencies and libraries
      - Integration points with other modules

    - **Thought 3: Implementation Approach**
      - Proposed technical approach
      - Architecture decisions
      - Algorithm selection
      - Design patterns to apply

    - **Thought 4: Potential Challenges**
      - Anticipated difficulties
      - Performance considerations
      - Edge cases to handle
      - Mitigation strategies

    - **Thought 5: Implementation Plan**
      - Step-by-step approach
      - Testing strategy
      - Integration plan
      - Documentation requirements

  - **Detailed Implementation Steps:**
    1. Specific implementation step 1
    2. Specific implementation step 2
    3. ...

  - **Current Status:** [pending/in-progress/done]
  - **Estimated Completion:** [date or TBD]
```

This template ensures that each task is thoroughly analyzed using Sequential Thinking before implementation begins, following the project's integrated workflow approach.

## Overall Status

- **Project Name:** Amortized Causal Bayesian Optimization Framework
- **Total Tasks:** 7
- **Research Direction:** Implementation of Amortized Causal Discovery as outlined in Löwe et al. (2022)

## High-Level Task Summary

1.  **Implement Reliable DAG Generation:** `done`
    - Essential foundation for generating training and test graphs.
2.  **Implement Task Family Generation:** `done`
    - Critical for meta-learning across related causal structures.
3.  **Integrate StructuralCausalModel Implementation:** `done`
    - Core component for data generation and intervention simulation.
4.  **Implement Neural Causal Discovery Components:** `in-progress`
    - Neural network components for graph structure inference.
5.  **Implement Amortized Causal Dynamics Modeling:** `in-progress`
    - Neural dynamics modeling for intervention prediction.
6.  **Implement Amortized Causal Bayesian Optimization:** `done`
    - End-to-end implementation combining neural graph inference and dynamics modeling.
7.  **Create Evaluation Framework and Benchmarks:** `pending`
    - Comprehensive benchmarking and visualization tools.
8.  **Demo Scripts for Supervisor Meeting:** `in-progress`
    - HIGH PRIORITY: Demonstration scripts for upcoming supervisor meeting.
    - Must be completed ahead of Tasks 6 and 7.

## Research Approach

This project implements Amortized Causal Discovery (ACD) within a Causal Bayesian Optimization framework. The approach uses neural networks to learn both causal graph structures and dynamics models from data, enabling scalability to larger graphs that wouldn't be feasible with traditional methods. Our implementation follows the tiered inference approach, focusing on neural approximation for both graph structure inference and dynamics modeling.

### Project Implementation Plan

#### Task 1: Implement Reliable DAG Generation (Done ✅)
- Creation of foundational DAG generation tools to provide training data for neural models.
- Subtask 1.1: Add create_random_dag method signature to GraphFactory (Done ✅)
- Subtask 1.2: Implement core DAG generation algorithm (Done ✅)
- Subtask 1.3: Implement edge probability logic (Done ✅)
- Subtask 1.4: Add DAG validation and verification (Done ✅)
- Subtask 1.5: Update example script to use the new DAG generator (Done ✅)

#### Task 2: Implement Task Family Generation (Done ✅)
- Generation of related causal structures for meta-learning and transfer learning.
- Subtask 2.1: Set up module structure and base function implementation (Done ✅)
- Subtask 2.2: Implement edge weight variation (Done ✅)
- Subtask 2.3: Implement structure variation with DAG preservation (Done ✅)
- Subtask 2.4: Implement node function variation (Deferred ⏸️)
- Subtask 2.5: Integrate with framework and implement comprehensive testing (Done ✅)

#### Task 3: Integrate StructuralCausalModel Implementation (Done ✅)
- Foundational SCM implementation that will be used for data generation in training neural models.
- Subtask 3.1: Review and analyze existing StructuralCausalModel implementation (Done ✅)
- Subtask 3.2: Implement or extend sample_data method (Done ✅) 
- Subtask 3.3: Implement or extend perform_intervention method (Done ✅)
- Subtask 3.4: Implement or extend get_adjacency_matrix method (Done ✅)
- Subtask 3.5: Update example workflow to use StructuralCausalModel (Done ✅)

#### Task 4: Implement Neural Causal Discovery Components (In Progress 🔄)
- **Description:** Implement neural network-based components for inferring causal structure from observational and interventional data.
- **Priority:** high
- **Status:** in-progress
- **Dependencies:** 1, 2, 3

- **Subtask 4.1: Implement GraphEncoder neural network (Done ✅)**
  - Create `GraphEncoder` class in `causal_meta/meta_learning/acd_models.py`
  - Implement attention-based GNN architecture for encoding time-series data into graph structures
  - Add sparsity regularization and structural constraints
  - Implement batched processing functionality
  - Create comprehensive unit tests

- **Subtask 4.2: Create graph inference utilities (Done ✅)**
  - Implement threshold mechanism for converting edge probabilities to discrete graphs
  - Create posterior sampling methods for uncertainty quantification
  - Implement graph validation to ensure DAG properties
  - Add metrics for graph recovery accuracy (SHD, precision/recall)
  - Integrate with existing `CausalGraph` representations

- **Subtask 4.3: Implement graph encoder training pipeline (Done ✅)**
  - Create loss functions for graph structure learning
  - Implement curriculum learning for increasingly complex graphs
  - Add regularization terms for sparsity and acyclicity
  - Create checkpoint and model saving utilities
  - Implement early stopping and performance tracking

- **Subtask 4.4: Create synthetic data generation for training (Done ✅)**
  - Implement `SyntheticDataGenerator` class in `causal_meta/meta_learning/data_generation.py`
  - Create observational data generation from SCMs
  - Implement interventional data simulation
  - Add noise models and data augmentation techniques
  - Create efficient data loaders and batching utilities

#### Task 5: Implement Amortized Causal Dynamics Modeling (In Progress 🔄)
- **Description:** Implement neural network components for modeling dynamics and predicting intervention outcomes.
- **Priority:** high
- **Status:** in-progress
- **Dependencies:** 3, 4

- **Subtask 5.1: Implement DynamicsDecoder neural network (Done ✅)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - The DynamicsDecoder needs to predict intervention outcomes based on inferred graph structure and observational data
      - Must be compatible with GraphEncoder outputs and handle different types of interventions
      - Should provide uncertainty estimates for predictions
      - Needs to be differentiable end-to-end for training

    - **Thought 2: Component Identification**
      - Neural network architecture that processes graph structure and node features
      - Intervention conditioning mechanism for different intervention types
      - Prediction capabilities for counterfactual node values
      - Uncertainty quantification system
      - Integration interface with GraphEncoder
      - Batched processing for efficient training

    - **Thought 3: Implementation Approach**
      - Use Graph Neural Network (GNN) with attention mechanisms
      - Implement message passing between nodes based on graph structure
      - Add attention layers to focus on relevant nodes for interventions
      - Create mechanisms to condition on different intervention types
      - Implement uncertainty estimation through ensemble or variational methods
      - Design output layer to generate predictions for all nodes

    - **Thought 4: Potential Challenges**
      - Ensuring differentiability: Use soft adjacency matrices
      - Handling different intervention types: Create abstract intervention representation
      - Computational efficiency: Implement sparse operations
      - Uncertainty calibration: Use ensemble methods with proper validation
      - Integration with GraphEncoder: Define clear interfaces
      - Preventing overfitting: Add graph-specific regularization
      - Gradient flow: Use skip connections and normalization

    - **Thought 5: Implementation Plan**
      - Create file structure and class definition
      - Implement core neural components
      - Add uncertainty quantification
      - Create test infrastructure
      - Integrate with existing components
      - Add documentation and examples

  - **Detailed Implementation Steps:**
    1. Create new file `causal_meta/meta_learning/dynamics_decoder.py`
    2. Define `DynamicsDecoder` class inheriting from `nn.Module`
    3. Implement Graph Attention layers for message passing
    4. Add intervention conditioning mechanism
    5. Create uncertainty quantification system
    6. Write comprehensive tests in `tests/meta_learning/test_dynamics_decoder.py`
    7. Create integration methods with GraphEncoder
    8. Add documentation and examples

  - **Current Status:** Done
  - **Estimated Completion:** 2025-06-22

- **Subtask 5.2: Implement AmortizedCausalDiscovery class (Done ✅)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - The AmortizedCausalDiscovery class needs to serve as the main interface for the amortized approach
      - It must combine GraphEncoder for structure inference and DynamicsDecoder for dynamics modeling
      - Should provide a unified training pipeline for both components
      - Needs to implement inference methods for graph structure and intervention outcomes
      - Should include evaluation metrics for assessing performance of both components

    - **Thought 2: Component Identification**
      - Integration Components: GraphEncoder, DynamicsDecoder, data handling utilities
      - Training Components: Joint loss function, training loop, curriculum learning, checkpointing
      - Inference Components: Graph structure inference, intervention outcome prediction, uncertainty quantification
      - Evaluation Components: Structure metrics (SHD, precision, recall), prediction metrics (MSE, MAE)
      - Interface Components: High-level API, configuration system, serialization

    - **Thought 3: Implementation Approach**
      - Create a new class inheriting from nn.Module to combine both components
      - Implement a train method that handles joint training with balanced losses
      - Create inference methods for both graph structure and dynamics
      - Implement data handling utilities for proper data flow
      - Add visualization and evaluation tools for both components
      - Create a clean, high-level API for easy use

    - **Thought 4: Potential Challenges**
      - Gradient Flow: Ensuring proper gradient flow between encoder and decoder
      - Balancing Learning: Finding the right balance between structure and dynamics objectives
      - Computational Efficiency: Joint training can be computationally expensive
      - Data Compatibility: Ensuring both components can process the same data format
      - Hyperparameter Management: Managing many hyperparameters across components
      - Evaluation Complexity: Need for different metrics for different aspects
      - Uncertainty Calibration: Ensuring well-calibrated uncertainty estimates

    - **Thought 5: Implementation Plan**
      - Set up class structure with proper initialization of components
      - Implement core methods for joint operation and training
      - Create combined loss functions with appropriate weighting
      - Add data processing utilities for joint training
      - Implement comprehensive testing and evaluation
      - Finalize with documentation, serialization, and error handling

  - **Detailed Implementation Steps:**
    1. Create new file `causal_meta/meta_learning/amortized_causal_discovery.py`
    2. Define `AmortizedCausalDiscovery` class inheriting from `nn.Module`
    3. Implement initialization to create and configure both components
    4. Create `forward` method for joint operation of the model
    5. Implement `train` method with joint training procedure
    6. Add `infer_causal_graph` method for structure inference
    7. Create `predict_intervention_outcomes` method for dynamics prediction
    8. Implement combined loss function with weighting mechanism
    9. Add data processing utilities for batching and preprocessing
    10. Create test file with comprehensive test cases
    11. Implement evaluation metrics and visualization tools
    12. Add model serialization and configuration validation

  - **Current Status:** Done
  - **Estimated Completion:** 2025-06-23

- **Subtask 5.3: Implement meta-learning capabilities (Done ✅)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - The goal is to add meta-learning functionality to our existing AmortizedCausalDiscovery class
      - Need to enable few-shot adaptation to new causal structures with minimal data
      - Implement a meta-learning approach based on MAML (Model-Agnostic Meta-Learning)
      - Develop task embedding utilities for representing different causal structures
      - Create adaptation strategies specifically for causal discovery
      - Test adaptation performance on families of related causal tasks

    - **Thought 2: Component Identification**
      - Task Representation Components: Task embedding network, task family generator, similarity metrics
      - Meta-Training Components: Inner loop optimization, outer loop optimization, batch sampling
      - Adaptation Components: Rapid adaptation methods, parameter initialization, fine-tuning procedures
      - Evaluation Components: Few-shot metrics, cross-task generalization measurements
      - Implementation Details: Differential inner loop, higher-order gradients, parameter isolation

    - **Thought 3: Implementation Approach**
      - Implement MAML-based approach for fast adaptation to new causal structures
      - Create task embedding network to encode graph structure into fixed-size representations
      - Design meta-training procedure with inner and outer optimization loops
      - Develop adaptation strategies for both GraphEncoder and DynamicsDecoder components
      - Extend AmortizedCausalDiscovery class while maintaining compatibility with existing code

    - **Thought 4: Potential Challenges**
      - Computational Complexity: Higher-order gradients are computationally intensive
      - Task Diversity: Ensuring sufficient diversity in meta-training tasks
      - Few-Shot Performance: Achieving good results with minimal adaptation steps
      - Balancing Learning: Appropriate balance between structure and dynamics adaptation
      - Overfitting: Meta-learning is prone to overfitting to the meta-training distribution
      - Implementation Complexity: MAML requires careful handling of computational graphs

    - **Thought 5: Implementation Plan**
      - Set up code structure for meta-learning components
      - Implement core MAML components with inner/outer loop optimization
      - Create task embedding and representation utilities
      - Develop complete meta-training procedure
      - Implement adaptation and evaluation methods
      - Create comprehensive testing suite
      - Add detailed documentation and examples

  - **Detailed Implementation Steps:**
    1. Create new module `causal_meta/meta_learning/meta_learning.py` for meta-learning utilities
    2. Add meta-learning methods to the AmortizedCausalDiscovery class
    3. Implement task embedding network as a separate component
    4. Create inner loop adaptation function with configurable steps and learning rate
    5. Implement outer loop meta-optimization with appropriate higher-order gradients
    6. Add task batch sampling for meta-training
    7. Implement meta_train method with support for episodes and tasks
    8. Create adapt method for few-shot adaptation to new tasks
    9. Implement evaluation procedures for adapted models
    10. Add metrics for measuring adaptation speed and performance
    11. Create test file with comprehensive tests for meta-learning components
    12. Add detailed documentation and example scripts

  - **Current Status:** Done
  - **Estimated Completion:** 2025-06-25
  - **Implementation Notes:**
    - Successfully implemented TaskEmbedding class with methods to encode graph structures
    - Leveraged existing GraphStructureRepresentation from task_representation.py for compatibility
    - Added MAMLForCausalDiscovery class to integrate with MAML algorithm
    - Implemented enable_meta_learning and meta_adapt methods in AmortizedCausalDiscovery
    - Added comprehensive documentation and appropriate type hints
    - Updated module exports in __init__.py to expose the new components

- **Subtask 5.4: Add uncertainty quantification (Pending ⏱️)**
  - Implement ensemble methods or dropout-based uncertainty estimation
  - Create confidence intervals for predictions
  - Add visualization tools for uncertainty
  - Implement metrics for calibration assessment
  - Test uncertainty estimates on held-out data

#### Task 6: Implement Amortized Causal Bayesian Optimization (Done ✅)
- **Description:** Create the end-to-end system for causal Bayesian optimization using amortized components.
- **Priority:** high
- **Status:** done
- **Dependencies:** 4, 5

- **Subtask 6.1: Implement AmortizedCBO class (Done ✅)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - The AmortizedCBO needs to efficiently select interventions in causal systems using our neural network amortized causal discovery framework
      - Must implement Bayesian optimization techniques specifically for causal structure learning
      - Needs to leverage the uncertainty in neural network predictions
      - Should integrate with the meta-learning capabilities for transfer learning
      - Must balance exploration and exploitation in the intervention selection process

    - **Thought 2: Component Identification**
      - Acquisition Functions: Expected Improvement, Upper Confidence Bound, Probability of Improvement, Thompson Sampling
      - Intervention Selection: Strategy for selecting optimal interventions with budget constraints
      - Model Update: Mechanism to update the model with new observations
      - Optimization Loop: Full loop for iterative intervention optimization
      - Meta-Learning Integration: TaskEmbedding and adaptation for transfer learning

    - **Thought 3: Implementation Approach**
      - Implement neural network compatible versions of standard acquisition functions
      - Create intervention selection strategy with budget constraints
      - Develop a mechanism for updating the model with new observations
      - Implement a full optimization loop with early stopping
      - Integrate with TaskEmbedding for meta-learning capabilities
      - Ensure proper error handling and device management

    - **Thought 4: Potential Challenges**
      - Handling uncertainty estimates from neural networks
      - Balancing exploration and exploitation
      - Efficiently integrating meta-learning for transfer
      - Managing computational resources for large graphs
      - Ensuring robust error handling for different model types
      - Balancing batch processing with sequential decision-making

    - **Thought 5: Implementation Plan**
      - Create file structure and class definition
      - Implement acquisition functions for neural network predictions
      - Add intervention selection with budget constraints
      - Implement optimization loop with early stopping
      - Add model update mechanism
      - Integrate meta-learning capabilities
      - Create comprehensive test suite
      - Implement error handling and edge cases

  - **Detailed Implementation Steps:**
    1. Create new file `causal_meta/meta_learning/amortized_cbo.py`
    2. Define `AmortizedCBO` class
    3. Implement acquisition functions (EI, UCB, PI, Thompson sampling)
    4. Add intervention selection mechanism with budget constraints
    5. Implement model update functionality with meta-learning integration
    6. Create optimization loop with early stopping
    7. Add comprehensive error handling and edge cases
    8. Create test file `tests/meta_learning/test_amortized_cbo.py`
    9. Update module exports in `causal_meta/meta_learning/__init__.py`
    10. Implement error handling for mock compatibility in tests
    11. Add device management for proper tensor operations
    12. Fix edge cases in type handling and error recovery

  - **Current Status:** Done
  - **Estimated Completion:** 2025-06-26
  - **Implementation Notes:**
    - Successfully implemented AmortizedCBO class with comprehensive acquisition functions
    - Added robust intervention selection with budget constraints
    - Implemented full optimization loop with early stopping
    - Added meta-learning integration with TaskEmbedding
    - Fixed issues with error handling and device management
    - Added comprehensive test suite with all tests passing
    - Ensured compatibility with mock objects for testing

- **Subtask 6.2: Implement acquisition functions (Done ✅)**
  - **Current Status:** Done (completed as part of 6.1)
  - **Estimated Completion:** 2025-06-26
  - **Implementation Notes:**
    - Implemented Expected Improvement, Upper Confidence Bound, Probability of Improvement
    - Added Thompson sampling with neural network uncertainty estimates
    - Created numerical stability improvements for acquisition functions
    - Ensured proper handling of edge cases and error conditions
    - All acquisition functions compatible with batched neural network outputs

- **Subtask 6.3: Create budget-aware intervention selection (Done ✅)**
  - **Current Status:** Done (completed as part of 6.1)
  - **Estimated Completion:** 2025-06-26
  - **Implementation Notes:**
    - Implemented intervention selection with budget constraints
    - Added per-node cost specification
    - Created flexible mechanism for handling single or multiple intervention values
    - Added comprehensive tests for budget constraints

- **Subtask 6.4: Update example workflow (Done ✅)**
  - Create end-to-end example workflow in `examples/amortized_cbo_workflow.py`
  - Add tutorial-style documentation and comments
  - Create parameter recommendations
  - Implement progress tracking and visualization
  - Create a Jupyter notebook tutorial
  - **Current Status:** Done
  - **Estimated Completion:** 2025-06-27
  - **Implementation Notes:**
    - Successfully implemented end-to-end workflow in `examples/amortized_cbo_workflow.py`
    - Added comprehensive documentation with step-by-step explanations
    - Implemented synthetic data generation, model setup, and configuration
    - Created simplified training approach for demonstration purposes
    - Added graph inference and visualization components
    - Implemented full optimization loop for intervention selection
    - Created results visualization and analysis tools
    - Ensured all tests are passing with the implemented example
    - Added parameter recommendations for optimal performance

#### Task 7: Create Evaluation Framework and Benchmarks (Pending ⏱️)
- **Description:** Develop comprehensive evaluation tools and benchmarks for the amortized approach.
- **Priority:** medium
- **Status:** pending
- **Dependencies:** 6

- **Subtask 7.1: Implement benchmark suite (Pending ⏱️)**
  - Create standardized benchmark problems
  - Implement performance metrics
  - Add baseline comparison methods
  - Create synthetic and semi-synthetic datasets
  - Implement automated benchmark execution

- **Subtask 7.2: Implement visualization components (Pending ⏱️)**
  - Create tools for visualizing graph inference results
  - Implement intervention outcome visualization
  - Add optimization progress visualization
  - Create performance comparison plots
  - Implement uncertainty visualization

- **Subtask 7.3: Implement scalability testing (Pending ⏱️)**
  - Create tools for testing performance on increasingly larger graphs
  - Implement memory and runtime profiling
  - Add automated scaling tests
  - Create performance vs. graph size plots
  - Document scalability limits and recommendations

- **Subtask 7.4: Create comprehensive documentation (Pending ⏱️)**
  - Write detailed API documentation
  - Create usage tutorials and examples
  - Add theoretical background and explanations
  - Document best practices
  - Create a project website

#### Task 8: Demo Scripts for Supervisor Meeting (In Progress 🔄)
- **Description:** Create demonstration scripts for showcasing the functionality of the implemented components.
- **Priority:** high
- **Status:** in-progress
- **Dependencies:** 4, 5, 6

- **Subtask 8.1: Create Simple Parent-Scaled ACD Demo (Status: `done`)**
  - **Implementation Details:**
    - Implemented `demos/parent_scale_acd_demo.py` with proper graph visualization and error handling
    - Created compatibility layer to gracefully handle missing dependencies
    - Added proper handling of graph objects for visualization functions
    - Implemented helper function `create_graph_from_adjacency` to convert adjacency matrices to proper graph objects
    - Added visualization of ground truth vs. inferred graphs
    - Successfully tested the script with minimal settings using the `--quick` flag
    - Identified and fixed interface mismatch issues in the `plot_graph` function
  - **Key Learning:**
    - Interface mismatches between components are a common source of errors
    - Always check expected parameter types and object interfaces
    - Implement fallback mechanisms for graceful degradation
    - Using proper error handling allows for more robust scripts
    - Visualization functions often expect specific object types/interfaces

- **Subtask 8.2: Create Full Amortized ACD Pipeline Demo (Status: `in-progress`)**
  - **Implementation Details:**
    - Created initial structure for `full_acd_pipeline_demo.py`
    - Identified issues with graph representation that need to be fixed
    - Need to apply similar visualization fixes as in parent_scale_acd_demo.py
    - Needs implementation of proper task family creation and visualization
    - Requires integration with meta-learning components
  - **Key Learning:**
    - Need to ensure consistent graph representations across the pipeline
    - Meta-learning components require additional error handling
    - Visualization of task families requires special attention
    - Training loop needs to be simplified for demonstration purposes

- **Subtask 8.3: Create Demo Documentation (Status: `in-progress`)**
  - **Implementation Details:**
    - Created basic structure for documentation
    - Need to document command-line arguments and usage instructions
    - Need to create README with example commands and expected outputs
    - Required clear explanation of key concepts and implementation details
  - **Key Learning:**
    - Documentation should include troubleshooting for common errors
    - Include examples of expected visualizations
    - Provide clear setup instructions for dependencies
    - Document limitations and assumptions of the demo scripts

- **Subtask 8.4: Restructure Demos to Leverage Existing Components (New)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - Current demos contain many duplicate implementations instead of using our existing codebase
      - Need to identify which components from our codebase can be used directly
      - Should make the demos shorter and more maintainable by reusing existing code
      - Must ensure demos are robust by properly handling errors and fallbacks

    - **Thought 2: Component Identification**
      - Core components to leverage:
        - `causal_meta.graph.causal_graph.CausalGraph` for graph representation
        - `causal_meta.environments.scm.StructuralCausalModel` for data generation
        - `causal_meta.graph.generators.factory.GraphFactory` for graph creation
        - `causal_meta.graph.visualization` for plotting functions
        - `causal_meta.meta_learning.acd_models.GraphEncoder` for graph structure inference
        - `causal_meta.meta_learning.dynamics_decoder.DynamicsDecoder` for dynamics modeling
        - `causal_meta.meta_learning.amortized_causal_discovery.AmortizedCausalDiscovery` for unified approach
        - `causal_meta.meta_learning.amortized_cbo.AmortizedCBO` for intervention optimization

    - **Thought 3: Implementation Approach**
      - Use direct imports from existing modules instead of duplicate implementations
      - Implement fallback mechanisms only where necessary for robustness
      - Replace dummy implementations with proper error handling
      - Create consistent interface between demos using shared utility functions
      - Ensure proper handling of tensor shapes and dimensions
      - Add better error messages and debugging capabilities

    - **Thought 4: Potential Challenges**
      - Node naming consistency between SCM and neural components
      - Handling tensor dimension issues between different components
      - Managing dependencies and ensuring proper component initialization
      - Graceful degradation when some components are not available
      - Path management for assets and model checkpoints
      - Ensuring reproducibility across different runs

    - **Thought 5: Implementation Plan**
      - Analyze both demo scripts to identify duplicated code
      - Replace duplicated implementations with imports from our codebase
      - Add proper error handling and fallback mechanisms
      - Create shared utility functions for common operations
      - Test demos with different settings to ensure robustness
      - Refactor for cleaner and more concise code

  - **Detailed Implementation Steps:**
    1. Analyze and map duplicated functionality to existing codebase components
    2. Update imports in both demo scripts to use existing implementations
    3. Replace dummy classes with proper error handling and fallbacks
    4. Fix tensor shape and dimension issues in data processing
    5. Implement consistent node naming and identifier handling
    6. Add better asset management and directory structure
    7. Create uniform logging and error reporting
    8. Test with various settings to ensure robustness

  - **Current Status:** pending
  - **Estimated Completion:** TBD