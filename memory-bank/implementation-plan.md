# Implementation Plan

This document outlines the detailed implementation plan for our **Causal Graph Structure Learning MVP** based on the new, urgent direction. The MVP will demonstrate that neural networks can learn causal graph structures from observational and interventional data.

## Priority: MVP Causal Graph Structure Learning

Our immediate focus is developing this MVP to demonstrate progressive learning of causal graph structure through iterative interventions using a simple neural network approach.

## MVP Tasks

### Task 1: Environment Setup and Scaffolding

- **Description:** Set up basic project structure and dependencies
- **Priority:** Highest
- **Status:** pending
- **Dependencies:** none

#### Subtask 1.1: Project Configuration

  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
    - Need to create a clean environment for the MVP implementation
    - Must ensure all dependencies are properly managed
    - Need to set up appropriate configuration for reproducibility
    - Should leverage existing codebase where possible

    - **Thought 2: Component Identification**
    - Project configuration files (pyproject.toml, requirements.txt)
    - Random seed management
    - Core imports from existing codebase
    - Basic project structure

    - **Thought 3: Implementation Approach**
    - Review existing `causal_meta` components to identify reusable parts
    - Update dependencies if needed
    - Set up configuration parameters based on PRD
    - Create constants and configuration management

    - **Thought 4: Potential Challenges**
    - Ensuring compatibility with existing codebase
    - Managing project scope to focus only on MVP requirements
    - Avoiding unnecessary complexity
    - Ensuring reproducibility across different seeds

    - **Thought 5: Implementation Plan**
    - Define all necessary constants and configuration parameters
    - Set up random seed management
    - Create utility functions for reproducibility
    - Establish clear imports from existing codebase
    - Build basic scaffolding for the MVP implementation

  - **Detailed Implementation Steps:**
  1. Review existing imports and dependencies in `causal_meta`
  2. Create configuration parameters based on PRD specifications
  3. Set up random seed management for reproducibility
  4. Define constants in a central configuration file
  5. Create main experiment script structure
  6. Set up logging and basic visualization utilities

- **Current Status:** pending
- **Estimated Completion:** 1 day

### Task 2: Graph Generation and SCM Implementation

- **Description:** Implement random DAG generation and linear SCM model
- **Priority:** High
- **Status:** pending
- **Dependencies:** Task 1

#### Subtask 2.1: Random DAG Generation

  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
    - Need to create random DAGs with specified number of nodes
    - Must ensure acyclicity during generation
    - Need to control edge probability for appropriate density
    - Should convert to appropriate adjacency matrix format

    - **Thought 2: Component Identification**
    - Existing `GraphFactory` and graph generators in `causal_meta.graph.generators`
    - NetworkX for graph operations
    - Adjacency matrix representation
    - Acyclicity validation

    - **Thought 3: Implementation Approach**
    - Leverage existing graph generation utilities where appropriate
    - Use topological sorting to ensure acyclicity
    - Implement method to generate random DAGs with specific properties
    - Provide conversion to adjacency matrix format

    - **Thought 4: Potential Challenges**
    - Ensuring consistent generation across different random seeds
    - Balancing graph complexity for the MVP
    - Proper validation of acyclicity
    - Interface compatibility with existing components

    - **Thought 5: Implementation Plan**
    - Review existing graph generation components
    - Implement random DAG generation function
    - Add validation for acyclicity
    - Create conversion to adjacency matrix
    - Add visualization utilities for generated graphs
    - Test with different parameters to ensure proper behavior

  - **Detailed Implementation Steps:**
  1. Review existing `GraphFactory` implementation in `causal_meta.graph.generators`
  2. Implement `generate_random_dag` function based on PRD
  3. Add acyclicity validation
  4. Create conversion to adjacency matrix
  5. Implement visualization for generated DAGs
  6. Test with different configurations

  - **Current Status:** pending
- **Estimated Completion:** 1 day

#### Subtask 2.2: Linear SCM Implementation

  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
    - Need to create linear structural causal models
    - Must implement random weight generation in specified range
    - Need to add Gaussian noise with configurable scale
    - Should ensure compatibility with existing SCM interface

    - **Thought 2: Component Identification**
    - Existing `StructuralCausalModel` in `causal_meta.environments.scm`
    - Linear equations and coefficients
    - Noise distributions
    - Data generation utilities

    - **Thought 3: Implementation Approach**
    - Leverage existing SCM implementation where possible
    - Create linear equation generator with random weights
    - Implement Gaussian noise with configurable scale
    - Ensure data generation for both observational and interventional settings

    - **Thought 4: Potential Challenges**
    - Ensuring compatibility with existing SCM interface
    - Proper implementation of linear mechanisms
    - Managing the complexity of the SCM implementation
    - Handling intervention mechanics correctly

    - **Thought 5: Implementation Plan**
    - Review existing SCM implementation
    - Create helper function for generating linear SCMs
    - Add utilities for random weight generation
    - Implement linear mechanisms with noise
    - Test data generation for both observational and interventional settings

  - **Detailed Implementation Steps:**
  1. Review existing `StructuralCausalModel` implementation
  2. Implement `generate_linear_scm` function based on PRD
  3. Create utilities for random weight generation in specified range
  4. Add Gaussian noise with configurable scale
  5. Test data generation for both observational and interventional data
  6. Validate generated data against expected distributions

  - **Current Status:** pending
- **Estimated Completion:** 1-2 days

### Task 3: Data Generation and Processing

- **Description:** Implement observational and interventional data generation and processing
- **Priority:** High
- **Status:** pending
- **Dependencies:** Task 2

#### Subtask 3.1: Data Generation Functions

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to generate observational data from the SCM
    - Must implement random intervention mechanism
    - Need to collect interventional data with proper tracking
    - Should prepare data for neural network input

  - **Thought 2: Component Identification**
    - Existing data generation in `StructuralCausalModel`
    - Intervention mechanism
    - Data formatting for neural network
    - Tensor conversion utilities

  - **Thought 3: Implementation Approach**
    - Leverage existing SCM data generation where possible
    - Implement random intervention selection
    - Create intervention data collection with proper tracking
    - Prepare data conversion for neural network input

  - **Thought 4: Potential Challenges**
    - Ensuring proper intervention representation
    - Managing data formats for the neural network
    - Tracking intervention information correctly
    - Balancing between using existing code and custom implementation

  - **Thought 5: Implementation Plan**
    - Review existing data generation utilities
    - Implement observational data generation function
    - Create random intervention mechanism
    - Add intervention data collection with tracking
    - Implement data conversion for neural network input
    - Test the entire data generation pipeline

  - **Detailed Implementation Steps:**
  1. Implement `generate_observational_data` function based on PRD
  2. Create `perform_random_intervention` function for intervention generation
  3. Implement data conversion utilities for neural network input
  4. Add intervention tracking mechanisms
  5. Test the entire data generation pipeline
  6. Validate generated data format and content

  - **Current Status:** pending
- **Estimated Completion:** 1 day

#### Subtask 3.2: Data Processing and Formatting

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to process raw data for neural network input
    - Must encode intervention information explicitly
    - Need to create appropriate tensor formats
    - Should manage batched data processing

  - **Thought 2: Component Identification**
    - Tensor conversion utilities
    - Intervention encoding mechanism
    - Batch processing
    - Neural network input format

  - **Thought 3: Implementation Approach**
    - Create explicit encoding for intervention information
    - Implement tensor conversion utilities
    - Add batch processing capabilities
    - Ensure compatibility with neural network input expectations

  - **Thought 4: Potential Challenges**
    - Properly encoding intervention information
    - Managing tensor shapes and dimensions
    - Ensuring efficient batch processing
    - Compatibility with neural network architecture

  - **Thought 5: Implementation Plan**
    - Design intervention encoding format
    - Implement tensor conversion utilities
    - Add batch processing capabilities
    - Test data format compatibility with neural network
    - Optimize for efficiency

  - **Detailed Implementation Steps:**
  1. Implement `convert_to_tensor` function for tensor conversion
  2. Create intervention encoding mechanism
  3. Implement batch processing utilities
  4. Test compatibility with neural network input expectations
  5. Optimize for efficiency

  - **Current Status:** pending
- **Estimated Completion:** 1 day

### Task 4: Neural Network Model Implementation

- **Description:** Implement the neural network for causal graph structure learning
- **Priority:** Highest
- **Status:** pending
- **Dependencies:** Task 1

#### Subtask 4.1: SimpleGraphLearner Implementation

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to create a simple MLP-based neural network
    - Must accept node values and intervention indicators
    - Need to output adjacency matrix (graph structure)
    - Should implement loss function with regularization

  - **Thought 2: Component Identification**
    - MLP architecture
    - Input encoding for node values and interventions
    - Output format for adjacency matrix
    - Loss function with appropriate regularization

  - **Thought 3: Implementation Approach**
    - Create a simple MLP architecture based on PRD
    - Implement input encoding for node values and interventions
    - Add output formatting for adjacency matrix
    - Implement comprehensive loss function

  - **Thought 4: Potential Challenges**
    - Properly handling intervention information
    - Balancing regularization terms
    - Ensuring acyclicity constraint is properly enforced
    - Managing tensor shapes and dimensions

  - **Thought 5: Implementation Plan**
    - Define neural network architecture
    - Implement forward pass with intervention information
    - Add output formatting for adjacency matrix
    - Create comprehensive loss function
    - Test model with sample data

- **Detailed Implementation Steps:**
  1. Create `SimpleGraphLearner` class with specified architecture
  2. Implement forward pass with node values and intervention indicators
  3. Add output formatting for adjacency matrix
  4. Implement comprehensive loss function with regularization
  5. Add methods for training and evaluation
  6. Test model with sample data

- **Current Status:** pending
- **Estimated Completion:** 2 days

#### Subtask 4.2: Loss Functions and Regularization

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to implement appropriate loss functions
    - Must add sparsity regularization for fewer edges
    - Need to enforce acyclicity for valid DAG structure
    - Should balance different regularization terms

  - **Thought 2: Component Identification**
    - Binary cross-entropy loss for supervised learning
    - Sparsity regularization
    - Acyclicity constraint
    - Regularization balancing

  - **Thought 3: Implementation Approach**
    - Implement BCE loss for supervised learning
    - Add sparsity regularization
    - Implement acyclicity constraint using matrix exponential
    - Create mechanism for balancing regularization terms

  - **Thought 4: Potential Challenges**
    - Properly implementing acyclicity constraint
    - Balancing different regularization terms
    - Ensuring stable gradient flow
    - Computing matrix exponential efficiently

  - **Thought 5: Implementation Plan**
    - Implement BCE loss for supervised learning
    - Add sparsity regularization
    - Create acyclicity constraint using matrix exponential
    - Implement regularization balancing
    - Test loss function behavior with different inputs

- **Detailed Implementation Steps:**
  1. Implement BCE loss for supervised learning
  2. Add sparsity regularization with configurable weight
  3. Create acyclicity constraint using matrix exponential
  4. Implement regularization balancing mechanism
  5. Test loss function behavior with different inputs
  6. Validate gradient flow and numerical stability

- **Current Status:** pending
- **Estimated Completion:** 1 day

### Task 5: Training and Evaluation Functions

- **Description:** Implement training and evaluation utilities for the model
- **Priority:** High
- **Status:** pending
- **Dependencies:** Task 3, Task 4

#### Subtask 5.1: Training Function Implementation

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to implement training step functionality
    - Must handle both observational and interventional data
    - Need to manage optimization process
    - Should track training progress and metrics

  - **Thought 2: Component Identification**
    - Training step function
    - Optimizer configuration
    - Progress tracking
    - Batch processing during training

  - **Thought 3: Implementation Approach**
    - Create training step function for single update
    - Implement batch processing during training
    - Add progress tracking and metrics
    - Configure optimizer properly

  - **Thought 4: Potential Challenges**
    - Managing both observational and interventional data
    - Ensuring stable training dynamics
    - Proper backpropagation and gradient flow
    - Tracking relevant metrics during training

  - **Thought 5: Implementation Plan**
    - Implement training step function
    - Add batch processing during training
    - Create progress tracking and metrics
    - Configure optimizer
    - Test training process with sample data

- **Detailed Implementation Steps:**
  1. Implement `train_step` function for single update
  2. Add batch processing during training
  3. Create progress tracking and metrics calculation
  4. Configure optimizer with appropriate parameters
  5. Test training process with sample data
  6. Validate training dynamics and stability

- **Current Status:** pending
- **Estimated Completion:** 1 day

#### Subtask 5.2: Evaluation Function Implementation

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to evaluate predicted graph against true graph
    - Must calculate appropriate metrics (accuracy, SHD)
    - Need to track progress during training
    - Should provide visualization of results

  - **Thought 2: Component Identification**
    - Structural Hamming Distance (SHD) calculation
    - Accuracy computation
    - Evaluation function
    - Visualization utilities

  - **Thought 3: Implementation Approach**
    - Implement evaluation function with multiple metrics
    - Add SHD calculation
    - Create accuracy computation
    - Implement visualization utilities for results

  - **Thought 4: Potential Challenges**
    - Properly calculating SHD
    - Creating informative visualizations
    - Managing binary threshold for edge prediction
    - Providing comprehensive evaluation metrics

  - **Thought 5: Implementation Plan**
    - Implement evaluation function with metrics
    - Add SHD calculation
    - Create accuracy computation
    - Implement visualization utilities
    - Test evaluation function with sample results

- **Detailed Implementation Steps:**
  1. Implement `evaluate_graph` function with comprehensive metrics
  2. Add SHD calculation functionality
  3. Create accuracy computation for edge prediction
  4. Implement visualization utilities for results presentation
  5. Test evaluation function with sample results
  6. Validate metrics against expected values

- **Current Status:** pending
- **Estimated Completion:** 1 day

### Task 6: Main Experiment Loop

- **Description:** Implement the main experiment loop for progressive learning
- **Priority:** High
- **Status:** pending
- **Dependencies:** Task 2, Task 3, Task 4, Task 5

#### Subtask 6.1: Progressive Intervention Loop

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to implement the main experiment loop
    - Must manage progressive interventions
    - Need to track learning progress
    - Should handle model updates with both observational and interventional data

  - **Thought 2: Component Identification**
    - Main experiment function
    - Progressive intervention loop
    - Learning progress tracking
    - Model update mechanism

  - **Thought 3: Implementation Approach**
    - Create main experiment function
    - Implement progressive intervention loop
    - Add learning progress tracking
    - Create model update mechanism
    - Implement evaluation during the loop

  - **Thought 4: Potential Challenges**
    - Managing the intervention sequence
    - Properly updating the model with different data types
    - Tracking and visualizing learning progress
    - Handling convergence criteria

  - **Thought 5: Implementation Plan**
    - Implement main experiment function
    - Create progressive intervention loop
    - Add learning progress tracking
    - Implement model update mechanism
    - Add evaluation during the loop
    - Create convergence checking

- **Detailed Implementation Steps:**
  1. Implement `run_experiment` function based on PRD
  2. Create progressive intervention loop
  3. Add learning progress tracking with history dictionary
  4. Implement model update mechanism for both data types
  5. Add evaluation during the loop
  6. Create convergence checking and early stopping
  7. Test the entire experiment pipeline

- **Current Status:** pending
- **Estimated Completion:** 2 days

#### Subtask 6.2: Experiment Configuration and Parameter Tuning

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to configure experiment parameters
    - Must tune model and training parameters
    - Need to ensure reproducibility across runs
    - Should optimize for performance and stability

  - **Thought 2: Component Identification**
    - Configuration parameters
    - Model hyperparameters
    - Training settings
    - Reproducibility management

  - **Thought 3: Implementation Approach**
    - Define optimal configuration parameters
    - Tune model hyperparameters
    - Configure training settings
    - Ensure reproducibility management

  - **Thought 4: Potential Challenges**
    - Finding optimal parameter values
    - Balancing performance and stability
    - Ensuring reproducibility
    - Managing configuration complexity

  - **Thought 5: Implementation Plan**
    - Define configuration parameters from PRD
    - Tune model hyperparameters for optimal performance
    - Configure training settings for stability
    - Implement reproducibility management
    - Test different configurations

- **Detailed Implementation Steps:**
  1. Define all configuration parameters from PRD
  2. Tune model hyperparameters (hidden dim, learning rate, etc.)
  3. Configure training settings for optimal performance
  4. Implement reproducibility management
  5. Test different configurations and document results
  6. Select optimal configuration for the final implementation

- **Current Status:** pending
- **Estimated Completion:** 1-2 days

### Task 7: Visualization and Results Analysis

- **Description:** Implement visualization utilities and results analysis
- **Priority:** Medium
- **Status:** pending
- **Dependencies:** Task 6

#### Subtask 7.1: Visualization Functions

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to visualize true and learned graphs
    - Must plot learning progress metrics
    - Need to create comprehensive visualizations
    - Should save visualizations for documentation

  - **Thought 2: Component Identification**
    - Graph visualization
    - Learning curves plotting
    - Results visualization
    - Saving functionality

  - **Thought 3: Implementation Approach**
    - Implement graph visualization using NetworkX
    - Create learning curves plotting
    - Add comprehensive results visualization
    - Implement saving functionality

  - **Thought 4: Potential Challenges**
    - Creating clear and informative visualizations
    - Managing multiple plots in a single figure
    - Properly formatting visualization elements
    - Saving high-quality figures

  - **Thought 5: Implementation Plan**
    - Implement graph visualization function
    - Create learning curves plotting utility
    - Add comprehensive results visualization
    - Implement saving functionality
    - Test visualizations with sample results

- **Detailed Implementation Steps:**
  1. Implement `plot_results` function based on PRD
  2. Create graph visualization using NetworkX
  3. Add learning curves plotting for SHD and accuracy
  4. Implement comprehensive results visualization
  5. Add saving functionality with configurable paths
  6. Test visualizations with sample results

- **Current Status:** pending
- **Estimated Completion:** 1 day

#### Subtask 7.2: Results Analysis and Documentation

- **Sequential Thinking Analysis:**
  - **Thought 1: Problem Understanding**
    - Need to analyze experimental results
    - Must document findings and insights
    - Need to compare different configurations
    - Should provide clear interpretation of results

  - **Thought 2: Component Identification**
    - Results analysis utilities
    - Documentation framework
    - Comparison methodology
    - Interpretation guidelines

  - **Thought 3: Implementation Approach**
    - Create results analysis utilities
    - Implement documentation framework
    - Add comparison methodology
    - Provide interpretation guidelines

  - **Thought 4: Potential Challenges**
    - Extracting meaningful insights from results
    - Creating clear and concise documentation
    - Comparing different configurations fairly
    - Providing objective interpretation

  - **Thought 5: Implementation Plan**
    - Implement results analysis utilities
    - Create documentation framework
    - Add comparison methodology
    - Provide interpretation guidelines
    - Test with experimental results

- **Detailed Implementation Steps:**
  1. Create results analysis utilities for experiment output
  2. Implement documentation framework with standardized format
  3. Add comparison methodology for different configurations
  4. Provide interpretation guidelines for results
  5. Test with actual experimental results
  6. Create comprehensive documentation of findings

- **Current Status:** pending
- **Estimated Completion:** 1 day

## Timeline and Priority

1. **Week 1 (Immediate Focus)**:
   - Task 1: Environment Setup and Scaffolding (1 day)
   - Task 2: Graph Generation and SCM Implementation (2 days)
   - Task 3: Data Generation and Processing (2 days)
   - Task 4: Neural Network Model Implementation (3 days, overlapping)

2. **Week 2**:
   - Task 5: Training and Evaluation Functions (2 days)
   - Task 6: Main Experiment Loop (3 days)
   - Task 7: Visualization and Results Analysis (2 days)

## Important Notes

1. **Interface Validation**: All implementations should validate interfaces against existing components in the `causal_meta` package before implementation to ensure compatibility.

2. **Template Functions**: The template functions provided in the PRD are meant as guides rather than "ground truth". They should be adapted based on the actual components and interfaces available in the codebase.

3. **Modularity**: Ensure implementations are modular and can be easily extended or modified for future work.

4. **Documentation**: Maintain comprehensive documentation throughout the implementation process.

5. **Testing**: Implement thorough testing at each stage to ensure components work as expected.