# Experiment Design Document

## Project Goal

Develop an Amortized Causal Bayesian Optimization (AmortizedCBO) framework. The core idea is to leverage neural networks to learn both causal structures and dynamics models from data, enabling efficient and scalable causal discovery and intervention optimization on large graphs. This approach will allow us to perform Bayesian optimization to find optimal interventions on complex causal systems without requiring exhaustive enumeration of possible graph structures.

## Current Direction & Approach: Amortized Causal Discovery

1.  **Represent Causal Tasks:** Use `StructuralCausalModel` (SCM) objects, built upon `CausalGraph` representations (DAGs) for ground truth data generation.
2.  **Neural Causal Structure Learning:**
    - Implement a `GraphEncoder` that infers causal edge probabilities from observational and interventional data.
    - Apply regularization to enforce sparsity and acyclicity constraints.
    - Implement uncertainty quantification for edge predictions.
3.  **Neural Dynamics Modeling:**
    - Implement a `DynamicsDecoder` that learns the functional relationships between variables.
    - Design the model to predict counterfactual outcomes under interventions.
    - Support both continuous and discrete node value predictions.
4.  **Unified Amortized Framework:**
    - Create an `AmortizedCausalDiscovery` class that combines the graph structure and dynamics components.
    - Implement joint training procedures with losses that balance structure learning and dynamics modeling.
    - Provide methods for inference of both graph structure and intervention outcomes.
5.  **Meta-Learning & Transfer:**
    - Generate task families using `TaskFamilyGenerator` to create sets of related causal graphs.
    - Implement meta-learning techniques to enable few-shot adaptation to new tasks.
    - Incorporate a curriculum learning approach to gradually increase graph complexity during training.
6.  **Bayesian Optimization with Neural Surrogates:**
    - Replace traditional GP-based BO with neural surrogate models.
    - Implement acquisition functions compatible with the neural predictions.
    - Develop intervention selection strategies that leverage the inferred causal structure.
    - Create a complete `AmortizedCBO` class that provides an end-to-end solution.

## Key Design Considerations & Trade-offs

-   **Neural Architecture Selection:** Balance between model expressivity and computational efficiency. Graph neural networks with attention mechanisms show promise for capturing complex causal relationships.
-   **Regularization:** Enforcing DAG constraints in neural networks is challenging. Need to explore spectral normalization, continuous optimization of acyclicity constraints, or other approaches.
-   **Uncertainty Quantification:** Essential for reliable decision-making. Will implement ensemble methods or dropout-based uncertainty estimates.
-   **Scalability vs. Accuracy:** Amortized approaches provide scalability but may sacrifice some accuracy compared to exact methods on small graphs. This trade-off is acceptable given our focus on larger graphs where exact methods are intractable.
-   **Training Data Requirements:** Neural methods require substantial training data. Will implement efficient data generation with curriculum learning.
-   **Adaptation Mechanisms:** Need to balance pre-training (general knowledge) and adaptation (task-specific fine-tuning) for optimal performance on new tasks.

## Scalability Advantages

-   **Graph Size:** Neural approaches can scale to much larger graphs (100+ nodes) compared to exact methods or GP-based approaches.
-   **Computational Efficiency:** Once trained, inference is extremely fast (milliseconds) compared to exact posterior computation.
-   **Memory Usage:** Fixed model size regardless of graph complexity, unlike traditional methods where memory requirements grow exponentially with graph size.
-   **Parallelization:** Neural network training and inference can leverage GPU acceleration.
-   **Inductive Bias:** The neural architecture can incorporate domain knowledge and structural priors about causal systems.

## Potential Challenges & Mitigations

-   **Training Stability:** Neural networks can be challenging to train. Will implement learning rate scheduling, gradient clipping, and early stopping.
-   **DAG Constraints:** Enforcing acyclicity is difficult. Will explore recent advances in differentiable DAG constraints and spectral approaches.
-   **Evaluation Metrics:** Need to develop comprehensive metrics for both structure discovery and intervention outcomes. Will implement established metrics (SHD, precision/recall) and develop custom ones as needed.
-   **Uncertainty Calibration:** Ensuring reliable uncertainty estimates is crucial. Will implement calibration techniques and validation procedures.
-   **Hyperparameter Tuning:** Neural approaches have many hyperparameters. Will use systematic search or Bayesian optimization for tuning.

## Implementation Roadmap

1.  **Foundation (Completed):**
    - Reliable DAG generation tools
    - Task family generation capabilities
    - StructuralCausalModel implementation
2.  **Neural Components (Next):**
    - Graph encoder implementation
    - Dynamics decoder implementation 
    - Training pipeline and data generation
3.  **Unified Framework:**
    - AmortizedCausalDiscovery integration
    - Meta-learning and adaptation mechanisms
    - Uncertainty quantification
4.  **Optimization Integration:**
    - AmortizedCBO implementation
    - Acquisition functions and intervention selection
    - Example workflows and tutorials
5.  **Evaluation & Benchmarking:**
    - Comprehensive benchmark suite
    - Scaling tests
    - Comparison to baseline methods

## Style and Form Recommendations

-   **Modular Neural Components:** Design neural components with clear interfaces that can be combined flexibly.
-   **Batched Operations:** Optimize for batch processing throughout the codebase for efficient GPU utilization.
-   **Clear Abstractions:** Define interfaces between physics-based (SCM) and neural components.
-   **Functional Style:** Where appropriate, use functional programming patterns for data transformations.
-   **Stateful Objects:** For trained models, use proper object-oriented design with clear serialization/deserialization.
-   **Testing:** Implement comprehensive unit tests, especially for numerical components.
-   **Type Hinting:** Use Python's `typing` module with `jaxtyping` extensions for tensor shapes.
-   **Documentation:** Clear docstrings and tutorials, especially for the neural components.
-   **Configuration:** Use Hydra for managing complex experiment configurations.

## Potential Alternatives

-   **Different Meta-Learning Algorithms:** Instead of MAML, consider Reptile, MetaOptNet, or others.
-   **Alternative BO Models:** Instead of GPs, consider Bayesian Neural Networks or Random Forests for the surrogate model.
-   **Direct Policy Learning:** Meta-learn an intervention *policy* directly instead of a GP prior.

## Potential Blockers

-   Difficulty in creating meaningful task representations.
-   Instability or slow convergence of MAML training.
-   Challenges in defining and optimizing a suitable causal acquisition function.
-   Computational cost limiting the complexity of tasks or the number of meta-training iterations.
-   Integrating complex causal effect estimation methods if needed.

## Adhere to Clean Code/Architecture principles:

-   Focus on readability, maintainability, and separation of concerns.

## Implementation Progress Summary (June 24, 2025)

We have made significant progress in implementing the Amortized Causal Discovery framework, following our research pivot from GP-based surrogate models to neural network-based approaches. This shift aligns with our goal of scaling causal discovery and intervention prediction to larger graphs.

### Key Accomplishments

1. **Core Components**: 
   - Implemented reliable DAG generation for synthetic data creation
   - Created task family generation for meta-learning scenarios
   - Integrated the StructuralCausalModel for data generation and intervention simulation

2. **Neural Causal Discovery**:
   - Implemented GraphEncoder with attention-based architecture for graph structure inference
   - Created graph inference utilities with thresholding and posterior sampling
   - Developed training pipeline with curriculum learning and regularization
   - Implemented synthetic data generation with observational and interventional data

3. **Dynamics Modeling**:
   - Implemented DynamicsDecoder with Graph Attention Network architecture
   - Added intervention conditioning mechanisms for counterfactual prediction
   - Incorporated ensemble-based uncertainty quantification
   - Used skip connections and layer normalization for stable training

4. **Unified Framework**:
   - Created AmortizedCausalDiscovery class integrating GraphEncoder and DynamicsDecoder
   - Implemented joint training with balanced loss functions
   - Added high-level API for graph inference and intervention prediction
   - Created utilities for converting between adjacency matrices and CausalGraph objects
   - Added model serialization and loading functionality

5. **Meta-Learning Design**:
   - Established test suite for meta-learning capabilities
   - Designed TaskEmbedding component for graph structure representation
   - Planned MAML-based approach for few-shot adaptation
   - Created test fixtures for synthetic meta-learning task data

### Methodological Improvements

1. **Workflow Integration**:
   - Successfully integrated Sequential Thinking with Memory Bank for structured problem-solving
   - Implemented TDD approach across all components
   - Created comprehensive test suites before implementation
   - Documented thinking process in implementation plan and code comments

2. **Architecture Decisions**:
   - Chose GATv2Conv for message passing to capture edge importance
   - Implemented ensemble methods for uncertainty quantification
   - Added intervention conditioning through feature modification
   - Created unified training approach with weighted objectives

### Next Steps

1. **Meta-Learning Implementation** (Highest priority):
   - Implement TaskEmbedding class for graph structure representation
   - Create MAML implementation with inner/outer optimization loops
   - Add few-shot adaptation capabilities to AmortizedCausalDiscovery
   - Develop meta-training procedures with task sampling

2. **Technical Improvements**:
   - Standardize APIs across the codebase
   - Fix test failures related to thresholds and expected values
   - Resolve tensor shape compatibility issues
   - Optimize acyclicity constraint implementation

3. **Evaluation and Benchmarking**:
   - Create comprehensive benchmark suite
   - Implement visualization tools for results analysis
   - Test scalability to larger graphs
   - Compare with baseline approaches

4. **Causal Bayesian Optimization**:
   - Implement acquisition functions compatible with neural predictions
   - Create budget-aware intervention selection
   - Add multi-objective optimization capabilities
   - Develop planning strategies for sequential interventions

This progress positions us well to complete the meta-learning capabilities in the next phase, enabling few-shot adaptation to new causal structures with minimal data. The integrated Sequential Thinking + Memory Bank workflow has proven effective and will continue to guide our implementation efforts.

## Benchmarking and Evaluation Framework

A comprehensive benchmarking framework has been implemented to rigorously evaluate our research hypotheses regarding amortized causal discovery and optimization methods. This framework allows systematic comparison of our neural approaches against traditional methods.

### Key Evaluation Objectives

1. **Structural Learning Performance**: Assess how well our neural approaches recover true causal structures compared to traditional methods
   - Metrics: Structural Hamming Distance (SHD), precision, recall, F1 score
   - Test environments: Varying graph sizes, densities, noise levels, and functional relationships

2. **Intervention Optimization Efficiency**: Evaluate sample efficiency and quality of intervention strategies
   - Metrics: Best value found, regret, improvement ratio, sample efficiency
   - Test scenarios: Varying intervention budgets, known vs. unknown graph structures

3. **Computational Requirements**: Measure runtime and memory usage to assess practical deployment feasibility
   - Analysis: Scaling behavior with increasing graph size
   - Metrics: Runtime, memory usage, complexity class estimation

4. **Transferability Benefits**: Quantify the advantages of amortized/meta-learning approaches in transfer scenarios
   - Scenario: Training on a family of related causal systems, testing on unseen but related systems
   - Metrics: Performance gap between seen and unseen environments

### Benchmarking Components

The benchmarking framework consists of several specialized components:

1. **CausalDiscoveryBenchmark**: For evaluating structure learning methods
2. **CBOBenchmark**: For evaluating intervention optimization approaches
3. **ScalabilityBenchmark**: For analyzing computational scaling behavior
4. **BenchmarkRunner**: For orchestrating comprehensive evaluations across multiple scenarios

Each benchmark generates standardized test environments with ground truth causal structures, enabling fair and reproducible comparisons between methods.

### Statistical Validity

To ensure the statistical validity of our findings:

1. All experiments use multiple random seeds for initialization
2. Results are aggregated across numerous test cases (typically 20+ per configuration)
3. Statistical significance testing is performed for method comparisons
4. Confidence intervals are reported for all performance metrics
5. Ablation studies isolate the contribution of each component

### Integration with Research Hypotheses

The benchmarking framework directly addresses our core research questions:

1. **Hypothesis 1**: Neural causal discovery methods can achieve competitive or superior structural accuracy compared to traditional methods.
   - Tested via: CausalDiscoveryBenchmark with structural metrics

2. **Hypothesis 2**: Amortized approaches provide significant sample efficiency advantages for intervention optimization.
   - Tested via: CBOBenchmark with sample efficiency metrics

3. **Hypothesis 3**: Neural approaches scale better to larger graphs than classical methods.
   - Tested via: ScalabilityBenchmark with scaling analysis

4. **Hypothesis 4**: Meta-learning enables effective transfer to unseen but related causal systems.
   - Tested via: Specialized transfer benchmarks comparing in-distribution and out-of-distribution performance

### Visualization and Reporting

The framework generates comprehensive visualizations and reports:

1. Bar charts and box plots for performance comparisons
2. Scaling curves for computational analysis
3. Precision-recall trade-off curves for structural learning
4. Optimization trajectory plots for intervention efficiency
5. Summary reports with statistical significance indicators

These visualizations enable clear communication of research findings and facilitate identifying strengths and limitations of different approaches.

The benchmarking framework serves as the foundation for validating our research claims and ensuring that our findings are robust, reproducible, and scientifically sound. 