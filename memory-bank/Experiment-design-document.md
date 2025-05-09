# Experiment Design Document

## Project Goal

Develop an Amortized Causal Meta-Learning Framework with an interface-first design pattern. The core idea is to leverage neural networks to learn both causal structures and dynamics models from data, enabling efficient and scalable causal discovery and intervention optimization on large graphs, while maintaining a flexible, decoupled architecture that allows for different model implementations.

## Current Direction & Approach: Interface-First Refactoring

1. **Interface-First Design Pattern:**
   - Define core interfaces for key functionalities that completely decouple algorithm logic from specific model implementations
   - Create adapter classes for existing GNN models to implement the new interfaces
   - Ensure algorithm logic makes no assumptions about implementation details
   - Enable greater flexibility in model implementation while maintaining coherent architectural design

2. **Core Interfaces:**
   - `CausalStructureInferenceModel`: Interface for models that infer causal structure from data
   - `InterventionOutcomeModel`: Interface for models that predict outcomes of interventions
   - `AcquisitionStrategy`: Interface for strategies that select interventions
   - `UncertaintyEstimator`: Interface for estimating uncertainty in predictions
   - `Updatable`: Interface for models that can be updated with new data

3. **Flexible Uncertainty Estimation & Model Updating:**
   - Implement a decoupled `UncertaintyEstimator` interface with multiple implementations
     - `EnsembleUncertaintyEstimator`: Using model ensembles
     - `DropoutUncertaintyEstimator`: Using MC dropout
     - `DirectUncertaintyEstimator`: For models that directly predict variance
     - `ConformalUncertaintyEstimator`: Using conformal prediction methods
   - Implement an `Updatable` interface for model updating with different strategies
     - `IncrementalUpdater`: For efficient updates with new data
     - `ExperienceReplayUpdater`: For balancing old/new data
     - `FullRetrainingUpdater`: For when full retraining is necessary
   - Create an `UncertaintyThresholdManager` to monitor uncertainty and trigger fallbacks

4. **Integration Components:**
   - `AmortizedCausalOptimizer`: High-level class that integrates the core interfaces to perform causal optimization
   - `AmortizedCausalDiscovery`: Combines structure inference and dynamics modeling
   - Various model implementations that satisfy the interfaces

5. **Configuration System:**
   - Implement a YAML-based configuration system for all components
   - Create a `ConfigurationManager` to load and validate configurations
   - Develop a `ConfigurableComponent` base class for components that can be configured using YAML
   - Define configuration schemas for validation and documentation

6. **Error Handling Strategy:**
   - Implement a unified error handling strategy with custom exception types for different error categories
   - Define standards for when to raise vs. when to handle errors
   - Add better error messages with context information

## Key Design Considerations & Trade-offs

- **Interface Design:** Balance between too restrictive (limiting flexibility) and too loose (making implementations complex)
- **Model Flexibility:** Support different model architectures (GNNs, MLPs, Transformers) while maintaining consistent interfaces
- **Uncertainty Representation:** Standardize uncertainty representation across different estimation methods
- **Performance vs. Modularity:** Ensure that the decoupled design doesn't significantly impact performance
- **Configuration Complexity:** Balance between detailed configuration options and user-friendly defaults
- **Backward Compatibility:** Maintain compatibility with existing codebase during incremental refactoring
- **Testing Strategy:** Develop comprehensive tests for interfaces and implementations

## Scalability Advantages

- **Architectural Scalability:** The interface-first design allows for easier extension to new model types and algorithms
- **Implementation Flexibility:** Different model implementations can be optimized for specific graph types or sizes
- **Configuration-Driven Experimentation:** YAML configuration enables rapid experimentation without code changes
- **Uncertainty-Aware Decision Making:** Flexible uncertainty estimation allows for better decision-making in large-scale problems
- **Model Updating Strategies:** Different update strategies can be selected based on computational constraints

## Potential Challenges & Mitigations

- **Interface Evolution:** As requirements change, interfaces may need to evolve. Use interface versioning and deprecation warnings.
- **Implementation Overhead:** Abstract interfaces add some overhead. Keep interfaces minimal and focused.
- **Testing Complexity:** More interfaces mean more testing complexity. Create comprehensive test suites and fixtures.
- **Learning Curve:** New developers need to understand interfaces before implementation. Create clear documentation and examples.
- **Performance Considerations:** Abstractions can impact performance. Use profiling to identify and optimize critical paths.

## Implementation Roadmap

Following a phased approach to the refactoring:

### Phase 0: Interface Definition (1 week)
- Define core interfaces (`CausalStructureInferenceModel`, `InterventionOutcomeModel`, etc.)
- Create reference implementations
- Build adapter patterns for existing code

### Phase 1: Core Components Refactoring (1-2 weeks)
- Refactor `CausalGraph` and `DirectedGraph` with standardized method naming
- Update `StructuralCausalModel` to implement relevant interfaces
- Improve validation logic and error handling

### Phase 2: Model Components Refactoring (2-3 weeks)
- Implement adapters for existing structure inference models
- Update dynamics prediction models to use the new interfaces
- Create pluggable acquisition strategies
- Implement uncertainty estimation and model updating components

### Phase 3: Integration Classes Refactoring (2-3 weeks)
- Create `AmortizedCausalOptimizer` using the interface-based design
- Update existing algorithm classes to use the new interfaces
- Refactor meta-learning components

### Phase 4: Demo Scripts & Testing (1-2 weeks)
- Update demo scripts to use the new interfaces
- Add comprehensive tests
- Create performance benchmarks

### Phase 5: YAML Configuration System (1-2 weeks)
- Implement YAML configuration infrastructure
- Update demo scripts to use configuration files
- Create configuration-driven training pipeline
- Add documentation for configuration options

## Style and Form Recommendations

- **Interface Documentation:** Clearly document the contract for each interface method
- **Implementation Guidelines:** Provide guidelines for implementing each interface
- **Configuration Templates:** Create template configurations for common scenarios
- **Error Handling Standards:** Define standards for error handling across the codebase
- **Type Annotations:** Use comprehensive type annotations for all interfaces and implementations
- **Testing Patterns:** Develop standard testing patterns for interface implementations
- **Documentation:** Create clear documentation with examples for each interface
- **Configuration Schema:** Document the configuration schema with examples

## Risk Management

### Potential Issues & Mitigations
1. **Breaking Changes**
   - Risk: Interface changes breaking dependent code
   - Mitigation: Deprecate old interfaces before removing them, provide adapters

2. **Performance Regression**
   - Risk: Refactoring causing performance degradation
   - Mitigation: Add performance tests for critical paths

3. **Test Coverage Gaps**
   - Risk: Missing tests allowing bugs to slip through
   - Mitigation: Require test coverage metrics for all changes

4. **Knowledge Silos**
   - Risk: Only certain team members understanding components
   - Mitigation: Implement pair programming and knowledge sharing

5. **Interface Design Challenges**
   - Risk: Designing interfaces that are too restrictive or too loose
   - Mitigation: Start with minimal interfaces and evolve based on feedback

6. **Configuration Complexity**
   - Risk: Creating overly complex configuration options
   - Mitigation: Layer configurations with sensible defaults and documentation

## Evaluation Framework

The evaluation framework will be enhanced to assess the benefits of the interface-first design pattern:

1. **Interface Compliance Testing:**
   - Verify that all implementations correctly satisfy their interface contracts
   - Test edge cases and error conditions
   - Check performance characteristics

2. **Flexibility Assessment:**
   - Evaluate ease of swapping different implementations
   - Measure development time for new implementations
   - Assess code reuse across different implementations

3. **Configuration Testing:**
   - Verify that all components can be correctly configured via YAML
   - Test configuration validation and error reporting
   - Measure configuration-driven experimentation efficiency

4. **Uncertainty Estimation Evaluation:**
   - Compare different uncertainty estimation methods
   - Assess calibration of uncertainty estimates
   - Evaluate decision quality based on uncertainty

5. **Model Updating Efficiency:**
   - Compare different update strategies
   - Measure computational efficiency of updates
   - Assess learning curve with incremental updates

## YAML Configuration System

The YAML Configuration System will provide a unified approach to managing component configurations:

### Key Features

1. **Schema Definition:**
   - Define clear schemas for all configurable components
   - Support validation of configuration values
   - Provide clear error messages for invalid configurations

2. **Configuration Inheritance:**
   - Allow configurations to extend base configurations
   - Support overriding specific values
   - Enable composition of configurations

3. **Environment Variable Substitution:**
   - Support environment variable references in configurations
   - Enable dynamic configuration based on environment

4. **Command Line Overrides:**
   - Allow overriding configuration values from command line
   - Support nested configuration paths

5. **Default Configurations:**
   - Provide sensible defaults for all components
   - Include example configurations for common scenarios

### Example Configuration Structure

```yaml
# Example configuration for a causal optimization experiment
structure_inference_model:
  type: GNNGraphEncoder
  params:
    hidden_dim: 64
    num_layers: 3
    attention_heads: 4
    dropout: 0.1

dynamics_model:
  type: GNNDynamicsDecoder
  params:
    hidden_dim: 64
    num_layers: 3
    message_passing_steps: 2

acquisition_strategy:
  type: ExpectedImprovement
  params:
    exploration_weight: 0.1

uncertainty_estimation:
  type: EnsembleUncertaintyEstimator
  params:
    num_models: 5
    bootstrap: true

model_updating:
  type: ExperienceReplayUpdater
  params:
    buffer_size: 1000
    sample_ratio: 0.3

training:
  optimizer:
    type: Adam
    params:
      lr: 0.001
      weight_decay: 1e-5
  num_epochs: 200
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001

evaluation:
  metrics:
    - shd
    - precision
    - recall
    - f1
    - mse
    - mae
  test_size: 0.2
  seed: 42
```

## Definition of Done

A component refactoring is considered complete when:

1. All identified issues are addressed
2. Code passes all existing and new tests
3. Documentation is updated
4. Performance is verified to be maintained or improved
5. Code review is completed with no major concerns
6. Integration tests pass with dependent components
7. The component properly implements any applicable interfaces
8. YAML configuration support is added where appropriate 