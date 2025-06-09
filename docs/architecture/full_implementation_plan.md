# Amortized Causal Bayesian Optimization (ACBO)
# Implementation Plan

## Overview

This document outlines the implementation plan for the Amortized Causal Bayesian Optimization framework, which combines AVICI's amortized inference with PARENT_SCALE's causal optimization approach to create an efficient system for causal discovery and intervention selection.

The system follows a cyclic workflow:
```
Surrogate Model → Acquisition Model → SCM Environment → Surrogate Model
```

## Implementation Principles

- **Functional Core, Imperative Shell**: Pure functions for algorithms, efficient mutable state where needed
- Use **selective immutability** based on usage patterns and performance requirements
- Design pure functions for data transformation and computation
- Maintain **clear component boundaries** with explicit interfaces
- Include comprehensive docstrings and type hints
- Choose data structures based on **performance characteristics** rather than ideology
- **Explicit state management** rather than hidden mutation
- Write minimal, focused functions rather than complex multi-purpose ones

## Selective Immutability Guidelines

**Immutable Where It Matters:**
- SCM definitions (structure, mechanisms) - rarely change, frequently compared
- Intervention specifications - small objects, frequently compared/cached
- Configuration objects and hyperparameters
- Algorithm results and derived data structures

**Mutable Where Performance Matters:**
- Experience buffers - use efficient append-only structures
- Neural network parameters - standard PyTorch/JAX mutable tensors
- Large intermediate computation results
- Training state and optimization variables

## Phase 1: Core Data Structures and Environment

### SCM Implementation
- Immutable core representation using frozen dataclasses (structure rarely changes)
- Mutable workspace for temporary modifications during sampling/intervention
- Graph exploration utilities as pure functions (parents, children, ancestors, descendants)
- Validation functions (cycle detection, topological sorting)
- Mechanism consistency checking
- Efficient caching of derived properties

### Sample Representation
- Lightweight immutable representation of variable assignments (small, frequently compared)
- Support for both observational and interventional data
- Type-safe interfaces with proper validation
- Utilities for sample manipulation and transformation
- Memory pooling for frequent allocations

### Mechanism Framework
- Function factories for common mechanisms (linear, nonlinear)
- Pure functional noise generation
- Composable mechanism building blocks
- Support for different functional forms and noise distributions

### Intervention Framework
- Function registry pattern for intervention handlers
- Immutable intervention specifications (small objects, frequently compared)
- Support for perfect, imperfect, and soft interventions
- Factory functions for intervention creation
- Efficient application with mutable workspaces when needed

### Sampling Functions
- Topological sampling from SCMs
- Batch sampling capabilities
- Intervention-aware sampling
- Utilities for generating observational and interventional data

### Experience Buffer
- Efficient mutable buffer with append-only primary operations
- Checkpoint/versioning system for experiment reproducibility
- Fast query and filtering with proper indexing
- Memory-efficient storage of intervention-outcome pairs
- Batch processing capabilities for neural network training
- Support for observational and interventional data

## Phase 2: Surrogate Model (AVICI Adaptation)

### GNN Architecture
- Graph Attention Networks (GAT) implementation using standard PyTorch
- Graph Transformers option
- GatedGCN implementation
- Node feature extraction pipeline
- Weighted message passing based on posterior probabilities

### AVICI Integration
- Adapter functions for data structures
- `encode_data` and `decode_posterior` functions
- Target-specific output layer modification
- Input encoding for observational and interventional data

### Posterior Representation
- `ParentSetPosterior` data structure (immutable - frequently analyzed/compared)
- Posterior manipulation utilities as pure functions
- Uncertainty quantification functions
- Visualization and analysis tools

### Multi-Loss Training Framework
- KL divergence loss implementation
- Edge-wise binary cross-entropy loss
- Calibration loss for well-calibrated uncertainty
- Weighted loss combination function
- Standard PyTorch training loops with mutable parameters

## Phase 3: Acquisition Model (RL with GRPO) ✅ COMPLETE

### Rich State Representation ✅
- `AcquisitionState` dataclass with optimization tracking
- Integration with ParentSetPosterior uncertainty
- Buffer statistics and marginal parent probabilities
- History format conversion for transformer input
- **File**: `src/causal_bayes_opt/acquisition/state.py`

### Policy Network with Alternating Attention ✅
- Two-headed architecture (variable + value selection)
- Alternating attention transformer encoder following CAASL
- Uncertainty-aware variable selection using marginal probabilities
- Optimization-aware value selection with best value context
- **File**: `src/causal_bayes_opt/acquisition/policy.py`

### Multi-Component Verifiable Reward System ✅
- Optimization reward (target variable improvement)
- Structure discovery reward (information gain from posteriors)
- Parent intervention reward (bonus for likely parents)
- Exploration bonus (diversity encouragement)
- Configurable weighted combination
- **File**: `src/causal_bayes_opt/acquisition/rewards.py`

### Enhanced GRPO Implementation ✅
- Literature-compliant GRPO with group-based advantages
- Open-r1 enhancements: sample reuse, configurable scaling, zero KL penalty
- Sample reuse manager for training efficiency
- JAX-compiled training functions
- Comprehensive test coverage (33 passing tests)
- **File**: `src/causal_bayes_opt/acquisition/grpo.py`

### Uncertainty-Guided Exploration ✅
- Expected information gain computation per intervention
- Count-based exploration for under-sampled interventions
- Variable uncertainty bonuses maximized at prob~0.5
- Adaptive temperature scheduling based on optimization progress
- **File**: `src/causal_bayes_opt/acquisition/exploration.py`

## Phase 4: Training Infrastructure

### Enhanced GRPO Training Configuration ✅
- Pydantic-based configuration system with comprehensive validation
- Open-r1 enhancements: sample reuse, configurable scaling, zero KL penalty
- Immutable configuration objects with type safety
- Backward compatibility with existing GRPOConfig
- **File**: `src/causal_bayes_opt/training/config.py`

### PARENT_SCALE Integration
- Expert demonstration collection from PARENT_SCALE algorithm
- SCM generation for training with diverse causal structures
- Data transformation utilities for multi-stage training
- Curriculum-based example generation

### Multidimensional Curriculum Learning
- Dynamic difficulty scaling along multiple dimensions:
  - Graph size and density (3-20+ variables)
  - Functional complexity (linear → nonlinear mechanisms)
  - Noise levels (low → high variance)
  - Observational sample size (sparse → dense)
- Parameter sampling from difficulty-controlled distributions
- Progressive complexity increase based on training progress

### Multi-Stage Training Pipeline
- Surrogate model pretraining with AVICI-compatible training loops
- GRPO-based acquisition model training with verifiable rewards
- End-to-end system training with dual objectives
- Integration of curriculum learning and exploration strategies
- Efficient training state management with explicit checkpointing

### Training Utilities
- Batch generation and processing for diverse SCM structures
- Checkpoint management with immutable configurations
- Progress visualization for dual-objective metrics
- Hyperparameter optimization and learning rate scheduling
- Training diagnostics and performance monitoring

## Phase 5: System Integration and Evaluation

### Complete System Integration
- End-to-end cyclic workflow: Surrogate → Acquisition → Environment → Buffer
- Comprehensive testing framework with integration tests
- Logging and debugging tools for training and deployment
- End-to-end examples and usage demonstrations
- Performance profiling and optimization

### Dual-Objective Evaluation Framework
- **Optimization Performance**: Target variable improvement, convergence metrics
- **Structure Discovery**: Parent set accuracy, uncertainty calibration
- **Combined Metrics**: Pareto efficiency, objective trade-off quality
- **Intervention Quality**: Diversity, exploration coverage, parent targeting
- **Training Efficiency**: Sample efficiency, convergence speed, stability

### Experimental Validation
- Benchmark against PARENT_SCALE on standard test cases
- Scaling analysis: 3-variable → 20+ variable graphs
- Transfer learning evaluation across different causal structures
- Computational efficiency benchmarks vs. exact methods
- Ablation studies on reward components and exploration strategies

### Visualization and Analysis Tools
- Parent set posterior uncertainty visualization
- Intervention effectiveness and optimization progress analysis
- Training diagnostics: loss curves, reward decomposition, exploration metrics
- Causal discovery accuracy and calibration plots
- Interactive system performance dashboards

## Dependencies Between Components

- **Sample representation** depends on **SCM implementation**
- **Sampling functions** depend on **mechanism framework** and **SCM implementation**
- **Experience buffer** depends on **sample representation**
- **Surrogate model** depends on **experience buffer**
- **Acquisition model** depends on **surrogate model** and **experience buffer**
- **Training pipeline** depends on all previous components
- **Evaluation framework** depends on the complete system integration

## Implementation Strategy

For each component, follow this development pattern:
1. Define the interfaces and type signatures
2. Implement core functionality with minimal dependencies
3. Write comprehensive tests for the component
4. Integrate with other components
5. Optimize performance and add advanced features

For parallel development, prioritize components with fewer dependencies.

## Architectural Decisions

See the Architecture Decision Records (ADRs) for detailed rationale on key decisions:
- ADR 001: Intervention Representation (Revised)
- ADR 002: GRPO Implementation
- ADR 003: Verifiable Rewards for GRPO