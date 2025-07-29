# ACBO Component Analysis

## Overview

This document provides a comprehensive analysis of each component in the ACBO (Amortized Causal Bayesian Optimization) feedback loop, identifying weaknesses, assumptions, brittle connections, and interface issues.

## The ACBO Feedback Loop

The ideal ACBO loop should be:
1. **Initialize**: Create initial state from observational data
2. **Predict Structure**: Use surrogate model to predict parent set posterior
3. **Create State**: Build rich acquisition state with uncertainty estimates
4. **Select Intervention**: Policy network chooses intervention based on state
5. **Apply Intervention**: Execute intervention and observe outcome
6. **Update Buffer**: Add new data to experience buffer
7. **Update Posterior**: Recompute structure beliefs with new data
8. **Repeat**: Continue until convergence or budget exhausted

## Component Analysis

### 1. **AcquisitionState** (`acquisition/state.py`)

**Purpose**: Central state representation combining structural uncertainty with optimization progress.

**Strengths**:
- Comprehensive state representation
- Immutable design following functional principles
- Rich derived properties (uncertainty_bits, exploration coverage, etc.)

**Weaknesses**:
- Creates circular dependency issues (imports from multiple modules)
- Mechanism prediction integration is incomplete (lots of try/except blocks)
- Validation in `__post_init__` is fragile
- Assumes fixed variable names throughout episode

**Interface Issues**:
- Requires `ParentSetPosterior` object which has its own complex creation
- Expects specific buffer format
- Target variable must match posterior's target

**Stub Implementations**:
- Mechanism confidence computation has fallback paths
- Some metrics return placeholder values

### 2. **ParentSetPosterior** (`avici_integration/parent_set/`)

**Purpose**: Represents belief distribution over causal structures.

**Weaknesses**:
- Multiple implementations (discrete enumeration vs continuous)
- Unclear which version to use when
- predict_parent_posterior has complex data format requirements

**Interface Issues**:
- Requires specific data format: [N, d, 3] tensor
- Variable ordering must be consistent
- Target variable handling is inconsistent

### 3. **ContinuousParentSetPredictionModel** (`avici_integration/continuous/model.py`)

**Purpose**: Variable-agnostic attention-based model for structure learning.

**Strengths**:
- Truly variable-agnostic design
- Uses attention mechanisms for flexible input sizes

**Weaknesses**:
- Training data format is rigid [N, d, 3]
- No clear documentation on what the 3 channels represent
- Integration with posterior creation is unclear

### 4. **EnhancedPolicyNetwork** (`acquisition/enhanced_policy_network.py`)

**Purpose**: Policy network for intervention selection.

**Strengths**:
- Variable-agnostic architecture using attention
- Modular design with separate components

**Weaknesses**:
- State enrichment expects specific channel counts (5, 32, etc.)
- RoleBasedProjection creates fixed-size projections
- Complex initialization with many hyperparameters

**Interface Issues**:
- `create_enhanced_policy_for_grpo` hardcodes num_variables at creation
- State tensor shape expectations are rigid
- Target variable index handling is inconsistent

### 5. **State Construction Pipeline**

**Major Issues**:
- `StateConverter` in enriched_trainer.py creates fixed-size tensors
- History builder expects specific temporal patterns
- Role-based projections assume fixed dimensions

**Key Problem**: The pipeline from AcquisitionState → enriched tensors → policy input loses variable-agnosticism.

### 6. **ExperienceBuffer** (`data_structures/buffer.py`)

**Strengths**:
- Clean immutable interface
- Good abstraction over sample storage

**Weaknesses**:
- No built-in support for different data modalities
- Intervention tracking is basic

### 7. **Training Infrastructure**

**EnrichedGRPOTrainer Issues**:
- Requires 15-20 nested config fields
- Complex SCM rotation logic
- State conversion is brittle
- Bootstrap surrogate integration is incomplete

**Key Problems**:
- Training creates states differently than evaluation
- No consistent way to create proper acquisition states during training
- Dummy states used instead of real posterior updates

### 8. **Evaluation Infrastructure**

**Issues**:
- GRPOEvaluator creates states without proper posterior updates
- StateConverter initialization is hacky (creates dummy configs)
- No integration between structure learning and intervention selection

## Critical Interface Mismatches

1. **Variable Agnosticism Lost**: Models are variable-agnostic but state construction isn't
2. **Fixed Channel Assumptions**: State enrichment assumes specific channel counts
3. **Target Variable Handling**: Inconsistent between components
4. **Data Format Rigidity**: [N, d, 3] format assumption throughout
5. **Posterior Update Gap**: No clear pipeline for updating beliefs during evaluation

## Stub/Placeholder Implementations

From grep analysis:
- Intervention handlers have stub implementations
- Trajectory validation uses placeholders
- GRPO training manager uses dummy experiences
- BC trainer creates dummy states for initialization
- Many TODOs around checkpointing and comprehensive evaluation

## Root Causes

1. **Premature Abstraction**: Components were abstracted before interfaces were stable
2. **Config Complexity**: Over-engineering with OmegaConf led to brittle connections
3. **Missing Integration Layer**: No clear service layer to properly connect components
4. **State Construction**: The pipeline from raw data to policy input is the weakest link
5. **Training vs Inference Gap**: Different code paths create inconsistencies

## Recommendations for Refactoring

1. **Simplify State Construction**: Create a single, clear pipeline for state creation
2. **Fix Variable Agnosticism**: Ensure all components handle variable-sized inputs
3. **Unify Training/Inference**: Same state construction for both paths
4. **Remove Config Dependencies**: Reduce configuration complexity
5. **Add Integration Tests**: Test full loop, not just components

## Files to Clean Up

Based on our analysis, these files from our previous implementation should be removed:
- `/scripts/test_bc_synthetic.py`
- `/scripts/test_simplified_trainers.py`
- `/src/causal_bayes_opt/training/simplified_grpo_trainer.py`
- `/src/causal_bayes_opt/training/simplified_bc_trainer.py`
- `/src/causal_bayes_opt/evaluation/simplified_grpo_evaluator.py`
- `/src/causal_bayes_opt/evaluation/simplified_bc_evaluator.py`
- `/scripts/run_simplified_demo.py`

These represent our failed attempt to bypass the complexity without understanding it.