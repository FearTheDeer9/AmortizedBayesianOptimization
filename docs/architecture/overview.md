# Amortized Causal Bayesian Optimization Architecture

## System Overview

This architecture integrates AVICI's amortized inference with PARENT_SCALE's causal optimization approach, following a clean cyclic workflow:

`Parent Set Model → Acquisition Model → SCM Environment → Experience Buffer → Parent Set Model`

## Implementation Principles

- **Functional Core, Imperative Shell**: Pure functions for algorithms, efficient mutable state where needed
- Use **selective immutability** based on usage patterns and performance requirements
- Design pure functions for data transformation and computation
- Maintain **clear component boundaries** with explicit interfaces
- Include comprehensive docstrings and type hints
- Choose data structures based on **performance characteristics** rather than ideology
- **Explicit state management** rather than hidden mutation

## Core Components

### 1. Parent Set Prediction Model (AVICI Adaptation)
- **Location**: `avici_integration/parent_set/model.py`
- **Class**: `ParentSetPredictionModel`
- Produces posterior distributions over parent sets using amortized inference
- Uses alternating attention transformers similar to AVICI architecture
- Supports target-aware conditioning for directed exploration

### 2. Acquisition Model (RL with GRPO)
- **Location**: `acquisition/policy.py`
- **Class**: `AcquisitionPolicyNetwork`
- Selects interventions based on dual objectives:
  - Structure learning (information gain)
  - Target optimization (exploitation)
- Uses Group Relative Policy Optimization without value networks
- Outputs both variable selection and intervention values

### 3. SCM Environment
- **Core**: `data_structures/scm.py` - Immutable SCM representation
- **Sampling**: `environments/sampling.py` - Intervention execution
- **Mechanisms**: `mechanisms/` - Currently supports linear mechanisms
- Executes interventions and returns outcomes according to the causal model

### 4. Experience Buffer System
Two complementary buffer implementations:

#### ExperienceBuffer
- **Location**: `data_structures/buffer.py`
- Mutable, append-only buffer for intervention-outcome pairs
- Optimized for fast appends and efficient querying
- Maintains indices for variable-based lookups

#### TrajectoryBuffer
- **Location**: `acquisition/trajectory.py`
- Specialized buffer for RL training
- Stores complete (state, action, reward, next_state) tuples
- Designed for GRPO training workflows

## Supporting Components

### 5. Acquisition State
- **Location**: `acquisition/state.py`
- Rich state representation for decision-making
- Tracks uncertainty estimates, optimization progress, and intervention history
- Provides context for intelligent intervention selection

### 6. Reward System
- **Location**: `acquisition/rewards.py`
- Verifiable multi-component rewards without human feedback
- Balances exploration (structure learning) and exploitation (optimization)
- Supports diverse objective functions

### 7. AVICI Integration Bridge
- **Location**: `avici_integration/core/`
- Data conversion between internal Sample format and AVICI's expected format
- Standardization and validation utilities
- Enables seamless integration with AVICI's neural architectures

### 8. Intervention Registry
- **Location**: `interventions/registry.py`
- Centralized intervention handling with registry pattern
- Currently supports perfect interventions
- Extensible for soft interventions and other types

## Data Flow

1. Initialize with observational data in experience buffer
2. Parent set model computes posterior distributions over causal structures
3. Acquisition state aggregates current knowledge and uncertainty
4. Acquisition model selects intervention based on state and objectives
5. SCM environment executes intervention and returns outcome
6. Experience buffer is updated with new intervention-outcome pair
7. Trajectory buffer stores RL training data if in training mode
8. Return to step 2 and repeat

## Key Architectural Decisions

- **GRPO for RL**: Chosen for stable training without value networks
- **Function registry pattern**: Enables extensible intervention handling
- **Selective immutability**: Immutable for SCMs/configs, mutable for buffers
- **Target-aware inference**: Conditions all models on optimization target
- **Verifiable rewards**: Eliminates need for human feedback in training