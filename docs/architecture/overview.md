# Amortized Causal Bayesian Optimization Architecture

## System Overview

This architecture integrates AVICI's amortized inference with PARENT_SCALE's causal optimization approach, following a clean cyclic workflow:

`Surrogate Model → Acquisition Model → SCM Environment → Surrogate Model`

## Implementation Principles

- **Functional Core, Imperative Shell**: Pure functions for algorithms, efficient mutable state where needed
- Use **selective immutability** based on usage patterns and performance requirements
- Design pure functions for data transformation and computation
- Maintain **clear component boundaries** with explicit interfaces
- Include comprehensive docstrings and type hintsW
- Choose data structures based on **performance characteristics** rather than ideology
- **Explicit state management** rather than hidden mutation

## Core Components

### 1. Surrogate Model (AVICI Adaptation)
Produces posterior distributions over parent sets using amortized inference.

### 2. Acquisition Model (RL with GRPO)
Selects interventions based on information gain and optimization objectives.

### 3. SCM Environment
Executes interventions and returns outcomes according to the underlying causal model.

### 4. Experience Buffer
Stores observational and interventional data using immutable data structures.

## Data Flow

1. Initialize with observational data in experience buffer
2. Surrogate model computes parent set posterior
3. Acquisition model selects intervention based on posterior
4. SCM environment executes intervention and returns outcome
5. Experience buffer is updated with new intervention-outcome pair
6. Return to step 2 and repeat

## Key Architectural Decisions

- Use of GRPO for reinforcement learning without value networks
- Function registry pattern for intervention handlers
- Immutable data structures for all components
- Decoupling of data representation from behavior