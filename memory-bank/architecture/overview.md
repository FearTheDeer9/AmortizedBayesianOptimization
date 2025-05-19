# Amortized Causal Bayesian Optimization Architecture

## System Overview

This architecture integrates AVICI's amortized inference with PARENT_SCALE's causal optimization approach, following a clean cyclic workflow:

`Surrogate Model → Acquisition Model → SCM Environment → Surrogate Model`

## Core Principles

- **Functional Design**: Pure functions, immutable data structures, minimal state
- **Separation of Concerns**: Clear component boundaries with explicit interfaces
- **Scalability**: Designed to handle increasing graph complexity
- **Extensibility**: Easily add new intervention types or reward functions

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