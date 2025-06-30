# Examples Directory

This directory contains example scripts demonstrating the usage of the Amortized Causal Bayesian Optimization (ACBO) framework, organized by complexity and use case.

## Directory Structure

### 📁 **Core Examples** (Root Level)
- `complete_workflow_demo.py` - **Main comprehensive ACBO demonstration**
- `parent_scale_demo.py` - PARENT_SCALE integration showcase
- `jax_native_demo.py` - JAX-native architecture performance demo
- `verify_intervention_strategies.py` - Quick intervention strategy validation

### 📁 **research/** - Research & Validation
- `run_comprehensive_validation.py` - Publication-quality statistical validation
- `bic_fix_demo.py` - Educational BIC scoring demonstration

### 📁 **advanced/** - Advanced Features
- `complete_acquisition_training_demo.py` - Complete GRPO training pipeline
- `hybrid_rewards_demo.py` - Mechanism-aware reward system
- `mechanism_aware_integration_demo.py` - End-to-end mechanism integration

### 📁 **development/** - Development Tools
- `migration_example.py` - Legacy to JAX-native migration guide
- `mechanism_aware_demo.py` - Mechanism-aware architecture demo
- `performance_comparison.py` - JAX-native vs legacy benchmarking
- `demo_hydra_wandb_integration.py` - Hydra/WandB integration examples
- `hydra_wandb_demo.py` - Minimal integration demo

### 📁 **experiments/** - Experiment Infrastructure
- `basic_usage.py` - New experiment infrastructure demo
- `quick_start.py` - Quick start guide for experiments

## Quick Start

**Main ACBO demonstration:**
```bash
poetry run python examples/complete_workflow_demo.py
```

**PARENT_SCALE integration:**
```bash
poetry run python examples/parent_scale_demo.py
```

**JAX-native performance showcase:**
```bash
poetry run python examples/jax_native_demo.py
```

## Available Examples

### 1. Complete Workflow Demo (`complete_workflow_demo.py`)
The main demonstration of the complete ACBO pipeline with all components integrated.

**Features:**
- Progressive learning with self-supervised causal discovery
- Difficulty comparative study across Easy/Medium/Hard SCMs
- BIC scoring to prevent overfitting
- Intervention strategy validation
- Full integration of all ACBO components

**Usage:**
```bash
python examples/complete_workflow_demo.py
```

### 2. PARENT_SCALE Integration Demo (`parent_scale_demo.py`)
A simple demonstration of using the integrated PARENT_SCALE algorithm.

**Features:**
- Shows how to create SCMs and run PARENT_SCALE
- Displays optimization trajectory and results
- Works standalone without requiring original implementation

**Usage:**
```bash
python examples/parent_scale_demo.py
```

### 3. PARENT_SCALE Algorithm Comparison (`parent_scale_comparison.py`)
Compares the original and integrated PARENT_SCALE implementations to validate consistent behavior.

**Features:**
- Side-by-side comparison of both algorithms
- Validates intervention sequences and optimization values
- Useful for testing integration correctness

**Usage:**
```bash
# Requires causal_bayes_opt_old/ directory with original implementation
python examples/parent_scale_comparison.py
```

**Note:** This example requires the original PARENT_SCALE implementation to be available in a `causal_bayes_opt_old/` directory. Without it, the script will run only the integrated algorithm as a demonstration.

### 4. Intervention Strategy Research Scripts
Located in the examples directory, these scripts demonstrate research findings on intervention strategies:

- `verify_intervention_strategies.py` - Quick validation of random vs fixed intervention strategies
- `run_comprehensive_validation.py` - Thorough statistical analysis with multiple trials
- `bic_fix_demo.py` - Educational demonstration of the BIC scoring fix

## Other Example Scripts

Additional demonstration and validation scripts are available in the examples directory. Check the individual scripts for their specific purposes and usage instructions.

## Running Examples

All examples can be run from the project root directory:

```bash
# From project root
python examples/<script_name>.py
```

Most examples use relative imports and expect to be run from the project root directory.

## Creating Your Own Examples

When creating new examples:
1. Place them in this `examples/` directory
2. Use proper import paths (see existing examples)
3. Include clear documentation and usage instructions
4. Follow the project's functional programming principles
5. Add an entry to this README

## Common Issues

### Import Errors
If you encounter import errors, ensure you're running the scripts from the project root directory, not from within the examples directory.

### Missing Dependencies
Ensure all project dependencies are installed:
```bash
poetry install
```

### PARENT_SCALE Not Available
Some examples require the external PARENT_SCALE library. If not available, the examples will still demonstrate the integrated implementation.