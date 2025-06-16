# ACBO Documentation Navigation Guide

## Overview

This documentation covers the complete Amortized Causal Bayesian Optimization (ACBO) framework, including its integration with PARENT_SCALE for expert demonstration collection and training.

## 📚 Documentation Structure

### 🏗️ Architecture & Design
- **[Overview](architecture/overview.md)** - System architecture and component overview
- **[Interface Specifications](architecture/interface_specifications.md)** - Complete API specifications and data structures
- **[Module Structure](architecture/module_structure.md)** - Code organization and dependencies
- **[Phase 4 Consolidated Plan](architecture/PHASE4_CONSOLIDATED_PLAN.md)** - ✅ **COMPLETE** implementation plan and achievements

#### Architecture Decision Records (ADRs)
- **[ADR 001: Intervention Representation](architecture/adr/001_intervention_representation.md)** - Design decisions for intervention handling
- **[ADR 002: GRPO Implementation](architecture/adr/002_grpo_implementation.md)** - Group Relative Policy Optimization approach
- **[ADR 003: Verifiable Rewards](architecture/adr/003_verifiable_rewards.md)** - Reward system without human feedback
- **[ADR 004: Custom Parent Set Model](architecture/adr/004_custom_parent_set_model.md)** - Parent set prediction architecture
- **[ADR 004: Target-Aware AVICI Strategy](architecture/adr/004_target_aware_avici_strategy.md)** - AVICI integration approach

### 🔗 Integration Documentation
- **[PARENT_SCALE Integration Complete](integration/parent_scale_integration_complete.md)** - 🎉 **Integration journey and 100% identical behavior achievement**
- **[PARENT_SCALE Data Bridge](integration/parent_scale_data_bridge.md)** - **Technical data bridge implementation and scaling validation**

### 🎓 Training & Expert Demonstrations
- **[Expert Demonstration Collection Implementation](training/expert_demonstration_collection_implementation.md)** - 📖 **COMPREHENSIVE GUIDE** for expert demonstration collection (19KB guide)
- **[Multi-Stage Training](training/multi_stage_training.md)** - Training pipeline for surrogate and acquisition models

### 📖 API Reference
- **[SCM API](api/scm.md)** - Structural Causal Model functions
- **[Sample API](api/sample.md)** - Sample data structures and operations
- **[Linear Mechanism API](api/linear_mechanism.md)** - Linear causal mechanisms
- **[AVICI Integration API](api/avici_integration.md)** - AVICI integration utilities
- **[GRPO API](api/grpo.md)** - Group Relative Policy Optimization implementation
- **[Buffer API](api/buffer.md)** - Experience buffer operations
- **[Rewards API](api/rewards.md)** - Reward computation functions

### 🧪 Examples & Research
- **[Complete Workflow Demo](examples/complete_workflow_demo.md)** - End-to-end ACBO demonstration
- **[Intervention Strategy Comparison](examples/intervention_strategy_comparison.md)** - Research on random vs fixed interventions
- **[BIC Scoring Fix](examples/bic_scoring_fix.md)** - Educational demo of overfitting prevention

## 🚀 Quick Start Guides

### For Expert Demonstration Collection
**Start here**: [Expert Demonstration Collection Implementation](training/expert_demonstration_collection_implementation.md)
```python
from causal_bayes_opt.integration import run_full_parent_scale_algorithm

# Collect a single expert demonstration
trajectory = run_full_parent_scale_algorithm(
    scm=None,  # Uses LinearColliderGraph for perfect fidelity
    target_variable='Y',
    T=10,
    seed=42
)
```

### For Understanding the Architecture
**Start here**: [Architecture Overview](architecture/overview.md) → [Interface Specifications](architecture/interface_specifications.md)

### For API Usage
**Start here**: Choose your component from the [API Reference](#-api-reference) section above

### For Research and Examples
**Start here**: [Complete Workflow Demo](examples/complete_workflow_demo.md)

## 🎯 Key Achievements & Status

### ✅ **PARENT_SCALE Integration (Complete)**
- **100% Identical Behavior**: Verified across multiple seeds and scenarios
- **Production API**: `run_full_parent_scale_algorithm()` ready for expert demonstration collection
- **Comprehensive Documentation**: Complete implementation guide available

### ✅ **Code Refactoring (Complete - 2025-06-16)**
- **Modular Architecture**: Focused modules following functional programming principles
- **Clean APIs**: Backward compatible with improved maintainability
- **Quality Validation**: Comprehensive testing and validation utilities

### ✅ **Research Validation (Complete)**
- **BIC Scoring**: Prevents likelihood overfitting in structure learning
- **Intervention Strategies**: Random interventions outperform fixed (validated)
- **Scaling Performance**: 20+ node graphs demonstrated

## 📋 Document Purpose Guide

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **Expert Demonstration Collection Implementation** | Complete practical guide | When implementing expert demo collection |
| **PARENT_SCALE Integration Complete** | Achievement story and journey | Understanding the integration process |
| **PARENT_SCALE Data Bridge** | Technical implementation details | Understanding data format conversions |
| **Architecture Overview** | System design and components | Getting started with the codebase |
| **Interface Specifications** | API reference and data structures | When implementing or integrating |
| **Phase 4 Consolidated Plan** | Project roadmap and achievements | Understanding project scope and status |

## 🔍 Finding What You Need

- **Using the system**: Start with [Expert Demonstration Collection](training/expert_demonstration_collection_implementation.md)
- **Understanding the code**: Start with [Architecture Overview](architecture/overview.md)
- **API reference**: Check [Interface Specifications](architecture/interface_specifications.md)
- **Research insights**: Browse [Examples & Research](#-examples--research) section
- **Integration details**: See [Integration Documentation](#-integration-documentation) section

## 📝 Documentation Standards

All documentation follows these principles:
- **Functional Programming**: Pure functions, explicit parameters, no side effects
- **Type Safety**: Comprehensive type hints and validation
- **Immutable Data**: Uses pyrsistent for data structures where appropriate
- **Comprehensive Examples**: Working code snippets in all guides
- **Clear Scope**: Each document clearly states its purpose and relationship to others

---

*Last Updated: 2025-06-16*  
*Status: Documentation Complete ✅*