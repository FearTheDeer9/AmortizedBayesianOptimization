# ACBO Training Pipeline Implementation Plan

**Created**: 2025-01-20  
**Status**: IN_PROGRESS  
**Last Updated**: 2025-01-20

## Overview

This document tracks the implementation of the ACBO training pipeline with simplified ground-truth verifiable rewards. The plan emphasizes clean, modular design following functional programming principles while building on existing components.

## Key Design Decisions

### Simplified Reward System
- **Binary rewards**: 1 (correct) or 0 (incorrect) following successful GRPO patterns
- **Ground truth based**: Uses true SCM structure, not model predictions  
- **Decoupled training**: Acquisition model training independent of surrogate quality
- **Anti-gaming**: Simple binary components resist exploitation

### Training Architecture
- **Phase 1**: Expert demonstration collection (PARENT_SCALE)
- **Phase 2**: Independent surrogate training (behavioral cloning on expert demos)
- **Phase 3**: Acquisition training (GRPO with frozen surrogate + verifiable rewards)
- **Phase 4**: Curriculum progression (difficulty_1 → difficulty_2 → ...)

---

## Phase 1: Foundation & Cleanup (Week 1)

### 1.1 Expert Demonstration Collection Consolidation
**Status**: COMPLETED ✅  
**Files**: 
- `src/causal_bayes_opt/training/expert_collection/` (clean up existing)
- `src/causal_bayes_opt/training/expert_demonstration_collection.py` (simplify to wrapper)

**Tasks**:
- [x] Archive `expert_demonstration_collection_original.py`
- [x] Simplify `expert_demonstration_collection.py` to thin wrapper around `expert_collection/`
- [x] Enhance `expert_collection/collector.py` for production batch collection
- [x] Add parallel PARENT_SCALE runs for efficiency
- [x] Validate expert demonstration quality

**Implementation Notes**: 
- Created `archive/` directory and moved `expert_demonstration_collection_original.py` there
- Found `expert_demonstration_collection.py` was already a clean wrapper - no changes needed
- Enhanced `expert_collection/collector.py` with:
  - Added `collect_demonstration_batch_parallel()` method using `ProcessPoolExecutor`
  - Added logging support with proper error handling
  - Enhanced `main.py` with CLI argument parsing for flexible collection
  - Supports both serial and parallel collection modes
- The collector already had excellent production features:
  - Quality validation with configurable accuracy thresholds
  - Comprehensive data requirements calculation using validated O(d^2.5) scaling
  - Proper error handling and retry logic
  - Multiple save formats (pickle/json)
  - Rich metadata tracking

**Plan Modifications**: No major changes needed - existing code was cleaner than expected

### 1.2 Simple Ground-Truth Verifiable Rewards
**Status**: COMPLETED ✅  
**Files**:
- `src/causal_bayes_opt/acquisition/verifiable_rewards.py` (new simple implementation)
- `tests/test_acquisition/test_verifiable_rewards.py` (comprehensive test suite)
- `src/causal_bayes_opt/acquisition/__init__.py` (updated exports)

**Tasks**:
- [x] Create `verifiable_rewards.py` with binary reward functions:
  - `target_improvement_reward()` - binary target improvement
  - `true_parent_intervention_reward()` - reward for intervening on actual parents  
  - `exploration_diversity_reward()` - diversity bonus
  - `compute_simple_verifiable_reward()` - weighted combination
- [x] Add anti-gaming monitoring functions (`validate_reward_consistency()`)
- [x] Add comprehensive tests for reward functions (39 tests, all passing)
- [x] Add integration convenience function (`compute_verifiable_reward_simple()`)

**Implementation Notes**: 
- Created completely new binary reward system following successful GRPO patterns
- All reward components are binary (0 or 1) to prevent gaming
- Ground truth based rewards using true SCM structure
- Comprehensive test coverage including gaming detection scenarios
- Anti-gaming measures:
  - Validate component balance (no single component >70% of total)
  - Detect mode collapse (low exploration diversity)
  - Monitor suspicious patterns (always hitting true parents)
  - Check reward variance (too consistent suggests gaming)
- Default weights: target_improvement=2.0, true_parent=1.0, exploration=0.5
- Configurable thresholds for improvement and diversity
- Clean integration with existing pyrsistent data structures

**Plan Modifications**: 
- Decided NOT to refactor existing `rewards.py` to maintain backward compatibility
- Both reward systems now coexist - old for research, new for production training

### 1.3 Training Directory Organization  
**Status**: COMPLETED ✅  
**Files**:
- `src/causal_bayes_opt/training/` (reorganized existing files)
- `src/causal_bayes_opt/training/archive/` (created)
- `src/causal_bayes_opt/training/master_trainer.py` (new)
- `src/causal_bayes_opt/training/curriculum.py` (new)

**Tasks**:
- [x] Archive redundant files to `archive/` subdirectory (created directory structure)
- [x] Create clear module responsibilities:
  - `master_trainer.py` - Main orchestrator ✅
  - `curriculum.py` - Difficulty progression ✅
  - `acquisition_trainer.py` - GRPO training (existing file kept)
  - `surrogate_training.py` - Keep existing JAX-optimized version ✅
  - `config.py` - Configuration system (existing enhanced) ✅
- [x] Update `__init__.py` exports for clean public API ✅

**Implementation Notes**: 
- Created `archive/` directory for future redundant files
- Built `master_trainer.py` with complete pipeline orchestration:
  - TrainingState immutable data structure
  - MasterTrainer class with checkpoint support
  - Three-stage pipeline: expert collection → surrogate → acquisition
  - Integration with curriculum learning
  - Placeholder implementations for Phase 2 development
- Built `curriculum.py` with progressive difficulty framework:
  - 5 difficulty levels from small (3-5 vars) to expert (20-30 vars)
  - Adaptive thresholds based on SCM characteristics
  - Clear advancement criteria (F1 thresholds, stability windows)
  - SCM generation for each difficulty level
  - Full integration with verifiable rewards system
- Updated `__init__.py` with proper exports and graceful import handling
- All imports tested and working correctly
- Basic functionality verified through test runs

**Plan Modifications**: 
- Decided not to archive existing files since they were well-organized already
- Used placeholder implementations in master_trainer to avoid Phase 2 dependencies
- All modules designed for Phase 2 implementation (they have proper TODO markers)

**API Integration Fixes Applied**:
- Fixed `TrainingState.replace()` → `dataclasses.replace()` for all state transitions
- Removed unused imports (`jax`, `jnp`, `Tuple`, `CurriculumManager`) flagged by Pylance
- Fixed config access patterns to use actual attributes (`config.grpo`, `config.surrogate`, etc.)
- Added comprehensive integration test validation (4/4 tests passing)
- Verified all function signatures and return types match expectations
- Confirmed API compatibility with existing reward and curriculum systems

### 1.4 Basic Training Infrastructure
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/master_trainer.py` (new)
- `scripts/train_acbo.py` (new)

**Tasks**:
- [ ] Create `MasterTrainer` class with clean interface
- [ ] Implement basic training pipeline coordination
- [ ] Add configuration loading and validation
- [ ] Create production training script entry point
- [ ] Add logging and progress monitoring

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

---

## Phase 2: Core Training Pipeline (Week 2)

### 2.1 Decoupled Surrogate Training
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/surrogate_trainer.py` (new wrapper around existing)

**Tasks**:
- [ ] Create clean interface around existing `surrogate_training.py`
- [ ] Implement expert demonstration loading and preprocessing
- [ ] Add BIC-penalized likelihood loss (existing)
- [ ] Create training loop with checkpointing
- [ ] Add validation metrics and early stopping

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 2.2 GRPO Acquisition Training with Verifiable Rewards
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/acquisition_trainer.py` (enhance existing)
- `src/causal_bayes_opt/acquisition/grpo.py` (integrate with new rewards)

**Tasks**:
- [ ] Integrate GRPO with new verifiable reward system
- [ ] Implement behavioral cloning warm-start phase
- [ ] Add GRPO fine-tuning with group_size=64
- [ ] Create training loop with frozen surrogate model
- [ ] Add comprehensive training diagnostics

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 2.3 Anti-Gaming Monitoring System
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/monitoring.py` (new)

**Tasks**:
- [ ] Implement reward component tracking
- [ ] Add gaming detection algorithms
- [ ] Create alerting system for exploitation patterns
- [ ] Add training stability metrics
- [ ] Implement automated intervention for detected issues

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 2.4 Curriculum Learning Framework
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/curriculum.py` (new)

**Tasks**:
- [ ] Define difficulty levels (difficulty_1, difficulty_2, ...)
- [ ] Implement progression criteria (F1 thresholds)
- [ ] Create SCM generation for each difficulty
- [ ] Add advancement validation
- [ ] Implement curriculum scheduling

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

---

## Phase 3: Production & Optimization (Week 3)

### 3.1 Master Training Orchestrator  
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/master_trainer.py` (enhance from Phase 1)

**Tasks**:
- [ ] Complete pipeline orchestration
- [ ] Add error handling and recovery
- [ ] Implement distributed training coordination
- [ ] Add experiment tracking and logging
- [ ] Create resume-from-checkpoint functionality

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 3.2 GPU/Distributed Training Support
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/distributed.py` (new)

**Tasks**:
- [ ] Add JAX device detection and GPU utilization
- [ ] Implement memory-efficient training (gradient checkpointing)
- [ ] Add batch size scaling based on available memory
- [ ] Create multi-GPU model parallelism
- [ ] Add mixed precision training for speed

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 3.3 Production Scripts and Configuration
**Status**: TODO  
**Files**:
- `scripts/train_acbo.py` (enhance from Phase 1)
- `scripts/collect_expert_batch.py` (new)
- `scripts/evaluate_acbo.py` (new)
- `configs/acbo_training.yaml` (new)

**Tasks**:
- [ ] Complete production training script with all features
- [ ] Create parallel expert collection script
- [ ] Implement comprehensive evaluation script
- [ ] Design configuration system with YAML support
- [ ] Add command-line argument parsing

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 3.4 Comprehensive Evaluation Framework
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/evaluation/` (new directory)

**Tasks**:
- [ ] Create evaluation metrics (F1, optimization improvement, etc.)
- [ ] Implement baseline comparisons (PARENT_SCALE)
- [ ] Add statistical significance testing
- [ ] Create visualization tools for results
- [ ] Implement automated evaluation pipelines

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

---

## Phase 4: Advanced Features (Week 4)

### 4.1 Transfer Learning Linear→Non-linear
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/training/transfer_learning.py` (new)

**Tasks**:
- [ ] Design progressive mechanism complexity training
- [ ] Implement selective layer freezing/fine-tuning
- [ ] Add non-linear mechanism support
- [ ] Create transfer learning validation
- [ ] Compare vs from-scratch training

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 4.2 Performance Optimization and Profiling
**Status**: TODO  
**Files**: Various (optimization across codebase)

**Tasks**:
- [ ] Profile training pipeline for bottlenecks
- [ ] Optimize JAX compilation and memory usage
- [ ] Add performance monitoring and metrics
- [ ] Implement training speed improvements
- [ ] Create performance comparison benchmarks

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 4.3 Documentation and Examples
**Status**: TODO  
**Files**:
- `docs/training/` (enhance existing)
- `examples/training_pipeline_demo.py` (new)

**Tasks**:
- [ ] Complete training pipeline documentation
- [ ] Create comprehensive examples and tutorials
- [ ] Add troubleshooting guides
- [ ] Document configuration options
- [ ] Create getting started guide

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

### 4.4 Final Validation Against Baselines
**Status**: TODO  
**Files**: 
- `experiments/validation_study.py` (new)

**Tasks**:
- [ ] Design comprehensive validation study
- [ ] Compare against PARENT_SCALE on multiple metrics
- [ ] Validate sample efficiency improvements
- [ ] Test generalization across different SCM types
- [ ] Document final performance results

**Implementation Notes**: *(To be filled as we implement)*

**Plan Modifications**: *(Document any changes to original plan)*

---

## Progress Tracking

### Overall Progress
- **Phase 1**: ⏳ 75% (3/4 sections complete)
- **Phase 2**: ⏳ 0% (0/4 sections complete)  
- **Phase 3**: ⏳ 0% (0/4 sections complete)
- **Phase 4**: ⏳ 0% (0/4 sections complete)

### Current Focus
**Active Task**: Phase 1.4 - Basic Training Infrastructure

### Completed Milestones
- [x] Created implementation plan document
- [x] Phase 1.1 - Expert Demonstration Collection Consolidation
- [x] Phase 1.2 - Simple Ground-Truth Verifiable Rewards
- [x] Phase 1.3 - Training Directory Organization

### Blocked Items
- *(None currently)*

### Implementation Discoveries
*(Document insights and challenges discovered during implementation)*

### Plan Evolution
*(Track how the plan changes as we learn from actual implementation)*

---

## Technical Notes

### Key Dependencies
- JAX/Haiku for neural networks
- Optax for optimization  
- Pyrsistent for immutable data structures
- PARENT_SCALE integration for expert demonstrations

### Performance Targets
- **Training Speed**: Complete curriculum in <24 hours on single GPU
- **Sample Efficiency**: Match PARENT_SCALE performance with 10x fewer samples
- **Structure Learning**: >90% F1 score on test SCMs
- **Optimization**: Match/exceed PARENT_SCALE target improvement

### Success Criteria
- [ ] All phases complete with working code
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance targets met
- [ ] Clean, maintainable codebase following project standards
- [ ] Complete documentation and examples

---

*This document will be updated regularly as implementation progresses.*