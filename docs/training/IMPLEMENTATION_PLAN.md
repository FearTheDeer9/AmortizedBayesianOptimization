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
**Status**: COMPLETED ✅  
**Files**:
- `src/causal_bayes_opt/training/master_trainer.py` (implemented)
- `src/causal_bayes_opt/training/curriculum.py` (implemented)
- `src/causal_bayes_opt/training/config.py` (enhanced existing)

**Tasks**:
- [x] Create `MasterTrainer` class with clean interface
- [x] Implement basic training pipeline coordination
- [x] Add configuration loading and validation
- [x] Create production training script entry point (placeholder ready)
- [x] Add logging and progress monitoring

**Implementation Notes**: 
- Successfully created `MasterTrainer` class with immutable `TrainingState` data structure
- Three-stage pipeline orchestration: expert collection → surrogate → acquisition
- Complete integration with curriculum learning framework
- Placeholder implementations marked for Phase 2 development (proper TODO markers)
- API integration fixes applied: fixed `dataclasses.replace()`, removed unused imports
- Comprehensive integration test validation (4/4 tests passing)
- All modules designed for Phase 2 implementation

**Plan Modifications**: 
- Used placeholder implementations to avoid Phase 2 dependencies
- Enhanced beyond original scope to include curriculum integration
- All function signatures and return types verified for API compatibility

---

## ARCHITECTURE ENHANCEMENT PIVOT (Added 2025-06-20)

### Context
During Phase 2.1 implementation planning, we identified a critical limitation in our current architecture: the `ParentSetPredictionModel` only predicts **which** variables are parents (topology) but not **how** they influence their children (mechanism type and parameters). This severely limits intervention effectiveness because:

1. **Missing Mechanism Information**: Knowing "X is a parent of Y" doesn't tell us the functional form (linear, polynomial, Gaussian) or effect magnitude (coefficient 0.1 vs 10.0)
2. **Suboptimal Intervention Selection**: Without mechanism information, acquisition policy cannot prioritize high-impact interventions or predict outcomes accurately
3. **Limited Generalization**: Real-world causal systems have diverse mechanism types beyond simple linear relationships

### Decision
**PIVOT**: Implement mechanism-aware architecture enhancement before continuing with Phase 2, ensuring we build the training pipeline on a fundamentally sound foundation.

**Design Principles**:
1. **Modular Design**: Easy switching between structure-only and mechanism-aware modes for scientific comparison
2. **Backward Compatibility**: Structure-only mode preserved as fallback to validate base system works
3. **Hybrid Rewards**: Combine supervised learning (valid use of ground truth) with observable signals for robustness
4. **TDD Approach**: Comprehensive testing following our established patterns

### Enhanced Plan: Mechanism-Aware ACBO System

#### Part A: Modular Model Architecture (3 days)
**Status**: COMPLETED ✅  
**Files**: 
- `src/causal_bayes_opt/avici_integration/parent_set/mechanism_aware.py` (new - 570 lines)
- `tests/test_avici_integration/test_mechanism_aware.py` (new - 533 lines, 20/20 tests passing)
- `examples/mechanism_aware_demo.py` (new - demo script)

**Tasks**:
- [x] Implement `ModularParentSetModel` with configurable mechanism prediction
- [x] Add mechanism prediction heads:
  - [x] Mechanism type classification (linear, polynomial, gaussian, neural)
  - [x] Parameter regression (coefficients, effect magnitudes, uncertainties)
- [x] Ensure backward compatibility with structure-only mode via feature flags
- [x] Create configuration switching system (`predict_mechanisms: bool`)
- [x] Comprehensive testing of both modes (20 tests, all passing)

**Implementation Notes**:
- Successfully created `ModularParentSetModel` with clean feature flag architecture
- Supports 4 mechanism types: linear, polynomial, gaussian, neural
- Full backward compatibility: structure-only mode works identically to existing system
- Enhanced mode adds mechanism type classification and parameter regression
- Factory functions `create_structure_only_config()` and `create_enhanced_config()` for easy switching
- Comprehensive test coverage validates both modes and integration with existing API
- Demo script shows real working examples with configurable mechanism types
- All 20 tests passing, demonstrates robust implementation

**Architecture**:
```python
class ModularParentSetModel(hk.Module):
    def __init__(self, predict_mechanisms: bool = False, mechanism_types: List[str] = None):
        # Feature flag controls complexity
    
    def __call__(self, x, variable_order, target_variable):
        structure_outputs = self._predict_structure(...)  # Always included
        
        if not self.predict_mechanisms:
            return structure_outputs  # Simple mode
        
        # Enhanced mode: add mechanism predictions
        mechanism_outputs = self._predict_mechanisms(...)
        return {**structure_outputs, **mechanism_outputs}
```

#### Part B: Hybrid Reward System (2 days)
**Status**: COMPLETED ✅  
**Files**:
- `src/causal_bayes_opt/acquisition/hybrid_rewards.py` (new - 799 lines)
- `tests/test_acquisition/test_hybrid_rewards.py` (new - 693 lines, 22/22 tests passing)
- `examples/hybrid_rewards_demo.py` (new - demo script)

**Tasks**:
- [x] Implement supervised mechanism rewards (using ground truth during training):
  - [x] `supervised_mechanism_impact_reward()` - weight by true effect magnitude
  - [x] `supervised_mechanism_discovery_reward()` - reward for high-uncertainty edges
- [x] Implement observable signal rewards (no ground truth, for robustness):
  - [x] `posterior_confidence_reward()` - uncertainty reduction
  - [x] `causal_effect_discovery_reward()` - outcome-based effect detection
  - [x] `mechanism_consistency_reward()` - prediction validation
- [x] Create configurable hybrid reward system with flexible weighting
- [x] Comprehensive testing including gaming detection

**Implementation Notes**:
- Successfully created comprehensive hybrid reward system combining supervised and observable signals
- Supports 3 configuration modes: training (both signals), deployment (observable only), research (supervised only)
- All reward components are pure functions with proper JAX integration
- Configurable weighting system allows flexible balance between signal types
- Comprehensive gaming detection with validation functions
- Integration with mechanism-aware architecture through MechanismPrediction objects
- All 22 tests passing, demonstrates robust implementation
- Demo script shows real working examples with different configurations
- Anti-gaming measures include variance checking, contribution balance, and suspicious pattern detection

**Reward Configuration**:
```python
@dataclass
class HybridRewardConfig:
    # Supervised signals (use ground truth in training)
    use_supervised_signals: bool = True
    supervised_parent_weight: float = 1.0
    supervised_mechanism_weight: float = 0.8
    
    # Observable signals (no ground truth, for robustness)  
    use_observable_signals: bool = True
    posterior_confidence_weight: float = 0.5
    mechanism_consistency_weight: float = 0.6
```

#### Part C: Integration & Testing (2 days)
**Status**: TODO  
**Files**:
- `src/causal_bayes_opt/acquisition/state.py` (enhance for mechanism predictions)
- `src/causal_bayes_opt/acquisition/policy.py` (enhance for mechanism features)
- `tests/test_integration/test_mechanism_aware_pipeline.py` (new)

**Tasks**:
- [ ] Update `AcquisitionState` to include mechanism predictions and uncertainties
- [ ] Enhance policy network to use mechanism information for variable selection
- [ ] Create comparative evaluation framework for structure-only vs mechanism-aware
- [ ] End-to-end integration testing
- [ ] Performance benchmarking and ablation studies

#### Part D: Scientific Validation (1 day)
**Status**: TODO  
**Files**:
- `examples/mechanism_aware_demo.py` (new)
- `docs/architecture/adr/005_mechanism_aware_architecture.md` (new ADR)

**Tasks**:
- [ ] Create demonstration showing both modes work
- [ ] Document architectural decision with alternatives considered
- [ ] Implement side-by-side performance comparison
- [ ] Validate 20%+ efficiency improvement to justify complexity

### Return to Original Plan
**Timeline**: After architecture enhancement (7 days total), return to Phase 2.1 with enhanced mechanism-aware system.

**Benefits Expected**:
- 20%+ improvement in intervention efficiency
- Better sample efficiency through targeted high-impact interventions  
- Generalization to diverse mechanism types
- Scientific validation of when added complexity is worthwhile

---

## Phase 2: Core Training Pipeline (Week 2)

### 2.1 Decoupled Surrogate Training
**Status**: COMPLETED ✅  
**Files**:
- `src/causal_bayes_opt/training/surrogate_trainer.py` (new wrapper around existing)
- `tests/test_training/test_surrogate_trainer.py` (comprehensive test suite)
- `tests/test_training/test_surrogate_trainer_integration.py` (integration tests)

**Tasks**:
- [x] Create clean interface around existing `surrogate_training.py`
- [x] Implement expert demonstration loading and preprocessing
- [x] Add BIC-penalized likelihood loss (existing)
- [x] Create training loop with checkpointing
- [x] Add validation metrics and early stopping

**Implementation Notes**: 
- Successfully created `SurrogateTrainer` class as clean wrapper around sophisticated `surrogate_training.py`
- Follows TDD approach with comprehensive test coverage (11/11 integration tests passing)
- Preserves JAX performance optimizations (250-3,386x speedup from existing infrastructure)
- Integrated with Master Trainer for orchestrated training pipeline
- Maintains backward compatibility while adding new functionality
- Graceful error handling and detailed logging

**Plan Modifications**: 
- Enhanced beyond original scope to include comprehensive testing and master trainer integration
- **READY**: Now enhanced with mechanism-aware architecture for Phase 2.2

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
- **Phase 1**: ✅ 100% (4/4 sections complete)
- **Phase 2**: ⏳ 25% (1/4 sections complete)  
- **Phase 3**: ⏳ 0% (0/4 sections complete)
- **Phase 4**: ⏳ 0% (0/4 sections complete)

### Current Focus
**Active Task**: Architecture Enhancement Pivot - Part C: Integration & Testing (next up after Part B completion)

### Completed Milestones
- [x] Created implementation plan document
- [x] Phase 1.1 - Expert Demonstration Collection Consolidation
- [x] Phase 1.2 - Simple Ground-Truth Verifiable Rewards
- [x] Phase 1.3 - Training Directory Organization
- [x] Phase 2.1 - Decoupled Surrogate Training (with JAX compilation speedup)
- [x] Architecture Enhancement Pivot - Part A: Modular Model Architecture (mechanism-aware parent set prediction)
- [x] Architecture Enhancement Pivot - Part B: Hybrid Reward System (supervised + observable signals)

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