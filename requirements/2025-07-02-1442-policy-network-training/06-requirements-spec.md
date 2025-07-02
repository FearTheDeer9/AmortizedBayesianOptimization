# GRPO Policy Network Training Validation Requirements

## Problem Statement
The GRPO policy training system has sophisticated architecture but contains critical integration issues, silent failure modes, and suboptimal reward design that prevent reliable training validation. Need to ensure the training loop works correctly and provides proper learning signals for both structure discovery and target optimization objectives.

## Solution Overview
Fix critical training pipeline issues, implement component-wise validation, design improved reward system, and create comprehensive validation tests to ensure GRPO policy training actually improves performance on intended tasks.

## Functional Requirements

### FR1: Fix Silent Failure Modes
- **Location**: `src/causal_bayes_opt/training/grpo_training_manager.py:298-320`
- **Requirement**: Remove fallback behaviors that return mock results when training fails
- **Acceptance**: Training failures raise explicit exceptions instead of returning mock data
- **Priority**: Critical

### FR2: Component-wise Reward Validation
- **Location**: New validation functions in training system
- **Requirement**: Create validation functions that can selectively disable reward components
- **Components**: optimization, structure discovery, parent intervention, exploration
- **Acceptance**: Can zero out individual components and verify optimization pressure works correctly for remaining components
- **Priority**: High

### FR3: Basic GRPO Training Validation
- **Location**: New simple validation tests
- **Requirement**: Create simple validation tests using basic policy networks before enhanced integration
- **Acceptance**: Basic GRPO training loop demonstrably improves policy performance on simple tasks
- **Priority**: High

### FR4: Improved Optimization Reward Design
- **Location**: `src/causal_bayes_opt/acquisition/rewards.py`
- **Requirement**: Replace binary relative-improvement reward with continuous, SCM-objective reward
- **Problem**: Current reward causes agent to avoid optimal interventions once found
- **Acceptance**: New reward system encourages optimal interventions regardless of prior history
- **Priority**: High

### FR5: Test Infrastructure Validation
- **Location**: `tests/test_training/` directory
- **Requirement**: Fix failing tests, ensuring they test genuinely important functionality
- **Acceptance**: Working test suite that validates training pipeline components
- **Note**: Ensure tests are not outdated and focus on meaningful sanity checks
- **Priority**: Medium

### FR6: State Tensor Creation Fix
- **Location**: `src/causal_bayes_opt/training/grpo_training_manager.py:370-420`
- **Requirement**: Fix state tensor conversion that falls back to zero tensors
- **Acceptance**: Proper state representation without fallback behaviors
- **Priority**: High

### FR7: Balanced Reward Weight Testing
- **Location**: Validation test configuration
- **Requirement**: Test with balanced weights between optimization and structure discovery
- **Current**: optimization=1.0, structure=0.5, parent=0.3, exploration=0.1
- **Test**: Both optimization and structure at 1.0 for validation
- **Priority**: Medium

## Technical Requirements

### TR1: JAX Compatibility
- **Requirement**: Ensure all GRPO components are JAX-compatible and compilable
- **Location**: `src/causal_bayes_opt/training/grpo_core.py`
- **Focus**: Value loss computation conditional branches

### TR2: Enhanced Policy Network Integration
- **Requirement**: Complete integration between enhanced policy networks and GRPO training
- **Location**: `src/causal_bayes_opt/acquisition/enhanced_policy_network.py`
- **Priority**: Medium (after basic validation works)

### TR3: Immutable Data Structures
- **Requirement**: Maintain pyrsistent patterns throughout training pipeline
- **Pattern**: Follow existing codebase conventions

### TR4: Functional Programming Principles
- **Requirement**: Pure functions, no side effects in core training logic
- **Pattern**: Follow CLAUDE.md standards

## Implementation Strategy

### Phase 1: Critical Fixes (Priority: Critical/High)
1. Remove silent fallback behaviors in GRPOTrainingManager
2. Fix state tensor creation without fallbacks
3. Design and implement continuous, SCM-objective optimization reward
4. Create component-wise reward validation functions

### Phase 2: Basic Validation (Priority: High)
1. Create simple GRPO training validation with basic networks
2. Implement zero-out reward component tests
3. Test with balanced reward weights

### Phase 3: Test Infrastructure (Priority: Medium)
1. Audit and fix failing tests in tests/test_training/
2. Ensure tests validate meaningful functionality
3. Remove or update outdated tests

### Phase 4: Enhanced Integration (Priority: Medium)
1. Fix enhanced policy network integration issues
2. Validate enhanced networks work with GRPO training

## Acceptance Criteria

### AC1: Training Pipeline Robustness
- No silent failures or mock result fallbacks
- Explicit error handling with meaningful exceptions
- Training either works or fails clearly

### AC2: Reward Component Validation
- Can isolate and test individual reward components
- Demonstrated optimization pressure for each component
- Continuous optimization reward that doesn't avoid optimal interventions

### AC3: Learning Demonstration
- Simple validation shows policies actually improve over time
- Both structure discovery and target optimization objectives reinforced
- Performance metrics demonstrate learning is occurring

### AC4: Test Coverage
- Working test suite for training pipeline
- Tests validate genuine functionality
- All critical training paths covered

## Assumptions

### A1: Reward Design
- Continuous, SCM-objective reward will be more effective than relative improvement
- Optimal interventions should always be preferred regardless of history

### A2: Training Architecture
- Basic GRPO training validation will establish foundation for enhanced networks
- Component isolation is sufficient to validate optimization pressure

### A3: Test Quality
- Some current test failures are due to outdated tests rather than real issues
- Focus on meaningful sanity checks rather than comprehensive coverage

## Dependencies
- JAX/Haiku for neural networks (existing)
- Pyrsistent for immutable data structures (existing)
- Existing GRPO core implementation (needs fixes)
- Enhanced policy network architecture (exists but needs integration fixes)

## Success Metrics
- Training pipeline runs without silent failures
- Policy performance demonstrably improves on validation tasks
- Both structure discovery and optimization objectives show learning
- Component-wise validation confirms reward system works correctly
- Test suite provides reliable validation of training functionality