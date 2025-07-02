# Context Findings

## GRPO Policy Training Setup Analysis

### Current State Summary
The GRPO policy training system has a sophisticated theoretical foundation but contains several critical integration and robustness issues that need to be addressed before reliable training validation can occur.

### Key Components Identified

#### 1. **GRPO Core Implementation** (`src/causal_bayes_opt/training/grpo_core.py`)
- **Strengths**: Pure functional JAX implementation with proper advantage estimation
- **Potential Issue**: Conditional branches in value loss computation may affect JAX compilation

#### 2. **Reward System** (`src/causal_bayes_opt/acquisition/rewards.py`)
- **Components**: optimization (1.0), structure discovery (0.5), parent intervention (0.3), exploration (0.1)
- **Issue**: Optimization dominates reward signal, may limit structure learning objectives
- **Lines 85-95**: Hardcoded reward weights need validation for dual objectives

#### 3. **GRPO Training Manager** (`src/causal_bayes_opt/training/grpo_training_manager.py`)
- **Critical Issues**:
  - Silent fallback to mock results when training fails (lines 298-320)
  - Enhanced policy network integration can fail silently
  - State tensor conversion falls back to zero tensors (lines 370-420)

#### 4. **Enhanced Policy Network** (`src/causal_bayes_opt/acquisition/enhanced_policy_network.py`)
- **Issue**: Integration with GRPO appears incomplete or problematic
- **Missing**: Proper connection between enhanced networks and training manager

#### 5. **Test Infrastructure**
- **Critical**: Extensive test failures across all training components
- **Files**: All test files in `tests/test_training/` are failing
- **Impact**: No validation that training pipeline actually works

### Specific Files Requiring Investigation

#### **Critical Priority**:
1. `src/causal_bayes_opt/training/grpo_training_manager.py:298-320` - Fix silent failures
2. `src/causal_bayes_opt/training/grpo_training_manager.py:370-420` - Fix state tensor creation
3. `src/causal_bayes_opt/acquisition/rewards.py:85-95` - Validate reward component balance
4. `src/causal_bayes_opt/acquisition/enhanced_policy_network.py` - Complete GRPO integration

#### **High Priority**:
5. `tests/test_training/test_grpo_training_manager.py` - Fix failing tests
6. `tests/test_training/test_end_to_end_training.py` - Validate full pipeline
7. `src/causal_bayes_opt/training/grpo_core.py` - Ensure JAX compatibility

### Training Validation Strategy Needed

Based on findings, we need:
1. **Fix silent failure modes** that mask training issues
2. **Validate reward component balance** for dual objectives (structure + optimization)
3. **Create simple GRPO validation tests** before complex scenarios
4. **Implement component isolation tests** (e.g., zero out one reward component)
5. **Fix enhanced policy network integration**

### Related Features/Patterns
- Uses JAX functional programming throughout
- Follows immutable data structure patterns with pyrsistent
- Has curriculum learning infrastructure but may not be properly connected
- Contains sophisticated attention-based policy networks but integration is problematic