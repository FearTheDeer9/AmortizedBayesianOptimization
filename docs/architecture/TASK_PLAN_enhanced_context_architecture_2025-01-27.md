# Task Plan: Enhanced Context Architecture Fix

## Date: 2025-01-27

## Objective

Fix the fundamental design flaw in the current policy network architecture where learned transformer representations are diluted with hand-crafted features through post-processing concatenation. Replace with an enriched transformer input approach that enables temporal integration of all contextual information.

## Current State Analysis

### Problem Statement
The current `AcquisitionPolicyNetwork` in `src/causal_bayes_opt/acquisition/policy.py` suffers from an architectural contradiction:
- **Stage 1**: Transformer learns good representations from intervention history
- **Stage 2**: 9 hand-crafted features are generated via complex feature engineering
- **Stage 3**: Learned representations are concatenated with hand-crafted features, wasting learned structure

### Evidence of Design Flaw
- **Location**: `src/causal_bayes_opt/acquisition/policy.py:434-442`
- **Pattern**: Post-transformer feature concatenation instead of enriched input
- **Impact**: Information loss, reduced temporal reasoning, architectural contradiction

### Current Dependencies (Found via Serena MCP)
**22 total references** across the codebase:

**Files importing/using `AcquisitionPolicyNetwork`:**
- `examples/complete_acquisition_training_demo.py` (import + usage)
- `src/causal_bayes_opt/acquisition/__init__.py` (export)
- `src/causal_bayes_opt/training/acquisition_training.py` (import)
- `tests/test_acquisition/test_policy.py` (import + tests)
- `tests/test_integration/test_mechanism_aware_comparative.py` (import)
- `tests/test_integration/test_mechanism_aware_pipeline.py` (import)

**Files using `AlternatingAttentionEncoder`:**
- `src/causal_bayes_opt/acquisition/__init__.py` (export)
- `tests/test_acquisition/test_policy.py` (import + 3 dedicated tests)

**Files using `create_acquisition_policy`:**
- `examples/complete_acquisition_training_demo.py` (usage)
- `src/causal_bayes_opt/training/acquisition_training.py` (usage)
- `tests/test_acquisition/test_policy.py` (16 test usages)
- `tests/test_integration/test_mechanism_aware_pipeline.py` (2 test usages)

### Key Insight
User confirmed the policy network **has never been trained or used in practice**, making complete redesign safe and practical.

## Implementation Plan

### Phase 1: Planning & Documentation ✓
1. ✓ Create this planning document
2. ✓ Analyze complete dependency graph using Serena MCP
3. ✓ Document all 22 references requiring updates

### Phase 2: Test-Driven Architecture Design
1. **Write comprehensive tests** for new `EnrichedTransformerEncoder`:
   - Property-based tests for enriched history generation
   - Integration tests comparing behavior with old architecture
   - Performance benchmarks for JAX compilation improvements
   - Edge case coverage for different input configurations

2. **Design enriched input format**:
   ```python
   # New format: [MAX_HISTORY_SIZE, n_vars, 10]
   enriched_history[:, :, 0] = standardized_values
   enriched_history[:, :, 1] = intervention_indicators  
   enriched_history[:, :, 2] = target_indicators
   enriched_history[:, :, 3] = marginal_parent_probabilities
   enriched_history[:, :, 4] = uncertainty_bits
   enriched_history[:, :, 5] = mechanism_confidence
   enriched_history[:, :, 6] = predicted_effect_magnitude
   enriched_history[:, :, 7] = mechanism_type_encoding
   enriched_history[:, :, 8] = best_value_progress
   enriched_history[:, :, 9] = stagnation_indicators
   ```

### Phase 3: Implementation
1. **Create new modules**:
   - `src/causal_bayes_opt/acquisition/enriched_encoder.py`
   - `src/causal_bayes_opt/acquisition/context_enrichment.py`

2. **Implement core classes**:
   - `EnrichedTransformerEncoder`: Processes 10-channel enriched input
   - `ContextEnrichmentBuilder`: Converts AcquisitionState to enriched tensor
   - `EnrichedAcquisitionPolicyNetwork`: New policy with enriched encoder

### Phase 4: Clean Migration (Delete-First Approach)
1. **Remove old architecture completely**:
   - Delete `AlternatingAttentionEncoder` class
   - Delete feature concatenation logic in `_variable_selection_head`
   - Remove hand-crafted feature engineering functions

2. **Replace policy implementation**:
   - Replace `AcquisitionPolicyNetwork` implementation
   - Update `create_acquisition_policy` factory function
   - Maintain interface compatibility for existing users

### Phase 5: Systematic Reference Updates
**Use Serena MCP to update all 22 references:**

1. **Import updates**:
   - `src/causal_bayes_opt/acquisition/__init__.py`: Update exports
   - All test files: Update imports

2. **Usage updates**:
   - Maintain `PolicyConfig` interface compatibility
   - Ensure `create_acquisition_policy` signature unchanged
   - Validate `sample_intervention_from_policy` still works

3. **Test updates**:
   - Update all 16 test usages in `test_policy.py`
   - Update integration tests to work with new architecture
   - Ensure all assertions still valid

### Phase 6: Integration & Validation
1. **Run comprehensive test suite**
2. **Validate JAX compilation improvements**
3. **Performance benchmarking**
4. **Integration testing with existing workflows**

## Design Decisions Log

### Decision 1: Enriched Input vs Feature Concatenation
**Date**: 2025-01-27  
**Decision**: Move all contextual information into transformer input channels rather than concatenating after encoding  
**Rationale**: Enables temporal integration, reduces feature engineering, leverages transformer's attention mechanisms  
**Alternative**: Keep concatenation but improve feature engineering - rejected due to architectural contradiction

### Decision 2: Delete-First Migration Strategy
**Date**: 2025-01-27  
**Decision**: Completely remove old architecture rather than gradual migration  
**Rationale**: Policy network never used in practice, clean slate enables better design, reduces code complexity  
**Alternative**: Feature flag approach - rejected due to unnecessary complexity given lack of production usage

### Decision 3: Interface Compatibility
**Date**: 2025-01-27  
**Decision**: Maintain existing `PolicyConfig` and `create_acquisition_policy` interfaces  
**Rationale**: Minimize disruption to existing code, enable drop-in replacement  
**Alternative**: New interface design - rejected to reduce migration complexity

## Problems & Solutions

*This section will be updated as problems are encountered during implementation*

## Progress Updates

### 2025-01-27 Initial Planning
- ✅ Created planning document
- ✅ Analyzed current architecture flaws
- ✅ Identified all 22 code references using Serena MCP
- ✅ Designed enriched input format
- ✅ Established test-first implementation strategy

*Additional progress updates will be appended here*

## Next Steps

1. Begin Phase 2: Write comprehensive tests for `EnrichedTransformerEncoder`
2. Implement `ContextEnrichmentBuilder` with property-based testing
3. Create new encoder with 10-channel input processing
4. Update planning document after each major milestone