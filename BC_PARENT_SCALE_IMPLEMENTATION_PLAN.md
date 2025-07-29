# BC Training with PARENT_SCALE Demonstrations - Implementation Plan

## Date: 2025-01-29

## Overview
This document captures our plan for implementing BC training using real PARENT_SCALE expert demonstrations from the `expert_demonstrations/` directory, replacing the current synthetic "oracle" approach.

## Key Discoveries

### Data Structure
1. **Expert demonstrations** are stored as `ExpertDemonstration` objects containing:
   - `observational_samples`: List of PMaps with structure `{'values': {var: value}, ...}`
   - `interventional_samples`: List of PMaps with intervention outcomes
   - `parent_posterior['trajectory']`: Contains `intervention_sequence` and `intervention_values`
   - `parent_posterior['posterior_history']`: Evolution of parent set posteriors

2. **3-Channel Tensor Format** (verified from `three_channel_converter.py`):
   - Channel 0: Variable values (standardized)
   - Channel 1: Target indicator (1.0 for target variable only)
   - Channel 2: Intervention indicator (1.0 if variable was intervened on)

### Critical Implementation Details

#### BC Training Examples
Extract (state, action) pairs where:
- **State**: 3-channel tensor from buffer at decision time
- **Action**: Expert's choice of (intervention_variable, intervention_value)

#### Surrogate Training Examples
Extract (input, posterior) pairs where:
- **Input**: 3-channel tensor of all data before intervention outcome
- **Output**: Marginal parent probabilities after seeing intervention outcome

#### Marginal Probability Extraction (TO VERIFY)
```python
def compute_marginal_parent_probs(posterior_dict, variables, target):
    """Convert parent set posterior to marginal probabilities."""
    marginals = {var: 0.0 for var in variables if var != target}
    
    for parent_set, prob in posterior_dict.items():
        for parent in parent_set:
            if parent in marginals:
                marginals[parent] += prob
    
    return marginals
```
**Note**: This sums probabilities over all parent sets containing each variable. Need to verify this is the correct interpretation.

## Implementation Steps

1. **Remove synthetic expert generation** from `clean_bc_trainer.py`
2. **Add demonstration loading** using `behavioral_cloning_adapter.py`
3. **Implement trajectory extraction** for both BC and surrogate training
4. **Ensure proper buffer state reconstruction** at each decision point
5. **Validate channel mapping** matches current tensor format
6. **Add quality filtering** based on demonstration accuracy

## Key Benefits
- Learns from real expert behavior (PARENT_SCALE)
- No SCM structure leakage into training
- Consistent with existing infrastructure
- Enables joint training of acquisition and surrogate

## Open Questions
1. Is the marginal probability extraction mathematically correct?
2. How to handle variable ordering consistency across demonstrations?
3. Should we filter demonstrations by SCM type or difficulty?

## Next Steps
After repository cleanup, implement this plan starting with demonstration loading and validation.