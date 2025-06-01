# ADR 004: Target-Aware AVICI Implementation Strategy

## Context

We need to implement a target-aware version of AVICI that can process [N, d, 3] input tensors (adding target conditioning to the standard [N, d, 2] format) for Phase 1.3 of the ACBO framework. This decision affects both immediate validation goals and long-term research objectives around transfer learning, scalability, and performance comparison.

Two primary approaches were considered:
1. **Adapter Pattern**: Wrap pretrained AVICI model to handle target conditioning
2. **Direct Modification**: Create target-aware AVICI model from scratch

The research goals include validating target conditioning effectiveness, transfer learning analysis, scaling to 100+ node graphs, and performance comparison against Parent_Scale_ACD baselines.

## Decision

**Phase 1.3 (Validation)**: Implement direct modification approach with target-aware BaseModel trained from scratch.

**Post-Validation (Phases 2+)**: Implement hybrid architecture that supports both target-aware training and pretrained weight loading for transfer learning research.

## Key Aspects

### Phase 1.3: Clean Validation Implementation
- Create `TargetAwareBaseModel` that accepts [N, d, 3] input 
- Preserve AVICI's proven transformer architecture completely
- Only modify input handling (first linear layer: 2 channels → 3 channels)
- Train from scratch on simple synthetic SCMs (3-5 variables)
- Focus on pure target-aware learning without transfer learning complications

### Post-Validation: Hybrid Architecture
- Design forward-compatible architecture that can load pretrained AVICI weights
- Enable comparison between scratch training vs. transfer learning
- Support scaling experiments with pretrained initialization
- Maintain ability to degrade to standard AVICI for baseline comparison

### Implementation Strategy
```python
# Phase 1.3: Direct target-aware implementation
class TargetAwareBaseModel(hk.Module):
    def __init__(self, target_conditioning=True, pretrained_compatible=False):
        # Clean target-aware implementation for validation
        
# Phase 2+: Hybrid for research
class HybridAVICIModel(hk.Module):
    def load_pretrained_weights(self, avici_params):
        # Enable transfer learning research
```

## Alternatives Considered

### Adapter Pattern for Phase 1.3
**Pros:**
- Could potentially leverage pretrained AVICI weights immediately
- Less code to write initially

**Cons:**
- Pretrained model expects [N, d, 2] input - unclear how to properly encode target information
- Would confound validation results (poor performance could be due to adapter vs. core approach)
- Creates debugging complexity with black-box interactions
- Doesn't provide clean test of target conditioning hypothesis

### Pure Direct Modification (No Hybrid)
**Pros:**
- Simpler architecture with single implementation path
- No complexity from supporting multiple modes

**Cons:**
- Cannot leverage pretrained weights for scaling research
- Limits competitive performance on large graphs (100+ nodes)
- Reduces research contribution potential for transfer learning analysis
- Makes comparison with strong baselines more difficult

### Pure Adapter Pattern
**Pros:**
- Maximum leverage of existing pretrained models
- Potentially faster time to competitive performance

**Cons:**
- Unclear how to properly implement target conditioning with [N, d, 2] constraint
- Less control over target-aware learning throughout the network
- Difficulty in making strong research claims about target conditioning benefits

## Consequences

### Positive
- **Clean validation signal**: Phase 1.3 provides unambiguous test of target conditioning effectiveness
- **Research flexibility**: Hybrid architecture enables comprehensive transfer learning analysis
- **Stronger baselines**: Can leverage pretrained weights for competitive scaling experiments
- **Clear contribution**: Can make strong claims about both target conditioning and transfer learning
- **Forward compatibility**: Validation implementation sets up post-validation research phases

### Negative
- **Additional complexity**: Need to implement both direct and hybrid approaches
- **Longer development time**: More architectural planning required upfront
- **Training time**: Phase 1.3 requires training from scratch (but on small SCMs)
- **Code maintenance**: Must maintain compatibility between different model modes

## Implementation Notes

### Phase 1.3 Success Criteria
- Target-aware model accepts [N, d, 3] input without errors
- Model outputs sensible edge probabilities  
- Training converges on synthetic SCMs
- Target parents receive higher probability than non-parents
- Clear demonstration that target conditioning improves causal discovery

### Validation Experiment Design
- Train on X → Y ← Z structure with different targets
- Verify target='Y' learns different parent probabilities than target='X'
- Compare against standard AVICI baseline on same data
- Measure both accuracy and calibration of posterior predictions

### Post-Validation Research Questions
- Transfer learning effectiveness: pretrained vs. scratch training
- Optimal fine-tuning strategies for target-aware tasks  
- Scaling behavior with pretrained initialization on 100+ node graphs
- Performance comparison against Parent_Scale_ACD with strongest possible baseline

### Technical Implementation
- Use JAX/Haiku for consistency with AVICI codebase
- Implement comprehensive validation and testing
- Design for easy switching between model modes
- Include proper logging and experiment tracking
- Maintain functional programming principles throughout

## Research Contribution Strategy

This phased approach enables multiple research contributions:

1. **Target Conditioning Validation**: Clean demonstration that target awareness improves causal discovery
2. **Transfer Learning Analysis**: Systematic comparison of training strategies for causal models
3. **Scalability Research**: Scaling to large graphs with strong baselines
4. **Performance Benchmarking**: Competitive comparison against state-of-the-art methods

The hybrid architecture positions us to make the strongest possible research claims while maintaining scientific rigor in the validation phase.

## References

- Related to ADR 002 (GRPO Implementation) - target-aware model will integrate with RL framework
- Related to ADR 003 (Verifiable Rewards) - posterior predictions will guide reward computation
- Supports research goals outlined in "Amortized Causal Bayesian Optimization for Large-Scale Graphs" proposal
