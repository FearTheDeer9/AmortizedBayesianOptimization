# ADR 004: Custom Parent Set Model Instead of AVICI Adaptation

## Status
Accepted

## Context

The original **AVICI Adaptation Implementation Plan** called for directly adapting AVICI's existing architecture to handle target-conditioned parent set prediction. This would involve:

1. Modifying AVICI's input format from `[N, d, 2]` to `[N, d, 3]` (adding target conditioning)
2. Creating `TargetAwareBaseModel` and `TargetAwareInferenceModel` classes
3. Adapting AVICI's training infrastructure for target-specific predictions
4. Leveraging AVICI's pre-trained components where possible

During development, we discovered several challenges with this approach and implemented an alternative solution.

## Decision

**We decided to build a custom `ParentSetPredictionModel` instead of directly adapting AVICI's architecture.**

The custom model:
- Predicts parent sets directly rather than full adjacency matrices
- Uses a transformer-based architecture similar to AVICI but purpose-built for parent set prediction
- Handles target conditioning natively from the ground up
- Provides a clean API through the `ParentSetPosterior` data structure

## Rationale

### Advantages of Custom Approach

**1. Direct Parent Set Optimization**
- AVICI predicts full `[d, d]` adjacency matrices, then we extract parent sets for one target
- Our model directly predicts top-k parent sets for the specified target
- More efficient: O(k) parent sets vs O(d²) adjacency entries
- Better optimization signal: loss directly on parent sets rather than full graph

**2. Target-Specific Design**
- Built from ground up for target-conditioned prediction
- No need to modify complex existing AVICI architecture
- Cleaner separation between target conditioning and graph structure prediction
- Can easily add target-specific inductive biases

**3. Simplified Architecture**
- No need to understand and modify AVICI's complex multi-loss training
- Full control over model architecture, training, and inference
- Easier to debug and extend
- Reduced dependency on external codebase changes

**4. Better API Design**
- Clean `ParentSetPosterior` output format with uncertainty quantification
- Structured utilities (marginal probabilities, summaries, comparisons)
- Composable functions following functional programming principles
- Clear separation between model prediction and posterior analysis

**5. Performance Benefits**
- Smaller model: Only needs to predict k parent sets, not full adjacency matrix
- Faster inference: Direct parent set enumeration and scoring
- More focused training: Loss computed directly on parent sets
- Better memory usage: O(k) vs O(d²) predictions

### Disadvantages of Custom Approach

**1. No Pre-trained Components**
- Cannot leverage AVICI's pre-trained models
- Need to train from scratch on each new domain
- Longer development time for initial working system

**2. Less Established**
- AVICI is well-validated on causal discovery benchmarks
- Our approach needs validation against standard baselines
- Fewer guarantees about performance on complex graphs

**3. Deviation from Plan**
- Original plan assumed AVICI adaptation would be straightforward
- Requires updating documentation and expectations
- Different skill set needed (custom architecture vs adaptation)

## Alternatives Considered

### Alternative 1: Direct AVICI Adaptation (Original Plan)
**Pros**: Leverage existing validation, pre-trained models, established architecture
**Cons**: Complex modification of existing codebase, inefficient for single-target prediction, harder to debug

**Decision**: Rejected due to implementation complexity and efficiency concerns

### Alternative 2: Hybrid Approach
Use AVICI for graph structure learning, custom model for parent set ranking
**Pros**: Best of both worlds, can leverage AVICI's structure learning
**Cons**: Added complexity, two models to maintain, unclear interface between components

**Decision**: Rejected in favor of unified custom approach for simplicity

### Alternative 3: Ensemble of Multiple Approaches
**Pros**: Robust predictions, can compare different methods
**Cons**: Much higher complexity, harder to interpret, computational overhead

**Decision**: Deferred to future work - focus on single working approach first

## Implementation Details

### Core Components

**1. ParentSetPredictionModel**
```python
class ParentSetPredictionModel(hk.Module):
    - Transformer-based architecture (similar to AVICI backbone)
    - Direct parent set enumeration and scoring
    - MLP-based compatibility scoring vs simple dot products
    - Adaptive k selection based on graph size
```

**2. ParentSetPosterior Data Structure**
```python
@dataclass(frozen=True)
class ParentSetPosterior:
    target_variable: str
    parent_set_probs: ParentSetProbs  # Immutable mapping
    uncertainty: float               # Entropy-based uncertainty
    top_k_sets: List[Tuple[ParentSet, float]]
    metadata: pyr.PMap[str, Any]
```

**3. Clean Functional API**
```python
# Prediction
posterior = predict_parent_posterior(net, params, data, vars, target)

# Analysis
marginals = get_marginal_parent_probabilities(posterior, all_variables)
summary = summarize_posterior(posterior)
comparison = compare_posteriors(predicted, ground_truth)
```

### Integration with Existing System

The custom approach integrates cleanly with the existing ACBO framework:

- **Data Bridge**: Uses the same `[N, d, 3]` format we developed for AVICI adaptation
- **SCM Interface**: Works with our existing SCM and sampling infrastructure  
- **Training Pipeline**: Compatible with existing training loop patterns
- **Evaluation**: Provides structured output for downstream optimization decisions

## Consequences

### Positive

**Immediate Benefits**:
- Working parent set prediction system (Phase 2 complete)
- Clean, maintainable codebase with clear responsibilities
- Efficient inference suitable for real-time optimization
- Strong foundation for Phase 3 (Acquisition Model integration)

**Long-term Benefits**:
- Full control over model architecture and training
- Easier to add domain-specific inductive biases
- Clear path to extensions (uncertainty calibration, multi-target prediction)
- Better integration with downstream RL components

### Negative

**Short-term Costs**:
- Need to validate performance against AVICI baselines
- Training from scratch required for each domain
- Documentation needs updating to reflect new approach

**Long-term Risks**:
- May not scale as well as AVICI to very large graphs
- Less community validation than established methods
- Need to build evaluation infrastructure from scratch

## Validation Plan

To mitigate risks, we will validate the custom approach through:

1. **Accuracy Benchmarks**: Compare against AVICI on standard causal discovery tasks
2. **Scaling Analysis**: Test performance on graphs of increasing size (3-20 variables)
3. **Uncertainty Calibration**: Ensure probability estimates are well-calibrated
4. **Integration Testing**: Verify compatibility with downstream ACBO components

## Migration Path

The implementation maintains backward compatibility:

- Old API (`predict_parent_sets`) still works for existing code
- New API (`predict_parent_posterior`) provides enhanced functionality
- Gradual migration possible as system components are updated
- Documentation covers both approaches during transition period

## Related Decisions

- **ADR 001**: Intervention Representation (supports our functional approach)
- **ADR 002**: GRPO Implementation (next phase, will use ParentSetPosterior outputs)
- **ADR 003**: Verifiable Rewards (will incorporate parent set prediction uncertainty)

## Future Considerations

This decision positions us well for:

1. **Phase 3**: GRPO acquisition model can use structured ParentSetPosterior outputs
2. **Uncertainty-Aware RL**: Posterior uncertainty can guide exploration strategies
3. **Multi-Target Extension**: Architecture can be extended for multiple targets
4. **Transfer Learning**: Model components can be pre-trained on synthetic data

The custom approach provides a solid foundation for the complete ACBO system while maintaining the flexibility to incorporate insights from AVICI and other causal discovery methods in the future.
