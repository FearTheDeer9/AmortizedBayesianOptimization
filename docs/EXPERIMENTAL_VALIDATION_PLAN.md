# ACBO Experimental Validation Plan

**Created**: 2025-06-26  
**Status**: ACTIVE  
**Purpose**: Validate neural network approach on standard benchmarks before full infrastructure development

## Executive Summary

This document outlines a comprehensive 3-week experimental validation plan to prove that the ACBO neural network approach provides significant advantages over traditional GP-based methods on standard causal discovery benchmarks. The validation serves as a critical go/no-go gate before proceeding with large-scale infrastructure development.

### Key Insight
Phase 2.2 revealed that structure learning F1 scores were **SIMULATED**, not from real trained models. We need to prove the core hypothesis works on real benchmarks before investing more resources.

## Background & Motivation

### The Problem
1. **Unproven Core Hypothesis**: No validation that NN approach actually works on real data
2. **Simulated Performance**: Previous "validation" used probability distributions, not trained models
3. **Infrastructure Risk**: Building production systems before proving core concept

### The Solution
Progressive experimental validation on standard benchmarks to demonstrate:
- Neural networks can learn causal structure from data
- Training provides clear benefits over untrained models  
- The approach scales to larger graphs
- Transfer learning works across graph types

## Experimental Design

### Phase 1: Library Integration & Baseline (Week 1)

#### 1.1 Standard Graph Generation
**Libraries to Use**:
```python
import networkx as nx  # Already available via PARENT_SCALE
import numpy as onp     # For I/O only (following CLAUDE.md)
import jax.numpy as jnp # For computation
```

**Graph Types**:
1. **Erdos-Renyi Random Graphs**: `nx.erdos_renyi_graph(n, p, directed=True)`
2. **Scale-Free Networks**: `nx.scale_free_graph(n)`  
3. **Small-World Networks**: `nx.watts_strogatz_graph(n, k, p)`

**Implementation**: `src/causal_bayes_opt/experiments/benchmark_graphs.py`

#### 1.2 Standard Causal Discovery Datasets
**Selected Benchmarks**:

1. **Sachs Protein Network** (11 nodes)
   - Source: Sachs et al. (2005) Science paper
   - Real protein signaling data with known interactions
   - Ground truth: Flow cytometry experiments
   - Why: Most cited causal discovery benchmark

2. **BnLearn Repository Networks**:
   - **Asia** (8 nodes): Simple medical diagnosis network
   - **Alarm** (37 nodes): Medical monitoring system
   - **Child** (20 nodes): Child development assessment
   - Source: http://www.bnlearn.com/bnrepository/

3. **DREAM Challenge Networks**:
   - Gene regulatory networks from DREAM4/5 challenges  
   - Sizes: 10, 50, 100 nodes
   - Source: Synapse platform or published datasets

**Data Format**:
```python
@dataclass
class BenchmarkDataset:
    name: str
    graph: pyr.PMap  # True causal structure as SCM
    data: jnp.ndarray  # Observational samples [N, d]
    interventional_data: Optional[Dict] = None
    ground_truth_parents: Dict[str, List[str]]
    target_variable: str
    metadata: Dict[str, Any]
```

#### 1.3 Baseline Experiments
**Current Method Validation**:
- Run `complete_workflow_demo.py` approach on each benchmark
- Measure: convergence time, F1 score, sample efficiency
- Establish performance floor for comparisons

### Phase 2: Model Training & Progressive Experiments (Week 2)

#### 2.1 Expert Demonstration Collection
**Small-Scale Collection**:
```python
# For each benchmark dataset
expert_trajectories = run_full_parent_scale_algorithm(
    scm=benchmark.graph,
    n_observational=int(0.85 * 1.2 * (n_nodes ** 2.5)),
    n_interventional=int(0.15 * 1.2 * (n_nodes ** 2.5)),
    n_trials=max(10, n_nodes),
    target_variable=benchmark.target_variable
)
```

**Quality Validation**:
- Structure discovery accuracy > 80%
- Target optimization improvement > 50%
- Consistent intervention strategies

#### 2.2 Progressive Training Experiments
**Experimental Sequence**:

1. **Untrained Baseline**:
   - Random policy + untrained surrogate
   - Establishes lower bound performance

2. **Active Learning Only**:
   - Random policy + self-supervised surrogate training
   - Tests current `complete_workflow_demo.py` approach

3. **Expert Policy Only**:
   - Trained policy (GRPO) + untrained surrogate
   - Isolates policy learning benefits

4. **Expert Surrogate Only**:
   - Random policy + trained surrogate (behavioral cloning)
   - Isolates surrogate learning benefits

5. **Fully Trained**:
   - Trained policy + trained surrogate
   - Best-case performance

6. **Joint Training**:
   - End-to-end training of both components
   - Ultimate system performance

#### 2.3 Training Protocols
**Surrogate Training**:
```python
# Behavioral cloning on expert demonstrations
surrogate_config = SurrogateConfig(
    model_hidden_dim=256,
    model_n_layers=8,
    learning_rate=1e-3,
    batch_size=32,
    max_epochs=100
)
```

**Policy Training**:
```python
# GRPO with verifiable rewards
acquisition_config = AcquisitionConfig(
    policy_hidden_dim=128,
    policy_n_layers=4,
    grpo_group_size=64,
    grpo_learning_rate=3e-4,
    max_grpo_epochs=50
)
```

### Phase 3: Scaling & Transfer Learning (Week 3)

#### 3.1 Scaling Analysis
**Graph Size Progression**:
- **Small**: 8-15 nodes (proof of concept)
- **Medium**: 20-50 nodes (practical scale)
- **Large**: 50-100 nodes (scalability test)

**Metrics vs Graph Size**:
- Structure discovery F1 score
- Sample efficiency (samples to convergence)
- Computational time per intervention
- Memory usage

#### 3.2 Transfer Learning Validation
**Cross-Dataset Transfer**:
- Train on Sachs, test on Asia
- Train on Asia, test on Alarm  
- Train on synthetic Erdos-Renyi, test on real networks

**Cross-Size Transfer**:
- Train on 10-node graphs, test on 20-node
- Train on 20-node graphs, test on 50-node

## Evaluation Metrics

### Primary Metrics
1. **Structure Discovery**:
   ```python
   f1_score = 2 * precision * recall / (precision + recall)
   structural_hamming_distance = sum(abs(true_graph - pred_graph))
   ```

2. **Optimization Performance**:
   ```python
   sample_efficiency = improvement / n_samples
   convergence_time = steps_to_threshold
   regret = optimal_value - achieved_value
   ```

3. **Training Benefit**:
   ```python
   training_improvement = (trained_f1 - untrained_f1) / untrained_f1
   ```

### Secondary Metrics
- Intervention diversity
- Computational efficiency
- Memory usage
- Training stability

## Success Criteria

### Go/No-Go Decision Thresholds

**Structure Discovery** (Must Achieve):
- F1 > 0.7 on Sachs dataset with real trained models
- F1 > 0.6 on synthetic Erdos-Renyi graphs (20+ nodes)
- Clear improvement over random baseline (>2x)

**Sample Efficiency** (Must Achieve):
- 2x+ improvement over random intervention policy
- Faster convergence with trained vs untrained models
- Consistent performance across multiple runs

**Scalability** (Should Achieve):
- Successful performance on 50+ node graphs  
- Sub-linear degradation with graph size
- Transfer learning to new graph types

### Validation Outcomes

**If Validation Succeeds**:
- Proceed with full Phase 4 implementation
- Large-scale expert demonstration collection (1000+ trajectories)
- Production infrastructure development
- GPU/distributed training systems

**If Validation Fails**:
- Focus on smaller graphs where method works
- Investigate hybrid approaches (NN + traditional methods)
- Consider architectural changes or alternative training
- Potentially pivot research direction

## Implementation Plan

### Week 1: Foundation
**Days 1-2**: Library integration
- Set up NetworkX integration for graph generation
- Create dataset loaders for standard benchmarks
- Validate data conversion to SCM format

**Days 3-5**: Baseline experiments
- Run current approach on all benchmarks
- Collect performance metrics
- Identify initial bottlenecks

**Days 6-7**: Expert demonstration collection
- Use PARENT_SCALE API on benchmarks
- Validate trajectory quality
- Create training/validation splits

### Week 2: Training & Comparison
**Days 8-10**: Model training
- Train surrogate models on expert demonstrations
- Train policy models with GRPO
- Implement joint training pipeline

**Days 11-12**: Progressive experiments
- Run all 6 experimental conditions
- Collect comprehensive metrics
- Initial analysis of results

**Days 13-14**: Optimization & debugging
- Address any training issues
- Optimize hyperparameters
- Replicate key results

### Week 3: Analysis & Decision
**Days 15-17**: Scaling experiments
- Test on larger graphs
- Measure computational requirements
- Evaluate transfer learning

**Days 18-19**: Comprehensive analysis
- Statistical significance testing
- Performance comparison tables
- Visualization of key results

**Days 20-21**: Documentation & decision
- Write up experimental results
- Make go/no-go recommendation
- Plan next steps based on outcome

## Risk Mitigation

### Technical Risks
- **Training Instability**: Use proven hyperparameters from Phase 2.2
- **Memory Limitations**: Start with smaller graphs, optimize incrementally
- **Poor Baselines**: Implement multiple baseline methods for comparison

### Timeline Risks  
- **Library Integration Issues**: Have backup manual implementations
- **Dataset Access Problems**: Use multiple sources, create synthetic alternatives
- **Training Time**: Parallelize experiments, use checkpointing

### Scientific Risks
- **Negative Results**: Design experiments to be informative regardless of outcome
- **Cherry-Picking**: Use multiple independent datasets and random seeds
- **Overfitting**: Strict train/validation splits, multiple cross-validation folds

## Expected Deliverables

### Experimental Results
1. **Performance Comparison Table**: All methods × all datasets
2. **Scaling Analysis Plots**: Performance vs graph size  
3. **Learning Curves**: Training progress over time
4. **Transfer Learning Matrix**: Cross-dataset performance

### Documentation
1. **Methodology Report**: Detailed experimental procedures
2. **Results Analysis**: Statistical analysis and interpretation
3. **Go/No-Go Recommendation**: Clear decision with reasoning
4. **Next Steps Plan**: Roadmap based on validation outcome

### Code Artifacts
1. **Benchmark Integration**: Reusable dataset loaders
2. **Experiment Runner**: Automated experiment framework
3. **Analysis Tools**: Metrics computation and visualization
4. **Trained Models**: Checkpoints for successful models

## Long-Term Impact

### If Successful
- **Scientific Validation**: Proof that NN approach works on real problems
- **Production Readiness**: Foundation for scalable ACBO systems
- **Research Direction**: Clear path for further development
- **Baseline Establishment**: Performance targets for future work

### If Unsuccessful  
- **Risk Reduction**: Avoid wasting resources on unproven approach
- **Learning Opportunity**: Understand limitations and failure modes
- **Alternative Exploration**: Informed decision on research pivots
- **Honest Assessment**: Transparent evaluation of method limitations

---

## Conclusion

This experimental validation plan provides a rigorous, time-bounded evaluation of the ACBO neural network approach. The 3-week timeline balances thoroughness with urgency, while the progressive experimental design ensures we understand exactly where the benefits come from. Success will provide strong evidence for proceeding with full infrastructure development; failure will save significant resources and inform alternative approaches.

The key insight is that we must **prove before we build** - validating the core hypothesis on standard benchmarks before investing in large-scale production systems.