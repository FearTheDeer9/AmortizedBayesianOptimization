# Real Performance Validation Experiments

## Overview

Phase 2.2 validation used **simulated/mocked components** instead of trained models. This document outlines experiments needed to validate **real performance** once we have fully trained systems.

## Current Status: What's Simulated vs. Real

### ❌ **Completely Simulated (Must Fix)**

#### Structure Learning Performance
- **Current**: F1 scores from probability distributions (coin flips)
- **Claimed**: 79.8% F1 score on causal structure discovery
- **Reality**: 0% - no trained AVICI models used

```python
# Current "validation" - pure simulation!
difficulty_factors = {"linear_chain": 0.95, "fork": 0.90}
if random.uniform(key) < detection_rate:
    predicted_edges.add(edge)  # Coin flip, not model prediction!
```

### ⚠️ **Partially Simulated (Needs Real Training)**

#### Sample Efficiency 
- **Current**: Mathematical learning curves based on config assumptions
- **Claimed**: 25x improvement vs PARENT_SCALE
- **Reality**: ~20% real (configs exist), 80% simulated (no actual learning)

#### Optimization Performance
- **Current**: Grid search with simplified SCM evaluation  
- **Claimed**: 7x improvement vs PARENT_SCALE
- **Reality**: ~60% real (SCM mechanics), 40% simulated (no trained policies)

#### GRPO Training
- **Current**: Hardcoded loss values, mock gradient updates
- **Claimed**: Functional GRPO acquisition training
- **Reality**: ~40% real (infrastructure), 60% simulated (no learning)

### ✅ **Actually Validated (Can Trust)**

#### Training Speed
- **Current**: Component timing + JAX compilation benchmarks
- **Claimed**: <24h curriculum training (83.7x JAX speedup measured)
- **Reality**: 80% real - credible performance estimates

## Required Experiments for Real Validation

### Experiment 1: Real Structure Learning Validation

**Objective**: Train AVICI models and measure genuine F1 scores

**Prerequisites**: 
- Trained parent set prediction models on diverse synthetic SCMs
- Inference pipeline for real-time structure discovery
- Benchmark dataset of test SCMs with ground truth

**Protocol**:
```python
# 1. Train AVICI models
models = {}
for problem_size in ["small", "medium", "large"]:
    training_data = generate_synthetic_scms(size=problem_size, n_samples=10000)
    models[problem_size] = train_avici_model(training_data, epochs=100)

# 2. Real inference on test SCMs  
test_scms = create_benchmark_scms()
results = {}
for scm_name, scm in test_scms.items():
    predicted_edges = models[scm.size].predict_parent_sets(scm.observational_data)
    f1_score = compute_f1(scm.true_edges, predicted_edges)
    results[scm_name] = f1_score

# 3. Compare against baselines
baseline_methods = ["PC", "GES", "NOTEARS", "AVICI-original"]
performance_comparison = benchmark_against_baselines(results, baseline_methods)
```

**Success Criteria**: 
- Achieve >90% F1 score on majority of test problems
- Outperform baseline structure learning methods
- Demonstrate scalability to 20+ variable SCMs

**Timeline**: 2-3 weeks (1 week training, 1 week validation, 1 week analysis)

### Experiment 2: Real Sample Efficiency Validation

**Objective**: Train GRPO policies and measure genuine sample efficiency

**Prerequisites**:
- End-to-end GRPO training pipeline  
- Diverse intervention environments
- PARENT_SCALE baseline implementations

**Protocol**:
```python
# 1. Train GRPO policies across configurations
policies = {}
learning_curves = {}

for config_name in ["debug", "standard", "production"]:
    config = get_grpo_config(config_name)
    environment = create_intervention_environment(scm_suite)
    
    policy, curve = train_grpo_policy(
        config=config,
        environment=environment, 
        max_episodes=5000,
        convergence_threshold=0.9
    )
    
    policies[config_name] = policy
    learning_curves[config_name] = curve

# 2. Compare sample efficiency
for problem_type in ["linear", "nonlinear", "confounded"]:
    parent_scale_baseline = get_parent_scale_samples(problem_type)
    our_samples = get_samples_to_convergence(policies, problem_type)
    efficiency_ratio = parent_scale_baseline / our_samples
    
    print(f"{problem_type}: {efficiency_ratio:.1f}x improvement")
```

**Success Criteria**:
- Achieve ≥10x sample efficiency vs PARENT_SCALE on majority of problems
- Demonstrate consistent improvement across problem types
- Learning curves show genuine convergence patterns

**Timeline**: 3-4 weeks (2 weeks training, 1 week baseline comparison, 1 week analysis)

### Experiment 3: Real Optimization Performance Validation  

**Objective**: Deploy trained acquisition functions and measure real optimization

**Prerequisites**:
- Fully trained GRPO acquisition policies
- Diverse target optimization problems
- PARENT_SCALE baseline results

**Protocol**:
```python
# 1. Deploy trained acquisition functions
trained_policies = load_trained_grpo_policies()
test_environments = create_optimization_benchmarks()

results = {}
for env_name, env in test_environments.items():
    # Our method
    acquisition_fn = trained_policies[env.complexity_level]
    our_interventions = acquisition_fn.optimize(env.scm, env.target, budget=50)
    our_improvement = env.evaluate_interventions(our_interventions)
    
    # PARENT_SCALE baseline  
    parent_scale_interventions = run_parent_scale_baseline(env)
    baseline_improvement = env.evaluate_interventions(parent_scale_interventions)
    
    results[env_name] = {
        "our_improvement": our_improvement,
        "baseline_improvement": baseline_improvement,
        "ratio": our_improvement / baseline_improvement
    }
```

**Success Criteria**:
- Match or exceed PARENT_SCALE performance (≥1.0x ratio)
- Achieve target 1.2x improvement on majority of problems
- Demonstrate robustness across problem types

**Timeline**: 2-3 weeks (1 week policy deployment, 1 week benchmarking, 1 week analysis)

### Experiment 4: End-to-End System Validation

**Objective**: Validate complete ACBO pipeline with real models

**Prerequisites**: 
- All individual components validated from Experiments 1-3
- Integration testing infrastructure
- Realistic deployment scenarios

**Protocol**:
```python
# Complete ACBO pipeline test
def validate_end_to_end_acbo():
    # 1. Structure discovery
    scm_data = collect_observational_data(real_system)
    discovered_structure = structure_learning_model.predict(scm_data)
    
    # 2. Acquisition training  
    acquisition_policy = train_grpo_acquisition(
        discovered_structure, 
        reward_rubric,
        training_episodes=1000
    )
    
    # 3. Intervention optimization
    optimal_interventions = acquisition_policy.optimize(
        target_variable="outcome",
        intervention_budget=20
    )
    
    # 4. Real-world validation
    actual_outcomes = deploy_interventions(real_system, optimal_interventions)
    improvement = measure_target_improvement(actual_outcomes)
    
    return improvement

# Test across multiple domains
domains = ["healthcare", "marketing", "manufacturing"] 
for domain in domains:
    improvement = validate_end_to_end_acbo(domain)
    print(f"{domain}: {improvement:.2f} target improvement")
```

**Success Criteria**:
- End-to-end pipeline completes without errors
- Achieves meaningful target improvements in real domains
- Performance matches or exceeds validated individual components

**Timeline**: 4-6 weeks (2 weeks integration, 2-4 weeks real-world validation)

## Validation Infrastructure Needed

### Data Generation
- Large-scale synthetic SCM generator (10,000+ diverse structures)
- Realistic intervention environments with known ground truth
- Benchmark datasets for consistent evaluation

### Training Infrastructure  
- GPU cluster for AVICI model training
- Distributed GRPO training across multiple environments
- Hyperparameter optimization frameworks

### Evaluation Frameworks
- Standardized metrics and comparison protocols
- Baseline method implementations (PC, GES, NOTEARS, PARENT_SCALE)
- Statistical significance testing

### Real-World Integration
- APIs for deploying trained models
- A/B testing infrastructure
- Performance monitoring systems

## Timeline Summary

| Experiment | Duration | Dependencies | Critical Path |
|------------|----------|--------------|---------------|
| Structure Learning | 2-3 weeks | AVICI training pipeline | Yes |
| Sample Efficiency | 3-4 weeks | GRPO training + baselines | Yes |  
| Optimization Performance | 2-3 weeks | Trained policies | No |
| End-to-End Validation | 4-6 weeks | All above experiments | Yes |

**Total Timeline**: 8-12 weeks for complete real performance validation

## Success Metrics

### Minimum Viable Performance
- Structure Learning: >80% F1 score average
- Sample Efficiency: >5x improvement vs PARENT_SCALE  
- Optimization: ≥1.0x improvement vs PARENT_SCALE
- End-to-End: Successful deployment in ≥1 real domain

### Target Performance (Phase 2.2 Claims)
- Structure Learning: >90% F1 score average
- Sample Efficiency: ≥10x improvement vs PARENT_SCALE
- Optimization: ≥1.2x improvement vs PARENT_SCALE  
- End-to-End: Successful deployment in ≥3 real domains

## Risk Mitigation

### If Performance Falls Short
1. **Structure Learning <90% F1**: 
   - Acceptable for deployment if >80%
   - Plan additional AVICI architecture improvements

2. **Sample Efficiency <10x**:
   - Validate that >5x is achieved (still significant)
   - Investigate curriculum learning optimizations

3. **Optimization <1.2x**:
   - Ensure ≥1.0x is maintained (matching baselines)
   - Consider ensemble methods

### Contingency Plans
- Fallback to proven baselines for underperforming components
- Hybrid approaches combining our innovations with existing methods
- Incremental deployment starting with best-performing domains

## Notes for Future Implementation

When implementing these experiments:

1. **Start with Experiment 1** - structure learning is foundational
2. **Use incremental validation** - validate on small problems first
3. **Maintain comparison datasets** - ensure reproducible baselines
4. **Document all assumptions** - avoid the simulation mistakes of Phase 2.2
5. **Plan for iteration** - expect 2-3 rounds of improvement needed

Remember: The goal is **honest performance assessment** with **real trained models**, not optimistic projections or simulated results.