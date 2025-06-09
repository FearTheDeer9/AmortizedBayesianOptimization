# ACBO Phase 4: Training Infrastructure Implementation Plan (Revised)

## Executive Summary

**Project Status**: ACBO (Amortized Causal Bayesian Optimization) implementation with Phases 1-3 complete and production-ready. Phase 4 implements training infrastructure to achieve amortized scaling across diverse causal graphs using a **scientifically rigorous, incremental validation approach**.

**Core Strategy**: Start with **minimal viable implementation** (1K expert demonstrations, single GPU, end-to-end training) to validate amortization benefits before scaling complexity. This approach prioritizes scientific rigor and practical validation over ambitious targets.

**Key Innovation**: Custom amortized inference pipeline that adapts AVICI principles for parent-set prediction and intervention handling, creating a purpose-built system optimized for dual-objective causal Bayesian optimization.

**Critical Success Criteria**: Demonstrate **2x speedup** over per-graph PARENT_SCALE with **<20% accuracy drop** on held-out graphs before any scaling investment.

---

## Part I: Implementation Strategy

### 1. Incremental Validation Approach

**Philosophy**: Validate core amortization assumptions at small scale before committing to large-scale infrastructure.

#### Phase 2A: Minimal Viable Dataset (Week 1-2)
- **Target**: 1,000 expert demonstrations from 100 PARENT_SCALE runs
- **Graph Sizes**: 3-8 nodes (start simple)
- **Mechanism Types**: Linear mechanisms only
- **Storage**: ~100MB, single GPU memory
- **Success Criterion**: 2x speedup over per-graph PARENT_SCALE

#### Phase 2B: Validation Dataset (Week 2)
- **Target**: 200 held-out demonstrations for generalization testing
- **Graph Types**: Same size range, different random structures
- **Purpose**: Validate amortization vs. per-graph performance
- **Success Criterion**: <20% performance drop on held-out graphs

#### Phase 2C: Conditional Scaling (Week 4+)
- **Scaling Gate**: Only proceed if Phase 2A/2B show clear amortization benefit
- **Next Scale**: 5K demonstrations if 1K proves beneficial
- **Maximum Scale**: 50K demonstrations only if justified by performance gains

### 2. Custom Amortized Inference Pipeline

**Design Philosophy**: Adapt AVICI principles for parent-set prediction and intervention handling, creating a custom pipeline optimized for dual-objective learning.

```python
# Minimal viable data format
@dataclass(frozen=True)
class TrainingExample:
    observational_data: jnp.ndarray      # [n_obs, n_vars]
    interventional_data: jnp.ndarray     # [n_int, n_vars, 2]  # values + indicators
    parent_posterior: jnp.ndarray        # [n_vars] parent probabilities
    target_variable: str
    graph_metadata: Dict[str, Any]

# Minimal viable amortized model
class ParentSetPredictor(hk.Module):
    """Predicts parent set posteriors from observational + interventional data."""
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def __call__(self, obs_data, int_data, target_idx):
        # Simple architecture for parent-set prediction
        # Returns: [n_vars] parent probabilities for target
        pass
```

### 3. Go/No-Go Decision Framework

#### Week 2 Decision Point: Proceed with Scaling

**Go Criteria** (All must be met):
1. **Amortization Benefit**: Demonstrated 2x speedup on held-out graphs
2. **Performance Maintenance**: <20% accuracy drop vs. expert performance
3. **Technical Feasibility**: Training pipeline working without major issues
4. **Resource Efficiency**: Memory usage within single-GPU constraints

**No-Go Indicators**:
- Speedup <1.5x consistently
- Accuracy drop >30% on held-out graphs
- Technical blockers requiring major architecture changes
- Memory usage exceeding available hardware

#### Week 4 Decision Point: Production Investment

**Go Criteria** (Majority must be met):
1. **Transfer Learning**: Graceful degradation to larger graphs (<30% drop)
2. **Scaling Benefit**: Improved performance with 5K vs. 1K demonstrations
3. **Integration Success**: Clean integration with Phase 3 acquisition model
4. **Cost Justification**: Clear ROI path for production deployment

---

## Part II: Minimal Viable Implementation

### 4. Data Generation Pipeline

#### Minimal Dataset Generation Strategy

```python
class MinimalDatasetGenerator:
    """Generate small-scale dataset for initial validation."""
    
    def __init__(self):
        self.graph_sizes = [3, 4, 5, 6, 7, 8]  # Start simple
        self.graphs_per_size = 16              # 100 total graphs
        self.demonstrations_per_graph = 10     # 1K total demonstrations
    
    def generate_training_dataset(self) -> List[TrainingExample]:
        """Generate 1K demonstrations for initial validation."""
        demonstrations = []
        
        for size in self.graph_sizes:
            for _ in range(self.graphs_per_size):
                # Generate random linear SCM structure
                scm = create_random_linear_scm(num_nodes=size)
                
                # Run PARENT_SCALE to generate expert trajectory
                parent_scale = PARENT_SCALE(scm)
                trajectory = parent_scale.run_algorithm(T=self.demonstrations_per_graph)
                
                # Extract training examples
                examples = extract_expert_demonstrations([trajectory])
                demonstrations.extend(examples)
        
        return demonstrations
```

#### PARENT_SCALE Integration Pipeline

```python
def extract_expert_demonstrations(
    parent_scale_runs: List[Dict],
    target_count: int = 1000
) -> List[TrainingExample]:
    """
    Extract expert demonstrations from PARENT_SCALE algorithm runs.
    
    PARENT_SCALE outputs:
    - global_opt: List[float] - optimization progress
    - current_y: List[float] - target outcomes
    - intervention_set: List[Tuple[str]] - intervention variables
    - intervention_values: List[Tuple[float]] - intervention values
    """
    demonstrations = []
    
    for run_data in parent_scale_runs:
        trajectory = []
        for step in range(len(run_data['intervention_set'])):
            example = TrainingExample(
                observational_data=run_data['D_O_standardized'],
                interventional_data=format_interventional_data(
                    run_data['intervention_set'][:step+1],
                    run_data['intervention_values'][:step+1]
                ),
                parent_posterior=run_data['posterior_probabilities'][step],
                target_variable=run_data['target'],
                graph_metadata=extract_graph_metadata(run_data)
            )
            trajectory.append(example)
        demonstrations.extend(trajectory)
    
    return demonstrations[:target_count]
```

### 5. Training Infrastructure

#### JAX-Optimized Training Pipeline (Minimal)

```python
@dataclass
class MinimalTrainingConfig:
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 4
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    
    # Hardware utilization (start simple)
    use_mixed_precision: bool = False  # Start simple
    gradient_accumulation_steps: int = 1  # No accumulation initially
    
    # Validation
    validation_frequency: int = 10
    early_stopping_patience: int = 20

@jax.jit
def training_step(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: TrainingExample,
    key: jax.Array
) -> Tuple[hk.Params, optax.OptState, Dict[str, float]]:
    """Single JAX-optimized training step."""
    
    def loss_fn(params):
        # Forward pass through amortized model
        predictions = model.apply(params, batch.observational_data, 
                                 batch.interventional_data, batch.target_variable)
        
        # Loss: KL divergence between predicted and expert posteriors
        kl_loss = compute_kl_divergence(predictions, batch.parent_posterior)
        
        # Additional losses for calibration (optional initially)
        calibration_loss = compute_calibration_loss(predictions, batch.parent_posterior)
        
        total_loss = kl_loss + 0.1 * calibration_loss
        return total_loss, {'kl_loss': kl_loss, 'calibration_loss': calibration_loss}
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics
```

#### Memory-Efficient Data Loading (Minimal)

```python
class SimpleDataLoader:
    """Simple data loader for minimal dataset."""
    
    def __init__(self, examples: List[TrainingExample], batch_size: int = 32):
        self.examples = examples
        self.batch_size = batch_size
    
    def get_batches(self, key: jax.Array) -> Iterator[TrainingExample]:
        """Generate batches with JAX-compatible format."""
        indices = jax.random.permutation(key, len(self.examples))
        
        for i in range(0, len(self.examples), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_examples = [self.examples[idx] for idx in batch_indices]
            yield self._stack_examples(batch_examples)
    
    def _stack_examples(self, examples: List[TrainingExample]) -> TrainingExample:
        """Stack multiple examples into batched arrays."""
        # Simple implementation for minimal viable approach
        pass
```

### 6. Validation Framework

#### Primary Success Metrics

**Amortization Benefit (Critical)**:
- **Target**: 2x speedup vs. per-graph PARENT_SCALE
- **Measurement**: Average inference time per graph
- **Baseline**: PARENT_SCALE runtime on same hardware
- **Success Threshold**: Consistently faster on 90%+ of test graphs

**Performance Maintenance**:
- **Target**: <20% accuracy drop on held-out graphs
- **Metrics**: Parent set prediction accuracy (F1-score, AUC-PR)
- **Baseline**: Expert PARENT_SCALE performance
- **Success Threshold**: Maintain >80% of expert performance

#### Validation Checkpoints

```python
# Week 1 Checkpoint: Data Pipeline
def validate_data_pipeline():
    """Validate PARENT_SCALE → training format conversion."""
    original_trajectory = run_parent_scale_example()
    converted_examples = extract_expert_demonstrations([original_trajectory])
    
    # Verify: No information loss in conversion
    assert validate_trajectory_consistency(original_trajectory, converted_examples)
    
    # Performance: Conversion speed
    conversion_time = measure_conversion_performance()
    assert conversion_time < 1.0  # seconds per trajectory

# Week 2 Checkpoint: Amortization Proof (CRITICAL)
def validate_amortization_benefit():
    """Demonstrate speedup on minimal dataset."""
    # Train on 1K examples
    model, params = train_minimal_model()
    
    # Test on 100 held-out graphs
    test_graphs = generate_test_graphs(n=100)
    
    # Compare amortized vs. per-graph performance
    amortized_time = measure_amortized_inference(model, params, test_graphs)
    per_graph_time = measure_parent_scale_performance(test_graphs)
    
    speedup = per_graph_time / amortized_time
    assert speedup > 2.0, f"Only achieved {speedup:.2f}x speedup"
```

---

## Part III: Integration with Phases 1-3

### 7. Phase Integration Specifications

#### ExperienceBuffer Integration

```python
def integrate_amortized_with_buffer(
    buffer: ExperienceBuffer,
    amortized_model: ParentSetPredictor,
    model_params: hk.Params,
    target_variable: str
) -> ParentSetPosterior:
    """
    Use trained amortized model to predict parent posteriors from buffer data.
    
    Args:
        buffer: Existing experience buffer with obs/interventional data
        amortized_model: Trained parent set predictor
        model_params: Trained model parameters
        target_variable: Target variable for prediction
        
    Returns:
        ParentSetPosterior compatible with Phase 3 acquisition model
    """
    # Extract data from buffer in amortized model format
    obs_data = buffer.get_observations_as_array()
    int_data = buffer.get_interventions_as_array()
    
    # Get predictions from amortized model
    predictions = amortized_model.apply(
        model_params, obs_data, int_data, target_variable
    )
    
    # Convert to ParentSetPosterior format
    return convert_amortized_to_posterior(predictions, buffer.get_variable_names(), target_variable)
```

#### ParentSetPosterior Integration

```python
def convert_amortized_to_posterior(
    amortized_predictions: jnp.ndarray,  # [n_vars] parent probabilities
    variable_names: List[str],
    target_variable: str
) -> ParentSetPosterior:
    """
    Convert amortized model output to Phase 3 ParentSetPosterior format.
    
    Args:
        amortized_predictions: Parent probabilities from amortized model
        variable_names: Ordered list of variable names
        target_variable: Name of target variable
        
    Returns:
        ParentSetPosterior object compatible with Phase 3
    """
    # Convert marginal probabilities to parent set posteriors
    parent_sets = generate_parent_sets_from_marginals(
        amortized_predictions, variable_names, target_variable
    )
    
    return ParentSetPosterior(
        target_variable=target_variable,
        posterior_probs=parent_sets,
        uncertainty=compute_entropy(amortized_predictions),
        metadata=pyr.m({'source': 'amortized_model'})
    )
```

#### GRPO Acquisition Integration

```python
def create_amortized_surrogate_update_fn(
    amortized_model: ParentSetPredictor,
    model_params: hk.Params
) -> Callable:
    """
    Create surrogate update function for GRPO that uses amortized model.
    
    Returns function compatible with existing Phase 3 GRPO implementation.
    """
    def surrogate_update_fn(
        state: AcquisitionState,
        intervention: pyr.PMap,
        outcome: pyr.PMap
    ) -> AcquisitionState:
        """Update acquisition state with amortized predictions."""
        
        # Update buffer with new intervention-outcome pair
        new_buffer = state.buffer.copy()
        new_buffer.add_intervention(intervention, outcome)
        
        # Get new posterior from amortized model
        new_posterior = integrate_amortized_with_buffer(
            new_buffer, amortized_model, model_params, state.current_target
        )
        
        # Create updated acquisition state
        return update_state_with_intervention(
            state, intervention, outcome, new_posterior
        )
    
    return surrogate_update_fn
```

### 8. Module Structure and Organization

#### Code Organization

```
src/causal_bayes_opt/
├── core/                          # Phase 1: ✅ EXISTING
│   ├── scm.py
│   ├── sample.py
│   ├── intervention.py
│   └── buffer.py
├── surrogate/                     # Phase 2: ✅ EXISTING
│   ├── avici_adapter.py
│   ├── posterior.py
│   └── training.py
├── acquisition/                   # Phase 3: ✅ EXISTING
│   ├── state.py
│   ├── policy.py
│   ├── rewards.py
│   └── grpo.py
└── training/                      # Phase 4: 🆕 NEW
    ├── __init__.py
    ├── data_generation.py         # SCM generation and PARENT_SCALE extraction
    ├── amortized_models.py         # ParentSetPredictor and training
    ├── integration.py              # Phase 1-3 integration interfaces
    ├── validation.py               # Training validation and checkpoints
    └── deployment.py               # Runtime usage patterns
```

#### Import Structure

```python
# Training infrastructure imports
from causal_bayes_opt.training.data_generation import (
    MinimalDatasetGenerator,
    extract_expert_demonstrations
)
from causal_bayes_opt.training.amortized_models import (
    ParentSetPredictor,
    training_step,
    MinimalTrainingConfig
)
from causal_bayes_opt.training.integration import (
    integrate_amortized_with_buffer,
    convert_amortized_to_posterior,
    create_amortized_surrogate_update_fn
)

# Integration with existing phases
from causal_bayes_opt.core.buffer import ExperienceBuffer
from causal_bayes_opt.surrogate.posterior import ParentSetPosterior
from causal_bayes_opt.acquisition.state import AcquisitionState
```

### 9. Deployment Patterns

#### Runtime Usage Pattern

```python
# Training phase (one-time)
def train_amortized_system():
    """Complete training pipeline for amortized system."""
    
    # Generate minimal dataset
    generator = MinimalDatasetGenerator()
    demonstrations = generator.generate_training_dataset()
    
    # Train amortized model
    model, params = train_amortized_model(demonstrations)
    
    # Validate amortization benefit
    validate_amortization_benefit(model, params)
    
    return model, params

# Deployment phase (runtime)
def deploy_trained_system(model, params):
    """Deploy trained amortized system for runtime use."""
    
    # Create integration functions
    surrogate_update_fn = create_amortized_surrogate_update_fn(model, params)
    
    # Replace Phase 2 surrogate model in existing ACBO pipeline
    acbo_pipeline = ACBOPipeline(
        surrogate_update_fn=surrogate_update_fn,  # Use amortized model
        acquisition_model=existing_grpo_model,    # Keep existing Phase 3
        experience_buffer=ExperienceBuffer()      # Keep existing Phase 1
    )
    
    return acbo_pipeline
```

---

## Part IV: Incremental Implementation Timeline

### Week 1: Foundation and Data Pipeline

**Day 1-2: PARENT_SCALE Integration**
- Implement expert demonstration extraction
- Validate data format conversions
- Test on small examples (10 graphs, 100 demonstrations)
- **Deliverable**: Working data pipeline with validation

**Day 3-4: Custom Model Architecture**
- Implement ParentSetPredictor model
- Basic training loop with dummy data
- Verify JAX compilation and memory usage
- **Deliverable**: Trainable model architecture

**Day 5-7: Minimal Dataset Generation**
- Generate 1K demonstrations from 100 simple graphs
- Implement efficient data loading
- **Checkpoint**: Validate data pipeline performance
- **Success Criteria**: <1 second per trajectory conversion

### Week 2: Minimal Viable Training

**Day 8-10: End-to-End Training**
- Train amortized model on 1K demonstrations
- Implement validation loop and metrics
- Basic hyperparameter optimization
- **Deliverable**: Trained model with performance metrics

**Day 11-12: Amortization Validation (CRITICAL CHECKPOINT)**
- Compare amortized vs. per-graph performance
- Measure speedup and accuracy trade-offs
- **Checkpoint**: Prove 2x speedup on held-out graphs
- **Success Criteria**: 2x speedup + <20% accuracy drop

**Day 13-14: Initial Analysis**
- Performance analysis and bottleneck identification
- Documentation of results and lessons learned
- **Decision Point**: Scale or pivot based on results

### Week 3: Optimization and Scaling (Conditional)

**Day 15-17: Performance Optimization** (Only if Week 2 successful)
- JAX compilation optimization
- Memory usage optimization
- Batch processing improvements
- **Deliverable**: Optimized training pipeline

**Day 18-19: Transfer Learning Validation**
- Test on larger graph sizes (9-12 nodes)
- Evaluate generalization across structures
- **Checkpoint**: Validate transfer learning bounds
- **Success Criteria**: <30% performance drop on larger graphs

**Day 20-21: Scaling Decision**
- Generate 5K demonstrations if Week 2 results justify
- Alternative: Focus on integration if scaling unnecessary
- **Decision Point**: Proceed to production or focus on integration

### Week 4: Advanced Features and Integration (Conditional)

**Day 22-24: Enhanced Training** (Only if scaling validated)
- Mixed precision training
- Advanced optimization techniques
- Curriculum learning implementation
- **Deliverable**: Production-ready training infrastructure

**Day 25-26: Integration Testing**
- Integration with Phase 3 acquisition model
- End-to-end ACBO pipeline testing
- **Checkpoint**: Complete system validation
- **Success Criteria**: Full ACBO pipeline working with amortized surrogate

**Day 27-28: Documentation and Handoff**
- Comprehensive documentation
- Integration guides for deployment
- **Deliverable**: Production-ready amortized ACBO system

---

## Part V: Resource Requirements & Risk Management

### 10. Incremental Hardware Requirements

#### Development Phase (Weeks 1-2)
- **Hardware**: Single GPU (RTX 4090 24GB or A100 40GB)
- **Cost**: $1,500-8,000
- **Justification**: Sufficient for 1K demonstration training
- **Usage**: Model development and initial validation

#### Scaling Phase (Weeks 3-4) - Conditional
- **Hardware**: 2-4 GPUs for larger datasets
- **Cost**: $3,000-15,000
- **Justification**: Only if Week 2 checkpoint shows clear benefit
- **Usage**: 5K+ demonstration training

#### Production Phase (Weeks 5-8) - Highly Conditional
- **Hardware**: 4-8x H100 80GB for large-scale training
- **Cost**: $80,000-200,000
- **Justification**: Only if demonstrated ROI vs. per-graph methods
- **Usage**: 50K+ demonstration training and deployment

### 11. Software Dependencies

```python
# Core dependencies (required for minimal implementation)
jax = ">=0.4.20"                    # Numerical computation
jax-lib = ">=0.4.20"               # JAX backend
haiku = ">=0.0.10"                 # Neural networks
optax = ">=0.1.7"                  # Optimizers
pyrsistent = ">=0.19.0"            # Immutable data structures

# Development tools
pytest = ">=7.0.0"                 # Testing
black = ">=23.0.0"                 # Code formatting
mypy = ">=1.0.0"                   # Type checking

# Optional (only for scaling)
zarr = ">=2.12.0"                  # Efficient storage (if scaling)
h5py = ">=3.7.0"                   # HDF5 storage (if scaling)
```

### 12. Cost-Benefit Analysis

#### Break-Even Analysis
- **Development Cost**: $10,000-25,000 (hardware + time)
- **Per-Graph Cost**: PARENT_SCALE runtime × number of graphs
- **Break-Even Point**: ~1,000 graphs for 2x speedup
- **ROI Timeline**: Positive after processing 2,000+ graphs

#### Value Proposition
- **One-Time Training**: Amortized model trains once, applies everywhere
- **Deployment Efficiency**: Real-time inference vs. minutes per graph
- **Scalability**: Linear cost scaling vs. quadratic for per-graph methods

### 13. Risk Mitigation and Contingencies

#### Technical Risks

**Risk**: Amortization shows no benefit over per-graph methods
- **Probability**: Medium
- **Mitigation**: Week 2 checkpoint designed to catch this early
- **Contingency**: Pivot to improving PARENT_SCALE directly

**Risk**: Transfer learning fails across graph types
- **Probability**: Medium
- **Mitigation**: Test diverse graph structures in minimal dataset
- **Contingency**: Focus on domain-specific models rather than general amortization

**Risk**: Integration complexity with existing codebase
- **Probability**: Low
- **Mitigation**: Clear interface specifications and incremental integration
- **Contingency**: Maintain separate training and runtime systems

#### Resource Risks

**Risk**: Hardware unavailable or cost-prohibitive
- **Probability**: Low
- **Mitigation**: Start with modest hardware requirements
- **Contingency**: Cloud compute alternatives (Google Colab, AWS)

**Risk**: Timeline exceeded due to technical complexity
- **Probability**: Medium
- **Mitigation**: Conservative timeline with explicit checkpoints
- **Contingency**: Reduce scope to essential features only

#### Scientific Risks

**Risk**: Expert demonstrations insufficient for learning
- **Probability**: Medium
- **Mitigation**: Quality validation of PARENT_SCALE trajectories
- **Contingency**: Augment with synthetic demonstrations or different expert sources

**Risk**: Custom AVICI adaptation performs poorly
- **Probability**: Low
- **Mitigation**: Extensive validation against PARENT_SCALE baselines
- **Contingency**: Simplify model architecture or return to per-graph methods

---

## Part VI: Conditional Scaling (Advanced Features)

### 14. Advanced Data Infrastructure (If 5K→50K Scaling Validated)

#### Large-Scale Data Generation
**Only implement if Week 2 and Week 4 checkpoints successful**

```python
# Large-scale curriculum learning (conditional)
@dataclass(frozen=True)
class AdvancedCurriculumConfig:
    """Configuration for large-scale curriculum learning."""
    node_progression: List[Tuple[int, int]] = [
        (3, 8),    # Stage 1: Validated minimal
        (9, 15),   # Stage 2: Medium graphs
        (16, 25),  # Stage 3: Large graphs
        (26, 50)   # Stage 4: Very large graphs
    ]
    mechanism_progression: List[List[str]] = [
        ['linear'],                              # Stage 1: Validated
        ['linear', 'polynomial'],                # Stage 2: Add complexity
        ['linear', 'polynomial', 'sigmoid'],     # Stage 3: Non-linear
        ['linear', 'polynomial', 'sigmoid', 'neural']  # Stage 4: Full complexity
    ]
    
    # Large-scale targets (conditional)
    target_demonstrations: int = 50000  # Only if validated at smaller scales
    demonstrations_per_stage: int = 12500
```

#### Sophisticated Storage and Loading
**Only implement if dataset size justifies complexity**

```python
# Advanced storage (conditional on large datasets)
class AdvancedDataLoader:
    """Zarr-based loader for large-scale datasets."""
    
    def __init__(self, data_path: str, batch_size: int = 64):
        self.data_path = data_path
        self.batch_size = batch_size
        self._load_metadata()
    
    def get_batches_efficient(self, key: jax.Array) -> Iterator[Dict[str, jnp.ndarray]]:
        """Zarr-based efficient loading for large datasets."""
        with zarr.open(self.data_path, mode='r') as store:
            # Efficient batch loading implementation
            pass
```

### 15. Multi-GPU Training Infrastructure (If Resource Requirements Validated)

#### Advanced Training Optimization
**Only implement if single-GPU approach validated and scaling needed**

```python
# Multi-device training (conditional)
@dataclass(frozen=True)
class AdvancedTrainingConfig:
    """Advanced training configuration for multi-GPU scaling."""
    # Multi-device settings (conditional)
    use_pmap: bool = True
    devices_per_host: int = 8
    gradient_accumulation_steps: int = 4
    
    # Advanced optimization (conditional)
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    dynamic_batching: bool = True

def create_advanced_training_step(config: AdvancedTrainingConfig):
    """Create multi-GPU training step (only if validated)."""
    # Implementation only if scaling requirements validated
    pass
```

### 16. Production Deployment (If ROI Validated)

#### Enterprise Integration
**Only implement if cost-benefit analysis supports production deployment**

- Real-time inference APIs
- Model versioning and updates
- Monitoring and alerting
- Integration with existing causal discovery pipelines

---

## Conclusion

This Phase 4 implementation plan provides a **scientifically rigorous, incrementally validated** approach to amortized causal discovery training infrastructure. By starting with minimal viable experiments and scaling only upon demonstrated benefit, we maintain research integrity while building toward a potentially transformative capability.

The plan balances **technical ambition with practical constraints**, ensuring each phase builds upon validated successes rather than unsupported assumptions. The comprehensive validation framework and clear success criteria provide objective measures for progress and decision-making.

**Key Success Metrics**:
- **Week 2**: 2x speedup over PARENT_SCALE with <20% accuracy drop
- **Week 4**: Successful integration with existing ACBO pipeline
- **Long-term**: Scalable, efficient framework for causal Bayesian optimization

Success in Phase 4 will establish ACBO as a **practical, scalable framework** for causal Bayesian optimization, opening new possibilities for automated causal discovery and optimization across diverse scientific and engineering domains.