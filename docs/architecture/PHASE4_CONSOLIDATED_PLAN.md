# ACBO Phase 4: Consolidated Training Infrastructure Plan

## Executive Summary

**Goal**: Create a training infrastructure that uses PARENT_SCALE expert demonstrations to improve both the surrogate and acquisition models, enabling faster and more effective causal Bayesian optimization.

**Key Insight**: Expert demonstrations provide training data for BOTH components:
1. **Surrogate Model**: Learn posterior prediction from (data → posterior) pairs
2. **Acquisition Model**: Warm start GRPO with expert (state → action) examples

**Success Criteria**:
- 2x+ speedup over per-graph PARENT_SCALE inference
- <20% accuracy drop on held-out graphs
- Clean integration with existing Phases 1-3
- Effective warm-start for GRPO on new domains

---

## Problem Statement & Motivation

### Current Limitation
PARENT_SCALE provides excellent causal Bayesian optimization but doesn't scale:
- Must train new surrogate model for each SCM (minutes per graph)
- Must run GRPO from scratch for each problem (slow convergence)
- Cannot transfer knowledge between similar causal structures

### Solution Vision
Use PARENT_SCALE expertise to create an amortized system:
- **Amortized Surrogate**: Pre-trained model that predicts posteriors instantly
- **Warm-Start Acquisition**: GRPO initialized with expert knowledge for faster convergence

### What Expert Demonstrations Provide

```python
# Each PARENT_SCALE run gives us:
ParentScaleTrajectory = {
    # For surrogate training (behavioral cloning)
    'observational_data': jnp.ndarray,           # Initial observations
    'intervention_history': List[Intervention],   # Sequence of interventions
    'posterior_evolution': List[ParentSetPosterior], # Expert's posterior updates
    
    # For acquisition warm-start (imitation learning + GRPO)
    'states': List[AcquisitionState],            # Decision contexts
    'actions': List[Intervention],               # Expert's choices
    'rewards': List[float],                      # Outcomes achieved
    'next_states': List[AcquisitionState],       # Resulting states
    
    # Metadata
    'scm': SCM,                                  # The causal model
    'target_variable': str,                      # Optimization target
    'final_performance': float                   # Final value achieved
}
```

---

## Architecture Overview

### Component Responsibilities

#### Phase 1 (Data Structures) - Existing ✅
- `ExperienceBuffer`: Store trajectories efficiently
- `Sample`: Represent observations/interventions
- `SCM`: Causal model representation

#### Phase 2 (Surrogate Model) - Existing + Enhanced
- **Existing**: `ParentSetPredictionModel` with transformer architecture
- **Enhanced**: Training pipeline using expert demonstrations
- **Output**: Pre-trained model for instant posterior inference

#### Phase 3 (Acquisition Model) - Existing + Enhanced  
- **Existing**: `AcquisitionPolicyNetwork` with GRPO training
- **Enhanced**: Warm-start initialization from expert demonstrations
- **Output**: Policy network with expert knowledge for faster GRPO convergence

#### Phase 4 (Training Infrastructure) - New
- Expert demonstration extraction from PARENT_SCALE
- Training pipelines for both models
- Integration and validation systems

### Data Flow Architecture

```
PARENT_SCALE Expert Runs
        ↓
Expert Demonstration Extraction
        ↓
    ┌─────────────────────────────┐
    ↓                             ↓
Surrogate Training           Acquisition Training
(Behavioral Cloning)         (Imitation + GRPO)
    ↓                             ↓
Pre-trained Surrogate    Warm-started Acquisition
    ↓                             ↓
        ┌─────────────────────────────┐
        ↓
Integrated ACBO System
(Fast inference + Fast convergence)
```

---

## Detailed Component Specifications

### 1. Expert Demonstration Extraction

#### Input: PARENT_SCALE Algorithm Output
```python
# What PARENT_SCALE provides (Jean's algorithm output)
parent_scale_output = {
    'D_O': jnp.ndarray,                    # Observational data
    'intervention_set': List[Tuple[str]],   # Variables intervened on
    'intervention_values': List[Tuple[float]], # Values used
    'current_y': List[float],              # Target outcomes
    'posterior_probabilities': List[Dict], # Posterior evolution
    'global_opt': List[float],             # Optimization progress
    'target': str,                         # Target variable
    # ... other PARENT_SCALE outputs
}
```

#### Output: Structured Trajectories
```python
@dataclass(frozen=True)
class ExpertTrajectory:
    """Structured expert demonstration for training."""
    
    # Problem setup
    scm: pyr.PMap
    target_variable: str
    variable_order: List[str]
    
    # Initial conditions
    observational_samples: List[Sample]
    
    # Trajectory data
    steps: List[TrajectoryStep]
    
    # Metadata
    final_value: float
    n_interventions: int
    source: str = "PARENT_SCALE"

@dataclass(frozen=True)
class TrajectoryStep:
    """Single step in expert trajectory."""
    
    # State at decision time
    buffer_state: ExperienceBuffer       # Data available to expert
    posterior: ParentSetPosterior        # Expert's posterior belief
    acquisition_state: AcquisitionState  # Rich decision context
    
    # Expert's decision
    chosen_intervention: pyr.PMap        # What expert selected
    
    # Outcome
    outcome_sample: Sample               # Result of intervention
    reward: float                        # Improvement achieved
    
    # Step metadata
    step_number: int
    timestamp: Optional[float] = None
```

#### Extraction Functions
```python
def extract_expert_trajectory(
    parent_scale_output: Dict[str, Any],
    scm: pyr.PMap
) -> ExpertTrajectory:
    """Convert PARENT_SCALE output to structured trajectory."""
    pass

def validate_trajectory_consistency(
    trajectory: ExpertTrajectory
) -> bool:
    """Ensure trajectory is self-consistent and complete."""
    pass

def extract_multiple_trajectories(
    parent_scale_runs: List[Dict[str, Any]],
    scms: List[pyr.PMap],
    max_trajectories: Optional[int] = None
) -> List[ExpertTrajectory]:
    """Extract trajectories from multiple PARENT_SCALE runs."""
    pass
```

### 2. Surrogate Model Training (Behavioral Cloning)

#### Training Data Format
```python
SurrogateTrainingExample = Tuple[
    ExperienceBuffer,      # Input: data state
    ParentSetPosterior     # Target: expert's posterior
]
```

#### Training Interface
```python
def extract_surrogate_training_data(
    trajectories: List[ExpertTrajectory]
) -> List[SurrogateTrainingExample]:
    """Extract (buffer → posterior) pairs from trajectories."""
    pass

def train_surrogate_with_expert_data(
    training_examples: List[SurrogateTrainingExample],
    model_config: SurrogateConfig,
    training_config: TrainingConfig
) -> SurrogateModelResult:
    """Train existing ParentSetPredictionModel on expert data."""
    pass

@dataclass(frozen=True)
class SurrogateModelResult:
    """Results from surrogate model training."""
    model: hk.Transformed
    params: hk.Params
    training_history: Dict[str, List[float]]
    validation_metrics: Dict[str, float]
    speedup_estimate: float
```

### 3. Acquisition Model Training (Imitation + GRPO)

#### Training Data Format  
```python
AcquisitionTrainingExample = {
    'state': AcquisitionState,           # Decision context
    'expert_action': pyr.PMap,           # What expert chose
    'reward': float,                     # Reward achieved
    'next_state': AcquisitionState,      # Resulting state
    'trajectory_id': str,                # Source trajectory
    'step_id': int                       # Step within trajectory
}
```

#### Training Interface
```python
def extract_acquisition_training_data(
    trajectories: List[ExpertTrajectory],
    surrogate_model: hk.Transformed,
    surrogate_params: hk.Params
) -> List[AcquisitionTrainingExample]:
    """Extract GRPO training data from expert trajectories."""
    pass

def train_acquisition_with_expert_warmstart(
    training_examples: List[AcquisitionTrainingExample],
    policy_config: PolicyConfig,
    grpo_config: GRPOConfig,
    imitation_config: ImitationConfig
) -> AcquisitionModelResult:
    """Train policy with expert warm-start."""
    pass

@dataclass(frozen=True)
class ImitationConfig:
    """Configuration for imitation learning component."""
    imitation_weight: float = 0.1       # Weight for expert matching
    imitation_epochs: int = 20           # Pre-training epochs
    use_behavioral_cloning: bool = True  # Pre-train with BC
    grpo_warmstart: bool = True          # Initialize GRPO with BC policy
```

### 4. Integrated System

#### System Architecture
```python
@dataclass
class TrainedACBOSystem:
    """Complete trained ACBO system."""
    
    # Trained components
    surrogate_model: hk.Transformed
    surrogate_params: hk.Params
    acquisition_model: hk.Transformed
    acquisition_params: hk.Params
    
    # Metadata
    variable_order: List[str]
    training_scms: List[pyr.PMap]
    performance_metrics: Dict[str, float]
    
    def predict_posterior(
        self, 
        buffer: ExperienceBuffer, 
        target: str
    ) -> ParentSetPosterior:
        """Fast posterior prediction using trained surrogate."""
        pass
    
    def select_intervention(
        self, 
        state: AcquisitionState,
        exploration_rate: float = 0.1
    ) -> pyr.PMap:
        """Intervention selection using warm-started policy."""
        pass
    
    def run_optimization(
        self,
        scm: pyr.PMap,
        n_steps: int = 30,
        use_online_grpo: bool = True
    ) -> OptimizationResult:
        """Complete optimization with optional online fine-tuning."""
        pass
```

---

## Interface Specifications

### 1. Data Generation Interface

```python
# Core extraction function
def extract_expert_demonstrations(
    parent_scale_runs: List[Dict[str, Any]],
    scms: List[pyr.PMap],
    validation_split: float = 0.2
) -> Tuple[List[ExpertTrajectory], List[ExpertTrajectory]]:
    """Extract and split trajectories into train/validation sets."""
    pass

# Data quality validation
def validate_demonstration_quality(
    trajectories: List[ExpertTrajectory],
    min_trajectory_length: int = 5,
    max_posterior_entropy: float = 2.0
) -> List[ExpertTrajectory]:
    """Filter trajectories based on quality criteria."""
    pass

# Data augmentation
def augment_expert_demonstrations(
    trajectories: List[ExpertTrajectory],
    augmentation_factor: int = 2
) -> List[ExpertTrajectory]:
    """Create additional training data through augmentation."""
    pass
```

### 2. Training Pipeline Interface

```python
# Main training entry point
def train_complete_acbo_system(
    expert_trajectories: List[ExpertTrajectory],
    surrogate_config: SurrogateConfig,
    acquisition_config: AcquisitionConfig,
    training_config: TrainingConfig
) -> TrainedACBOSystem:
    """Train both components and return integrated system."""
    pass

# Individual component training
def train_surrogate_component(
    trajectories: List[ExpertTrajectory],
    config: SurrogateConfig
) -> SurrogateModelResult:
    """Train only the surrogate model."""
    pass

def train_acquisition_component(
    trajectories: List[ExpertTrajectory],
    surrogate_result: SurrogateModelResult,
    config: AcquisitionConfig
) -> AcquisitionModelResult:
    """Train only the acquisition model."""
    pass
```

### 3. Evaluation Interface

```python
# Performance evaluation
def evaluate_system_performance(
    system: TrainedACBOSystem,
    test_scms: List[pyr.PMap],
    baseline_method: str = "per_graph_training"
) -> EvaluationReport:
    """Comprehensive evaluation against baselines."""
    pass

# Component-specific evaluation
def evaluate_surrogate_accuracy(
    surrogate: SurrogateModelResult,
    test_trajectories: List[ExpertTrajectory]
) -> Dict[str, float]:
    """Evaluate surrogate model accuracy."""
    pass

def evaluate_acquisition_performance(
    acquisition: AcquisitionModelResult,
    test_scenarios: List[OptimizationScenario]
) -> Dict[str, float]:
    """Evaluate acquisition model performance."""
    pass
```

### 4. Integration Interface

```python
# Integration with existing phases
def integrate_with_existing_pipeline(
    trained_system: TrainedACBOSystem
) -> ACBOPipeline:
    """Create pipeline compatible with existing Phase 1-3 interfaces."""
    pass

# Deployment interface  
def deploy_trained_system(
    system: TrainedACBOSystem,
    deployment_config: DeploymentConfig
) -> DeployedACBOSystem:
    """Prepare system for production deployment."""
    pass
```

---

## Configuration Specifications

### Training Configurations

```python
@dataclass(frozen=True)
class SurrogateConfig:
    """Configuration for surrogate model training."""
    model_hidden_dim: int = 256
    model_n_layers: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_frequency: int = 5

@dataclass(frozen=True)  
class AcquisitionConfig:
    """Configuration for acquisition model training."""
    policy_hidden_dim: int = 128
    policy_n_layers: int = 4
    grpo_group_size: int = 64
    grpo_learning_rate: float = 3e-4
    imitation_config: ImitationConfig
    max_grpo_epochs: int = 50
    online_finetuning: bool = True

@dataclass(frozen=True)
class TrainingConfig:
    """Overall training configuration."""
    n_expert_trajectories: int = 1000
    validation_split: float = 0.2
    random_seed: int = 42
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    use_mixed_precision: bool = False
    log_level: str = "INFO"
```

---

## Deliverables & Success Criteria

### Immediate Deliverables (Week 1-2)

1. **Expert Demonstration Extraction** ✅
   - Extract 100 trajectories from PARENT_SCALE runs
   - Validate data quality and consistency
   - Create train/validation splits

2. **Surrogate Model Training** ✅
   - Train ParentSetPredictionModel on expert data
   - Achieve 2x+ speedup over per-graph training
   - Maintain <20% accuracy drop on validation set

3. **Basic Integration** ✅
   - Integrate trained surrogate with existing pipeline
   - Validate end-to-end functionality
   - Document integration interfaces

### Extended Deliverables (Week 3-4)

4. **Acquisition Model Warm-Start** ✅
   - Implement imitation learning pre-training
   - Warm-start GRPO with expert demonstrations
   - Validate faster convergence vs cold start

5. **Complete System Training** ✅
   - Train both components on same expert data
   - Integrate into unified ACBO system
   - Validate system-level performance

6. **Evaluation & Validation** ✅
   - Comprehensive evaluation against baselines
   - Transfer learning validation (larger graphs)
   - Performance analysis and optimization

### Success Metrics

#### Primary Metrics (Must Achieve)
- **Surrogate Speedup**: ≥2x faster than per-graph training
- **Surrogate Accuracy**: ≤20% accuracy drop vs expert
- **Integration Success**: Clean integration with Phases 1-3
- **System Stability**: Robust performance across different SCMs

#### Secondary Metrics (Stretch Goals)
- **Acquisition Convergence**: 50%+ faster GRPO convergence with warm-start
- **Transfer Learning**: <30% performance drop on 2x larger graphs
- **Memory Efficiency**: Single GPU training and inference
- **Expert Matching**: ≥70% agreement with expert intervention choices

---

## Implementation Plan

### Phase 4.1: Data Pipeline (Week 1)
- Implement expert demonstration extraction
- Create data validation and quality checks
- Build train/validation splitting utilities
- Test with 100 PARENT_SCALE trajectories

### Phase 4.2: Surrogate Training (Week 1-2)  
- Adapt existing ParentSetPredictionModel training
- Implement behavioral cloning on expert posteriors
- Validate speedup and accuracy metrics
- Integrate with existing inference pipeline

### Phase 4.3: Acquisition Warm-Start (Week 2-3)
- Implement imitation learning pre-training
- Adapt GRPO training for warm-start initialization
- Validate convergence speed improvements
- Test on held-out optimization problems

### Phase 4.4: System Integration (Week 3-4)
- Build unified training pipeline
- Create complete system evaluation framework
- Implement deployment utilities
- Comprehensive testing and validation

### Phase 4.5: Evaluation & Optimization (Week 4)
- Run comprehensive evaluation suite
- Optimize performance bottlenecks
- Document results and lessons learned
- Prepare for production deployment

---

## Risk Mitigation

### Technical Risks
- **Expert Data Quality**: Implement rigorous validation and filtering
- **Domain Transfer**: Test on diverse graph structures and sizes
- **Integration Complexity**: Define clear interfaces and extensive testing
- **Performance Gaps**: Set realistic expectations and fallback strategies

### Implementation Risks  
- **Scope Creep**: Stick to defined deliverables and success criteria
- **Integration Issues**: Test integration points early and frequently
- **Time Overrun**: Prioritize core functionality over nice-to-have features

### Scientific Risks
- **Limited Improvement**: Have backup plans if benefits are marginal
- **Overfitting to Experts**: Validate on diverse test scenarios
- **Scaling Challenges**: Test scalability assumptions early

---

## Conclusion

This consolidated plan provides a clear roadmap for Phase 4 implementation:

1. **Clear Problem Statement**: Amortize PARENT_SCALE expertise for both components
2. **Well-Defined Interfaces**: Precise specifications for all major functions
3. **Realistic Deliverables**: Achievable goals with measurable success criteria
4. **Risk Awareness**: Identified potential issues with mitigation strategies

The plan balances ambition with practicality, building on proven components while adding clear value through expert knowledge transfer and amortization.