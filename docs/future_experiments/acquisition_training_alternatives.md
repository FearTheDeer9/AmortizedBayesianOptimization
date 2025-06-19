# Acquisition Training Alternatives

## Overview

This document outlines alternative approaches to acquisition model training that could be explored once the core BC + GRPO pipeline is operational. These alternatives represent different philosophical approaches to the training problem and could potentially offer improvements in performance, efficiency, or robustness.

## Alternative Loss Functions

### 1. Wasserstein Loss for Behavioral Cloning

**Motivation**: Traditional cross-entropy loss for BC only considers action probabilities, not the underlying policy distribution structure. Wasserstein loss could provide better distribution matching.

**Implementation Approach**:
```python
def wasserstein_bc_loss(
    policy_params: Any,
    expert_actions: List[Any],
    states: List[AcquisitionState],
    policy_network: Any
) -> float:
    """Wasserstein distance between expert and policy action distributions."""
    
    # Sample actions from both policy and expert
    policy_actions = [sample_intervention_from_policy(...) for state in states]
    
    # Compute optimal transport distance
    # (requires representing actions in continuous space)
    transport_cost = compute_optimal_transport(expert_actions, policy_actions)
    
    return transport_cost
```

**Expected Benefits**:
- Better handling of action space geometry
- More stable training with fewer mode collapse issues
- Better generalization to unseen states

**Experimental Protocol**:
- Replace BC cross-entropy loss with Wasserstein loss
- Compare convergence speed and final BC accuracy
- Evaluate downstream GRPO performance
- Measure intervention diversity in trained policy

**Success Metrics**:
- BC accuracy improvement: >5% over baseline
- Training stability: reduced variance in loss curves
- GRPO fine-tuning: faster convergence from BC initialization

### 2. InfoNCE Contrastive Learning

**Motivation**: Learn intervention representations that distinguish good interventions from poor ones, rather than directly mimicking expert actions.

**Implementation Approach**:
```python
def infoNCE_loss(
    state_embeddings: jnp.ndarray,      # [batch_size, hidden_dim]
    positive_actions: jnp.ndarray,      # [batch_size, action_dim] 
    negative_actions: jnp.ndarray,      # [batch_size * k, action_dim]
    temperature: float = 0.1
) -> float:
    """InfoNCE contrastive loss for intervention learning."""
    
    # Compute similarities
    pos_sim = compute_action_similarity(state_embeddings, positive_actions)
    neg_sim = compute_action_similarity(state_embeddings, negative_actions)
    
    # InfoNCE loss
    logits = jnp.concatenate([pos_sim, neg_sim]) / temperature
    labels = jnp.zeros(pos_sim.shape[0])  # Positive samples are at index 0
    
    return optax.softmax_cross_entropy(logits, labels)
```

**Expected Benefits**:
- Learn meaningful action representations
- Better handling of multiple valid expert actions
- Improved robustness to expert demonstration quality

**Experimental Protocol**:
- Generate negative samples using random interventions
- Compare against BC baseline on same expert data
- Evaluate learned action representations via clustering analysis
- Test generalization to new SCM structures

**Success Metrics**:
- Action representation quality: measured via intervention clustering
- Downstream task performance: >10% improvement in final ACBO performance
- Robustness: better performance with noisy expert demonstrations

### 3. Uncertainty-Weighted Cross-Entropy

**Motivation**: Weight BC loss by posterior uncertainty - learn more from states where structure is uncertain, less from states where it's already well-determined.

**Implementation Approach**:
```python
def uncertainty_weighted_bc_loss(
    expert_actions: List[Any],
    states: List[AcquisitionState],
    policy_outputs: List[Dict[str, jnp.ndarray]]
) -> float:
    """Weight BC loss by structural uncertainty."""
    
    losses = []
    weights = []
    
    for state, action, output in zip(states, expert_actions, policy_outputs):
        # Standard cross-entropy loss
        ce_loss = compute_action_cross_entropy(output, action, state)
        
        # Weight by uncertainty (high uncertainty = learn more)
        uncertainty_weight = state.uncertainty_bits / 10.0  # Normalize
        uncertainty_weight = jnp.clip(uncertainty_weight, 0.1, 2.0)  # Reasonable range
        
        losses.append(ce_loss)
        weights.append(uncertainty_weight)
    
    # Weighted average
    total_weight = jnp.sum(jnp.array(weights))
    weighted_loss = jnp.sum(jnp.array(losses) * jnp.array(weights)) / total_weight
    
    return weighted_loss
```

**Expected Benefits**:
- Focus learning on informative states
- Better sample efficiency with limited expert data
- Improved performance on structure discovery tasks

**Experimental Protocol**:
- Implement uncertainty weighting in existing BC pipeline
- Compare learning curves with and without weighting
- Analyze which types of states receive highest weights
- Measure structure discovery performance improvement

**Success Metrics**:
- Sample efficiency: achieve same BC performance with 20% less data
- Structure focus: improved performance on structure discovery rewards
- Convergence speed: faster BC training with better final accuracy

## Alternative Training Methodologies

### 1. Pure GRPO (Skip Behavioral Cloning)

**Motivation**: Following DeepSeek R1's approach of skipping SFT and training directly with RL on the base model.

**Implementation Approach**:
```python
def pure_grpo_training(
    policy_network: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    environments: List[Any],
    config: PureGRPOConfig
) -> Any:
    """Train acquisition policy purely with GRPO, no BC warm-start."""
    
    # Initialize policy randomly
    params = initialize_random_policy(policy_network)
    
    # Extended GRPO training (more episodes needed)
    enhanced_config = GRPOConfig(
        max_episodes=config.bc_equivalent_episodes + config.grpo_episodes,
        exploration_bonus=0.2,  # Higher exploration for cold start
        group_size=config.group_size
    )
    
    # Direct GRPO training
    final_params = grpo_train_from_scratch(
        params, policy_network, surrogate_model, enhanced_config
    )
    
    return final_params
```

**Expected Benefits**:
- Avoid BC biases from expert demonstrations
- More exploration and potentially better final performance
- Simpler pipeline (one training phase instead of two)

**Experimental Protocol**:
- Train policy purely with GRPO for 3x normal episode count
- Use higher exploration bonuses to compensate for cold start
- Compare final performance against BC + GRPO baseline
- Analyze exploration patterns in early training

**Success Metrics**:
- Final performance: match or exceed BC + GRPO approach
- Training efficiency: total compute time competitive with two-phase approach
- Exploration quality: discover interventions not in expert demonstrations

### 2. Joint End-to-End Training

**Motivation**: Train surrogate and acquisition models simultaneously rather than sequentially, allowing co-adaptation.

**Implementation Approach**:
```python
def joint_end_to_end_training(
    surrogate_model: Any,
    acquisition_model: Any,
    expert_trajectories: List[ParentScaleTrajectory],
    config: JointTrainingConfig
) -> Tuple[Any, Any]:
    """Train surrogate and acquisition models jointly."""
    
    surrogate_params = initialize_surrogate(surrogate_model)
    acquisition_params = initialize_acquisition(acquisition_model)
    
    for episode in range(config.max_episodes):
        # Collect experience with current models
        batch_data = collect_joint_batch(
            surrogate_model, surrogate_params,
            acquisition_model, acquisition_params,
            expert_trajectories
        )
        
        # Update both models
        surrogate_params = update_surrogate(surrogate_params, batch_data)
        acquisition_params = update_acquisition(acquisition_params, batch_data)
        
        # Co-adaptation: surrogate learns to help acquisition, acquisition learns to use surrogate
    
    return surrogate_params, acquisition_params
```

**Expected Benefits**:
- Better co-adaptation between components
- Potentially superior final performance
- Single training loop instead of separate phases

**Experimental Protocol**:
- Design joint loss function balancing both objectives
- Compare against sequential training baseline
- Analyze co-adaptation effects via ablation studies
- Measure computational overhead of joint training

**Success Metrics**:
- Performance improvement: >15% improvement in final ACBO metrics
- Training stability: convergent joint training without instability
- Computational efficiency: reasonable overhead for joint approach

### 3. Curriculum Learning Approach

**Motivation**: Gradually increase problem difficulty during training, starting with simple SCMs and progressing to complex ones.

**Implementation Approach**:
```python
def curriculum_acquisition_training(
    policy_network: Any,
    surrogate_model: Any,
    curriculum_schedule: List[CurriculumStage],
    config: CurriculumConfig
) -> Any:
    """Train acquisition model with progressive difficulty curriculum."""
    
    params = initialize_policy(policy_network)
    
    for stage in curriculum_schedule:
        # Generate SCMs appropriate for current difficulty
        training_scms = generate_scms_for_difficulty(stage.difficulty_level)
        
        # Collect expert demonstrations at this difficulty
        expert_trajs = collect_expert_demos(training_scms, stage.n_trajectories)
        
        # Train for this stage
        if stage.use_bc:
            params = behavioral_cloning_phase(params, expert_trajs, stage.bc_config)
        
        params = grpo_phase(params, training_scms, stage.grpo_config)
        
        # Evaluate readiness for next stage
        if not evaluate_stage_completion(params, training_scms, stage.success_criteria):
            # Repeat stage with adjusted parameters
            stage = adjust_stage_difficulty(stage)
    
    return params
```

**Expected Benefits**:
- More stable learning progression
- Better final performance on complex problems
- Reduced training time through guided learning

**Experimental Protocol**:
- Design curriculum from 3-variable to 10-variable SCMs
- Define difficulty progression metrics (graph complexity, noise levels)
- Compare against flat training on mixed difficulties
- Analyze learning transfer between curriculum stages

**Success Metrics**:
- Complex problem performance: >25% improvement on 10+ variable SCMs
- Training stability: reduced variance in learning curves
- Transfer learning: faster adaptation to new problem sizes

## Alternative Network Architectures

### 1. Graph Neural Network (GNN) Architecture

**Motivation**: Explicitly represent causal graph structure in the policy network rather than using generic transformers.

**Implementation Approach**:
```python
class GraphAcquisitionPolicy(hk.Module):
    """GNN-based acquisition policy that explicitly models graph structure."""
    
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def __call__(self, state: AcquisitionState) -> Dict[str, jnp.ndarray]:
        # Extract graph structure from state
        graph_structure = extract_graph_from_posterior(state.posterior)
        
        # GNN message passing
        node_embeddings = self.gnn_layers(graph_structure, self.num_layers)
        
        # Output intervention probabilities based on graph structure
        variable_logits = self.variable_selection_head(node_embeddings)
        value_params = self.value_selection_head(node_embeddings)
        
        return {'variable_logits': variable_logits, 'value_params': value_params}
```

**Expected Benefits**:
- Better inductive biases for causal reasoning
- Improved generalization to different graph structures
- More interpretable intervention selection process

**Experimental Protocol**:
- Implement GNN-based policy architecture
- Compare against transformer baseline on diverse graph structures
- Evaluate generalization to unseen graph topologies
- Analyze intervention selection patterns vs graph structure

**Success Metrics**:
- Generalization: >20% better performance on unseen graph structures
- Interpretability: clear correlation between graph structure and interventions
- Efficiency: competitive training time with architectural benefits

### 2. Mixture of Experts (MoE) Architecture

**Motivation**: Different types of SCMs (chains, forks, colliders) may require different intervention strategies.

**Implementation Approach**:
```python
class MoEAcquisitionPolicy(hk.Module):
    """Mixture of Experts policy with specialized experts for different SCM types."""
    
    def __init__(self, n_experts: int, expert_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.expert_dim = expert_dim
    
    def __call__(self, state: AcquisitionState) -> Dict[str, jnp.ndarray]:
        # Route to appropriate expert based on state characteristics
        routing_weights = self.router(state)  # [n_experts]
        
        # Get predictions from all experts
        expert_outputs = []
        for i in range(self.n_experts):
            expert_output = self.experts[i](state)
            expert_outputs.append(expert_output)
        
        # Weighted combination of expert outputs
        combined_output = self.combine_expert_outputs(expert_outputs, routing_weights)
        
        return combined_output
```

**Expected Benefits**:
- Specialized strategies for different problem types
- Better performance across diverse SCM structures
- Potential for expert interpretability

**Experimental Protocol**:
- Train separate experts on different SCM types (chains, forks, colliders)
- Compare against single policy baseline
- Analyze expert routing decisions vs SCM characteristics
- Evaluate specialization effectiveness

**Success Metrics**:
- Diverse performance: improved performance across all SCM types
- Specialization: clear expert preferences for different structures
- Efficiency: competitive training time despite increased complexity

## Implementation Priority

### High Priority (Phase A)
1. **Uncertainty-Weighted Cross-Entropy**: Low implementation complexity, high potential impact
2. **Pure GRPO Training**: Tests fundamental assumption about BC necessity
3. **InfoNCE Contrastive Learning**: Novel approach with good theoretical foundation

### Medium Priority (Phase B)
1. **Curriculum Learning**: Requires careful curriculum design but high potential
2. **GNN Architecture**: Significant architecture change but strong theoretical motivation
3. **Wasserstein Loss**: Complex implementation but potentially high impact

### Low Priority (Phase C)
1. **Joint End-to-End Training**: Very complex, requires major pipeline changes
2. **Mixture of Experts**: High complexity, unclear whether benefits justify costs

## Expected Research Impact

### Academic Contributions
- **Novel loss functions** adapted for causal intervention selection
- **Architecture innovations** for causal reasoning tasks
- **Training methodology** comparisons for domain-specific RL

### Practical Benefits
- **Improved performance** on challenging causal discovery problems
- **Better sample efficiency** with limited expert demonstrations
- **Enhanced generalization** to diverse problem characteristics

## Resource Requirements

### Computational Resources
- **Development Time**: 2-4 weeks per alternative approach
- **Training Compute**: 2-5x baseline training time for comparative studies
- **Evaluation Compute**: Extensive hyperparameter search and cross-validation

### Data Requirements
- **Expert Demonstrations**: Same as baseline (500-1000 trajectories)
- **Evaluation SCMs**: Diverse set covering different structures and sizes
- **Ablation Studies**: Multiple runs with statistical significance testing