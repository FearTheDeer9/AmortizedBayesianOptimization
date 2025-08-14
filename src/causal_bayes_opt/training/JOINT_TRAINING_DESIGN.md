# Joint Training Architecture for ACBO Models

## Executive Summary

This document outlines the design and implementation plan for a GAN-like joint training system where the policy and surrogate models are trained together to optimize their communication and collaborative performance in causal Bayesian optimization.

## 1. Architecture Overview

### 1.1 Core Concept
- **Alternating Training**: Policy and surrogate take turns being frozen/trained
- **Shared Experience**: Both models work on the same trajectory data
- **Communication Channel**: 5-channel tensors with posterior predictions
- **Principled Losses**: Each model has well-defined optimization objectives

### 1.2 Key Components
```python
JointACBOTrainer(UnifiedGRPOTrainer):
    - Manages alternating training phases
    - Coordinates SCM rotation via curriculum factory
    - Maintains trajectory replay buffer
    - Tracks joint performance metrics
```

## 2. Training Phases and Loss Functions

### 2.1 Policy Training Phase (Surrogate Frozen)

**Loss Components:**
```python
policy_loss = λ₁ * absolute_target_loss + λ₂ * information_gain_loss
```

**Absolute Target Loss:**
- Measures actual target node value (not relative improvement)
- Uses initial observational value as baseline
- Directly optimizes for MINIMIZE/MAXIMIZE direction
- Implementation exists in `compute_better_clean_reward` with `reward_type='absolute'`

**Information Gain Loss:**
- Measures reduction in posterior entropy: `H(P|D) - H(P|D,I)`
- Requires surrogate predictions before and after intervention
- Already partially implemented in reward computation

### 2.2 Surrogate Training Phase (Policy Frozen)

**Loss Components:**
```python
surrogate_loss = structure_prediction_loss  # Standard BCE
```

**Structure Prediction Loss:**
- Binary cross-entropy on parent relationships
- Well-established and principled
- Already implemented in `SurrogateBCTrainer`

**Note:** We avoid unprincipled losses (intervention utility, calibration) that lack clear implementation.

## 3. SCM Management and Curriculum Learning

### 3.1 SCM Generation Strategy
- Use `SCMCurriculumFactory` for progressive difficulty
- Generate fresh SCMs on rotation (not fixed set)
- Track performance per SCM characteristics

### 3.2 Rotation Logic
```python
AdaptiveSCMGenerator:
    - request_new_scm(performance_metrics) -> (scm, metadata)
    - should_rotate() based on:
        * Episodes on current SCM (> 20)
        * Convergence metrics (F1 > 0.9)
        * Performance plateau detection
```

### 3.3 Curriculum Integration
- Leverage existing `curriculum_factory.py`
- Test thoroughly as it hasn't been integrated yet
- Parameters: graph_type, n_vars, edge_prob, noise_scale

## 4. Experience Replay System

### 4.1 Trajectory Storage
```python
TrajectoryReplayBuffer:
    - Stores complete trajectories (not mixed samples)
    - Maintains SCM metadata for each trajectory
    - Capacity: 100 full trajectories
    - Sampling: Can sample same-SCM or mixed
```

### 4.2 Replay Strategy
- Store full episode trajectories to maintain consistency
- Never mix samples from different trajectories
- Prioritize recent and high-learning-value trajectories

## 5. Training Algorithm

### 5.1 Main Training Loop
```python
for episode in range(max_episodes):
    # Determine training phase
    if should_switch_phase():
        current_phase = "policy" if current_phase == "surrogate" else "surrogate"
    
    # Get SCM (may rotate based on performance)
    if should_rotate_scm():
        scm, metadata = scm_generator.request_new_scm(performance)
    
    # Collect trajectory with current models
    trajectory = collect_episode(policy_params, surrogate_params, scm)
    
    # Train appropriate model
    if current_phase == "policy":
        policy_params = train_policy_with_grpo(
            trajectory, 
            frozen_surrogate=surrogate_params
        )
    else:
        surrogate_params = train_surrogate_with_bce(
            trajectory,
            frozen_policy=policy_params
        )
    
    # Store trajectory for replay
    replay_buffer.add(trajectory, scm_metadata, performance_delta)
    
    # Mix with replay data
    if len(replay_buffer) > min_replay_size:
        replay_batch = replay_buffer.sample(batch_size=4)
        # Use replay_batch in training
```

### 5.2 GRPO Integration
- Policy uses GRPO training (already implemented)
- Surrogate uses standard supervised learning (not GRPO)
- Reason: Surrogate doesn't take "actions" in RL sense

## 6. Adaptive Reward Weighting

### 6.1 Within-Trajectory Adaptation
```python
def get_trajectory_weights(step_in_trajectory):
    if step_in_trajectory < 5:  # Early: explore
        return {'absolute_target': 0.3, 'information_gain': 0.7}
    else:  # Late: exploit
        return {'absolute_target': 0.8, 'information_gain': 0.2}
```

### 6.2 Training-Wide Adaptation (Optional)
- Early training (< 100 episodes): More exploration
- Mid training (100-500): Balanced
- Late training (> 500): Exploitation focused

**Note:** Start with fixed weights, add adaptation later if needed.

## 7. Performance-Based Rotation

### 7.1 Model Training Balance
```python
def should_switch_training_focus():
    if policy_plateaued and not surrogate_plateaued:
        return "train_surrogate"
    elif surrogate_plateaued and not policy_plateaued:
        return "train_policy"
    else:
        return "alternate_normally"
```

### 7.2 SCM Rotation Criteria
- Converged on current SCM (high F1, good target value)
- Spent too many episodes (> 20)
- Performance plateaued (no improvement in 5 episodes)

### 7.3 Curriculum Advancement
- Both models performing well (F1 > 0.8, target improved)
- Consistent performance across multiple SCMs
- Ready for increased complexity

## 8. Implementation Stages

### Stage 1: Foundation (Week 1)
**Goal:** Basic joint training infrastructure

**Tasks:**
1. Create `JointACBOTrainer` class extending `UnifiedGRPOTrainer`
2. Implement basic alternating training logic
3. Add trajectory replay buffer
4. Create simple test with fixed SCMs

**Validation:**
- [ ] Both models update alternately
- [ ] No crashes or memory leaks
- [ ] Basic metrics logged

### Stage 2: SCM Management (Week 1-2)
**Goal:** Dynamic SCM generation and rotation

**Tasks:**
1. Integrate `SCMCurriculumFactory`
2. Implement rotation logic
3. Add performance tracking per SCM
4. Test curriculum progression

**Validation:**
- [ ] SCMs rotate appropriately
- [ ] Curriculum advances based on performance
- [ ] No variable mapping errors

### Stage 3: Loss Implementation (Week 2)
**Goal:** Proper loss functions for joint training

**Tasks:**
1. Implement absolute target reward
2. Add information gain calculation
3. Ensure proper gradient flow
4. Add loss component tracking

**Validation:**
- [ ] Losses decrease over time
- [ ] Both models show learning
- [ ] Information gain correlates with structure learning

### Stage 4: Replay System (Week 3)
**Goal:** Experience replay for stability

**Tasks:**
1. Implement trajectory buffer
2. Add replay sampling
3. Mix replay with online data
4. Test forgetting prevention

**Validation:**
- [ ] Old trajectories retrieved correctly
- [ ] No trajectory mixing bugs
- [ ] Performance improves with replay

### Stage 5: Advanced Features (Week 3-4)
**Goal:** Optimization and debugging

**Tasks:**
1. Add adaptive weighting (if needed)
2. Implement performance-based rotation
3. Add comprehensive logging
4. Create visualization tools

**Validation:**
- [ ] Adaptive features work as intended
- [ ] Can debug training issues
- [ ] Clear performance improvements

### Stage 6: Testing & Validation (Week 4)
**Goal:** Ensure correctness and performance

**Tasks:**
1. Run ablation studies
2. Compare against baselines
3. Test on diverse SCMs
4. Document results

**Validation:**
- [ ] Joint training outperforms independent
- [ ] No silent failures
- [ ] Reproducible results

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each component in isolation
- Verify loss calculations
- Check gradient flow
- Validate buffer operations

### 9.2 Integration Tests
```python
class JointTrainingValidator:
    def validate_episode(self, episode_data):
        checks = [
            check_variable_mapping_consistency(),
            check_both_models_updating(),
            check_frozen_model_unchanged(),
            check_trajectory_completeness(),
            check_scm_rotation_proper(),
            check_reward_computation_correct(),
            check_no_data_leakage(),
            check_posteriors_stored_correctly()
        ]
        assert all(checks), "Validation failed"
```

### 9.3 Debug Checks
- Log all model decisions
- Track gradient norms
- Monitor loss components
- Visualize learning curves

## 10. Potential Issues and Mitigations

### 10.1 Variable Mapping Errors
**Issue:** Variables ordered differently in different components
**Mitigation:** Always use `VariableMapper`, validate consistency

### 10.2 Catastrophic Forgetting
**Issue:** Models forget earlier learning
**Mitigation:** Experience replay, elastic weight consolidation

### 10.3 Training Instability
**Issue:** Loss explosions, NaN values
**Mitigation:** Gradient clipping, proper initialization, loss monitoring

### 10.4 SCM Rotation Bugs
**Issue:** Wrong SCM used, metadata mismatch
**Mitigation:** Strict tracking, validation checks

### 10.5 Silent Logic Errors
**Issue:** Models appear to train but don't actually learn
**Mitigation:** Comprehensive metrics, ablation studies, baseline comparisons

## 11. Success Metrics

### 11.1 Training Metrics
- Policy loss decreasing
- Surrogate accuracy improving
- Information gain positive
- Target values improving

### 11.2 Joint Performance
- Models use each other's information
- Better than independent training
- Faster convergence
- Higher final performance

### 11.3 Robustness
- Works across diverse SCMs
- Handles curriculum progression
- No catastrophic failures
- Reproducible results

## 12. Code Organization

```
src/causal_bayes_opt/training/
├── joint_acbo_trainer.py          # Main joint trainer
├── trajectory_replay_buffer.py    # Replay system
├── adaptive_scm_generator.py      # SCM management
├── joint_training_utils.py        # Helper functions
└── tests/
    ├── test_joint_training.py     # Unit tests
    ├── test_replay_buffer.py      # Buffer tests
    └── test_integration.py        # Full system tests
```

## 13. Next Steps

1. **Review this design** with stakeholders
2. **Create skeleton classes** with interfaces
3. **Implement Stage 1** (basic infrastructure)
4. **Test after each stage** before proceeding
5. **Document issues** as they arise
6. **Update this document** with learnings

## Appendix A: Configuration Example

```yaml
joint_training:
  # Phase control
  episodes_per_phase: 5
  initial_phase: "policy"
  
  # SCM management
  use_curriculum: true
  rotation_episodes: 20
  convergence_f1_threshold: 0.9
  
  # Replay buffer
  replay_capacity: 100
  replay_batch_size: 4
  min_replay_size: 10
  
  # Loss weights
  policy_weights:
    absolute_target: 0.7
    information_gain: 0.3
  
  # Adaptive features (optional)
  use_adaptive_weights: false
  use_performance_rotation: true
  
  # Debug
  log_every: 10
  validate_every: 50
  save_checkpoints: true
```

## Appendix B: Monitoring Dashboard

Key metrics to track during training:
1. **Policy Performance**: Loss, entropy, target improvement
2. **Surrogate Performance**: BCE loss, F1 score, calibration
3. **Joint Metrics**: Information gain, coordination score
4. **System Health**: Memory usage, gradient norms, validation checks

---

**Document Version:** 1.0
**Last Updated:** [Current Date]
**Author:** Joint ACBO Team
**Status:** DRAFT - Ready for Review