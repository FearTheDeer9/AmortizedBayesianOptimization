# Training Loop API

## Overview

The training loop coordinates the interaction between the surrogate model, acquisition model, and SCM environment.

## Core Functions

### initialize_models

```python
def initialize_models(
    buffer: ExperienceBuffer,
    target: str,
    random_seed: Optional[int] = None
) -> Tuple[SurrogateParams, AcquisitionParams]:
    """
    Initialize surrogate and acquisition models
    
    Args:
        buffer: Initial experience buffer with observational data
        target: Target variable of interest
        random_seed: Optional seed for reproducibility
        
    Returns:
        Tuple of initialized model parameters
    """
### intervention_step
pythondef intervention_step(
    scm: SCM,
    acquisition_params: AcquisitionParams,
    buffer: ExperienceBuffer,
    posterior: ParentSetPosterior
) -> Tuple[Intervention, Sample]:
    """
    Select intervention using acquisition model and apply to SCM
    
    Args:
        scm: Current structural causal model
        acquisition_params: Acquisition model parameters
        buffer: Current experience buffer
        posterior: Current parent set posterior
        
    Returns:
        selected_intervention: The intervention chosen
        outcome: The resulting outcome from the SCM
    """
### train
pythondef train(
    scm: SCM,
    initial_buffer: ExperienceBuffer,
    num_steps: int,
    surrogate_lr: float,
    acquisition_lr: float,
    true_parent_set: Optional[FrozenSet[str]] = None
) -> Tuple[SurrogateParams, AcquisitionParams, ExperienceBuffer]:
    """
    Main training loop for amortized CBO
    
    Args:
        scm: Target structural causal model
        initial_buffer: Initial experience buffer with observational data
        num_steps: Number of training steps to perform
        surrogate_lr: Learning rate for surrogate model
        acquisition_lr: Learning rate for acquisition model
        true_parent_set: Optional ground truth for evaluation
        
    Returns:
        surrogate_params: Trained surrogate model parameters
        acquisition_params: Trained acquisition model parameters
        final_buffer: Updated experience buffer with all data
    """
### Training Process Flow
1. Initialize models with observational data
2. For each training step:
   a. Update surrogate model and compute posterior
   b. Select intervention using acquisition model
   c. Apply intervention to SCM and observe outcome
   d. Add intervention-outcome pair to buffer
   e. Update acquisition model using reward signal
3. Return trained models and final buffer
### Expert Demonstration Integration
pythondef expert_demonstration(
    scm: SCM,
    buffer: ExperienceBuffer,
    num_interventions: int
) -> ExperienceBuffer:
    """
    Generate expert demonstrations using PARENT_SCALE
    
    Args:
        scm: Target structural causal model
        buffer: Current experience buffer
        num_interventions: Number of interventions to perform
        
    Returns:
        Updated buffer with expert demonstrations
    """