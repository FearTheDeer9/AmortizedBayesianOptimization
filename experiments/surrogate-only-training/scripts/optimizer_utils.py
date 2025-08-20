#!/usr/bin/env python3
"""Advanced optimizer configurations for curriculum learning."""

import optax
import jax.numpy as jnp
from typing import Dict, Any, Optional


def create_warmup_cosine_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> optax.Schedule:
    """Create a warmup + cosine decay schedule.
    
    Args:
        init_value: Initial learning rate
        peak_value: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps after warmup
        end_value: Final learning rate
    """
    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps
        ),
        optax.cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps,
            alpha=end_value/peak_value
        )
    ]
    return optax.join_schedules(schedules, [warmup_steps])


def create_stage_aware_schedule(
    base_lr: float,
    stage_multipliers: Dict[int, float],
    steps_per_stage: int
) -> optax.Schedule:
    """Create a schedule that adjusts learning rate per curriculum stage.
    
    Args:
        base_lr: Base learning rate
        stage_multipliers: Multiplier for each stage (e.g., {1: 1.0, 2: 0.8, 3: 0.6})
        steps_per_stage: Approximate steps per stage
    """
    def schedule_fn(step):
        # Use JAX-compatible indexing instead of dict.get()
        stage_idx = jnp.minimum(step // steps_per_stage, len(stage_multipliers) - 1)
        # Convert dict to array for JAX compatibility
        multipliers_array = jnp.array([stage_multipliers[i+1] for i in range(max(stage_multipliers.keys()))])
        multiplier = multipliers_array[stage_idx]
        return base_lr * multiplier
    
    return schedule_fn


def create_adaptive_optimizer(
    config: Dict[str, Any],
    num_training_steps: Optional[int] = None,
    use_curriculum_aware: bool = True
) -> optax.GradientTransformation:
    """Create an advanced optimizer with smart scheduling.
    
    Args:
        config: Configuration dictionary with optimizer settings
        num_training_steps: Total number of training steps (for cosine schedule)
        use_curriculum_aware: Whether to use curriculum-aware scheduling
    """
    
    base_lr = config.get('learning_rate', 0.001)
    
    # Choose schedule based on configuration
    if use_curriculum_aware and 'stage_multipliers' in config:
        # Curriculum-aware schedule - reduce LR as SCMs get larger
        schedule = create_stage_aware_schedule(
            base_lr=base_lr,
            stage_multipliers=config['stage_multipliers'],
            steps_per_stage=config.get('steps_per_stage', 1000)
        )
    elif num_training_steps is not None:
        # Warmup + Cosine schedule (best for known total steps)
        warmup_steps = min(1000, num_training_steps // 10)
        schedule = create_warmup_cosine_schedule(
            init_value=base_lr * 0.1,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=num_training_steps - warmup_steps,
            end_value=base_lr * 0.01
        )
    else:
        # Exponential decay (current default)
        schedule = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=config.get('decay_steps', 2000),
            decay_rate=config.get('decay_rate', 0.9),
            staircase=True
        )
    
    # Build optimizer chain
    optimizer_chain = []
    
    # Gradient clipping
    clip_norm = config.get('gradient_clip', 1.0)
    if clip_norm > 0:
        optimizer_chain.append(optax.clip_by_global_norm(clip_norm))
    
    # Optional gradient accumulation for large models
    if config.get('accumulate_gradients', False):
        optimizer_chain.append(
            optax.MultiSteps(
                optax.adamw(learning_rate=schedule, 
                          weight_decay=config.get('weight_decay', 1e-4)),
                every_k_schedule=config.get('accumulation_steps', 4)
            )
        )
    else:
        # Main optimizer
        optimizer_type = config.get('optimizer_type', 'adamw')
        
        if optimizer_type == 'adamw':
            optimizer_chain.append(
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.get('adam_b1', 0.9),
                    b2=config.get('adam_b2', 0.999),
                    eps=config.get('adam_eps', 1e-8),
                    weight_decay=config.get('weight_decay', 1e-4)
                )
            )
        elif optimizer_type == 'lion':
            # Lion optimizer - often better for large models
            optimizer_chain.append(
                optax.lion(
                    learning_rate=schedule * 0.1,  # Lion typically needs lower LR
                    b1=config.get('lion_b1', 0.9),
                    b2=config.get('lion_b2', 0.99),
                    weight_decay=config.get('weight_decay', 1e-4)
                )
            )
        elif optimizer_type == 'sgd':
            # SGD with momentum - sometimes more stable
            optimizer_chain.append(
                optax.sgd(
                    learning_rate=schedule,
                    momentum=config.get('momentum', 0.9),
                    nesterov=config.get('nesterov', True)
                )
            )
    
    return optax.chain(*optimizer_chain)


def create_curriculum_optimizer_config(
    model_size: str,
    max_stages: int = 5,
    estimated_steps: int = 10000
) -> Dict[str, Any]:
    """Create optimizer configuration based on model size and curriculum.
    
    Args:
        model_size: 'small', 'medium', 'large', or 'xlarge'
        max_stages: Number of curriculum stages
        estimated_steps: Estimated total training steps
    """
    
    # Base configurations for different model sizes
    base_configs = {
        'small': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'optimizer_type': 'adamw'
        },
        'medium': {
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'optimizer_type': 'adamw'
        },
        'large': {
            'learning_rate': 0.0003,
            'weight_decay': 2e-4,
            'gradient_clip': 0.5,
            'optimizer_type': 'adamw',
            'adam_b2': 0.98  # Less aggressive for stability
        },
        'xlarge': {
            'learning_rate': 0.0002,
            'weight_decay': 3e-4,
            'gradient_clip': 0.3,
            'optimizer_type': 'lion',  # Lion often better for very large models
            'accumulate_gradients': True,
            'accumulation_steps': 4
        }
    }
    
    config = base_configs.get(model_size, base_configs['medium'])
    
    # Add curriculum-aware scheduling
    # Reduce LR as we progress through stages (larger SCMs need more careful updates)
    config['stage_multipliers'] = {
        1: 1.0,    # Full LR for small SCMs
        2: 0.8,    # 80% for medium
        3: 0.6,    # 60% for large
        4: 0.4,    # 40% for very large
        5: 0.3     # 30% for 100-var SCMs
    }
    config['steps_per_stage'] = estimated_steps // max_stages
    
    return config


def get_optimizer_for_resume(
    checkpoint: Dict,
    config: Optional[Dict] = None
) -> optax.GradientTransformation:
    """Create optimizer for resuming training from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        config: Optional override configuration
    """
    
    if config is None:
        config = checkpoint.get('optimizer_config', {})
    
    # Estimate remaining steps
    completed_steps = checkpoint.get('total_steps', 0)
    estimated_total = checkpoint.get('estimated_total_steps', 20000)
    remaining_steps = max(1000, estimated_total - completed_steps)
    
    # Create optimizer with adjusted schedule
    return create_adaptive_optimizer(
        config=config,
        num_training_steps=remaining_steps,
        use_curriculum_aware=True
    )


# Example usage in training script:
if __name__ == "__main__":
    # Test different configurations
    for model_size in ['small', 'medium', 'large', 'xlarge']:
        print(f"\n{model_size.upper()} Model Configuration:")
        config = create_curriculum_optimizer_config(model_size)
        print(f"  Base LR: {config['learning_rate']}")
        print(f"  Optimizer: {config['optimizer_type']}")
        print(f"  Weight decay: {config['weight_decay']}")
        print(f"  Gradient clip: {config['gradient_clip']}")
        print(f"  Stage multipliers: {config['stage_multipliers']}")
        if config.get('accumulate_gradients'):
            print(f"  Gradient accumulation: {config['accumulation_steps']} steps")