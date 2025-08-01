# Principled Reward Computation Refactor

## Current Problem
- Module-level imports make testing difficult
- Need monkey patching to intercept reward computations
- No built-in observability for reward calculations

## Proposed Solution 1: Dependency Injection

### Change the trainer to accept reward function as parameter:

```python
class UnifiedGRPOTrainer:
    def __init__(self, 
                 config: Config,
                 reward_fn: Optional[Callable] = None,
                 **kwargs):
        # Use provided reward function or default
        self.compute_reward = reward_fn or compute_clean_reward
```

### Benefits:
- Easy to inject custom reward functions for testing
- No monkey patching needed
- Clear dependency management

## Proposed Solution 2: Reward Hooks/Callbacks

### Add hooks for reward computation:

```python
class UnifiedGRPOTrainer:
    def __init__(self, config: Config, **kwargs):
        self.reward_hooks = []
    
    def add_reward_hook(self, hook: Callable):
        """Add a callback that gets called after each reward computation."""
        self.reward_hooks.append(hook)
    
    def _compute_reward_with_hooks(self, *args, **kwargs):
        reward_info = compute_clean_reward(*args, **kwargs)
        
        # Call all hooks with reward details
        for hook in self.reward_hooks:
            hook(reward_info, *args, **kwargs)
        
        return reward_info
```

### Usage in tests:
```python
captured_rewards = []

def capture_hook(reward_info, buffer_before, intervention, outcome, 
                 target_variable, config, posterior_before, posterior_after):
    captured_rewards.append({
        'reward_info': reward_info,
        'has_posteriors': posterior_before is not None and posterior_after is not None,
        # ... other details
    })

trainer.add_reward_hook(capture_hook)
```

## Proposed Solution 3: Built-in Logging

### Add structured logging for rewards:

```python
def compute_clean_reward(...):
    # ... existing computation ...
    
    # Structured logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("reward_computation", extra={
            'reward_total': float(total_reward),
            'reward_components': {
                'target': float(target_reward),
                'diversity': float(diversity_reward),
                'exploration': float(exploration_reward),
                'info_gain': float(info_gain_reward),
            },
            'has_posteriors': posterior_before is not None and posterior_after is not None,
            'entropy_before': posterior_before.get('entropy') if posterior_before else None,
            'entropy_after': posterior_after.get('entropy') if posterior_after else None,
        })
    
    return rewards
```

Then tests can capture logs instead of monkey patching.

## Proposed Solution 4: Reward Registry Pattern

### Create a registry for reward functions:

```python
# In clean_rewards.py
class RewardRegistry:
    def __init__(self):
        self._functions = {}
        self._default = 'clean'
        
    def register(self, name: str, fn: Callable):
        self._functions[name] = fn
        
    def get(self, name: str = None) -> Callable:
        name = name or self._default
        return self._functions.get(name, compute_clean_reward)

reward_registry = RewardRegistry()
reward_registry.register('clean', compute_clean_reward)
reward_registry.register('structure_aware', compute_structure_aware_reward)

# In trainer
class UnifiedGRPOTrainer:
    def __init__(self, config: Config, reward_type: str = 'clean', **kwargs):
        self.compute_reward = reward_registry.get(reward_type)
```

## Recommendation

I recommend **Solution 2 (Reward Hooks)** because:
1. Minimal changes to existing code
2. Flexible for testing and monitoring
3. Can be added without breaking existing interfaces
4. Allows multiple observers (testing, logging, metrics)

Implementation would be:
1. Add `reward_hooks` list to trainer
2. Add `add_reward_hook()` method
3. Wrap reward computation to call hooks
4. Update tests to use hooks instead of monkey patching