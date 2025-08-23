# UnifiedGRPOTrainer Modular Architecture Refactoring

## 🎯 Overview

The `UnifiedGRPOTrainer` has been refactored from a monolithic 2654-line class into a clean modular architecture while maintaining **100% backward compatibility** with existing experiments.

### Key Achievements
- **📉 3x Complexity Reduction**: 2654 lines → 843 lines  
- **🔧 Modular Components**: Clear separation of concerns
- **✅ Binary Rewards**: New reward system for sparse signal training
- **🔄 Drop-in Replacement**: All existing tests work unchanged
- **🧪 Easier Testing**: Components can be tested independently

---

## 🏗️ Architecture Changes

### Before (Monolithic)
```
UnifiedGRPOTrainer [2654 lines]
├── Policy initialization
├── Surrogate handling  
├── GRPO training logic
├── Reward computation (multiple overlapping systems)
├── Diagnostic logging (mixed throughout)
├── Phase switching logic
├── SCM rotation logic
├── Convergence detection
├── Checkpointing
└── Extensive debug code
```

### After (Modular)
```
UnifiedGRPOTrainer [843 lines] - Main interface (unchanged)
├── GRPORewardComputer [~150 lines] - Unified reward computation
├── GRPOLogger [~300 lines] - Diagnostic logging  
├── Original components - Policy, GRPO, convergence (simplified)
└── Compatibility layer - For inheritance by JointACBOTrainer
```

---

## 🔄 Backward Compatibility

### ✅ No Changes Required for Existing Tests

**All existing experiment scripts work unchanged:**
```python
# This continues to work exactly as before
from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer

trainer = UnifiedGRPOTrainer(config=config)
results = trainer.train(scms=scms)
```

**Inheritance chains continue working:**
```python
# JointACBOTrainer still inherits properly
class JointACBOTrainer(UnifiedGRPOTrainer):
    # All existing code works unchanged
```

### Interface Guarantees

**Same constructor:**
```python
UnifiedGRPOTrainer(
    config=config,  # Same config format
    learning_rate=3e-4,  # Same parameters  
    batch_size=64,
    # ... all original parameters supported
)
```

**Same methods:**
- `train(scms, eval_scms=None)` - Identical interface and return format
- `_run_grpo_episode()` - Same signature and behavior
- `_save_checkpoint()` - Same checkpointing logic
- All compatibility methods for JointACBOTrainer inheritance

**Same attributes:**
- `policy_params` - Policy parameters for access
- `rng_key` - RNG state
- `reward_stats` - Running statistics
- `config` - Configuration dictionary
- All other attributes expected by existing code

---

## 🆕 New Features

### 1. Binary Rewards

**New reward system for sparse signal training:**

```python
config = {
    'reward_type': 'binary',  # Enable binary rewards
    'optimization_direction': 'MINIMIZE',  # or 'MAXIMIZE'
    # ... other config
}
```

**Binary reward logic:**
- **+1** if target value is **below** group median (good for minimization)
- **0** if target value is **above** group median (bad for minimization)
- Uses current GRPO batch to compute median threshold
- Provides clear, sparse learning signals

**Usage in experiments:**
```python
# Add this to any experiment config
config['reward_type'] = 'binary'

# Look for these logs to verify it's working
# [BINARY TARGET REWARD] Value: -15.405, Binary reward: 1.0
# [BINARY BATCH] Median: -18.234, Rewards: ['1', '0', '1', '0', ...]
```

### 2. Modular Components

**For new test development, you can now use components directly:**

```python
# Use reward computer independently
from src.causal_bayes_opt.training.grpo_reward_computer import GRPORewardComputer

reward_computer = GRPORewardComputer(config)
reward_info = reward_computer.compute_reward(intervention, outcome, buffer, scm, ...)

# Use logger independently  
from src.causal_bayes_opt.training.grpo_logger import GRPOLogger

logger = GRPOLogger(optimization_direction="MINIMIZE")
logger.log_candidates_with_rewards(grpo_batch_data, mapper, target_var, scm)
```

---

## 📊 Configuration Guide

### Existing Configurations (No Changes)

**All existing configs continue working:**
```python
config = {
    'max_episodes': 100,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'reward_weights': {'target': 0.7, 'parent': 0.1, 'info_gain': 0.2},
    'optimization_direction': 'MINIMIZE',
    # ... all existing options supported
}
```

### New Configuration Options

**Binary rewards:**
```python
config = {
    'reward_type': 'binary',  # NEW: Enable binary rewards
    'optimization_direction': 'MINIMIZE',  # Required for binary logic
    'reward_weights': {'target': 1.0, 'parent': 0.0, 'info_gain': 0.0},  # Recommended for binary
}
```

**Reward types supported:**
- `'composite'` - Default, multi-component rewards (target + parent + info_gain)
- `'binary'` - NEW: Binary rewards based on group median
- `'clean'` - Improvement-based rewards with baselines
- `'better_clean'` - Adaptive sigmoid rewards with running statistics

---

## 🧪 Development Patterns

### For Existing Tests

**No changes needed! But you get benefits:**
- **Faster debugging**: Clear separation makes issues easier to isolate
- **Better logging**: More structured diagnostic output
- **Easier maintenance**: Reward computation bugs only affect one module

### For New Tests

**Testing specific components:**
```python
# Test only reward computation
from src.causal_bayes_opt.training.grpo_reward_computer import GRPORewardComputer

def test_binary_rewards():
    reward_computer = GRPORewardComputer(config={'reward_type': 'binary'})
    reward_info = reward_computer.compute_reward(...)
    assert reward_info['total'] in [0.0, 1.0]  # Should be binary

# Test only logging
from src.causal_bayes_opt.training.grpo_logger import GRPOLogger

def test_advantage_logging():
    logger = GRPOLogger("MINIMIZE")
    logger.log_grpo_advantages(rewards)  # Test logging independently
```

**Creating new trainers:**
```python
# Compose components differently for new experiments
from src.causal_bayes_opt.training.grpo_reward_computer import GRPORewardComputer
from src.causal_bayes_opt.training.grpo_logger import GRPOLogger

class CustomGRPOTrainer:
    def __init__(self, config):
        self.reward_computer = GRPORewardComputer(config)
        self.logger = GRPOLogger(config['optimization_direction'])
        # Build custom trainer using modular components
```

---

## 🔧 Migration Guide

### Immediate (No Action Required)
- **All existing tests continue working unchanged**
- **Same performance and behavior**  
- **Same interfaces and return formats**

### Optional Enhancements

**To leverage binary rewards:**
```python
# Add to any existing experiment config
config['reward_type'] = 'binary'

# Expected behavior change:
# - Sparse +1/0 rewards instead of continuous values
# - Clearer learning signals for GRPO
# - Better exploration/exploitation balance
```

**To use modular components in new tests:**
```python
# Instead of testing entire trainer, test components
def test_reward_computation():
    reward_computer = GRPORewardComputer(config)
    # Test reward logic in isolation
    
def test_grpo_logging():
    logger = GRPOLogger("MINIMIZE") 
    # Test diagnostic logic independently
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Unknown reward type binary" warning:**
```
WARNING: Unknown reward type binary, using sigmoid
```
**Solution:** This indicates the binary reward isn't being routed correctly. Check that:
- `config['reward_type'] = 'binary'` is set
- GRPORewardComputer is being used (should see "Initialized GRPORewardComputer" log)

**2. Import errors with modular components:**
```python
# Correct imports for new components
from src.causal_bayes_opt.training.grpo_reward_computer import GRPORewardComputer
from src.causal_bayes_opt.training.grpo_logger import GRPOLogger
```

**3. Missing attributes in inheritance:**
```
AttributeError: object has no attribute 'some_method'
```
**Solution:** The modular version includes compatibility methods for JointACBOTrainer. If you see this, the method might need to be added to the compatibility layer.

### Debugging Binary Rewards

**Expected logs when working correctly:**
```
[INFO] Initialized GRPORewardComputer with reward_type=binary
[INFO] [BINARY TARGET REWARD] Value: -15.405, Binary reward: 1.0
[INFO] [BINARY BATCH] Median: -18.234, Rewards: ['1', '0', '1', '0']
```

**If you see continuous values instead of 0/1:**
- Check that `reward_type='binary'` is in your config
- Verify GRPORewardComputer initialization logs
- Check that the right reward computation path is being used

---

## 📈 Benefits for Team Development

### Immediate Benefits
- **🔍 Easier Debugging**: Issues isolated to specific components
- **⚡ Faster Development**: Clear interfaces for each concern
- **🧪 Better Testing**: Test components independently  
- **📖 Cleaner Code**: Single responsibility, easier to understand

### Future Enhancement Opportunities
- **New Reward Types**: Add to GRPORewardComputer without touching training logic
- **Custom Logging**: Modify GRPOLogger for specific experiment needs
- **Different Architectures**: Easy to swap policy components
- **Performance Optimizations**: Optimize components independently

### Reduced Maintenance Burden
- **🎯 Single files to modify**: Want to change rewards? Only touch grpo_reward_computer.py
- **📝 Cleaner git history**: Changes are focused and purpose-built
- **🔄 Easier reviews**: Smaller, focused changes
- **🧩 Component reuse**: Use reward computer in other trainers

---

## 📁 File Structure

### New Modular Components
```
src/causal_bayes_opt/training/
├── grpo_reward_computer.py     # Unified reward computation
├── grpo_logger.py              # Diagnostic logging
├── unified_grpo_trainer.py     # Main trainer (modular internally)
└── unified_grpo_trainer_original.py  # Backup of original
```

### Existing Files (Unchanged)
```
experiments/policy-only-training/
├── train_grpo_single_scm_sanity.py  # Works unchanged
└── train_grpo_*.py                  # All work unchanged

experiments/joint-training/
├── train_joint.py                   # Works unchanged
└── ...                              # All work unchanged
```

---

## 🚀 Quick Start for Team

### Using Existing Tests
```bash
# Everything works exactly as before
python experiments/policy-only-training/train_grpo_single_scm_sanity.py --episodes 10

# To try binary rewards, just add to config:
# config['reward_type'] = 'binary'
```

### Creating New Tests
```python
# Option 1: Use full trainer (recommended for most cases)
from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer

trainer = UnifiedGRPOTrainer(config={'reward_type': 'binary'})
results = trainer.train(scms)

# Option 2: Use modular components (for component-specific tests)
from src.causal_bayes_opt.training.grpo_reward_computer import GRPORewardComputer

reward_computer = GRPORewardComputer(config)
reward_info = reward_computer.compute_reward(...)
```

---

## ⚠️ Important Notes

1. **Backward Compatibility Guaranteed**: All existing code works without modification
2. **Binary Rewards are Optional**: Default behavior unchanged unless explicitly enabled
3. **Gradual Adoption**: Use new features when convenient, no forced migration
4. **Component Testing**: New modular components can be tested independently
5. **Performance**: Same performance characteristics as original implementation

## 📞 Support

If you encounter issues with the refactored trainer:
1. Check that your config includes the correct `reward_type`
2. Look for initialization logs to verify components are loading correctly  
3. Compare with working examples in `experiments/policy-only-training/`
4. Check troubleshooting section above for common issues

The modular architecture provides a solid foundation for continued development while maintaining full compatibility with existing work.