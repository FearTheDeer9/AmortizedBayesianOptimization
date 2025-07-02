# Hydra + WandB Integration Demo

This directory contains a minimal but powerful demonstration of integrating Hydra configuration management and enhanced WandB experiment tracking with your existing causal Bayesian optimization framework.

## What's New

### Enhanced WandB Integration
- **Real WandB Logging**: The `GRPOTrainingManager` now actually logs metrics to WandB (not just console messages)
- **Causal Discovery Metrics**: Specialized logging for parent set accuracy, intervention efficiency, structural hamming distance
- **Comprehensive Metrics**: GRPO losses, diversity metrics, training progress all logged with proper prefixes
- **Error Handling**: Graceful fallback if WandB is unavailable

### Hydra Configuration Management
- **Compositional Configs**: Mix and match training, logging, and experiment configurations
- **Easy Overrides**: Change any parameter from command line without editing files
- **Hyperparameter Sweeps**: Built-in support for systematic parameter exploration
- **Environment Configs**: Different settings for development vs production

## Directory Structure

```
config/
├── config.yaml                           # Main configuration
├── training/
│   ├── grpo_base.yaml                    # Base GRPO configuration
│   └── grpo_high_performance.yaml       # High-performance variant
├── logging/
│   ├── wandb_enabled.yaml               # WandB enabled logging
│   └── local_dev.yaml                   # Local development logging
└── experiment/
    └── causal_discovery.yaml            # Causal discovery experiment setup

scripts/
└── hydra_wandb_demo.py                  # Complete integration demonstration
```

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install hydra-core wandb
   ```

2. **Run the basic demo**:
   ```bash
   python scripts/hydra_wandb_demo.py
   ```

3. **Try configuration overrides**:
   ```bash
   # Change learning rate
   python scripts/hydra_wandb_demo.py training.algorithm.learning_rate=0.001
   
   # Use high-performance config
   python scripts/hydra_wandb_demo.py training=grpo_high_performance
   
   # Disable WandB for local testing
   python scripts/hydra_wandb_demo.py logging=local_dev
   ```

4. **Run hyperparameter sweeps**:
   ```bash
   python scripts/hydra_wandb_demo.py --multirun \
     training.algorithm.learning_rate=0.0001,0.001,0.01 \
     training.algorithm.batch_size=32,64,128
   ```

## Integration with Existing Code

The demo shows how to integrate Hydra with your existing Pydantic configuration system:

```python
# Convert Hydra config to existing Pydantic config
logging_config = LoggingConfig(
    enable_wandb=cfg.logging.wandb.enabled,
    project_name=cfg.logging.wandb.project,
    # ... other fields
)

# Use with existing training manager
training_manager = GRPOTrainingManager(config=comprehensive_config)
training_manager.train()  # Will now log to WandB!
```

## Key Benefits Demonstrated

### 1. **Experiment Organization**
- Automatic experiment grouping and tagging
- Config versioning and reproducibility
- Hyperparameter sweep management

### 2. **Configuration Flexibility**
- Environment-specific configs (dev vs cluster)
- Easy parameter exploration
- Configuration composition and inheritance

### 3. **Enhanced Monitoring**
- Causal discovery specific metrics in WandB
- Real-time training progress visualization
- Automatic config logging for reproducibility

### 4. **Minimal Disruption**
- Works alongside existing Pydantic configs
- Optional integration - existing code unchanged
- Graceful fallbacks if tools unavailable

## Example Output

When you run the demo, you'll see:

```
INFO - Starting Hydra + WandB Integration Demo
INFO - WandB initialized: https://wandb.ai/your-username/causal_bayes_opt/runs/abc123
INFO - Step 0: Loss=1.9234, SHD=4.87, Precision=0.312
INFO - Step 10: Loss=1.2456, SHD=3.21, Precision=0.456
...
INFO - Demo completed successfully!
```

And in WandB, you'll see metrics like:
- `loss`, `shd`, `precision`, `recall`, `f1`
- `intervention_efficiency`
- `learning_rate`, `batch_size`
- Full configuration logged for reproducibility

## Next Steps

To integrate this into your actual training:

1. **Add to existing scripts**: Import and use the enhanced `GRPOTrainingManager`
2. **Create project-specific configs**: Add configs for your specific experiments
3. **Set up WandB sweeps**: Use WandB's sweep functionality with Hydra configs
4. **Cluster integration**: Add Hydra configs to your cluster deployment scripts

## Configuration Examples

### High-Performance Training
```bash
python scripts/your_training_script.py \
  training=grpo_high_performance \
  training.algorithm.batch_size=256 \
  logging.wandb.tags=[cluster,high_performance]
```

### Development Testing
```bash
python scripts/your_training_script.py \
  logging=local_dev \
  max_steps=100 \
  experiment.problem.difficulty=easy
```

### Systematic Exploration
```bash
python scripts/your_training_script.py --multirun \
  experiment.problem.difficulty=easy,medium,hard \
  training.algorithm.learning_rate=0.0001,0.001,0.01
```

## Notes

- The demo simulates training metrics - replace with your actual training loop
- WandB configuration is optional - the system works without it
- Hydra outputs are saved to `outputs/` directory by default
- All existing code continues to work unchanged
