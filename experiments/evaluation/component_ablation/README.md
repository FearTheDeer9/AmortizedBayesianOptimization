# Component Ablation Experiment

This experiment validates each design choice through systematic component removal and comparison. This implements **Experiments 3.1, 3.2, and 3.3** from the research plan.

## Overview

### Experiment 3.1: Component Ablation Table
- **Objective**: Compare full model vs variants with removed components
- **Variants**:
  - Without GRPO (vanilla policy gradient)
  - Without behavioral cloning initialization  
  - Without joint training (independent training)
  - Without alternating attention (standard transformer)
  - Random policy with learned posterior
- **Metrics**: Optimization regret, Parent F1, Inference time
- **Contribution**: Justifies each architectural choice

### Experiment 3.2: Reward Component Analysis
- **Objective**: Compare different reward configurations in GRPO
- **Variants**:
  - Optimization only: R_opt
  - Structure only: R_struct  
  - Combined: α*R_opt + β*R_struct
  - With parent supervision: +γ*R_parent
- **Analysis**: Convergence of posterior accuracy, target trajectories
- **Contribution**: Demonstrates multi-objective balance importance

### Experiment 3.3: Training Paradigm Comparison  
- **Objective**: Compare training approaches
- **Variants**:
  - Joint training vs independent training
  - Different phase lengths in alternating training
  - With/without replay buffer
- **Contribution**: Validates coupled training necessity

## Features

### Systematic Ablation Framework
- **Component isolation**: Remove one component at a time
- **Fair comparison**: Same training data and evaluation protocol
- **Multiple metrics**: Optimization + structure learning + efficiency

### Reward Configuration Testing
- **Flexible reward weighting**: Test different α, β, γ values
- **Component isolation**: Pure optimization vs pure structure learning
- **Multi-objective analysis**: Trade-off visualization

### Training Paradigm Analysis
- **Joint vs independent**: Simultaneous vs separate training
- **Phase length sensitivity**: Optimal alternation frequency
- **Replay buffer impact**: Memory vs performance trade-offs

## Usage

### Full Ablation Study
```bash
cd experiments/evaluation/component_ablation
python scripts/run_ablation_experiment.py
```

### Specific Component Test
```bash
python scripts/run_ablation_experiment.py \
  --component grpo \
  --baseline vanilla_pg
```

### Reward Analysis Only
```bash
python scripts/run_ablation_experiment.py \
  --experiment reward_analysis \
  --reward-configs configs/reward_configurations.yaml
```

### Training Paradigm Comparison
```bash
python scripts/run_ablation_experiment.py \
  --experiment training_paradigms \
  --include-joint-training
```

## Expected Results

### Component Importance Ranking
1. **GRPO vs Vanilla PG**: GRPO should show better exploration/exploitation balance
2. **Joint vs Independent**: Joint training should improve coordination
3. **BC Initialization**: Should provide better starting point than random
4. **Alternating Attention**: Should improve parent identification

### Reward Configuration Analysis
- **Pure optimization** (R_opt only): Good target values, poor structure learning
- **Pure structure** (R_struct only): Good parent identification, slower optimization  
- **Balanced combination**: Best overall performance
- **Parent supervision**: Additional improvement in structure accuracy

### Training Paradigm Insights
- **Joint training**: Better coordination between policy and surrogate
- **Optimal phase length**: Balance between stability and adaptation
- **Replay buffer**: Improves sample efficiency but increases memory

## Success Criteria

1. **Clear component ranking**: Each component should show measurable benefit
2. **Reward balance validation**: Combined rewards outperform single objectives
3. **Training paradigm justification**: Joint training shows clear advantages
4. **Statistical significance**: Differences should be statistically meaningful

## Troubleshooting

### Component Implementation Issues
- Some ablated components may require architectural changes
- Fallback to simplified implementations when needed
- Document any implementation limitations

### Training Time Considerations  
- Ablation study requires training multiple model variants
- Use smaller graphs/shorter training for rapid iteration
- Scale up to full evaluation once framework is validated