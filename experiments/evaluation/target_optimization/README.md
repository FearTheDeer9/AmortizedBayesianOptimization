# Target Optimization Experiment

This experiment evaluates core optimization performance by tracking target variable trajectories across different methods and graph sizes. This is **Experiment 2.4** from the research plan.

## Overview

**Experiment 2.4: Target Value Trajectories**
- **Objective**: Plot target node value over intervention sequence
- **Scope**: Multiple trajectories for different graph sizes (10, 20, 50, 100)  
- **Comparison**: Our method vs CBO-U vs random vs oracle
- **Metrics**: Convergence rate and final performance
- **Contribution**: Core optimization performance across scales

## Key Features

### Dynamic Model Pairing
- Test arbitrary combinations of policy and surrogate models
- Compare trained vs untrained models
- Evaluate impact of different model architectures

### Comprehensive Trajectory Analysis
- Track target value evolution over interventions
- Measure convergence rates and final performance
- Analyze parent selection accuracy over time
- Compare optimization strategies across methods

### Multi-Scale Evaluation
- Test on graphs of increasing size (5 to 100+ variables)
- Evaluate graceful degradation vs catastrophic failure
- Identify scaling limits for different approaches

## Directory Structure

```
target_optimization/
├── configs/
│   ├── trajectory_config.yaml      # Main trajectory experiment
│   ├── scaling_config.yaml         # Multi-size evaluation
│   └── quick_test_config.yaml      # Fast validation
├── src/
│   ├── trajectory_analyzer.py      # Core trajectory analysis
│   ├── optimization_metrics.py     # Optimization-specific metrics
│   └── convergence_analysis.py     # Convergence rate computation
├── scripts/
│   ├── run_trajectory_experiment.py # Main entry point
│   ├── analyze_convergence.py      # Post-processing analysis
│   └── compare_optimization.py     # Method comparison
└── results/                        # Output directory
```

## Usage

### Basic Trajectory Analysis
```bash
cd experiments/evaluation/target_optimization
python scripts/run_trajectory_experiment.py
```

### With Specific Model Combinations
```bash
python scripts/run_trajectory_experiment.py \
  --pairing-config configs/custom_pairings.yaml \
  --include-baselines
```

### Multi-Scale Analysis
```bash
python scripts/run_trajectory_experiment.py \
  --config configs/scaling_config.yaml \
  --track-convergence
```

### Comprehensive Comparison
```bash
python scripts/run_trajectory_experiment.py \
  --all-available-models \
  --generate-summary-plots
```

## Configuration

### Model Pairings
Define custom model combinations:
```yaml
pairings:
  - name: "Joint Trained"
    policy_checkpoint: "../../joint-training/checkpoints/joint_ep2/policy.pkl"
    surrogate_checkpoint: "../../joint-training/checkpoints/joint_ep2/surrogate.pkl"
    
  - name: "Trained Policy + Untrained Surrogate"
    policy_checkpoint: "../../policy-only-training/checkpoints/.../policy.pkl"
    surrogate_type: "untrained"
    
  - name: "Pure Untrained"
    policy_type: "untrained"
    surrogate_type: "untrained"
```

### Experiment Settings
```yaml
evaluation:
  graph_sizes: [10, 20, 50, 100]
  n_graphs_per_size: 10
  n_interventions: 50
  track_parent_accuracy: true
  
metrics:
  - "final_target_value"
  - "best_target_value" 
  - "convergence_rate"
  - "parent_selection_accuracy"
  - "exploration_diversity"
```

## Expected Results

### Convergence Patterns
- **Trained models**: Fast convergence to good solutions
- **Untrained models**: Slower, more exploratory behavior
- **Oracle**: Optimal convergence (upper bound)
- **Random**: No systematic convergence

### Scaling Behavior
- **Small graphs (≤20)**: All methods should perform reasonably
- **Medium graphs (20-50)**: Trained models show advantage
- **Large graphs (50+)**: Clear separation between methods

### Performance Metrics
- **Final target value**: How good is the final solution?
- **Best target value**: What's the best solution found?
- **Convergence rate**: How quickly do methods improve?
- **Sample efficiency**: Interventions needed to reach thresholds

## Output Files

### Trajectory Data
- `trajectory_results.csv`: Complete intervention sequences
- `convergence_metrics.csv`: Convergence analysis per method
- `parent_accuracy.csv`: Parent selection tracking

### Visualizations
- `plots/trajectories_by_size.png`: Trajectories grouped by graph size
- `plots/convergence_comparison.png`: Convergence rate comparison
- `plots/final_performance.png`: Final performance comparison
- `plots/parent_accuracy_evolution.png`: Parent selection over time

### Analysis Reports
- `optimization_report.txt`: Human-readable summary
- `convergence_analysis.json`: Detailed convergence statistics
- `method_rankings.csv`: Relative method performance

## Key Insights

This experiment should demonstrate:

1. **Optimization Capability**: Neural methods can effectively optimize targets
2. **Learning Evidence**: Trained models outperform untrained initialization
3. **Scaling Robustness**: Performance degrades gracefully with graph size
4. **Parent Discovery**: Methods learn to identify relevant variables
5. **Sample Efficiency**: Faster convergence than random baseline

## Success Criteria

- Trained models achieve significantly better final values than random
- Convergence rate improves with training
- Parent selection accuracy increases over interventions
- Performance scales reasonably to larger graphs
- Clear separation between trained/untrained/baseline methods