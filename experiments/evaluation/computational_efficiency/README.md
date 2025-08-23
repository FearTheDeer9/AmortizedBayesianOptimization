# Computational Efficiency Experiment

This experiment validates the primary motivation of our approach: scaling to large causal graphs through amortized inference. It demonstrates orders of magnitude speedup compared to exact methods while maintaining reasonable accuracy.

## Overview

The experiment implements three core efficiency analyses:

### Experiment 1.1: Inference Time Scaling
- **Objective**: Compare inference time vs number of variables (10 to 200 nodes)
- **Methods**: Our method (constant), CBO-U (exponential), Random (constant)
- **Expected**: Orders of magnitude speedup for n > 30
- **Contribution**: Validates feasibility claim for large-scale deployment

### Experiment 1.2: Memory Usage Scaling  
- **Objective**: Measure peak memory consumption during inference
- **Expected**: CBO-U fails/OOM at ~30-50 nodes, our method scales linearly O(d) vs O(2^d)
- **Contribution**: Demonstrates practical scalability limits

### Experiment 1.3: Training Time Analysis
- **Objective**: Amortization benefit analysis - one-time training vs per-problem CBO-U cost
- **Analysis**: Break-even point when training investment pays off
- **Contribution**: Justifies amortized approach for repeated problems

## Directory Structure

```
computational_efficiency/
├── configs/
│   ├── scaling_config.yaml        # Main scaling experiment
│   ├── memory_config.yaml         # Memory-focused testing
│   └── quick_test_config.yaml     # Fast validation
├── src/
│   ├── scaling_analyzer.py        # Core scaling analysis logic
│   ├── memory_profiler.py         # Memory usage profiling
│   └── efficiency_metrics.py      # Efficiency calculation utilities
├── scripts/
│   ├── run_scaling_experiment.py  # Main entry point
│   ├── analyze_efficiency.py      # Post-processing analysis
│   └── compare_methods.py         # Method comparison utilities
└── results/                       # Output directory (created at runtime)
```

## Usage

### Basic Scaling Analysis
```bash
cd experiments/evaluation/computational_efficiency
python scripts/run_scaling_experiment.py
```

### Memory-Focused Analysis
```bash
python scripts/run_scaling_experiment.py --config configs/memory_config.yaml
```

### With Specific Models
```bash
python scripts/run_scaling_experiment.py \
  --policy-checkpoint ../../joint-training/checkpoints/joint_ep2/policy.pkl \
  --compare-exact  # Include CBO-U comparison where feasible
```

### Quick Validation
```bash
python scripts/run_scaling_experiment.py --config configs/quick_test_config.yaml
```

## Expected Results

### Inference Time Scaling
- **Our method**: ~Constant time (slight linear growth due to tensor size)
- **CBO-U**: Exponential growth, becomes infeasible around 30-50 nodes
- **Random**: Constant time (baseline)

### Memory Usage
- **Our method**: Linear growth O(n) 
- **CBO-U**: Exponential growth O(2^n), OOM around 30-50 nodes
- **Random**: Constant memory

### Training Amortization
- **Break-even point**: Training cost recovered after ~10-20 optimization problems
- **Long-term benefit**: 100x+ speedup for repeated use

## Output Files

### Performance Data
- `scaling_results.csv`: Raw timing and memory data
- `complexity_analysis.json`: Estimated complexity classes
- `performance_comparison.csv`: Method-to-method comparison

### Visualizations  
- `plots/inference_time_scaling.png`: Time vs graph size
- `plots/memory_usage_scaling.png`: Memory vs graph size
- `plots/amortization_analysis.png`: Training cost break-even
- `plots/efficiency_summary.png`: Multi-metric comparison

### Analysis Reports
- `efficiency_report.txt`: Human-readable summary
- `complexity_estimates.txt`: Computational complexity analysis
- `scaling_limits.json`: Maximum feasible sizes per method

## Interpretation Guide

### Success Criteria
1. **Constant-time inference**: Our method should show minimal time increase with graph size
2. **Linear memory scaling**: Memory usage should grow proportionally to graph size
3. **Feasibility advantage**: Should handle 100+ node graphs where CBO-U fails
4. **Amortization benefit**: Training cost should be recovered within reasonable problem count

### Key Metrics
- **Inference time ratio**: Our_time / CBO-U_time (should be >> 1 for large graphs)
- **Memory efficiency**: Peak_memory / Graph_size (should be linear for our method)
- **Scaling exponent**: Time complexity exponent (should be ~1.0 for our method, >2.0 for CBO-U)
- **Maximum feasible size**: Largest graph each method can handle

## Troubleshooting

### Common Issues
1. **CBO-U timeouts**: Expected for large graphs, validates our efficiency claim
2. **Memory errors**: May need to adjust batch sizes or reduce test sizes
3. **GPU memory**: Ensure CUDA is available for fair comparison

### Performance Tips
- Run on dedicated machine for consistent timing
- Use multiple repetitions for statistical significance
- Monitor system resources during execution
- Consider thermal throttling effects on long runs