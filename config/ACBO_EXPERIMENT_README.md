# ACBO Active Learning vs Surrogate Experiment with WandB

This directory contains a **complete experiment setup** that compares active learning vs surrogate-based approaches for causal discovery, with full **WandB experiment tracking**.

## 🎯 What This Experiment Does

**Compares multiple approaches:**
- **Random Policy**: Selects interventions randomly (baseline)
- **Greedy Policy**: Uses simple heuristics to select interventions  
- **Oracle Policy**: Has perfect knowledge of true causal structure (upper bound)
- [Ready for your ACBO methods]: Active learning vs surrogate approaches

**Tracks causal discovery metrics:**
- **SHD (Structural Hamming Distance)**: Graph structure accuracy
- **Precision/Recall/F1**: Parent set prediction accuracy
- **Target optimization**: How well the intervention policy optimizes the target
- **Intervention efficiency**: How quickly methods converge to truth

**Provides comprehensive analysis:**
- Statistical significance testing (t-tests, effect sizes)
- Learning curves with confidence intervals
- Publication-ready visualizations
- Full WandB experiment tracking with plots and metrics

## 🚀 Quick Start

### 1. **Run the Complete Experiment**
```bash
python scripts/acbo_wandb_experiment.py
```

This will:
- Generate test SCMs with 5 variables
- Run 3 independent trials per method
- Compare Random vs Greedy vs Oracle policies
- Track all metrics in WandB
- Generate comparison plots
- Save results locally

### 2. **Customize the Experiment**
```bash
# Test with larger graphs
python scripts/acbo_wandb_experiment.py experiment.environment.num_variables=10

# More runs for better statistics  
python scripts/acbo_wandb_experiment.py n_runs=5

# Different intervention budget
python scripts/acbo_wandb_experiment.py experiment.target.max_interventions=30

# Custom WandB project
python scripts/acbo_wandb_experiment.py logging.wandb.project="my_causal_research"
```

### 3. **Hyperparameter Sweeps**
```bash
python scripts/acbo_wandb_experiment.py --multirun \\
  experiment.environment.num_variables=5,10,15 \\
  experiment.problem.edge_density=0.2,0.4,0.6 \\
  n_runs=3
```

This creates **9 separate experiments** (3×3 combinations) automatically tracked in WandB!

## 📊 What You'll See in WandB

### **Real-time Metrics**
- `{Method}/target_value`: Target variable optimization over time
- `{Method}/true_parent_likelihood`: Convergence to true causal structure  
- `{Method}/f1_score`: Structure recovery accuracy
- `{Method}/uncertainty`: Model confidence

### **Summary Statistics**
- `summary/{Method}_target_improvement_mean/std`: Final performance statistics
- `summary/{Method}_sample_efficiency_mean/std`: Learning efficiency
- `comparison/{Method1}_vs_{Method2}_p_value`: Statistical significance

### **Learning Curves**
- `curves/{Method}_likelihood_mean/std`: Aggregated learning curves with confidence intervals

### **Visualizations**
- Method comparison plots with error bars
- Intervention efficiency boxplots
- Calibration curves (when applicable)
- Complete experiment dashboard

## 📁 Generated Outputs

### **Local Files**
```
experiment_results/
├── acbo_comparison_TIMESTAMP.json    # Complete results
└── experiment_plots/
    ├── method_comparison.png          # Learning curves comparison
    ├── intervention_efficiency.png    # Efficiency analysis
    └── dashboard.png                  # Complete summary

outputs/acbo_comparison/              # Hydra outputs
└── YYYY-MM-DD/HH-MM-SS/
    ├── .hydra/
    └── acbo_wandb_experiment.log
```

### **WandB Artifacts**
- All plots automatically uploaded as artifacts
- Full configuration logged for reproducibility
- Statistical test results
- Method performance summaries

## 🔧 Integration with Your ACBO Methods

To add your actual ACBO active learning and surrogate methods:

### **1. Add your methods to the comparison:**
```python
# In scripts/acbo_wandb_experiment.py, update methods_to_compare:
methods_to_compare = {
    "Random Policy": BaselineType.RANDOM_POLICY,
    "Greedy Policy": BaselineType.GREEDY_POLICY, 
    "Oracle Policy": BaselineType.ORACLE_POLICY,
    
    # Add your methods here:
    "ACBO Active Learning": "acbo_active_learning",
    "ACBO Surrogate": "acbo_surrogate_model",
}
```

### **2. Add method execution logic:**
```python
# In run_single_experiment(), add cases for your methods:
if method_type == "acbo_active_learning":
    result = run_acbo_active_learning_experiment(scm, cfg, run_idx, scm_idx)
elif method_type == "acbo_surrogate_model":
    result = run_acbo_surrogate_experiment(scm, cfg, run_idx, scm_idx)
```

### **3. Ensure results follow the expected format:**
```python
def run_acbo_active_learning_experiment(scm, cfg, run_idx, scm_idx):
    # Your ACBO implementation here
    
    return {
        'method_name': 'acbo_active_learning',
        'final_target_value': final_value,
        'target_improvement': improvement,
        'structure_accuracy': accuracy,  
        'intervention_count': num_interventions,
        'detailed_results': {
            'learning_history': [
                {
                    'step': step,
                    'outcome_value': value,
                    'target_improvement': improvement,
                    'uncertainty': uncertainty,
                    'marginals': parent_probabilities  # Key for causal discovery metrics
                }
                # ... for each step
            ]
        }
    }
```

## 🎛️ Configuration Options

### **Experiment Settings**
```yaml
# config/experiment/acbo_comparison.yaml
environment:
  num_variables: 5        # Graph size
  noise_scale: 0.1        # SCM noise level
  
target:
  max_interventions: 20   # Budget per experiment

problem:
  edge_density: 0.3       # Graph sparsity
  difficulty: "medium"    # Problem complexity

n_runs: 3                 # Independent trials per method
n_scms: 2                 # Different SCMs to test
```

### **WandB Settings**
```yaml
# config/logging/wandb_enabled.yaml
wandb:
  project: "acbo_comparison"
  tags: ["acbo", "comparison"] 
  group: "active_vs_surrogate"
```

## 📈 Understanding the Results

### **Key Metrics to Watch**
1. **Target Improvement**: How much did each method improve the target variable?
2. **True Parent Likelihood**: How quickly did methods discover the true causal structure?
3. **Sample Efficiency**: How much improvement per intervention?
4. **Statistical Significance**: Are the differences between methods real?

### **What Good Results Look Like**
- **ACBO Active Learning** should outperform random/greedy baselines
- **Oracle Policy** provides the theoretical upper bound
- **Statistical significance** (p < 0.05) shows reliable differences
- **Confidence intervals** that don't overlap indicate clear winners

### **Interpreting WandB Plots**
- **Learning curves trending up**: Methods are learning successfully
- **Tight confidence intervals**: Consistent performance across runs  
- **Early convergence**: Sample-efficient methods
- **Higher final values**: Better ultimate performance

## 🔬 Statistical Analysis

The experiment automatically computes:
- **t-tests** between all method pairs
- **Effect sizes** (Cohen's d) to measure practical significance
- **Confidence intervals** for all metrics
- **Multiple comparison corrections** when testing many methods

## 🚨 Troubleshooting

### **Common Issues**
1. **"wandb not available"**: Install with `pip install wandb` and run `wandb login`
2. **"scipy not available"**: Install with `pip install scipy` for statistical tests
3. **Memory issues**: Reduce `n_runs` or `experiment.environment.num_variables`
4. **Import errors**: Ensure all dependencies in `pyproject.toml` are installed

### **Debug Mode**
```bash
# Run single experiment for debugging
python scripts/acbo_wandb_experiment.py n_runs=1 n_scms=1 experiment.target.max_interventions=5

# Disable WandB for local testing
python scripts/acbo_wandb_experiment.py logging.wandb.enabled=false
```

## 🎯 Next Steps

1. **Run the baseline experiment** to test your setup
2. **Add your ACBO methods** to the comparison
3. **Scale up** with larger graphs and more runs
4. **Share results** using WandB project links with collaborators
5. **Publish findings** using the generated visualizations

This experiment framework provides everything you need for **rigorous comparison** of causal discovery methods with **publication-ready results**! 🚀
