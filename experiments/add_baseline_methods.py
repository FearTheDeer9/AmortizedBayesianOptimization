"""
Add baseline methods for meaningful BC evaluation comparison.

This script creates baseline methods that can be added to Cell 8 of the BC notebook
to provide meaningful performance comparisons.
"""

def create_random_baseline_method():
    """Create a pure random baseline - no learning, random interventions."""
    def random_baseline(scm, config, scm_idx, seed):
        """Random baseline: random interventions with no learning."""
        import numpy as np
        import jax.random as random
        from src.causal_bayes_opt.experiments.experiments import run_causal_experiment
        from src.causal_bayes_opt.acquisition.policies import create_random_policy
        from src.causal_bayes_opt.models.parametric_scm_surrogate import create_surrogate_model
        
        # Use the actual CBO framework
        np.random.seed(seed)
        key = random.PRNGKey(seed)
        
        # Create untrained (random) components
        surrogate_model = create_surrogate_model(
            model_type='fixed',  # No learning
            prior_graph=None,
            use_prior=False
        )
        
        acquisition_fn = create_random_policy()
        
        # Run actual experiment
        results = run_causal_experiment(
            scm=scm,
            surrogate_model=surrogate_model,
            acquisition_fn=acquisition_fn,
            n_interventions=config.experiment.target.max_interventions,
            n_observational_samples=config.experiment.target.n_observational_samples,
            random_key=key
        )
        
        # Extract real performance metrics
        target_progress = results.get('target_progress', [])
        final_value = target_progress[-1] if target_progress else 0.0
        
        # Calculate actual improvement
        initial_value = target_progress[0] if target_progress else 0.0
        actual_improvement = final_value - initial_value
        
        return {
            'success': True,
            'final_value': final_value,
            'initial_value': initial_value,
            'actual_improvement': actual_improvement,
            'target_progress': target_progress,
            'method_name': 'random_baseline',
            'is_baseline': True,
            'used_bc_surrogate': False,
            'used_bc_acquisition': False
        }
    
    return random_baseline


def create_oracle_baseline_method():
    """Create an oracle baseline - knows true causal graph."""
    def oracle_baseline(scm, config, scm_idx, seed):
        """Oracle baseline: knows true graph, optimal interventions."""
        import numpy as np
        import jax.random as random
        from src.causal_bayes_opt.experiments.experiments import run_causal_experiment
        from src.causal_bayes_opt.acquisition.policies import create_oracle_policy
        from src.causal_bayes_opt.models.parametric_scm_surrogate import create_surrogate_model
        
        np.random.seed(seed)
        key = random.PRNGKey(seed)
        
        # Create oracle components (knows true graph)
        true_graph = scm.get('graph')  # Extract true causal graph
        
        surrogate_model = create_surrogate_model(
            model_type='oracle',  # Perfect knowledge
            prior_graph=true_graph,
            use_prior=True
        )
        
        acquisition_fn = create_oracle_policy(scm=scm)
        
        # Run actual experiment
        results = run_causal_experiment(
            scm=scm,
            surrogate_model=surrogate_model,
            acquisition_fn=acquisition_fn,
            n_interventions=config.experiment.target.max_interventions,
            n_observational_samples=config.experiment.target.n_observational_samples,
            random_key=key
        )
        
        # Extract real performance metrics
        target_progress = results.get('target_progress', [])
        final_value = target_progress[-1] if target_progress else 0.0
        
        # Calculate actual improvement
        initial_value = target_progress[0] if target_progress else 0.0
        actual_improvement = final_value - initial_value
        
        return {
            'success': True,
            'final_value': final_value,
            'initial_value': initial_value,
            'actual_improvement': actual_improvement,
            'target_progress': target_progress,
            'method_name': 'oracle_baseline',
            'is_baseline': True,
            'is_oracle': True,
            'used_bc_surrogate': False,
            'used_bc_acquisition': False
        }
    
    return oracle_baseline


def create_real_bc_method(surrogate_model, acquisition_model):
    """Create BC method that runs actual CBO experiments."""
    def real_bc_method(scm, config, scm_idx, seed):
        """BC method using actual trained models in CBO framework."""
        import numpy as np
        import jax.random as random
        from src.causal_bayes_opt.experiments.experiments import run_causal_experiment
        from src.causal_bayes_opt.models.bc_surrogate_wrapper import wrap_bc_surrogate
        from src.causal_bayes_opt.acquisition.bc_acquisition_wrapper import wrap_bc_acquisition
        
        np.random.seed(seed)
        key = random.PRNGKey(seed)
        
        # Wrap BC models for CBO framework
        wrapped_surrogate = wrap_bc_surrogate(surrogate_model)
        wrapped_acquisition = wrap_bc_acquisition(acquisition_model)
        
        # Run actual experiment
        results = run_causal_experiment(
            scm=scm,
            surrogate_model=wrapped_surrogate,
            acquisition_fn=wrapped_acquisition,
            n_interventions=config.experiment.target.max_interventions,
            n_observational_samples=config.experiment.target.n_observational_samples,
            random_key=key
        )
        
        # Extract real performance metrics
        target_progress = results.get('target_progress', [])
        final_value = target_progress[-1] if target_progress else 0.0
        
        # Calculate actual improvement
        initial_value = target_progress[0] if target_progress else 0.0
        actual_improvement = final_value - initial_value
        
        # Calculate improvement ratio vs random baseline
        # This will be calculated in the evaluation by comparing to random_baseline results
        
        return {
            'success': True,
            'final_value': final_value,
            'initial_value': initial_value,
            'actual_improvement': actual_improvement,
            'target_progress': target_progress,
            'method_name': 'bc_trained_real',
            'is_baseline': False,
            'used_bc_surrogate': True,
            'used_bc_acquisition': True
        }
    
    return real_bc_method


# Example code to add to Cell 8 of the notebook:
"""
# Add baseline methods for meaningful comparison
print("\\n📊 Adding Baseline Methods for Comparison...")

# 1. Random Baseline
random_baseline_method = create_random_baseline_method()
random_baseline_experiment = ExperimentMethod(
    name="Random Baseline",
    type="random_baseline",
    description="Pure random interventions with no learning",
    run_function=random_baseline_method,
    config={},
    requires_checkpoint=False
)
method_registry.register_method(random_baseline_experiment)
print("✅ Registered random_baseline method")

# 2. Oracle Baseline
oracle_baseline_method = create_oracle_baseline_method()
oracle_baseline_experiment = ExperimentMethod(
    name="Oracle Baseline",
    type="oracle_baseline",
    description="Perfect knowledge of causal graph",
    run_function=oracle_baseline_method,
    config={},
    requires_checkpoint=False
)
method_registry.register_method(oracle_baseline_experiment)
print("✅ Registered oracle_baseline method")

# Update registered methods list
registered_methods.extend(['random_baseline', 'oracle_baseline'])

print(f"\\n✅ Total methods for comparison: {len(registered_methods)}")
print(f"   Baselines: random_baseline, oracle_baseline")
print(f"   BC methods: {[m for m in registered_methods if 'bc' in m]}")
"""

# Metric interpretation guide:
"""
## Understanding the Metrics

### Absolute Performance
- **Random Baseline**: Shows performance with no causal knowledge
- **Oracle Baseline**: Shows theoretical maximum performance with perfect knowledge
- **BC Methods**: Should fall between random and oracle

### Relative Performance
- **Improvement Ratio**: (BC_improvement - Random_improvement) / (Oracle_improvement - Random_improvement)
  - 0% = No better than random
  - 100% = As good as oracle
  - 50% = Halfway between random and oracle

### What to Look For
1. **BC vs Random**: BC methods should significantly outperform random baseline
2. **BC vs Oracle**: Gap shows room for improvement
3. **BC Surrogate vs BC Acquisition**: Which component adds more value
4. **Learning Curves**: How quickly methods converge to good solutions

### Typical Real-World Values
- Random baseline: 0-20% improvement
- Good BC methods: 40-70% improvement  
- Oracle: 80-100% improvement
"""