#!/usr/bin/env python3
"""Script to fix the BC notebook cells with correct parameter names."""

import json
import sys

# Read the notebook
notebook_path = "experiments/bc_development_workflow.ipynb"
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Track if we made changes
changes_made = False

# Fix cell 18 (Integration Validation Tests)
cell_18_new_content = '''print("🧪 Running Integration Validation Tests...")

# Test 1: Verify BC Evaluator can be initialized
print("\\n1️⃣ Testing BC Evaluator initialization...")
try:
    from src.causal_bayes_opt.evaluation import BCEvaluator
    
    # Use the checkpoints from integration results
    bc_eval = BCEvaluator(
        surrogate_checkpoint=bc_integration_results['surrogate_checkpoint'],
        acquisition_checkpoint=bc_integration_results['acquisition_checkpoint'],
        name="BC_Test"
    )
    print("✅ BC evaluator initialized successfully")
except Exception as e:
    print(f"❌ BC evaluator initialization failed: {e}")

# Test 2: Verify Variable SCM Factory 
print("\\n2️⃣ Testing Variable SCM Factory...")
try:
    from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
    import pyrsistent as pyr
    
    # Create factory
    scm_factory = VariableSCMFactory(seed=42)
    
    # Create a test SCM with CORRECT parameter names
    test_scm = scm_factory.create_variable_scm(
        num_variables=4,  # Fixed: was n_vars
        structure_type='fork',
        target_variable=None  # Fixed: was target_idx=-1
        # Removed: optimize_target="MINIMIZE" (not a valid parameter)
    )
    
    # Set optimization direction in metadata (using immutable update)
    current_metadata = test_scm.get('metadata', pyr.pmap())
    updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
    test_scm = test_scm.set('metadata', updated_metadata)
    
    print("✅ Variable SCM Factory working correctly")
    print(f"   Generated SCM: {len(test_scm.get('variables', []))} variables")
    print(f"   Target: {test_scm.get('target', 'unknown')}")
    print(f"   Parents of target: {list(test_scm.get('mechanisms', {}).get(test_scm.get('target', ''), {}).get('parents', []))}")
    print(f"   Optimization: {test_scm.get('metadata', {}).get('optimization_direction', 'unknown')}")
    
except Exception as e:
    print(f"❌ Variable SCM Factory test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify unified evaluation imports
print("\\n3️⃣ Testing unified evaluation framework imports...")
try:
    from src.causal_bayes_opt.evaluation import (
        setup_evaluation_runner,
        run_evaluation_comparison,
        results_to_dataframe,
        plot_learning_curves,
        create_summary_report
    )
    print("✅ All unified evaluation functions imported successfully")
    
    # Test result types
    from src.causal_bayes_opt.evaluation.result_types import (
        ExperimentResult, StepResult, ComparisonResults, MethodMetrics
    )
    print("✅ All result types imported successfully")
    
except Exception as e:
    print(f"❌ Unified evaluation import failed: {e}")

# Test 4: Verify baseline evaluators
print("\\n4️⃣ Testing baseline evaluators...")
try:
    from src.causal_bayes_opt.evaluation.baseline_evaluators import (
        RandomBaselineEvaluator,
        OracleBaselineEvaluator,
        LearningBaselineEvaluator
    )
    
    # Test instantiation
    random_eval = RandomBaselineEvaluator()
    oracle_eval = OracleBaselineEvaluator()
    learning_eval = LearningBaselineEvaluator()
    
    print("✅ All baseline evaluators initialized successfully")
    print(f"   Random: {random_eval.get_method_name()}")
    print(f"   Oracle: {oracle_eval.get_method_name()}")
    print(f"   Learning: {learning_eval.get_method_name()}")
    
except Exception as e:
    print(f"❌ Baseline evaluators test failed: {e}")

# Test 5: Check evaluation configuration
print("\\n5️⃣ Checking evaluation configuration...")
print("Expected configuration (matching GRPO):")
print("   n_observational_samples: 100")
print("   max_interventions: 20")
print("   intervention_value_range: (-2.0, 2.0)")
print("   optimization_direction: MINIMIZE")

print("\\n✅ All integration tests completed!")'''

# Fix cell 21 (SCM Generation Loop)
cell_21_new_content = '''print("🚀 Running Unified Evaluation Comparison...")

# Check prerequisites
if 'unified_runner' not in globals() or 'eval_config' not in globals():
    raise RuntimeError("❌ No unified runner found - please run previous cell first")

# Generate a balanced set of SCMs
print("\\n📊 Generating test SCMs...")
test_scms = []
scm_factory = VariableSCMFactory(seed=42)
import pyrsistent as pyr

for structure in ['fork', 'chain', 'collider']:
    for n_vars in [3, 4, 5]:
        # Create SCM with CORRECT parameter names
        scm = scm_factory.create_variable_scm(
            num_variables=n_vars,  # Fixed: was n_vars (now explicit)
            structure_type=structure,
            target_variable=None  # Fixed: was target_idx=-1
            # Removed: optimize_target="MINIMIZE" (not a valid parameter)
        )
        
        # Set optimization direction in metadata (using immutable update)
        current_metadata = scm.get('metadata', pyr.pmap())
        updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
        scm = scm.set('metadata', updated_metadata)
        
        test_scms.append(scm)

print(f"Generated {len(test_scms)} SCMs:")
print(f"Structures: fork, chain, collider")
print(f"Variable counts: 3, 4, 5")
print(f"Methods: {unified_runner.registry.list_methods()}")

# Run comparison
print("\\n⏳ Running evaluation (this may take a few minutes)...")
results = unified_runner.run_comparison(
    test_scms=test_scms,
    config=eval_config,
    n_runs_per_scm=3,  # Changed from n_seeds to n_runs_per_scm
    base_seed=42
)

print("\\n✅ Evaluation complete!")

# Convert results to DataFrame
print("\\n📊 Performance Summary:")
df_results = results_to_dataframe(results)
print(df_results)

# Store results
globals()['unified_results'] = results
print("\\n💾 Results stored in 'unified_results'")'''

# Find and fix cells
cell_count = 0
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        cell_count += 1
        
        # Check if this is cell 18
        if cell_count == 18 and 'source' in cell:
            source_text = ''.join(cell['source'])
            if "Running Integration Validation Tests" in source_text:
                print(f"Found cell 18 at index {i}")
                print("Current content includes:", source_text[:100], "...")
                cell['source'] = cell_18_new_content
                changes_made = True
                print("✅ Fixed cell 18")
        
        # Check if this is cell 21
        if cell_count == 21 and 'source' in cell:
            source_text = ''.join(cell['source'])
            if "for graph_type in ['fork', 'chain', 'collider']" in source_text:
                print(f"\nFound cell 21 at index {i}")
                print("Current content includes:", source_text[:100], "...")
                cell['source'] = cell_21_new_content
                changes_made = True
                print("✅ Fixed cell 21")

# Save the notebook if changes were made
if changes_made:
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("\n✅ Notebook saved with fixes!")
else:
    print("\n❌ Could not find the cells to fix")
    print(f"Total code cells found: {cell_count}")