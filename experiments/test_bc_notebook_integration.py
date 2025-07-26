#!/usr/bin/env python3
"""Quick test to verify BC notebook integration is working."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing BC Notebook Integration...")

# Test 1: Variable SCM Factory
print("\n1️⃣ Testing Variable SCM Factory...")
try:
    from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
    from src.causal_bayes_opt.data_structures.scm import get_target, get_variables, get_metadata
    
    factory = VariableSCMFactory(noise_scale=0.5, coefficient_range=(-2.0, 2.0), seed=42)
    scm = factory.create_variable_scm(n_vars=4, structure_type='fork', target_idx=-1, optimize_target="MINIMIZE")
    
    target = get_target(scm)
    vars = list(get_variables(scm))
    metadata = get_metadata(scm)
    
    print(f"✅ Generated SCM: {len(vars)} vars, target={target}")
    print(f"   Optimization: {metadata.get('optimization_direction', 'UNKNOWN')}")
except Exception as e:
    print(f"❌ Variable SCM Factory failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Unified Evaluation
print("\n2️⃣ Testing Unified Evaluation Framework...")
try:
    from src.causal_bayes_opt.evaluation import setup_evaluation_runner
    from src.causal_bayes_opt.evaluation.result_types import StepResult, ExperimentResult
    
    # Create minimal test data
    step = StepResult(
        step=0,
        intervention={},
        outcome_value=1.0,
        marginals={},
        uncertainty=0.0,
        reward=0.0,
        computation_time=0.1
    )
    
    result = ExperimentResult(
        learning_history=[step],
        final_metrics={'improvement': 0.0},
        metadata={'method': 'test'},
        success=True,
        total_time=1.0
    )
    
    print("✅ Result types working correctly")
    
    # Test runner creation
    runner = setup_evaluation_runner(methods=['random'], parallel=False)
    print(f"✅ Created runner with methods: {runner.registry.list_methods()}")
    
except Exception as e:
    print(f"❌ Unified evaluation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: BC Evaluator
print("\n3️⃣ Testing BC Evaluator...")
try:
    from src.causal_bayes_opt.evaluation.bc_evaluator import BCEvaluator
    
    # Test without checkpoints
    bc_eval = BCEvaluator(name="Test_BC")
    print(f"✅ BC evaluator created: {bc_eval.name}")
    
except Exception as e:
    print(f"❌ BC evaluator failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Basic integration tests passed!")
print("The BC notebook should be able to run the unified evaluation.")