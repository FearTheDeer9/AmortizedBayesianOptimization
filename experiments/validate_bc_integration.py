#!/usr/bin/env python3
"""
Validation script for BC notebook integration with unified evaluation framework.

This script checks that all components are properly integrated without running
the full training pipeline.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check that all required modules can be imported."""
    logger.info("🔍 Checking imports...")
    
    required_imports = [
        ("Unified evaluation runner", "src.causal_bayes_opt.evaluation.unified_runner", "UnifiedEvaluationRunner"),
        ("BC evaluator", "src.causal_bayes_opt.evaluation.bc_evaluator", "BCEvaluator"),
        ("Baseline evaluators", "src.causal_bayes_opt.evaluation.baseline_evaluators", "RandomBaselineEvaluator"),
        ("Result types", "src.causal_bayes_opt.evaluation.result_types", "ComparisonResults"),
        ("Notebook helpers", "src.causal_bayes_opt.evaluation.notebook_helpers", "setup_evaluation_runner"),
        ("Variable SCM factory", "src.causal_bayes_opt.experiments.variable_scm_factory", "create_variable_scm_factory"),
        ("BC data pipeline", "src.causal_bayes_opt.training.bc_data_pipeline", "process_all_demonstrations"),
    ]
    
    all_passed = True
    for name, module_path, class_name in required_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            logger.info(f"  ✅ {name}: {cls}")
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")
            all_passed = False
    
    return all_passed


def check_bc_evaluator():
    """Check BC evaluator functionality."""
    logger.info("\n🔍 Checking BC evaluator...")
    
    try:
        from src.causal_bayes_opt.evaluation.bc_evaluator import BCEvaluator
        
        # Test initialization without checkpoints (should work)
        bc_eval = BCEvaluator(name="Test_BC_Empty")
        logger.info("  ✅ BC evaluator initialized without checkpoints")
        
        # Check methods
        assert hasattr(bc_eval, 'evaluate_single_run'), "Missing evaluate_single_run method"
        assert hasattr(bc_eval, 'initialize'), "Missing initialize method"
        logger.info("  ✅ BC evaluator has required methods")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ BC evaluator check failed: {e}")
        return False


def check_variable_scm_factory():
    """Check Variable SCM Factory functionality."""
    logger.info("\n🔍 Checking Variable SCM Factory...")
    
    try:
        from src.causal_bayes_opt.experiments.variable_scm_factory import create_variable_scm_factory
        from src.causal_bayes_opt.data_structures.scm import get_target, get_variables, get_metadata
        
        # Create factory
        factory = create_variable_scm_factory(
            n_vars_list=[3, 4],
            edge_prob=0.5,
            coeff_range=(-2.0, 2.0),
            noise_scale=0.5,
            n_scms_per_config=1,
            optimize_target="MINIMIZE",
            random_seed=42
        )
        logger.info("  ✅ Factory created successfully")
        
        # Test SCM generation
        scm = factory.create_variable_scm(n_vars=4, graph_type='fork', target_idx=-1)
        
        # Verify SCM structure
        target = get_target(scm)
        vars = list(get_variables(scm))
        metadata = get_metadata(scm)
        
        assert len(vars) == 4, f"Expected 4 variables, got {len(vars)}"
        assert target in vars, f"Target {target} not in variables"
        assert metadata.get('optimization_direction') == 'MINIMIZE', "Wrong optimization direction"
        
        logger.info(f"  ✅ Generated valid SCM: {len(vars)} vars, target={target}")
        logger.info(f"  ✅ Optimization direction: {metadata.get('optimization_direction')}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Variable SCM Factory check failed: {e}")
        return False


def check_unified_evaluation_setup():
    """Check unified evaluation framework setup."""
    logger.info("\n🔍 Checking unified evaluation setup...")
    
    try:
        from src.causal_bayes_opt.evaluation.notebook_helpers import setup_evaluation_runner
        from src.causal_bayes_opt.evaluation.unified_runner import UnifiedEvaluationRunner
        
        # Test runner creation with baselines only (no checkpoints needed)
        runner = setup_evaluation_runner(
            methods=['random', 'oracle', 'learning'],
            parallel=False,
            output_dir=Path('test_output')
        )
        
        assert isinstance(runner, UnifiedEvaluationRunner), "Wrong runner type"
        
        # Check registered methods
        methods = runner.registry.list_methods()
        logger.info(f"  ✅ Created runner with methods: {methods}")
        
        # Verify each method
        for method in methods:
            evaluator = runner.registry.get_method(method)
            logger.info(f"  ✅ {method}: {type(evaluator).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Unified evaluation setup failed: {e}")
        return False


def check_result_types():
    """Check result type definitions."""
    logger.info("\n🔍 Checking result types...")
    
    try:
        from src.causal_bayes_opt.evaluation.result_types import (
            ExperimentResult, StepResult, ComparisonResults, MethodMetrics
        )
        
        # Test creating instances
        step = StepResult(
            step=0,
            intervention={},
            outcome_value=1.0,
            marginals={},
            uncertainty=1.0,
            reward=0.0,
            computation_time=0.1
        )
        logger.info("  ✅ StepResult created successfully")
        
        exp_result = ExperimentResult(
            learning_history=[step],
            final_metrics={'improvement': 0.0},
            metadata={'method': 'test'},
            success=True,
            total_time=1.0
        )
        logger.info("  ✅ ExperimentResult created successfully")
        
        method_metrics = MethodMetrics(
            mean_improvement=0.0,
            std_improvement=0.0,
            mean_final_value=1.0,
            std_final_value=0.0,
            success_rate=1.0,
            mean_time=1.0,
            n_successful=1,
            n_runs=1
        )
        logger.info("  ✅ MethodMetrics created successfully")
        
        comp_results = ComparisonResults(
            method_metrics={'test': method_metrics},
            raw_results={'test': [exp_result]},
            statistical_tests={},
            config={'test': True}
        )
        logger.info("  ✅ ComparisonResults created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Result types check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("BC NOTEBOOK INTEGRATION VALIDATION")
    logger.info("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("BC Evaluator", check_bc_evaluator),
        ("Variable SCM Factory", check_variable_scm_factory),
        ("Unified Evaluation Setup", check_unified_evaluation_setup),
        ("Result Types", check_result_types),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All validation checks passed!")
        logger.info("The BC notebook is ready to run with unified evaluation.")
    else:
        logger.info("\n❌ Some validation checks failed.")
        logger.info("Please fix the issues before running the notebook.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)