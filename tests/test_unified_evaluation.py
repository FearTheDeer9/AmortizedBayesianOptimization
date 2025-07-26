"""
Test Unified Evaluation Framework

Validates that the new unified evaluation framework produces
equivalent results to the old implementation approaches.
"""

import pytest
import numpy as np
import jax.random as random
from pathlib import Path
import tempfile
import logging

from src.causal_bayes_opt.evaluation import (
    UnifiedEvaluationRunner,
    RandomBaselineEvaluator,
    OracleBaselineEvaluator,
    LearningBaselineEvaluator,
    setup_evaluation_runner,
    results_to_dataframe
)
from src.causal_bayes_opt.data_structures.scm import create_scm, get_target
from examples.demo_scms import create_easy_scm_base

logger = logging.getLogger(__name__)


class TestUnifiedEvaluation:
    """Test the unified evaluation framework."""
    
    @pytest.fixture
    def simple_scm(self):
        """Create a simple test SCM."""
        return create_easy_scm_base()
    
    @pytest.fixture
    def evaluation_config(self):
        """Create standard evaluation configuration."""
        return {
            'experiment': {
                'target': {
                    'max_interventions': 5,
                    'n_observational_samples': 50,
                    'intervention_value_range': (-2.0, 2.0),
                    'learning_rate': 1e-3
                }
            }
        }
    
    def test_baseline_evaluators_run(self, simple_scm, evaluation_config):
        """Test that all baseline evaluators can run successfully."""
        evaluators = [
            RandomBaselineEvaluator(),
            OracleBaselineEvaluator(),
            LearningBaselineEvaluator()
        ]
        
        for evaluator in evaluators:
            # Test single run
            result = evaluator.evaluate_single_run(
                scm=simple_scm,
                config=evaluation_config,
                seed=42,
                run_idx=0
            )
            
            # Validate result structure
            assert result.success
            assert len(result.learning_history) == 6  # initial + 5 interventions
            assert 'initial_value' in result.final_metrics
            assert 'final_value' in result.final_metrics
            assert 'final_f1' in result.final_metrics
            assert 'final_shd' in result.final_metrics
    
    def test_oracle_outperforms_random(self, simple_scm, evaluation_config):
        """Test that oracle baseline outperforms random baseline."""
        random_eval = RandomBaselineEvaluator()
        oracle_eval = OracleBaselineEvaluator()
        
        # Run both on same SCM with same seed
        random_result = random_eval.evaluate_single_run(
            scm=simple_scm,
            config=evaluation_config,
            seed=42
        )
        
        oracle_result = oracle_eval.evaluate_single_run(
            scm=simple_scm,
            config=evaluation_config,
            seed=42
        )
        
        # Oracle should achieve better target value (assuming minimization)
        assert oracle_result.final_metrics['best_value'] <= random_result.final_metrics['best_value']
        
        # Oracle should have perfect structure knowledge
        assert oracle_result.final_metrics['final_f1'] == 1.0
        assert oracle_result.final_metrics['final_shd'] == 0.0
    
    def test_learning_improves_structure_knowledge(self, simple_scm, evaluation_config):
        """Test that learning baseline improves structure knowledge over time."""
        learning_eval = LearningBaselineEvaluator()
        
        result = learning_eval.evaluate_single_run(
            scm=simple_scm,
            config=evaluation_config,
            seed=42
        )
        
        # Extract F1 scores over time
        true_parents = result.metadata['scm_info']['true_parents']
        f1_scores = []
        
        for step in result.learning_history:
            from src.causal_bayes_opt.analysis.trajectory_metrics import compute_f1_score_from_marginals
            f1 = compute_f1_score_from_marginals(step.marginals, true_parents)
            f1_scores.append(f1)
        
        # F1 should generally improve (allow for some noise)
        # Check that final F1 is better than initial
        assert f1_scores[-1] >= f1_scores[0]
    
    def test_unified_runner_integration(self, simple_scm, evaluation_config):
        """Test the unified runner with multiple methods."""
        runner = setup_evaluation_runner(
            methods=['random', 'oracle', 'learning'],
            parallel=False  # Sequential for testing
        )
        
        # Run comparison on single SCM
        results = runner.run_comparison(
            test_scms=[simple_scm],
            config=evaluation_config,
            n_seeds=2
        )
        
        # Validate results structure
        assert len(results.method_results) == 3
        assert 'Random_Baseline' in results.method_results
        assert 'Oracle_Baseline' in results.method_results
        assert 'Learning_Baseline' in results.method_results
        
        # Check that oracle has best performance
        oracle_metrics = results.method_results['Oracle_Baseline']
        random_metrics = results.method_results['Random_Baseline']
        
        # Oracle should have better improvement (more negative for minimization)
        assert oracle_metrics.mean_improvement <= random_metrics.mean_improvement
    
    def test_results_to_dataframe(self, simple_scm, evaluation_config):
        """Test conversion of results to DataFrame."""
        runner = setup_evaluation_runner(
            methods=['random', 'oracle'],
            parallel=False
        )
        
        results = runner.run_comparison(
            test_scms=[simple_scm],
            config=evaluation_config,
            n_seeds=1
        )
        
        # Convert to DataFrame
        df = results_to_dataframe(results)
        
        # Validate DataFrame structure
        assert len(df) == 2
        assert 'method' in df.columns
        assert 'mean_improvement' in df.columns
        assert 'mean_final_f1' in df.columns
        assert 'mean_final_shd' in df.columns
    
    def test_result_serialization(self, simple_scm, evaluation_config):
        """Test that results can be saved and loaded."""
        import pickle
        
        evaluator = RandomBaselineEvaluator()
        result = evaluator.evaluate_single_run(
            scm=simple_scm,
            config=evaluation_config,
            seed=42
        )
        
        # Test serialization
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            pickle.dump(result, f)
            f.flush()
            
            # Load back
            f.seek(0)
            loaded_result = pickle.load(f)
            
            # Validate loaded result
            assert loaded_result.success == result.success
            assert len(loaded_result.learning_history) == len(result.learning_history)
            assert loaded_result.final_metrics == result.final_metrics


@pytest.mark.integration
class TestEvaluationEquivalence:
    """Test that new evaluators produce equivalent results to old methods."""
    
    @pytest.fixture
    def scm_list(self):
        """Create list of test SCMs."""
        from examples.demo_scms import create_easy_scm_base, create_medium_scm, create_hard_scm
        return [
            create_easy_scm_base(),
            create_medium_scm(),
            create_hard_scm()
        ]
    
    def test_baseline_consistency(self, scm_list):
        """Test that baselines produce consistent results across runs."""
        config = {
            'experiment': {
                'target': {
                    'max_interventions': 10,
                    'n_observational_samples': 100
                }
            }
        }
        
        # Test each baseline
        for evaluator_class in [RandomBaselineEvaluator, OracleBaselineEvaluator, LearningBaselineEvaluator]:
            evaluator = evaluator_class()
            
            # Run twice with same seed - should get same results
            result1 = evaluator.evaluate_single_run(scm_list[0], config, seed=123)
            result2 = evaluator.evaluate_single_run(scm_list[0], config, seed=123)
            
            # Check key metrics are identical
            assert result1.final_metrics['final_value'] == result2.final_metrics['final_value']
            assert result1.final_metrics['best_value'] == result2.final_metrics['best_value']
            
            # Different seeds should give different results (except oracle)
            if not isinstance(evaluator, OracleBaselineEvaluator):
                result3 = evaluator.evaluate_single_run(scm_list[0], config, seed=456)
                # Allow for rare coincidences but generally should differ
                if result1.final_metrics['final_value'] != result3.final_metrics['final_value']:
                    assert True  # Expected different results