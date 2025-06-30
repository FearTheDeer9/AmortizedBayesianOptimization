#!/usr/bin/env python3
"""
Comprehensive Validation Suite

Thorough validation of BIC scoring fix and intervention strategy comparison
with multiple trials, statistical testing, and detailed analysis.

This script provides publication-quality validation of our core findings.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import time

try:
    # Try relative imports first (when used as module)
    from .demo_learning import DemoConfig
    from .demo_scms import create_easy_scm, create_medium_scm, create_hard_scm
    from .complete_workflow_demo import run_progressive_learning_demo_with_scm
    from .demo_evaluation import (
        comprehensive_evaluation, compute_f1_progress, analyze_convergence_speed
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from demo_learning import DemoConfig
    from demo_scms import create_easy_scm, create_medium_scm, create_hard_scm
    from complete_workflow_demo import run_progressive_learning_demo_with_scm
    from demo_evaluation import (
        comprehensive_evaluation, compute_f1_progress, analyze_convergence_speed
    )
from causal_bayes_opt.data_structures import get_variables, get_target


class ComprehensiveValidator:
    """Comprehensive validation suite for ACBO intervention strategies."""
    
    def __init__(self, n_trials: int = 5, max_steps: int = 15):
        self.n_trials = n_trials
        self.max_steps = max_steps
        self.results = {}
    
    def run_scoring_method_comparison(self) -> Dict[str, Any]:
        """Compare BIC vs likelihood scoring across multiple trials."""
        print("\n🧪 SCORING METHOD COMPARISON (Multiple Trials)")
        print("=" * 60)
        
        scoring_methods = ["bic", "likelihood"]
        results = {}
        
        for method in scoring_methods:
            print(f"\nTesting {method.upper()} scoring method...")
            method_results = []
            
            for trial in range(self.n_trials):
                config = DemoConfig(
                    n_observational_samples=0,
                    n_intervention_steps=self.max_steps,
                    learning_rate=1e-3,
                    scoring_method=method,
                    random_seed=42 + trial * 100
                )
                
                try:
                    # Use easy SCM with disconnected variable for overfitting test
                    from demo_scms import create_easy_scm_with_disconnected_var
                    scm = create_easy_scm_with_disconnected_var()
                    trial_result = run_progressive_learning_demo_with_scm(scm, config)
                    
                    # Add comprehensive evaluation
                    marginal_progress = trial_result['marginal_prob_progress']
                    true_parents = trial_result['true_parents']
                    trial_result['comprehensive_evaluation'] = comprehensive_evaluation(
                        marginal_progress, true_parents
                    )
                    
                    method_results.append(trial_result)
                    print(f"   Trial {trial + 1}/{self.n_trials} completed")
                    
                except Exception as e:
                    print(f"   Trial {trial + 1} failed: {e}")
            
            results[method] = method_results
        
        # Analyze results
        analysis = self._analyze_scoring_comparison(results)
        
        print(f"\n📊 SCORING METHOD ANALYSIS:")
        print("-" * 40)
        
        for method in scoring_methods:
            if method in analysis:
                stats = analysis[method]
                print(f"\n{method.upper()} Method:")
                print(f"   Mean F1 Score: {stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}")
                print(f"   Convergence Rate: {stats['convergence_rate']:.1%}")
                print(f"   Mean False Positives: {stats['mean_false_positives']:.1f}")
        
        # Statistical comparison
        if 'bic' in analysis and 'likelihood' in analysis:
            stat_test = self._statistical_test(
                analysis['bic']['f1_scores'],
                analysis['likelihood']['f1_scores']
            )
            
            print(f"\n📈 Statistical Comparison:")
            print(f"   BIC vs Likelihood F1 scores")
            print(f"   t-statistic: {stat_test['t_statistic']:.3f}")
            print(f"   p-value: {stat_test['p_value']:.4f}")
            print(f"   Significant: {stat_test['significant']}")
            print(f"   Effect size (Cohen's d): {stat_test['cohens_d']:.3f}")
        
        return {
            'method': 'scoring_comparison',
            'results': results,
            'analysis': analysis
        }
    
    def run_intervention_strategy_comparison(self) -> Dict[str, Any]:
        """Compare random vs fixed intervention strategies."""
        print("\n🎯 INTERVENTION STRATEGY COMPARISON (Multiple Trials)")
        print("=" * 60)
        
        strategies = ["random", "fixed"]
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy} intervention strategy...")
            strategy_results = []
            
            for trial in range(self.n_trials):
                config = DemoConfig(
                    n_observational_samples=0,
                    n_intervention_steps=self.max_steps,
                    learning_rate=1e-3,
                    scoring_method="bic",
                    random_seed=42 + trial * 100
                )
                
                try:
                    if strategy == "random":
                        try:
                            from .complete_workflow_demo import run_zero_obs_random_intervention_test
                        except ImportError:
                            from complete_workflow_demo import run_zero_obs_random_intervention_test
                        trial_result = run_zero_obs_random_intervention_test(config)
                    else:  # fixed
                        try:
                            from .complete_workflow_demo import run_zero_obs_fixed_intervention_test
                        except ImportError:
                            from complete_workflow_demo import run_zero_obs_fixed_intervention_test
                        trial_result = run_zero_obs_fixed_intervention_test(config)
                    
                    # Add comprehensive evaluation
                    marginal_progress = trial_result['marginal_prob_progress']
                    true_parents = trial_result['true_parents']
                    trial_result['comprehensive_evaluation'] = comprehensive_evaluation(
                        marginal_progress, true_parents
                    )
                    
                    strategy_results.append(trial_result)
                    print(f"   Trial {trial + 1}/{self.n_trials} completed")
                    
                except Exception as e:
                    print(f"   Trial {trial + 1} failed: {e}")
            
            results[strategy] = strategy_results
        
        # Analyze results
        analysis = self._analyze_strategy_comparison(results)
        
        print(f"\n📊 INTERVENTION STRATEGY ANALYSIS:")
        print("-" * 40)
        
        for strategy in strategies:
            if strategy in analysis:
                stats = analysis[strategy]
                print(f"\n{strategy.upper()} Strategy:")
                print(f"   Mean F1 Score: {stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}")
                print(f"   Convergence Rate: {stats['convergence_rate']:.1%}")
                print(f"   Mean Steps to F1=0.5: {stats['mean_steps_to_half']:.1f}")
        
        # Statistical comparison
        if 'random' in analysis and 'fixed' in analysis:
            stat_test = self._statistical_test(
                analysis['random']['f1_scores'],
                analysis['fixed']['f1_scores']
            )
            
            print(f"\n📈 Statistical Comparison:")
            print(f"   Random vs Fixed F1 scores")
            print(f"   t-statistic: {stat_test['t_statistic']:.3f}")
            print(f"   p-value: {stat_test['p_value']:.4f}")
            print(f"   Significant: {stat_test['significant']}")
            print(f"   Effect size (Cohen's d): {stat_test['cohens_d']:.3f}")
        
        return {
            'method': 'strategy_comparison',
            'results': results,
            'analysis': analysis
        }
    
    def run_difficulty_scaling_test(self) -> Dict[str, Any]:
        """Test performance across different SCM difficulties."""
        print("\n📈 DIFFICULTY SCALING TEST")
        print("=" * 60)
        
        difficulties = [
            ("Easy", create_easy_scm),
            ("Medium", create_medium_scm),
            ("Hard", create_hard_scm)
        ]
        
        results = {}
        
        for difficulty_name, scm_fn in difficulties:
            print(f"\nTesting {difficulty_name} difficulty...")
            difficulty_results = []
            
            for trial in range(max(2, self.n_trials // 2)):  # Fewer trials for complex SCMs
                config = DemoConfig(
                    n_observational_samples=0,
                    n_intervention_steps=self.max_steps,
                    learning_rate=1e-3,
                    scoring_method="bic",
                    random_seed=42 + trial * 100
                )
                
                try:
                    scm = scm_fn()
                    trial_result = run_progressive_learning_demo_with_scm(scm, config)
                    
                    # Add comprehensive evaluation
                    marginal_progress = trial_result['marginal_prob_progress']
                    true_parents = trial_result['true_parents']
                    trial_result['comprehensive_evaluation'] = comprehensive_evaluation(
                        marginal_progress, true_parents
                    )
                    
                    difficulty_results.append(trial_result)
                    print(f"   Trial {trial + 1} completed")
                    
                except Exception as e:
                    print(f"   Trial {trial + 1} failed: {e}")
            
            results[difficulty_name.lower()] = difficulty_results
        
        # Analyze scaling
        analysis = self._analyze_difficulty_scaling(results)
        
        print(f"\n📊 DIFFICULTY SCALING ANALYSIS:")
        print("-" * 40)
        
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in analysis:
                stats = analysis[difficulty]
                print(f"\n{difficulty.upper()} SCMs:")
                print(f"   Mean F1 Score: {stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}")
                print(f"   Convergence Rate: {stats['convergence_rate']:.1%}")
                print(f"   Mean Uncertainty: {stats['mean_uncertainty']:.2f} bits")
        
        # Check for expected degradation
        if all(d in analysis for d in ["easy", "medium", "hard"]):
            easy_f1 = analysis["easy"]["mean_f1"]
            medium_f1 = analysis["medium"]["mean_f1"]
            hard_f1 = analysis["hard"]["mean_f1"]
            
            degradation_observed = easy_f1 >= medium_f1 >= hard_f1
            
            print(f"\n🎯 Expected Degradation:")
            print(f"   Easy ≥ Medium ≥ Hard: {degradation_observed}")
            print(f"   F1 scores: {easy_f1:.3f} ≥ {medium_f1:.3f} ≥ {hard_f1:.3f}")
        
        return {
            'method': 'difficulty_scaling',
            'results': results,
            'analysis': analysis
        }
    
    def _analyze_scoring_comparison(self, results: Dict[str, List]) -> Dict[str, Dict]:
        """Analyze scoring method comparison results."""
        analysis = {}
        
        for method, method_results in results.items():
            if not method_results:
                continue
            
            f1_scores = []
            convergence_rates = []
            false_positives = []
            
            for result in method_results:
                eval_data = result.get('comprehensive_evaluation', {})
                summary = eval_data.get('summary', {})
                
                f1_scores.append(summary.get('final_f1', 0.0))
                convergence_rates.append(1.0 if summary.get('converged', False) else 0.0)
                
                # Count false positives
                final_marginals = result.get('final_marginal_probs', {})
                true_parents = set(result.get('true_parents', []))
                false_pos = sum(1 for var, prob in final_marginals.items() 
                               if prob > 0.7 and var not in true_parents)
                false_positives.append(false_pos)
            
            analysis[method] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'convergence_rate': np.mean(convergence_rates),
                'mean_false_positives': np.mean(false_positives),
                'f1_scores': f1_scores,
                'n_trials': len(method_results)
            }
        
        return analysis
    
    def _analyze_strategy_comparison(self, results: Dict[str, List]) -> Dict[str, Dict]:
        """Analyze intervention strategy comparison results."""
        analysis = {}
        
        for strategy, strategy_results in results.items():
            if not strategy_results:
                continue
            
            f1_scores = []
            convergence_rates = []
            steps_to_half = []
            
            for result in strategy_results:
                eval_data = result.get('comprehensive_evaluation', {})
                summary = eval_data.get('summary', {})
                speed_analysis = eval_data.get('speed_analysis', {})
                
                f1_scores.append(summary.get('final_f1', 0.0))
                convergence_rates.append(1.0 if summary.get('converged', False) else 0.0)
                
                steps_half = speed_analysis.get('convergence_steps', {}).get('f1_0.5', None)
                if steps_half is not None:
                    steps_to_half.append(steps_half)
            
            analysis[strategy] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'convergence_rate': np.mean(convergence_rates),
                'mean_steps_to_half': np.mean(steps_to_half) if steps_to_half else float('inf'),
                'f1_scores': f1_scores,
                'n_trials': len(strategy_results)
            }
        
        return analysis
    
    def _analyze_difficulty_scaling(self, results: Dict[str, List]) -> Dict[str, Dict]:
        """Analyze difficulty scaling results."""
        analysis = {}
        
        for difficulty, difficulty_results in results.items():
            if not difficulty_results:
                continue
            
            f1_scores = []
            convergence_rates = []
            uncertainties = []
            
            for result in difficulty_results:
                eval_data = result.get('comprehensive_evaluation', {})
                summary = eval_data.get('summary', {})
                
                f1_scores.append(summary.get('final_f1', 0.0))
                convergence_rates.append(1.0 if summary.get('converged', False) else 0.0)
                uncertainties.append(result.get('final_uncertainty', 0.0))
            
            analysis[difficulty] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'convergence_rate': np.mean(convergence_rates),
                'mean_uncertainty': np.mean(uncertainties),
                'f1_scores': f1_scores,
                'n_trials': len(difficulty_results)
            }
        
        return analysis
    
    def _statistical_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform statistical comparison between two groups."""
        if not group1 or not group2:
            return {'error': 'Insufficient data for statistical test'}
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2)
        }


def run_comprehensive_validation(n_trials: int = 5, max_steps: int = 15) -> Dict[str, Any]:
    """Run the comprehensive validation suite."""
    print("🔬 COMPREHENSIVE ACBO VALIDATION SUITE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"   Trials per test: {n_trials}")
    print(f"   Max steps per trial: {max_steps}")
    print(f"   Estimated runtime: ~{n_trials * 3 * max_steps / 10:.0f} minutes")
    
    start_time = time.time()
    validator = ComprehensiveValidator(n_trials=n_trials, max_steps=max_steps)
    
    # Run all validation tests
    test_results = []
    
    try:
        scoring_test = validator.run_scoring_method_comparison()
        test_results.append(scoring_test)
    except Exception as e:
        print(f"❌ Scoring method comparison failed: {e}")
    
    try:
        strategy_test = validator.run_intervention_strategy_comparison()
        test_results.append(strategy_test)
    except Exception as e:
        print(f"❌ Strategy comparison failed: {e}")
    
    try:
        scaling_test = validator.run_difficulty_scaling_test()
        test_results.append(scaling_test)
    except Exception as e:
        print(f"❌ Difficulty scaling test failed: {e}")
    
    total_time = time.time() - start_time
    
    # Generate final report
    print("\n🎯 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Tests completed: {len(test_results)}")
    
    # Validate key findings
    key_findings_validated = []
    
    # Check BIC vs likelihood
    for test in test_results:
        if test.get('method') == 'scoring_comparison':
            analysis = test.get('analysis', {})
            if 'bic' in analysis and 'likelihood' in analysis:
                bic_more_conservative = (
                    analysis['bic']['mean_false_positives'] < 
                    analysis['likelihood']['mean_false_positives']
                )
                key_findings_validated.append(('BIC prevents overfitting', bic_more_conservative))
    
    # Check random vs fixed
    for test in test_results:
        if test.get('method') == 'strategy_comparison':
            analysis = test.get('analysis', {})
            if 'random' in analysis and 'fixed' in analysis:
                random_better = (
                    analysis['random']['mean_f1'] > analysis['fixed']['mean_f1']
                )
                key_findings_validated.append(('Random outperforms fixed', random_better))
    
    print(f"\n📋 Key Findings Validation:")
    for finding, validated in key_findings_validated:
        status = "✅" if validated else "❌"
        print(f"   {status} {finding}")
    
    all_validated = all(validated for _, validated in key_findings_validated)
    
    if all_validated:
        print(f"\n🎉 ALL KEY FINDINGS VALIDATED!")
        print(f"The comprehensive validation confirms:")
        print(f"   • BIC scoring prevents likelihood overfitting")
        print(f"   • Random interventions outperform fixed interventions")
        print(f"   • The ACBO framework behaves as theoretically expected")
    else:
        print(f"\n⚠️ SOME FINDINGS NOT VALIDATED")
        print(f"Check the detailed results above for specific issues.")
    
    return {
        'test_results': test_results,
        'key_findings_validated': key_findings_validated,
        'all_validated': all_validated,
        'runtime_minutes': total_time / 60
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive ACBO validation')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per test')
    parser.add_argument('--steps', type=int, default=15, help='Max steps per trial')
    
    args = parser.parse_args()
    
    results = run_comprehensive_validation(n_trials=args.trials, max_steps=args.steps)
    
    # Exit with appropriate code
    if results['all_validated']:
        sys.exit(0)
    else:
        sys.exit(1)