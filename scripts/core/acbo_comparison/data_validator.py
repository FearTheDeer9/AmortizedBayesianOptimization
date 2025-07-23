#!/usr/bin/env python3
"""
Data Validator for ACBO Comparison Results

This module validates consistency between:
- Individual run results
- Aggregated trajectories
- Summary statistics

It helps identify data flow issues and ensures metrics are correctly aggregated.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates ACBO comparison results for consistency."""
    
    def __init__(self, results_file: Path):
        """Initialize validator with results file."""
        self.results_file = results_file
        with open(results_file, 'r') as f:
            self.data = json.load(f)
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        validation_results = {
            'trajectory_consistency': self._validate_trajectory_consistency(),
            'aggregation_accuracy': self._validate_aggregation_accuracy(),
            'summary_statistics': self._validate_summary_statistics(),
            'structure_metrics': self._validate_structure_metrics(),
            'data_completeness': self._validate_data_completeness()
        }
        
        # Overall pass/fail
        validation_results['passed'] = all(
            result['passed'] for result in validation_results.values()
        )
        
        return validation_results
    
    def _validate_trajectory_consistency(self) -> Dict[str, Any]:
        """Validate that trajectory data is consistent across storage locations."""
        issues = []
        
        if 'trajectory_data' not in self.data:
            return {'passed': False, 'error': 'No trajectory_data found'}
        
        if 'aggregated_trajectories' not in self.data:
            return {'passed': False, 'error': 'No aggregated_trajectories found'}
        
        # Check each method
        methods = set()
        for key in self.data['trajectory_data'].keys():
            method = key.rsplit('_', 2)[0]  # Remove _scm_run suffixes
            methods.add(method)
        
        for method in methods:
            # Get individual trajectories
            method_trajectories = [
                v for k, v in self.data['trajectory_data'].items() 
                if k.startswith(method)
            ]
            
            if not method_trajectories:
                issues.append(f"{method}: No individual trajectories found")
                continue
            
            # Check if aggregated data exists
            if method not in self.data['aggregated_trajectories']:
                issues.append(f"{method}: Missing from aggregated_trajectories")
                continue
            
            # Validate trajectory lengths
            traj_lengths = []
            for traj in method_trajectories:
                if 'target_values_trajectory' in traj:
                    traj_lengths.append(len(traj['target_values_trajectory']))
            
            if len(set(traj_lengths)) > 1:
                issues.append(f"{method}: Inconsistent trajectory lengths: {set(traj_lengths)}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'methods_checked': list(methods)
        }
    
    def _validate_aggregation_accuracy(self) -> Dict[str, Any]:
        """Validate that aggregated values match individual trajectories."""
        issues = []
        tolerance = 1e-6
        
        for method in self.data.get('aggregated_trajectories', {}):
            # Get individual trajectories for this method
            method_trajectories = [
                v for k, v in self.data['trajectory_data'].items()
                if k.startswith(method)
            ]
            
            if not method_trajectories:
                continue
            
            # Check final target value aggregation
            final_values = [
                t.get('target_values_final', 0) for t in method_trajectories
                if 'target_values_final' in t
            ]
            
            if final_values:
                computed_mean = np.mean(final_values)
                stored_mean = self.data['aggregated_trajectories'][method].get(
                    'final_target_value_mean', None
                )
                
                if stored_mean is not None:
                    if abs(computed_mean - stored_mean) > tolerance:
                        issues.append(
                            f"{method}: Final value mismatch - "
                            f"computed: {computed_mean:.6f}, stored: {stored_mean:.6f}"
                        )
            
            # Check improvement calculation
            improvements = []
            for traj in method_trajectories:
                if 'target_values_trajectory' in traj:
                    traj_values = traj['target_values_trajectory']
                    if traj_values:
                        improvement = traj_values[-1] - traj_values[0]
                        improvements.append(improvement)
            
            if improvements:
                computed_improvement = np.mean(improvements)
                stored_improvement = self.data['aggregated_trajectories'][method].get(
                    'target_improvement_mean', None
                )
                
                if stored_improvement is not None:
                    if abs(computed_improvement - stored_improvement) > tolerance:
                        issues.append(
                            f"{method}: Improvement mismatch - "
                            f"computed: {computed_improvement:.6f}, "
                            f"stored: {stored_improvement:.6f}"
                        )
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_summary_statistics(self) -> Dict[str, Any]:
        """Validate that summary statistics match aggregated trajectories."""
        issues = []
        tolerance = 1e-6
        
        if 'statistical_analysis' not in self.data:
            return {'passed': False, 'error': 'No statistical_analysis found'}
        
        summary_stats = self.data['statistical_analysis'].get('summary_statistics', {})
        aggregated = self.data.get('aggregated_trajectories', {})
        
        for method in summary_stats:
            if method not in aggregated:
                issues.append(f"{method}: Missing from aggregated_trajectories")
                continue
            
            # Check target improvement
            summary_improvement = summary_stats[method].get('target_improvement_mean', None)
            aggregated_improvement = aggregated[method].get('target_improvement_mean', None)
            
            if summary_improvement is not None and aggregated_improvement is not None:
                if abs(summary_improvement - aggregated_improvement) > tolerance:
                    issues.append(
                        f"{method}: Target improvement mismatch - "
                        f"summary: {summary_improvement:.6f}, "
                        f"aggregated: {aggregated_improvement:.6f}"
                    )
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_structure_metrics(self) -> Dict[str, Any]:
        """Validate that structure metrics are properly collected."""
        issues = []
        
        for method in self.data.get('aggregated_trajectories', {}):
            agg_data = self.data['aggregated_trajectories'][method]
            
            # Check if structure metrics exist
            has_f1 = 'f1_scores_mean' in agg_data
            has_shd = 'shd_values_mean' in agg_data
            
            # Learning methods should have structure metrics
            if 'Learning' in method and not (has_f1 and has_shd):
                issues.append(f"{method}: Missing structure metrics (F1/SHD)")
            
            # Check if values are non-zero
            if has_f1 and isinstance(agg_data['f1_scores_mean'], dict):
                f1_values = list(agg_data['f1_scores_mean'].values())
                if all(v == 0 for v in f1_values):
                    issues.append(f"{method}: All F1 scores are zero")
            
            # Untrained method should have zero F1
            if 'Untrained' in method and has_f1:
                if isinstance(agg_data['f1_scores_mean'], dict):
                    f1_values = list(agg_data['f1_scores_mean'].values())
                    if any(v > 0 for v in f1_values):
                        issues.append(f"{method}: Untrained method has non-zero F1 scores")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_data_completeness(self) -> Dict[str, Any]:
        """Check for missing or incomplete data."""
        required_keys = [
            'experiment_config',
            'method_results',
            'statistical_analysis',
            'aggregated_trajectories',
            'trajectory_data'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.data]
        
        # Check method coverage
        if 'experiment_config' in self.data:
            expected_methods = self.data['experiment_config']['experiment']['methods']
            actual_methods = set()
            
            if 'method_results' in self.data:
                actual_methods.update(self.data['method_results'].keys())
            
            missing_methods = set(expected_methods.keys()) - actual_methods
        else:
            missing_methods = []
        
        return {
            'passed': len(missing_keys) == 0 and len(missing_methods) == 0,
            'missing_keys': missing_keys,
            'missing_methods': list(missing_methods)
        }
    
    def print_report(self, validation_results: Dict[str, Any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print("ACBO Data Validation Report")
        print("="*60)
        print(f"Results file: {self.results_file}")
        print(f"Overall: {'✅ PASSED' if validation_results['passed'] else '❌ FAILED'}")
        print()
        
        for check_name, result in validation_results.items():
            if check_name == 'passed':
                continue
            
            print(f"\n{check_name.replace('_', ' ').title()}:")
            print("-" * 40)
            
            if isinstance(result, dict):
                if 'passed' in result:
                    print(f"Status: {'✅ Passed' if result['passed'] else '❌ Failed'}")
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                
                if 'issues' in result and result['issues']:
                    print("Issues found:")
                    for issue in result['issues']:
                        print(f"  - {issue}")
                
                if 'missing_keys' in result and result['missing_keys']:
                    print(f"Missing keys: {', '.join(result['missing_keys'])}")
                
                if 'missing_methods' in result and result['missing_methods']:
                    print(f"Missing methods: {', '.join(result['missing_methods'])}")


def main():
    """Run validation on a results file."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate ACBO comparison results')
    parser.add_argument('results_file', type=Path, help='Path to results JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    validator = DataValidator(args.results_file)
    results = validator.validate_all()
    validator.print_report(results)
    
    return 0 if results['passed'] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())