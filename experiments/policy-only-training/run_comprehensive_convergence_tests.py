#!/usr/bin/env python3
"""
Comprehensive convergence tests using fixed coefficient SCMs.

This script runs convergence tests on multiple graph types and sizes
using deterministic coefficients for reproducible analysis.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from create_fixed_coefficient_scms import FixedCoefficientSCMFactory, create_standard_fixed_scms


class ConvergenceTestSuite:
    """Comprehensive convergence testing framework."""
    
    def __init__(self, output_dir: str = 'thesis_results/comprehensive_convergence'):
        """Initialize test suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def run_single_test(self, 
                       structure_type: str,
                       num_variables: int,
                       interventions: int = 100,
                       coefficient_pattern: Optional[str] = None,
                       seed: int = 42) -> Optional[Path]:
        """
        Run a single convergence test.
        
        Args:
            structure_type: Type of graph structure
            num_variables: Number of variables
            interventions: Number of interventions
            coefficient_pattern: Pattern for coefficients
            seed: Random seed
            
        Returns:
            Path to convergence data file or None if failed
        """
        # Create unique test name
        test_name = f"{structure_type}_{num_variables}var"
        if coefficient_pattern:
            test_name += f"_{coefficient_pattern}"
        test_name += f"_{datetime.now().strftime('%H%M%S')}"
        
        # Build command
        cmd = [
            sys.executable,
            'experiments/policy-only-training/train_grpo_single_scm_with_surrogate.py',
            '--scm-type', structure_type,
            '--num-vars', str(num_variables),
            '--interventions', str(interventions),
            '--fixed-coefficients',
            '--no-early-stopping',
            '--seed', str(seed)
        ]
        
        if coefficient_pattern:
            cmd.extend(['--coefficient-pattern', coefficient_pattern])
        
        print(f"Running test: {test_name}")
        
        # Run training
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent.parent.parent
        )
        
        if result.returncode != 0:
            print(f"  ❌ Test failed: {test_name}")
            print(f"  Error: {result.stderr[:200]}")
            return None
        
        # Find convergence data
        checkpoint_dir = Path('checkpoints/grpo_single_scm')
        if checkpoint_dir.exists():
            latest_runs = sorted(
                checkpoint_dir.glob('single_scm_*'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for run_dir in latest_runs[:3]:  # Check last 3 runs
                convergence_file = run_dir / 'convergence_data.json'
                if convergence_file.exists():
                    # Copy to output directory
                    import shutil
                    output_file = self.output_dir / f'{test_name}_convergence.json'
                    shutil.copy(convergence_file, output_file)
                    print(f"  ✅ Test complete: {test_name}")
                    return output_file
        
        print(f"  ⚠️ No convergence data found for {test_name}")
        return None
    
    def run_structure_comparison(self, 
                                num_variables: int = 6,
                                interventions: int = 100):
        """
        Compare convergence across different structure types.
        
        Args:
            num_variables: Number of variables for all structures
            interventions: Number of interventions
        """
        structures = ['fork', 'true_fork', 'chain', 'collider', 'mixed', 'scale_free', 'random']
        
        print(f"\n{'='*60}")
        print(f"Structure Comparison Test ({num_variables} variables)")
        print(f"{'='*60}\n")
        
        results = {}
        for structure in structures:
            result_file = self.run_single_test(
                structure_type=structure,
                num_variables=num_variables,
                interventions=interventions,
                coefficient_pattern='decreasing'
            )
            if result_file:
                results[structure] = result_file
        
        # Generate comparison plots
        if results:
            self._plot_structure_comparison(results, num_variables)
        
        return results
    
    def run_scaling_test(self,
                        structure_type: str = 'scale_free',
                        variable_counts: List[int] = [4, 8, 16, 32],
                        interventions: int = 100):
        """
        Test how convergence scales with graph size.
        
        Args:
            structure_type: Structure to test
            variable_counts: List of variable counts
            interventions: Number of interventions
        """
        print(f"\n{'='*60}")
        print(f"Scaling Test for {structure_type}")
        print(f"{'='*60}\n")
        
        results = {}
        for num_vars in variable_counts:
            result_file = self.run_single_test(
                structure_type=structure_type,
                num_variables=num_vars,
                interventions=interventions,
                coefficient_pattern='decreasing'
            )
            if result_file:
                results[num_vars] = result_file
        
        # Generate scaling plots
        if results:
            self._plot_scaling_analysis(results, structure_type)
        
        return results
    
    def run_coefficient_pattern_test(self,
                                    structure_type: str = 'mixed',
                                    num_variables: int = 8,
                                    interventions: int = 100):
        """
        Test effect of different coefficient patterns.
        
        Args:
            structure_type: Structure to test
            num_variables: Number of variables
            interventions: Number of interventions
        """
        patterns = ['decreasing', 'alternating', 'strong', 'mixed', None]
        pattern_names = ['decreasing', 'alternating', 'strong', 'mixed', 'default']
        
        print(f"\n{'='*60}")
        print(f"Coefficient Pattern Test ({structure_type}, {num_variables} vars)")
        print(f"{'='*60}\n")
        
        results = {}
        for pattern, name in zip(patterns, pattern_names):
            result_file = self.run_single_test(
                structure_type=structure_type,
                num_variables=num_variables,
                interventions=interventions,
                coefficient_pattern=pattern
            )
            if result_file:
                results[name] = result_file
        
        # Generate pattern comparison plots
        if results:
            self._plot_coefficient_patterns(results, structure_type)
        
        return results
    
    def run_parallel_tests(self, test_configs: List[Dict]):
        """
        Run multiple tests in parallel.
        
        Args:
            test_configs: List of test configuration dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Running {len(test_configs)} tests in parallel")
        print(f"{'='*60}\n")
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}
            for config in test_configs:
                future = executor.submit(
                    self.run_single_test,
                    **config
                )
                futures[future] = config
            
            results = {}
            for future in as_completed(futures):
                config = futures[future]
                try:
                    result_file = future.result()
                    if result_file:
                        test_name = f"{config['structure_type']}_{config['num_variables']}var"
                        results[test_name] = result_file
                except Exception as e:
                    print(f"Test failed: {config}")
                    print(f"Error: {e}")
        
        return results
    
    def _plot_structure_comparison(self, results: Dict[str, Path], num_variables: int):
        """Create comparison plots for different structures."""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Convergence trajectories
        plt.subplot(2, 2, 1)
        for structure, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data:
                interventions = data['convergence_data']
                x = [d['intervention_idx'] for d in interventions]
                y = [d['probability'] if d['is_optimal'] else 0 for d in interventions]
                plt.plot(x, y, label=structure, linewidth=2)
        
        plt.xlabel('Intervention Index')
        plt.ylabel('P(Select Optimal)')
        plt.title(f'Convergence Trajectories ({num_variables} variables)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Convergence speed (interventions to 90%)
        plt.subplot(2, 2, 2)
        convergence_speeds = {}
        for structure, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data:
                interventions = data['convergence_data']
                for i, d in enumerate(interventions):
                    if d.get('is_optimal', False) and d.get('probability', 0) >= 0.9:
                        convergence_speeds[structure] = i
                        break
                else:
                    convergence_speeds[structure] = len(interventions)
        
        if convergence_speeds:
            structures = list(convergence_speeds.keys())
            speeds = list(convergence_speeds.values())
            colors = ['green' if s < 50 else 'orange' if s < 75 else 'red' for s in speeds]
            
            plt.bar(structures, speeds, color=colors, alpha=0.7)
            plt.xlabel('Structure Type')
            plt.ylabel('Interventions to 90% Convergence')
            plt.title('Convergence Speed Comparison')
            plt.xticks(rotation=45)
        
        # Subplot 3: Final performance
        plt.subplot(2, 2, 3)
        final_performance = {}
        for structure, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data and data['convergence_data']:
                # Last 10 interventions
                last_n = min(10, len(data['convergence_data']))
                last_interventions = data['convergence_data'][-last_n:]
                optimal_count = sum(1 for d in last_interventions if d.get('is_optimal', False))
                final_performance[structure] = (optimal_count / last_n) * 100
        
        if final_performance:
            structures = list(final_performance.keys())
            performances = list(final_performance.values())
            colors = ['green' if p >= 80 else 'orange' if p >= 50 else 'red' for p in performances]
            
            plt.bar(structures, performances, color=colors, alpha=0.7)
            plt.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Structure Type')
            plt.ylabel('Final Performance (%)')
            plt.title('Final Convergence Quality')
            plt.xticks(rotation=45)
            plt.ylim([0, 105])
        
        # Subplot 4: Target value improvement
        plt.subplot(2, 2, 4)
        for structure, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data:
                interventions = data['convergence_data']
                target_values = [d['target_outcome'] for d in interventions]
                if target_values:
                    # Smooth with rolling mean
                    window = 5
                    smoothed = np.convolve(target_values, np.ones(window)/window, mode='valid')
                    plt.plot(range(len(smoothed)), smoothed, label=structure, linewidth=2, alpha=0.7)
        
        plt.xlabel('Intervention Index')
        plt.ylabel('Target Outcome')
        plt.title('Target Value Progression')
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'structure_comparison_{num_variables}var.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to {output_file}")
    
    def _plot_scaling_analysis(self, results: Dict[int, Path], structure_type: str):
        """Create scaling analysis plots."""
        plt.figure(figsize=(12, 8))
        
        # Extract convergence metrics for each size
        sizes = sorted(results.keys())
        convergence_speeds = []
        final_performances = []
        
        for size in sizes:
            data = self._load_convergence_data(results[size])
            if data:
                # Find convergence speed
                for i, d in enumerate(data['convergence_data']):
                    if d.get('is_optimal', False) and d.get('probability', 0) >= 0.9:
                        convergence_speeds.append(i)
                        break
                else:
                    convergence_speeds.append(len(data['convergence_data']))
                
                # Calculate final performance
                last_n = min(10, len(data['convergence_data']))
                last_interventions = data['convergence_data'][-last_n:]
                optimal_count = sum(1 for d in last_interventions if d.get('is_optimal', False))
                final_performances.append((optimal_count / last_n) * 100)
        
        # Subplot 1: Scaling of convergence speed
        plt.subplot(2, 2, 1)
        plt.plot(sizes, convergence_speeds, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Variables')
        plt.ylabel('Interventions to 90% Convergence')
        plt.title(f'Convergence Speed Scaling ({structure_type})')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Log-scale plot
        plt.subplot(2, 2, 2)
        if all(s > 0 for s in convergence_speeds):
            plt.loglog(sizes, convergence_speeds, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Number of Variables (log)')
            plt.ylabel('Interventions to Converge (log)')
            plt.title('Log-Log Scaling Analysis')
            plt.grid(True, alpha=0.3, which='both')
        
        # Subplot 3: Final performance vs size
        plt.subplot(2, 2, 3)
        plt.plot(sizes, final_performances, 'o-', linewidth=2, markersize=8, color='green')
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        plt.xlabel('Number of Variables')
        plt.ylabel('Final Performance (%)')
        plt.title('Final Performance vs Graph Size')
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Efficiency metric
        plt.subplot(2, 2, 4)
        efficiency = [p / s if s > 0 else 0 for p, s in zip(final_performances, convergence_speeds)]
        plt.plot(sizes, efficiency, 'o-', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Number of Variables')
        plt.ylabel('Efficiency (Performance / Convergence Time)')
        plt.title('Learning Efficiency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'scaling_analysis_{structure_type}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved scaling plot to {output_file}")
    
    def _plot_coefficient_patterns(self, results: Dict[str, Path], structure_type: str):
        """Create coefficient pattern comparison plots."""
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Convergence trajectories
        plt.subplot(1, 2, 1)
        for pattern, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data:
                interventions = data['convergence_data']
                x = [d['intervention_idx'] for d in interventions]
                y = [d['probability'] if d['is_optimal'] else 0 for d in interventions]
                plt.plot(x, y, label=pattern, linewidth=2)
        
        plt.xlabel('Intervention Index')
        plt.ylabel('P(Select Optimal)')
        plt.title(f'Effect of Coefficient Patterns ({structure_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Convergence speed comparison
        plt.subplot(1, 2, 2)
        convergence_speeds = {}
        for pattern, file_path in results.items():
            data = self._load_convergence_data(file_path)
            if data:
                for i, d in enumerate(data['convergence_data']):
                    if d.get('is_optimal', False) and d.get('probability', 0) >= 0.9:
                        convergence_speeds[pattern] = i
                        break
                else:
                    convergence_speeds[pattern] = len(data['convergence_data'])
        
        if convergence_speeds:
            patterns = list(convergence_speeds.keys())
            speeds = list(convergence_speeds.values())
            plt.bar(patterns, speeds, alpha=0.7)
            plt.xlabel('Coefficient Pattern')
            plt.ylabel('Interventions to 90% Convergence')
            plt.title('Pattern Effect on Convergence Speed')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_file = self.output_dir / f'coefficient_patterns_{structure_type}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved pattern plot to {output_file}")
    
    def _load_convergence_data(self, file_path: Path) -> Optional[Dict]:
        """Load convergence data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def generate_summary_report(self):
        """Generate a summary report of all tests."""
        report = []
        report.append("=" * 60)
        report.append("CONVERGENCE TEST SUITE SUMMARY")
        report.append("=" * 60)
        report.append(f"Output Directory: {self.output_dir}")
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Count result files
        json_files = list(self.output_dir.glob('*_convergence.json'))
        png_files = list(self.output_dir.glob('*.png'))
        
        report.append(f"Total convergence files: {len(json_files)}")
        report.append(f"Total plots generated: {len(png_files)}")
        report.append("")
        
        # Analyze each convergence file
        if json_files:
            report.append("Test Results:")
            report.append("-" * 40)
            
            for file_path in sorted(json_files):
                data = self._load_convergence_data(file_path)
                if data:
                    test_name = file_path.stem.replace('_convergence', '')
                    convergence_data = data.get('convergence_data', [])
                    
                    # Find convergence point
                    convergence_idx = None
                    for i, d in enumerate(convergence_data):
                        if d.get('is_optimal', False) and d.get('probability', 0) >= 0.9:
                            convergence_idx = i
                            break
                    
                    # Calculate final performance
                    last_n = min(10, len(convergence_data))
                    if convergence_data:
                        last_interventions = convergence_data[-last_n:]
                        optimal_count = sum(1 for d in last_interventions if d.get('is_optimal', False))
                        final_perf = (optimal_count / last_n) * 100
                    else:
                        final_perf = 0
                    
                    report.append(f"\n{test_name}:")
                    report.append(f"  Total interventions: {len(convergence_data)}")
                    report.append(f"  Convergence at: {convergence_idx if convergence_idx else 'Not converged'}")
                    report.append(f"  Final performance: {final_perf:.1f}%")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_file = self.output_dir / 'summary_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        print(f"\nReport saved to {report_file}")


def main():
    """Main function to run comprehensive convergence tests."""
    parser = argparse.ArgumentParser(description='Comprehensive convergence testing suite')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'structures', 'scaling', 'patterns', 'custom'],
                       help='Type of test to run')
    parser.add_argument('--interventions', type=int, default=100,
                       help='Number of interventions per test')
    parser.add_argument('--output-dir', type=str,
                       default='thesis_results/comprehensive_convergence',
                       help='Output directory for results')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    
    args = parser.parse_args()
    
    # Initialize test suite
    suite = ConvergenceTestSuite(output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE CONVERGENCE TEST SUITE")
    print("="*60)
    
    if args.test == 'all' or args.test == 'structures':
        # Test different structures
        suite.run_structure_comparison(
            num_variables=6,
            interventions=args.interventions
        )
    
    if args.test == 'all' or args.test == 'scaling':
        # Test scaling behavior
        suite.run_scaling_test(
            structure_type='scale_free',
            variable_counts=[4, 8, 16, 32],
            interventions=args.interventions
        )
    
    if args.test == 'all' or args.test == 'patterns':
        # Test coefficient patterns
        suite.run_coefficient_pattern_test(
            structure_type='mixed',
            num_variables=8,
            interventions=args.interventions
        )
    
    if args.test == 'custom':
        # Run custom parallel tests
        test_configs = [
            {'structure_type': 'fork', 'num_variables': 10, 'interventions': 50},
            {'structure_type': 'chain', 'num_variables': 10, 'interventions': 50},
            {'structure_type': 'scale_free', 'num_variables': 20, 'interventions': 75},
            {'structure_type': 'random', 'num_variables': 15, 'interventions': 75},
        ]
        
        if args.parallel:
            suite.run_parallel_tests(test_configs)
        else:
            for config in test_configs:
                suite.run_single_test(**config)
    
    # Generate summary report
    suite.generate_summary_report()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print(f"Results saved to: {suite.output_dir}")


if __name__ == "__main__":
    main()