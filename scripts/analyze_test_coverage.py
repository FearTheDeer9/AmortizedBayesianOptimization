#!/usr/bin/env python3
"""Test Coverage Analysis Script for Phase 2.2.

This script analyzes test coverage for the entire codebase with focus on
Phase 2.2 components, ensuring we meet the >90% coverage requirement.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple


class CoverageAnalyzer:
    """Analyzes test coverage for the codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_data = {}
        self.phase22_modules = [
            "causal_bayes_opt.acquisition.reward_rubric",
            "causal_bayes_opt.acquisition.hybrid_rewards",
            "causal_bayes_opt.training.diversity_monitor",
            "causal_bayes_opt.environments.intervention_env",
            "causal_bayes_opt.training.async_training",
            "causal_bayes_opt.training.grpo_core",
            "causal_bayes_opt.training.experience_management",
            "causal_bayes_opt.training.grpo_config",
            "causal_bayes_opt.training.grpo_training_manager",
        ]
    
    def run_coverage(self) -> Dict[str, any]:
        """Run pytest with coverage and parse results."""
        print("Running test coverage analysis...")
        print("=" * 80)
        
        # Run coverage with JSON output
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=causal_bayes_opt",
            "--cov-report=json:coverage.json",
            "--cov-report=term",
            "-q"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse coverage JSON
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, "r") as f:
                    self.coverage_data = json.load(f)
            
            # Print overall summary
            self._print_summary()
            
            # Analyze Phase 2.2 coverage
            self._analyze_phase22_coverage()
            
            # Identify uncovered critical paths
            self._identify_uncovered_paths()
            
            return self.coverage_data
            
        except Exception as e:
            print(f"Error running coverage: {e}")
            return {}
    
    def _print_summary(self):
        """Print overall coverage summary."""
        if not self.coverage_data:
            print("No coverage data available")
            return
        
        totals = self.coverage_data.get("totals", {})
        percent_covered = totals.get("percent_covered", 0)
        
        print(f"\nOVERALL TEST COVERAGE: {percent_covered:.1f}%")
        print(f"Total lines: {totals.get('num_statements', 0)}")
        print(f"Covered lines: {totals.get('covered_lines', 0)}")
        print(f"Missing lines: {totals.get('missing_lines', 0)}")
        
        # Check if we meet 90% requirement
        if percent_covered >= 90:
            print("\n✅ MEETS 90% COVERAGE REQUIREMENT!")
        else:
            print(f"\n❌ BELOW 90% REQUIREMENT (need {90 - percent_covered:.1f}% more)")
    
    def _analyze_phase22_coverage(self):
        """Analyze coverage specifically for Phase 2.2 modules."""
        print("\n" + "=" * 80)
        print("PHASE 2.2 MODULE COVERAGE")
        print("=" * 80)
        
        files = self.coverage_data.get("files", {})
        phase22_coverage = []
        
        for module in self.phase22_modules:
            # Convert module to file path
            file_path = module.replace(".", "/") + ".py"
            
            # Find matching file in coverage data
            for file_key, file_data in files.items():
                if file_path in file_key:
                    summary = file_data.get("summary", {})
                    percent = summary.get("percent_covered", 0)
                    phase22_coverage.append((module, percent))
                    
                    status = "✅" if percent >= 90 else "❌"
                    print(f"{status} {module}: {percent:.1f}%")
                    break
            else:
                print(f"⚠️  {module}: No coverage data")
        
        # Calculate Phase 2.2 average
        if phase22_coverage:
            avg_coverage = sum(cov[1] for cov in phase22_coverage) / len(phase22_coverage)
            print(f"\nPhase 2.2 Average Coverage: {avg_coverage:.1f}%")
    
    def _identify_uncovered_paths(self):
        """Identify critical uncovered code paths."""
        print("\n" + "=" * 80)
        print("UNCOVERED CRITICAL PATHS")
        print("=" * 80)
        
        files = self.coverage_data.get("files", {})
        critical_uncovered = []
        
        # Look for uncovered lines in Phase 2.2 modules
        for module in self.phase22_modules:
            file_path = module.replace(".", "/") + ".py"
            
            for file_key, file_data in files.items():
                if file_path in file_key:
                    missing_lines = file_data.get("missing_lines", [])
                    if missing_lines:
                        critical_uncovered.append({
                            "module": module,
                            "file": file_key,
                            "missing_lines": missing_lines[:10],  # First 10
                            "total_missing": len(missing_lines)
                        })
                    break
        
        if critical_uncovered:
            print("\nModules with uncovered lines:")
            for item in critical_uncovered:
                print(f"\n{item['module']}:")
                print(f"  File: {Path(item['file']).name}")
                print(f"  Missing lines: {item['missing_lines']}")
                if item['total_missing'] > 10:
                    print(f"  ... and {item['total_missing'] - 10} more")
        else:
            print("\n✅ All Phase 2.2 critical paths are covered!")
    
    def generate_coverage_report(self):
        """Generate detailed HTML coverage report."""
        print("\n" + "=" * 80)
        print("GENERATING HTML COVERAGE REPORT")
        print("=" * 80)
        
        cmd = [
            "python", "-m", "coverage", "html",
            "--include=src/causal_bayes_opt/*",
            "--omit=*/tests/*,*/scripts/*"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.project_root, check=True)
            print("✅ HTML coverage report generated in htmlcov/")
            print("   Open htmlcov/index.html in a browser for details")
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
    
    def suggest_improvements(self):
        """Suggest specific improvements to reach 90% coverage."""
        print("\n" + "=" * 80)
        print("COVERAGE IMPROVEMENT SUGGESTIONS")
        print("=" * 80)
        
        totals = self.coverage_data.get("totals", {})
        current_percent = totals.get("percent_covered", 0)
        
        if current_percent >= 90:
            print("✅ Already meeting 90% coverage requirement!")
            return
        
        # Calculate how many more lines need coverage
        total_lines = totals.get("num_statements", 0)
        covered_lines = totals.get("covered_lines", 0)
        target_lines = int(total_lines * 0.9)
        needed_lines = target_lines - covered_lines
        
        print(f"Current coverage: {current_percent:.1f}%")
        print(f"Need to cover {needed_lines} more lines to reach 90%")
        
        # Find modules with lowest coverage
        files = self.coverage_data.get("files", {})
        low_coverage = []
        
        for file_path, file_data in files.items():
            summary = file_data.get("summary", {})
            percent = summary.get("percent_covered", 0)
            missing = summary.get("missing_lines", 0)
            
            if percent < 80 and missing > 10:
                low_coverage.append({
                    "file": Path(file_path).name,
                    "percent": percent,
                    "missing": missing
                })
        
        if low_coverage:
            print("\nFocus on these low-coverage modules:")
            low_coverage.sort(key=lambda x: x["percent"])
            for item in low_coverage[:5]:
                print(f"  - {item['file']}: {item['percent']:.1f}% ({item['missing']} lines missing)")


def main():
    """Run coverage analysis."""
    project_root = Path(__file__).parent.parent
    
    analyzer = CoverageAnalyzer(project_root)
    
    # Run coverage analysis
    coverage_data = analyzer.run_coverage()
    
    if coverage_data:
        # Generate HTML report
        analyzer.generate_coverage_report()
        
        # Provide improvement suggestions
        analyzer.suggest_improvements()
    
    # Return appropriate exit code
    totals = coverage_data.get("totals", {})
    if totals.get("percent_covered", 0) >= 90:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())