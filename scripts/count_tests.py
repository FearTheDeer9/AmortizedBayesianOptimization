#!/usr/bin/env python3
"""Count tests across all test files."""

from pathlib import Path
import re

def count_tests():
    """Count all test functions across the test suite."""
    project_root = Path(__file__).parent.parent
    test_files = list((project_root / "tests").rglob("test_*.py"))
    
    total_tests = 0
    phase22_tests = 0
    
    phase22_modules = [
        "reward_rubric", "hybrid_rewards", "diversity_monitor",
        "intervention_env", "async_training", "grpo_core",
        "experience_management", "grpo_config", "grpo_training_manager"
    ]
    
    for test_file in test_files:
        try:
            content = test_file.read_text()
            
            # Count test functions
            test_functions = re.findall(r'def test_\w+', content)
            file_test_count = len(test_functions)
            total_tests += file_test_count
            
            # Check if it's a Phase 2.2 test file
            is_phase22 = any(module in content for module in phase22_modules)
            if is_phase22:
                phase22_tests += file_test_count
                print(f"Phase 2.2: {test_file.name} - {file_test_count} tests")
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
    
    print(f"\nTEST SUMMARY:")
    print(f"Total test files: {len(test_files)}")
    print(f"Total test functions: {total_tests}")
    print(f"Phase 2.2 test functions: {phase22_tests}")
    print(f"Phase 2.2 coverage: {phase22_tests/total_tests*100:.1f}%")
    
    return total_tests, phase22_tests

if __name__ == "__main__":
    count_tests()