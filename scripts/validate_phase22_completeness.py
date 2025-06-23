#!/usr/bin/env python3
"""Phase 2.2 Code Completeness Validation Script.

This script validates that all Phase 2.2 components are properly implemented,
have no placeholder code, and all imports work correctly.
"""

import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Tuple, Any
import ast


class CodeCompletenessValidator:
    """Validates code completeness for Phase 2.2 implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src" / "causal_bayes_opt"
        self.results = {
            "components": {},
            "todos": [],
            "missing_implementations": [],
            "import_errors": [],
            "placeholder_code": []
        }
    
    def validate_all_components(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("=" * 80)
        print("PHASE 2.2 CODE COMPLETENESS VALIDATION")
        print("=" * 80)
        
        # Define Phase 2.2 components to validate
        components = {
            # Task 1: Reward Rubric System
            "reward_rubric": [
                "acquisition.reward_rubric.CausalRewardRubric",
                "acquisition.reward_rubric.RewardComponent",
                "acquisition.reward_rubric.RewardResult",
                "acquisition.reward_rubric.create_training_rubric",
                "acquisition.reward_rubric.create_research_rubric",
            ],
            
            # Task 2: Enhanced Configuration (part of hybrid_rewards)
            "hybrid_rewards": [
                "acquisition.hybrid_rewards.HybridRewardConfig",
                "acquisition.hybrid_rewards.create_hybrid_reward_config",
            ],
            
            # Task 3: Diversity Monitoring
            "diversity_monitor": [
                "training.diversity_monitor.DiversityMonitor",
                "training.diversity_monitor.DiversityMetrics",
                "training.diversity_monitor.DiversityAlert",
                "training.diversity_monitor.create_diversity_monitor",
            ],
            
            # Task 4: Intervention Environment
            "intervention_env": [
                "environments.intervention_env.InterventionEnvironment",
                "environments.intervention_env.EnvironmentConfig",
                "environments.intervention_env.EnvironmentInfo",
                "environments.intervention_env.create_intervention_environment",
                "environments.intervention_env.create_batch_environments",
            ],
            
            # Task 5: Async Training Infrastructure
            "async_training": [
                "training.async_training.AsyncTrainingManager",
                "training.async_training.AsyncTrainingConfig",
                "training.async_training.TrainingProgress",
                "training.async_training.create_async_training_manager",
            ],
            
            # Task 6: GRPO Integration
            "grpo_core": [
                "training.grpo_core.GRPOConfig",
                "training.grpo_core.GRPOTrajectory",
                "training.grpo_core.GRPOUpdateResult",
                "training.grpo_core.create_grpo_update_fn",
                "training.grpo_core.create_default_grpo_config",
            ],
            
            "experience_management": [
                "training.experience_management.Experience",
                "training.experience_management.ExperienceBatch",
                "training.experience_management.ExperienceManager",
                "training.experience_management.ExperienceConfig",
                "training.experience_management.create_experience_manager",
            ],
            
            "grpo_config": [
                "training.grpo_config.ComprehensiveGRPOConfig",
                "training.grpo_config.TrainingMode",
                "training.grpo_config.OptimizationLevel",
                "training.grpo_config.create_standard_grpo_config",
                "training.grpo_config.create_production_grpo_config",
                "training.grpo_config.create_debug_grpo_config",
            ],
            
            "grpo_training_manager": [
                "training.grpo_training_manager.GRPOTrainingManager",
                "training.grpo_training_manager.TrainingStep",
                "training.grpo_training_manager.TrainingSession",
                "training.grpo_training_manager.create_grpo_training_manager",
            ],
        }
        
        # Validate each component
        for category, items in components.items():
            print(f"\nValidating {category}...")
            self.results["components"][category] = {}
            
            for item in items:
                result = self._validate_component(item)
                self.results["components"][category][item] = result
                
                status = "✅" if result["exists"] and result["importable"] else "❌"
                print(f"  {status} {item}")
                
                if not result["exists"]:
                    self.results["missing_implementations"].append(item)
                elif not result["importable"]:
                    self.results["import_errors"].append({
                        "component": item,
                        "error": result.get("import_error", "Unknown error")
                    })
        
        # Check for TODOs and placeholder code
        print("\nScanning for TODOs and placeholder code...")
        self._scan_for_todos()
        self._scan_for_placeholders()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _validate_component(self, component_path: str) -> Dict[str, Any]:
        """Validate a single component."""
        parts = component_path.split(".")
        module_path = ".".join(["causal_bayes_opt"] + parts[:-1])
        component_name = parts[-1]
        
        result = {
            "exists": False,
            "importable": False,
            "has_docstring": False,
            "is_implemented": False
        }
        
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            
            # Check if component exists
            if hasattr(module, component_name):
                result["exists"] = True
                component = getattr(module, component_name)
                
                # Check if it has documentation
                if hasattr(component, "__doc__") and component.__doc__:
                    result["has_docstring"] = True
                
                # Check if it's implemented (not just a stub)
                if inspect.isclass(component):
                    # For classes, check if it has methods
                    methods = [m for m in dir(component) if not m.startswith("_")]
                    result["is_implemented"] = len(methods) > 0
                elif inspect.isfunction(component):
                    # For functions, check if it has a body
                    source = inspect.getsource(component)
                    result["is_implemented"] = "pass" not in source and "..." not in source
                else:
                    result["is_implemented"] = True
                
                result["importable"] = True
        except ImportError as e:
            result["import_error"] = str(e)
        except Exception as e:
            result["import_error"] = f"Unexpected error: {str(e)}"
        
        return result
    
    def _scan_for_todos(self):
        """Scan codebase for TODO comments."""
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, "r") as f:
                    content = f.read()
                    
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "TODO" in line or "FIXME" in line or "XXX" in line:
                        self.results["todos"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": i,
                            "content": line.strip()
                        })
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
    
    def _scan_for_placeholders(self):
        """Scan for placeholder implementations."""
        placeholder_patterns = [
            "raise NotImplementedError",
            "pass  # TODO",
            "...  # TODO",
            "return None  # TODO",
            "# PLACEHOLDER",
            "# STUB"
        ]
        
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, "r") as f:
                    content = f.read()
                    
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    for pattern in placeholder_patterns:
                        if pattern in line:
                            self.results["placeholder_code"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i,
                                "content": line.strip(),
                                "pattern": pattern
                            })
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
    
    def _generate_summary(self):
        """Generate validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        # Count statistics
        total_components = sum(len(items) for items in self.results["components"].values())
        implemented_components = sum(
            1 for category in self.results["components"].values()
            for result in category.values()
            if result["exists"] and result["importable"]
        )
        
        print(f"\nComponent Implementation Status:")
        print(f"  Total components: {total_components}")
        print(f"  Implemented: {implemented_components}")
        print(f"  Missing: {len(self.results['missing_implementations'])}")
        print(f"  Import errors: {len(self.results['import_errors'])}")
        
        print(f"\nCode Quality Issues:")
        print(f"  TODO comments: {len(self.results['todos'])}")
        print(f"  Placeholder code: {len(self.results['placeholder_code'])}")
        
        # List issues if any
        if self.results['missing_implementations']:
            print("\n❌ Missing Implementations:")
            for item in self.results['missing_implementations']:
                print(f"    - {item}")
        
        if self.results['import_errors']:
            print("\n❌ Import Errors:")
            for error in self.results['import_errors']:
                print(f"    - {error['component']}: {error['error']}")
        
        if self.results['todos']:
            print(f"\n⚠️  Found {len(self.results['todos'])} TODO comments")
            for todo in self.results['todos'][:5]:  # Show first 5
                print(f"    - {todo['file']}:{todo['line']} - {todo['content']}")
            if len(self.results['todos']) > 5:
                print(f"    ... and {len(self.results['todos']) - 5} more")
        
        if self.results['placeholder_code']:
            print(f"\n⚠️  Found {len(self.results['placeholder_code'])} placeholder implementations")
            for placeholder in self.results['placeholder_code'][:5]:  # Show first 5
                print(f"    - {placeholder['file']}:{placeholder['line']} - {placeholder['pattern']}")
            if len(self.results['placeholder_code']) > 5:
                print(f"    ... and {len(self.results['placeholder_code']) - 5} more")
        
        # Overall assessment
        print("\n" + "=" * 80)
        if (implemented_components == total_components and 
            not self.results['import_errors'] and
            len(self.results['todos']) == 0 and
            len(self.results['placeholder_code']) == 0):
            print("✅ PHASE 2.2 CODE IS COMPLETE!")
        else:
            print("❌ PHASE 2.2 CODE HAS ISSUES TO ADDRESS")
            completion_percentage = (implemented_components / total_components) * 100
            print(f"   Completion: {completion_percentage:.1f}%")
        print("=" * 80)


def main():
    """Run code completeness validation."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Add src to path
    sys.path.insert(0, str(project_root / "src"))
    
    # Run validation
    validator = CodeCompletenessValidator(project_root)
    results = validator.validate_all_components()
    
    # Return appropriate exit code
    if results['missing_implementations'] or results['import_errors']:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())