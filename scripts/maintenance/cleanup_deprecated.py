#!/usr/bin/env python3
"""
Cleanup Script for Deprecated Legacy Components

This script helps identify, mark, and optionally remove deprecated components
after the JAX-native migration in Phase 1.5. It provides safe cleanup with
backup and rollback capabilities.

Usage:
    python scripts/cleanup_deprecated.py --analyze    # Analyze deprecated code
    python scripts/cleanup_deprecated.py --mark       # Add deprecation warnings
    python scripts/cleanup_deprecated.py --cleanup    # Remove deprecated files
    python scripts/cleanup_deprecated.py --rollback   # Restore from backup
"""

import os
import sys
import argparse
import shutil
import json
import warnings
from pathlib import Path
from typing import List, Dict, Set, Tuple
from datetime import datetime
import ast
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Deprecated components mapping
DEPRECATED_COMPONENTS = {
    # Legacy acquisition components
    "src/causal_bayes_opt/acquisition/state_enhanced.py": {
        "replacement": "causal_bayes_opt.jax_native.state.JAXAcquisitionState",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md",
        "removal_date": "2024-02-01",
        "reason": "Non-JAX compatible, replaced by JAX-native implementation"
    },
    "src/causal_bayes_opt/acquisition/state_tensor_converter.py": {
        "replacement": "causal_bayes_opt.jax_native.state.JAXAcquisitionState",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md", 
        "removal_date": "2024-02-01",
        "reason": "Conversion layer no longer needed with JAX-native architecture"
    },
    "src/causal_bayes_opt/acquisition/state_tensor_ops.py": {
        "replacement": "causal_bayes_opt.jax_native.operations",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md",
        "removal_date": "2024-02-01", 
        "reason": "Replaced by JAX-compiled tensor operations"
    },
    "src/causal_bayes_opt/acquisition/tensor_features.py": {
        "replacement": "causal_bayes_opt.jax_native.operations.compute_policy_features_jax",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md",
        "removal_date": "2024-02-01",
        "reason": "Feature extraction integrated into JAX-native operations"
    },
    "src/causal_bayes_opt/acquisition/vectorized_attention.py": {
        "replacement": "JAX vmap operations in jax_native module",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md", 
        "removal_date": "2024-02-01",
        "reason": "Replaced by JAX-native vectorized operations"
    },
    
    # Legacy parent set components
    "src/causal_bayes_opt/avici_integration/parent_set/model.py": {
        "replacement": "causal_bayes_opt.avici_integration.parent_set.unified.model",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md",
        "removal_date": "2024-01-15",
        "reason": "Replaced by unified JAX-compatible model"
    },
    
    # Legacy training components  
    "src/causal_bayes_opt/training/surrogate_training.py": {
        "replacement": "causal_bayes_opt.training.surrogate_trainer",
        "migration_guide": "docs/migration/MIGRATION_GUIDE.md",
        "removal_date": "2024-01-15", 
        "reason": "Replaced by JAX-compiled training implementation"
    }
}

# Components that should have deprecation warnings added
COMPONENTS_TO_WARN = {
    "src/causal_bayes_opt/acquisition/rewards.py": {
        "preferred": "causal_bayes_opt.acquisition.hybrid_rewards",
        "reason": "Use hybrid_rewards for mechanism-aware reward computation"
    },
    "src/causal_bayes_opt/acquisition/policy.py": {
        "preferred": "JAX-native policy operations", 
        "reason": "Contains non-JAX compatible components"
    }
}

# Safe to remove files (no external dependencies)
SAFE_TO_REMOVE = {
    "src/causal_bayes_opt/acquisition/jax_utils.py",  # Functionality moved to jax_native
    "test_jax_robustness.py",  # Temporary test file
    "test_task3_validation.py"  # Temporary test file
}


class DeprecationAnalyzer:
    """Analyze codebase for deprecated component usage."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"
        
    def find_deprecated_imports(self) -> Dict[str, List[str]]:
        """Find all imports of deprecated components."""
        deprecated_usage = {}
        
        # Search Python files for imports
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find imports
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                
                # Check for deprecated imports
                for imp in imports:
                    if self._is_deprecated_import(imp):
                        if str(py_file) not in deprecated_usage:
                            deprecated_usage[str(py_file)] = []
                        deprecated_usage[str(py_file)].append(imp)
                        
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")
                
        return deprecated_usage
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        for directory in [self.src_dir, self.test_dir]:
            if directory.exists():
                python_files.extend(directory.rglob("*.py"))
        return python_files
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
                        
        return imports
    
    def _is_deprecated_import(self, import_name: str) -> bool:
        """Check if import refers to deprecated component."""
        for deprecated_path in DEPRECATED_COMPONENTS.keys():
            # Convert file path to module path
            module_path = deprecated_path.replace("src/", "").replace("/", ".").replace(".py", "")
            if import_name.startswith(module_path):
                return True
        return False
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        deprecated_usage = self.find_deprecated_imports()
        
        # Analyze file sizes and modification dates
        file_stats = {}
        for dep_file in DEPRECATED_COMPONENTS.keys():
            file_path = self.project_root / dep_file
            if file_path.exists():
                stat = file_path.stat()
                file_stats[dep_file] = {
                    "size_kb": stat.st_size / 1024,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "exists": True
                }
            else:
                file_stats[dep_file] = {"exists": False}
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "deprecated_usage": deprecated_usage,
            "file_statistics": file_stats,
            "total_deprecated_files": len([f for f in file_stats.values() if f.get("exists")]),
            "total_usage_locations": sum(len(usages) for usages in deprecated_usage.values())
        }


class DeprecationMarker:
    """Add deprecation warnings to legacy components."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def add_deprecation_warnings(self, dry_run: bool = True) -> Dict[str, bool]:
        """Add deprecation warnings to components."""
        results = {}
        
        for file_path, info in DEPRECATED_COMPONENTS.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                success = self._add_warning_to_file(full_path, info, dry_run)
                results[file_path] = success
            else:
                results[file_path] = False
                
        return results
    
    def _add_warning_to_file(self, file_path: Path, info: Dict, dry_run: bool) -> bool:
        """Add deprecation warning to specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if warning already exists
            if 'DeprecationWarning' in content or 'DEPRECATED' in content:
                print(f"Warning already exists in {file_path}")
                return True
            
            # Create deprecation warning
            warning_text = self._create_warning_text(info)
            
            # Insert warning after docstring or at top of file
            new_content = self._insert_warning(content, warning_text)
            
            if not dry_run:
                # Backup original file
                backup_path = file_path.with_suffix('.py.backup')
                shutil.copy2(file_path, backup_path)
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
                print(f"Added deprecation warning to {file_path}")
                
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def _create_warning_text(self, info: Dict) -> str:
        """Create deprecation warning text."""
        return f'''
import warnings

warnings.warn(
    "This module is deprecated as of Phase 1.5. "
    "Use {info['replacement']} instead. "
    "See {info['migration_guide']} for migration instructions. "
    "This module will be removed on {info['removal_date']}.",
    DeprecationWarning,
    stacklevel=2
)
'''
    
    def _insert_warning(self, content: str, warning_text: str) -> str:
        """Insert warning text at appropriate location."""
        lines = content.split('\n')
        
        # Find insertion point (after module docstring)
        insert_idx = 0
        in_docstring = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                elif stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                    insert_idx = i + 1
                    break
            elif not in_docstring and stripped and not stripped.startswith('#'):
                insert_idx = i
                break
        
        # Insert warning
        lines.insert(insert_idx, warning_text)
        return '\n'.join(lines)


class DeprecationCleaner:
    """Safely remove deprecated components with backup."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / "backups" / f"deprecated_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def create_backup(self) -> bool:
        """Create backup of all deprecated files."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in DEPRECATED_COMPONENTS.keys():
                full_path = self.project_root / file_path
                if full_path.exists():
                    # Create backup subdirectory structure
                    backup_file = self.backup_dir / file_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(full_path, backup_file)
                    
            # Create backup manifest
            manifest = {
                "backup_date": datetime.now().isoformat(),
                "files_backed_up": list(DEPRECATED_COMPONENTS.keys()),
                "project_root": str(self.project_root),
                "backup_reason": "JAX-native migration cleanup"
            }
            
            with open(self.backup_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
                
            print(f"Backup created at: {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return False
    
    def remove_deprecated_files(self, dry_run: bool = True) -> Dict[str, bool]:
        """Remove deprecated files after creating backup."""
        if not dry_run and not self.create_backup():
            print("Backup failed, aborting removal")
            return {}
        
        results = {}
        
        for file_path in DEPRECATED_COMPONENTS.keys():
            full_path = self.project_root / file_path
            
            if full_path.exists():
                try:
                    if not dry_run:
                        full_path.unlink()
                        print(f"Removed: {file_path}")
                    else:
                        print(f"Would remove: {file_path}")
                    results[file_path] = True
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
                    results[file_path] = False
            else:
                results[file_path] = False
                
        return results
    
    def rollback_from_backup(self, backup_path: Path) -> bool:
        """Restore files from backup."""
        try:
            manifest_path = backup_path / "manifest.json"
            if not manifest_path.exists():
                print("Invalid backup: no manifest found")
                return False
                
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Restore files
            for file_path in manifest["files_backed_up"]:
                backup_file = backup_path / file_path
                target_file = self.project_root / file_path
                
                if backup_file.exists():
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
                    print(f"Restored: {file_path}")
                    
            print("Rollback completed successfully")
            return True
            
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Cleanup deprecated legacy components")
    parser.add_argument('--analyze', action='store_true', help='Analyze deprecated code usage')
    parser.add_argument('--mark', action='store_true', help='Add deprecation warnings')
    parser.add_argument('--cleanup', action='store_true', help='Remove deprecated files')
    parser.add_argument('--rollback', type=str, help='Rollback from backup directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.mark, args.cleanup, args.rollback]):
        parser.print_help()
        return
    
    project_root = Path(__file__).parent.parent
    
    if args.analyze:
        print("Analyzing deprecated component usage...")
        analyzer = DeprecationAnalyzer(project_root)
        report = analyzer.generate_analysis_report()
        
        print(f"\nDeprecation Analysis Report:")
        print(f"Analysis Date: {report['analysis_date']}")
        print(f"Deprecated Files Found: {report['total_deprecated_files']}")
        print(f"Usage Locations: {report['total_usage_locations']}")
        
        if report['deprecated_usage']:
            print("\nFiles using deprecated components:")
            for file_path, imports in report['deprecated_usage'].items():
                print(f"  {file_path}:")
                for imp in imports:
                    print(f"    - {imp}")
        else:
            print("\nNo deprecated component usage found!")
            
        # Save detailed report
        report_path = project_root / "deprecation_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
    
    if args.mark:
        print("Adding deprecation warnings...")
        marker = DeprecationMarker(project_root)
        results = marker.add_deprecation_warnings(dry_run=args.dry_run)
        
        print(f"\nDeprecation marking results:")
        for file_path, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {file_path}: {status}")
    
    if args.cleanup:
        print("Removing deprecated files...")
        cleaner = DeprecationCleaner(project_root)
        results = cleaner.remove_deprecated_files(dry_run=args.dry_run)
        
        print(f"\nCleanup results:")
        for file_path, success in results.items():
            status = "REMOVED" if success else "FAILED/NOT_FOUND"
            print(f"  {file_path}: {status}")
    
    if args.rollback:
        print(f"Rolling back from backup: {args.rollback}")
        cleaner = DeprecationCleaner(project_root)
        success = cleaner.rollback_from_backup(Path(args.rollback))
        if success:
            print("Rollback completed successfully")
        else:
            print("Rollback failed")


if __name__ == "__main__":
    main()