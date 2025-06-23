#!/usr/bin/env python3
"""
SFT Dataset Validation Script

Comprehensive validation and quality analysis for SFT datasets collected from
PARENT_SCALE expert demonstrations. Provides detailed reports on:

- Data integrity and format validation
- Parent discovery accuracy analysis  
- Dataset distribution statistics
- Quality metrics and recommendations

Usage:
    python scripts/validate_dataset.py sft_datasets/my_dataset
    python scripts/validate_dataset.py sft_datasets/my_dataset --detailed --export-report
"""

import argparse
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing infrastructure
from causal_bayes_opt.training.expert_collection.data_structures import (
    ExpertDemonstration, DemonstrationBatch
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for SFT dataset."""
    
    # Basic statistics
    total_demonstrations: int = 0
    total_batches: int = 0
    avg_accuracy: float = 0.0
    accuracy_std: float = 0.0
    
    # Distribution analysis
    node_size_distribution: Dict[int, int] = field(default_factory=dict)
    graph_type_distribution: Dict[str, int] = field(default_factory=dict)
    accuracy_distribution: List[float] = field(default_factory=list)
    
    # Quality metrics
    low_accuracy_demos: int = 0
    high_accuracy_demos: int = 0
    accuracy_threshold_low: float = 0.6
    accuracy_threshold_high: float = 0.9
    
    # Data integrity
    corrupted_demonstrations: int = 0
    missing_fields: List[str] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    
    # Diversity metrics
    unique_scm_structures: int = 0
    intervention_diversity: float = 0.0
    target_variable_coverage: Dict[int, int] = field(default_factory=dict)
    
    # Performance metrics
    avg_optimization_improvement: float = 0.0
    parent_discovery_precision: float = 0.0
    parent_discovery_recall: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "basic_statistics": {
                "total_demonstrations": self.total_demonstrations,
                "total_batches": self.total_batches,
                "avg_accuracy": self.avg_accuracy,
                "accuracy_std": self.accuracy_std
            },
            "distribution_analysis": {
                "node_size_distribution": self.node_size_distribution,
                "graph_type_distribution": self.graph_type_distribution,
                "accuracy_quartiles": self._compute_accuracy_quartiles()
            },
            "quality_metrics": {
                "low_accuracy_demos": self.low_accuracy_demos,
                "high_accuracy_demos": self.high_accuracy_demos,
                "quality_score": self._compute_quality_score()
            },
            "data_integrity": {
                "corrupted_demonstrations": self.corrupted_demonstrations,
                "missing_fields": self.missing_fields,
                "format_errors": self.format_errors,
                "integrity_score": self._compute_integrity_score()
            },
            "diversity_metrics": {
                "unique_scm_structures": self.unique_scm_structures,
                "intervention_diversity": self.intervention_diversity,
                "target_variable_coverage": self.target_variable_coverage,
                "diversity_score": self._compute_diversity_score()
            },
            "performance_metrics": {
                "avg_optimization_improvement": self.avg_optimization_improvement,
                "parent_discovery_precision": self.parent_discovery_precision,
                "parent_discovery_recall": self.parent_discovery_recall
            }
        }
    
    def _compute_accuracy_quartiles(self) -> Dict[str, float]:
        """Compute accuracy quartiles."""
        if not self.accuracy_distribution:
            return {}
        accuracies = onp.array(self.accuracy_distribution)
        return {
            "q25": float(onp.percentile(accuracies, 25)),
            "q50": float(onp.percentile(accuracies, 50)),
            "q75": float(onp.percentile(accuracies, 75)),
            "min": float(onp.min(accuracies)),
            "max": float(onp.max(accuracies))
        }
    
    def _compute_quality_score(self) -> float:
        """Compute overall quality score (0-1)."""
        if self.total_demonstrations == 0:
            return 0.0
        high_quality_ratio = self.high_accuracy_demos / self.total_demonstrations
        avg_accuracy_score = self.avg_accuracy
        return (high_quality_ratio + avg_accuracy_score) / 2
    
    def _compute_integrity_score(self) -> float:
        """Compute data integrity score (0-1)."""
        if self.total_demonstrations == 0:
            return 0.0
        corruption_ratio = self.corrupted_demonstrations / self.total_demonstrations
        return max(0.0, 1.0 - corruption_ratio)
    
    def _compute_diversity_score(self) -> float:
        """Compute diversity score (0-1)."""
        if self.total_demonstrations == 0:
            return 0.0
        # Combine multiple diversity factors
        structure_diversity = min(1.0, self.unique_scm_structures / 100)  # Normalize to 100 structures
        intervention_diversity = min(1.0, self.intervention_diversity)
        target_coverage = len(self.target_variable_coverage) / max(10, len(self.target_variable_coverage))
        return (structure_diversity + intervention_diversity + target_coverage) / 3


class SFTDatasetValidator:
    """Comprehensive validator for SFT datasets."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.metrics = ValidationMetrics()
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        logger.info(f"Initializing validator for: {self.dataset_path}")
    
    def validate_dataset(self, detailed: bool = False) -> ValidationMetrics:
        """
        Perform comprehensive dataset validation.
        
        Args:
            detailed: Whether to perform detailed analysis (slower)
            
        Returns:
            ValidationMetrics object with results
        """
        logger.info("Starting dataset validation...")
        
        # Load dataset summary if available
        summary_path = self.dataset_path / "dataset_summary.json"
        if summary_path.exists():
            self._load_dataset_summary(summary_path)
        
        # Find and validate batch files
        batch_files = self._find_batch_files()
        logger.info(f"Found {len(batch_files)} batch files")
        
        # Validate each batch
        all_demonstrations = []
        for i, batch_file in enumerate(batch_files):
            logger.info(f"Validating batch {i+1}/{len(batch_files)}: {batch_file.name}")
            demonstrations = self._validate_batch(batch_file)
            all_demonstrations.extend(demonstrations)
        
        # Compute overall metrics
        self._compute_basic_statistics(all_demonstrations)
        self._compute_distribution_analysis(all_demonstrations)
        self._compute_quality_metrics(all_demonstrations)
        
        if detailed:
            self._compute_detailed_metrics(all_demonstrations)
        
        logger.info("Dataset validation complete")
        return self.metrics
    
    def _load_dataset_summary(self, summary_path: Path):
        """Load dataset summary information."""
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            collection_results = summary.get("collection_results", {})
            self.metrics.total_demonstrations = collection_results.get("total_demonstrations", 0)
            self.metrics.total_batches = collection_results.get("total_batches", 0)
            
        except Exception as e:
            logger.warning(f"Could not load dataset summary: {e}")
    
    def _find_batch_files(self) -> List[Path]:
        """Find all batch files in the dataset."""
        batch_files = []
        
        # Look in raw_demonstrations subdirectory
        raw_demo_dir = self.dataset_path / "raw_demonstrations"
        if raw_demo_dir.exists():
            batch_files.extend(raw_demo_dir.glob("*.pkl"))
        
        # Look in root dataset directory
        batch_files.extend(self.dataset_path.glob("*.pkl"))
        
        # Filter out checkpoint files
        batch_files = [f for f in batch_files if "checkpoint" not in f.name.lower()]
        
        return sorted(batch_files)
    
    def _validate_batch(self, batch_file: Path) -> List[ExpertDemonstration]:
        """Validate a single batch file and return demonstrations."""
        try:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)
            
            # Handle different batch formats
            if isinstance(batch, DemonstrationBatch):
                demonstrations = batch.demonstrations
            elif isinstance(batch, list):
                demonstrations = batch
            else:
                self.metrics.format_errors.append(f"Unknown batch format in {batch_file.name}")
                return []
            
            # Validate each demonstration
            valid_demonstrations = []
            for demo in demonstrations:
                if self._validate_demonstration(demo):
                    valid_demonstrations.append(demo)
                else:
                    self.metrics.corrupted_demonstrations += 1
            
            return valid_demonstrations
            
        except Exception as e:
            self.metrics.format_errors.append(f"Could not load {batch_file.name}: {str(e)}")
            logger.error(f"Error validating batch {batch_file}: {e}")
            return []
    
    def _validate_demonstration(self, demo: ExpertDemonstration) -> bool:
        """Validate a single demonstration."""
        required_fields = ["accuracy", "n_nodes", "graph_type"]
        
        for field in required_fields:
            if not hasattr(demo, field) or getattr(demo, field) is None:
                self.metrics.missing_fields.append(field)
                return False
        
        # Validate accuracy range
        if not (0.0 <= demo.accuracy <= 1.0):
            return False
        
        # Validate node count
        if demo.n_nodes < 2:
            return False
        
        return True
    
    def _compute_basic_statistics(self, demonstrations: List[ExpertDemonstration]):
        """Compute basic statistical metrics."""
        if not demonstrations:
            return
        
        accuracies = [demo.accuracy for demo in demonstrations]
        
        self.metrics.total_demonstrations = len(demonstrations)
        self.metrics.avg_accuracy = float(onp.mean(accuracies))
        self.metrics.accuracy_std = float(onp.std(accuracies))
        self.metrics.accuracy_distribution = accuracies
    
    def _compute_distribution_analysis(self, demonstrations: List[ExpertDemonstration]):
        """Compute distribution analysis."""
        # Node size distribution
        for demo in demonstrations:
            size = demo.n_nodes
            self.metrics.node_size_distribution[size] = \
                self.metrics.node_size_distribution.get(size, 0) + 1
        
        # Graph type distribution  
        for demo in demonstrations:
            graph_type = demo.graph_type
            self.metrics.graph_type_distribution[graph_type] = \
                self.metrics.graph_type_distribution.get(graph_type, 0) + 1
    
    def _compute_quality_metrics(self, demonstrations: List[ExpertDemonstration]):
        """Compute quality metrics."""
        for demo in demonstrations:
            if demo.accuracy < self.metrics.accuracy_threshold_low:
                self.metrics.low_accuracy_demos += 1
            elif demo.accuracy >= self.metrics.accuracy_threshold_high:
                self.metrics.high_accuracy_demos += 1
    
    def _compute_detailed_metrics(self, demonstrations: List[ExpertDemonstration]):
        """Compute detailed metrics (slower analysis)."""
        logger.info("Computing detailed metrics...")
        
        # Unique SCM structures (simplified analysis)
        unique_structures = set()
        for demo in demonstrations:
            # Create a simple hash of the SCM structure
            structure_key = (demo.n_nodes, demo.graph_type, 
                           getattr(demo, 'n_edges', 0) if hasattr(demo, 'n_edges') else 0)
            unique_structures.add(structure_key)
        
        self.metrics.unique_scm_structures = len(unique_structures)
        
        # Intervention diversity (placeholder - would need access to intervention data)
        self.metrics.intervention_diversity = min(1.0, len(unique_structures) / 50)
        
        # Target variable coverage (placeholder)
        max_nodes = max(demo.n_nodes for demo in demonstrations)
        for i in range(max_nodes):
            self.metrics.target_variable_coverage[i] = max_nodes // 5  # Simplified
    
    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate comprehensive validation report."""
        if output_path is None:
            output_path = self.dataset_path / "validation_report.json"
        
        report = {
            "dataset_path": str(self.dataset_path),
            "validation_timestamp": onp.datetime64('now').astype(str),
            "metrics": self.metrics.to_dict(),
            "recommendations": self._generate_recommendations(),
            "summary": self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved: {output_path}")
        return output_path
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        if self.metrics.avg_accuracy < 0.7:
            recommendations.append("Consider increasing min_accuracy threshold for data collection")
        
        if self.metrics.corrupted_demonstrations > 0:
            recommendations.append(f"Clean {self.metrics.corrupted_demonstrations} corrupted demonstrations")
        
        if len(self.metrics.graph_type_distribution) < 3:
            recommendations.append("Increase graph type diversity for better generalization")
        
        if self.metrics.low_accuracy_demos / self.metrics.total_demonstrations > 0.2:
            recommendations.append("High proportion of low-accuracy demonstrations - review collection parameters")
        
        quality_score = self.metrics._compute_quality_score()
        if quality_score < 0.8:
            recommendations.append(f"Overall quality score ({quality_score:.2f}) could be improved")
        
        return recommendations
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        quality_score = self.metrics._compute_quality_score()
        integrity_score = self.metrics._compute_integrity_score()
        
        status = "✅ EXCELLENT" if quality_score > 0.9 else \
                "🟡 GOOD" if quality_score > 0.7 else \
                "🔴 NEEDS IMPROVEMENT"
        
        return (f"Dataset Status: {status} | "
                f"Quality: {quality_score:.2f} | "
                f"Integrity: {integrity_score:.2f} | "
                f"Demos: {self.metrics.total_demonstrations}")
    
    def create_visualizations(self, output_dir: Optional[Path] = None):
        """Create visualization plots for the dataset."""
        if output_dir is None:
            output_dir = self.dataset_path / "visualizations"
        
        output_dir.mkdir(exist_ok=True)
        
        # Accuracy distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.metrics.accuracy_distribution, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Parent Discovery Accuracy')
        plt.axvline(self.metrics.avg_accuracy, color='red', linestyle='--', 
                   label=f'Mean: {self.metrics.avg_accuracy:.3f}')
        plt.legend()
        plt.savefig(output_dir / "accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Node size distribution
        if self.metrics.node_size_distribution:
            plt.figure(figsize=(10, 6))
            sizes = list(self.metrics.node_size_distribution.keys())
            counts = list(self.metrics.node_size_distribution.values())
            plt.bar(sizes, counts, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Nodes')
            plt.ylabel('Number of Demonstrations')
            plt.title('Distribution of Graph Sizes')
            plt.savefig(output_dir / "node_size_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Graph type distribution
        if self.metrics.graph_type_distribution:
            plt.figure(figsize=(10, 6))
            types = list(self.metrics.graph_type_distribution.keys())
            counts = list(self.metrics.graph_type_distribution.values())
            plt.bar(types, counts, alpha=0.7, edgecolor='black')
            plt.xlabel('Graph Type')
            plt.ylabel('Number of Demonstrations')
            plt.title('Distribution of Graph Types')
            plt.xticks(rotation=45)
            plt.savefig(output_dir / "graph_type_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to: {output_dir}")


def main():
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate SFT datasets for quality and integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python scripts/validate_dataset.py sft_datasets/medium_dataset
  
  # Detailed validation with report
  python scripts/validate_dataset.py sft_datasets/large_dataset --detailed --export-report
  
  # Validation with visualizations
  python scripts/validate_dataset.py sft_datasets/test_dataset --visualizations
        """
    )
    
    parser.add_argument("dataset_path", type=Path,
                       help="Path to dataset directory")
    parser.add_argument("--detailed", action="store_true",
                       help="Perform detailed analysis (slower)")
    parser.add_argument("--export-report", action="store_true",
                       help="Export JSON validation report")
    parser.add_argument("--visualizations", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for reports and visualizations")
    
    args = parser.parse_args()
    
    try:
        # Create validator
        validator = SFTDatasetValidator(args.dataset_path)
        
        # Validate dataset
        metrics = validator.validate_dataset(detailed=args.detailed)
        
        # Print summary
        print(f"\n📊 Dataset Validation Results")
        print(f"{'='*50}")
        print(f"Total Demonstrations: {metrics.total_demonstrations:,}")
        print(f"Average Accuracy: {metrics.avg_accuracy:.3f} ± {metrics.accuracy_std:.3f}")
        print(f"Quality Score: {metrics._compute_quality_score():.3f}")
        print(f"Integrity Score: {metrics._compute_integrity_score():.3f}")
        print(f"Summary: {validator._generate_summary()}")
        
        # Export report if requested
        if args.export_report:
            report_path = validator.generate_report(
                args.output_dir / "validation_report.json" if args.output_dir else None
            )
            print(f"\n📄 Report exported: {report_path}")
        
        # Create visualizations if requested
        if args.visualizations:
            validator.create_visualizations(
                args.output_dir / "visualizations" if args.output_dir else None
            )
            print(f"📈 Visualizations created")
        
        # Print recommendations
        recommendations = validator._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n✅ Validation complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logger.exception("Validation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())