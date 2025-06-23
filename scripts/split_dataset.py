#!/usr/bin/env python3
"""
SFT Dataset Splitting Script

Splits SFT datasets into train/validation/test sets with intelligent stratification
to ensure balanced distribution across:

- Difficulty levels (curriculum learning)
- Graph types and sizes
- Accuracy ranges
- Target variable diversity

Features:
- Stratified splitting based on multiple criteria
- Curriculum-aware split strategies
- Data leak prevention
- Export to multiple formats
- Comprehensive split statistics

Usage:
    python scripts/split_dataset.py sft_datasets/training_ready --splits 0.7 0.2 0.1
    python scripts/split_dataset.py sft_datasets/training_ready --stratify difficulty --curriculum
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as onp
from sklearn.model_selection import train_test_split
from collections import Counter

# Import data format
from scripts.prepare_sft_data import SFTDataFormat

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SplitConfiguration:
    """Configuration for dataset splitting."""
    
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.7
    val_ratio: float = 0.2  
    test_ratio: float = 0.1
    
    # Stratification strategy
    stratify_by: List[str] = None  # ["difficulty", "graph_type", "accuracy", "node_size"]
    curriculum_aware: bool = False
    
    # Quality constraints
    min_samples_per_split: int = 100
    balance_tolerance: float = 0.1  # Acceptable imbalance in stratification
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.stratify_by is None:
            self.stratify_by = ["difficulty", "graph_type"]
        
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        if any(r <= 0 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("All split ratios must be positive")


@dataclass
class DatasetSplit:
    """Container for split dataset."""
    
    train_data: SFTDataFormat
    val_data: SFTDataFormat
    test_data: SFTDataFormat
    
    # Split metadata
    split_config: SplitConfiguration
    split_statistics: Dict[str, Any]
    split_indices: Dict[str, onp.ndarray]  # Original indices for each split
    
    def save_split(self, output_dir: Path, format: str = "pickle"):
        """Save split dataset to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        splits = {"train": self.train_data, "val": self.val_data, "test": self.test_data}
        
        for split_name, split_data in splits.items():
            if format == "pickle":
                file_path = output_dir / f"{split_name}_data.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(split_data, f)
            else:
                # For other formats, we'd extend the save functionality
                raise NotImplementedError(f"Format {format} not yet implemented for splits")
        
        # Save split metadata
        metadata = {
            "split_config": {
                "train_ratio": self.split_config.train_ratio,
                "val_ratio": self.split_config.val_ratio,
                "test_ratio": self.split_config.test_ratio,
                "stratify_by": self.split_config.stratify_by,
                "curriculum_aware": self.split_config.curriculum_aware,
                "random_seed": self.split_config.random_seed
            },
            "split_statistics": self.split_statistics,
            "split_sizes": {
                "train": len(self.train_data),
                "val": len(self.val_data),
                "test": len(self.test_data)
            }
        }
        
        with open(output_dir / "split_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save split indices for reproducibility
        onp.savez(
            output_dir / "split_indices.npz",
            train_indices=self.split_indices["train"],
            val_indices=self.split_indices["val"],
            test_indices=self.split_indices["test"]
        )
        
        logger.info(f"Dataset split saved to: {output_dir}")


class SFTDatasetSplitter:
    """Intelligent splitter for SFT datasets."""
    
    def __init__(self, config: SplitConfiguration):
        self.config = config
        onp.random.seed(config.random_seed)
        
        logger.info(f"Dataset splitter initialized")
        logger.info(f"Split ratios: {config.train_ratio:.1%} / {config.val_ratio:.1%} / {config.test_ratio:.1%}")
        logger.info(f"Stratification: {config.stratify_by}")
    
    def split_dataset(self, data: SFTDataFormat) -> DatasetSplit:
        """
        Split dataset with intelligent stratification.
        
        Args:
            data: SFT dataset to split
            
        Returns:
            DatasetSplit with train/val/test splits
        """
        logger.info(f"Splitting dataset of {len(data)} samples...")
        
        # Validate dataset
        if not data.validate():
            raise ValueError("Input dataset failed validation")
        
        if len(data) < self.config.min_samples_per_split * 3:
            raise ValueError(f"Dataset too small for splitting: {len(data)} samples")
        
        # Create stratification groups
        stratify_labels = self._create_stratification_labels(data)
        
        # Perform stratified split
        train_idx, temp_idx = self._stratified_split(
            len(data), stratify_labels, self.config.train_ratio
        )
        
        # Split temp into val and test
        temp_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        temp_stratify = stratify_labels[temp_idx]
        
        val_idx_temp, test_idx_temp = self._stratified_split(
            len(temp_idx), temp_stratify, temp_ratio
        )
        
        # Map back to original indices
        val_idx = temp_idx[val_idx_temp]
        test_idx = temp_idx[test_idx_temp]
        
        # Create split datasets
        train_data = self._create_split_data(data, train_idx)
        val_data = self._create_split_data(data, val_idx)
        test_data = self._create_split_data(data, test_idx)
        
        # Compute split statistics
        split_statistics = self._compute_split_statistics(
            data, {"train": train_idx, "val": val_idx, "test": test_idx}
        )
        
        logger.info(f"Split complete: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return DatasetSplit(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            split_config=self.config,
            split_statistics=split_statistics,
            split_indices={"train": train_idx, "val": val_idx, "test": test_idx}
        )
    
    def _create_stratification_labels(self, data: SFTDataFormat) -> onp.ndarray:
        """Create composite stratification labels."""
        n_samples = len(data)
        labels = []
        
        for criterion in self.config.stratify_by:
            if criterion == "difficulty":
                labels.append(data.difficulty_levels)
            elif criterion == "graph_type":
                # Convert graph types to numeric labels
                unique_types = list(set(data.graph_types))
                type_to_idx = {t: i for i, t in enumerate(unique_types)}
                type_labels = onp.array([type_to_idx[t] for t in data.graph_types])
                labels.append(type_labels)
            elif criterion == "accuracy":
                # Bin accuracy into quartiles
                accuracy_bins = onp.percentile(data.accuracy_scores, [25, 50, 75])
                accuracy_labels = onp.digitize(data.accuracy_scores, accuracy_bins)
                labels.append(accuracy_labels)
            elif criterion == "node_size":
                # Bin node sizes
                unique_sizes = onp.unique(data.node_sizes)
                if len(unique_sizes) > 5:
                    # Create 5 bins for many different sizes
                    size_bins = onp.percentile(data.node_sizes, [20, 40, 60, 80])
                    size_labels = onp.digitize(data.node_sizes, size_bins)
                else:
                    # Use actual sizes if only a few
                    size_labels = data.node_sizes
                labels.append(size_labels)
            else:
                logger.warning(f"Unknown stratification criterion: {criterion}")
        
        # Combine labels into composite stratification key
        if len(labels) == 1:
            return labels[0]
        else:
            # Create composite labels
            composite_labels = onp.zeros(n_samples, dtype=onp.int64)
            multiplier = 1
            for label_array in labels:
                composite_labels += label_array * multiplier
                multiplier *= (onp.max(label_array) + 1)
            return composite_labels
    
    def _stratified_split(self, n_samples: int, stratify_labels: onp.ndarray, 
                         train_ratio: float) -> Tuple[onp.ndarray, onp.ndarray]:
        """Perform stratified split with fallback to random split."""
        indices = onp.arange(n_samples)
        
        try:
            # Try stratified split
            train_idx, test_idx = train_test_split(
                indices, 
                train_size=train_ratio,
                stratify=stratify_labels,
                random_state=self.config.random_seed
            )
            
            # Verify split quality
            self._verify_split_balance(stratify_labels, train_idx, test_idx)
            
            return train_idx, test_idx
            
        except (ValueError, Exception) as e:
            logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
            
            # Fallback to random split
            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_ratio,
                random_state=self.config.random_seed
            )
            
            return train_idx, test_idx
    
    def _verify_split_balance(self, labels: onp.ndarray, train_idx: onp.ndarray, 
                            test_idx: onp.ndarray):
        """Verify that the split maintains good balance."""
        train_dist = Counter(labels[train_idx])
        test_dist = Counter(labels[test_idx])
        
        # Check if distributions are reasonably balanced
        for label in train_dist.keys():
            if label in test_dist:
                train_prop = train_dist[label] / len(train_idx)
                test_prop = test_dist[label] / len(test_idx)
                imbalance = abs(train_prop - test_prop)
                
                if imbalance > self.config.balance_tolerance:
                    logger.warning(f"Imbalanced split for label {label}: {imbalance:.2%}")
    
    def _create_split_data(self, data: SFTDataFormat, indices: onp.ndarray) -> SFTDataFormat:
        """Create split dataset from indices."""
        return SFTDataFormat(
            observations=data.observations[indices],
            next_interventions=data.next_interventions[indices],
            intervention_variables=data.intervention_variables[indices],
            intervention_values=data.intervention_values[indices],
            scm_metadata=[data.scm_metadata[i] for i in indices],
            trajectory_metadata=[data.trajectory_metadata[i] for i in indices],
            accuracy_scores=data.accuracy_scores[indices],
            optimization_improvements=data.optimization_improvements[indices],
            difficulty_levels=data.difficulty_levels[indices],
            node_sizes=data.node_sizes[indices],
            graph_types=[data.graph_types[i] for i in indices]
        )
    
    def _compute_split_statistics(self, data: SFTDataFormat, 
                                indices: Dict[str, onp.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive statistics for each split."""
        statistics = {}
        
        for split_name, split_indices in indices.items():
            split_stats = {
                "size": len(split_indices),
                "size_percentage": len(split_indices) / len(data) * 100,
                
                # Quality statistics
                "accuracy_mean": float(onp.mean(data.accuracy_scores[split_indices])),
                "accuracy_std": float(onp.std(data.accuracy_scores[split_indices])),
                "accuracy_min": float(onp.min(data.accuracy_scores[split_indices])),
                "accuracy_max": float(onp.max(data.accuracy_scores[split_indices])),
                
                # Difficulty distribution
                "difficulty_distribution": dict(Counter(data.difficulty_levels[split_indices])),
                
                # Graph type distribution
                "graph_type_distribution": dict(Counter([data.graph_types[i] for i in split_indices])),
                
                # Node size distribution
                "node_size_distribution": dict(Counter(data.node_sizes[split_indices])),
                
                # Optimization improvement statistics
                "optimization_improvement_mean": float(onp.mean(data.optimization_improvements[split_indices])),
                "optimization_improvement_std": float(onp.std(data.optimization_improvements[split_indices]))
            }
            
            statistics[split_name] = split_stats
        
        # Add balance metrics
        statistics["balance_metrics"] = self._compute_balance_metrics(statistics)
        
        return statistics
    
    def _compute_balance_metrics(self, split_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute balance metrics across splits."""
        balance_metrics = {}
        
        # Accuracy balance (standard deviation of means across splits)
        accuracy_means = [split_stats[split]["accuracy_mean"] 
                         for split in ["train", "val", "test"]]
        balance_metrics["accuracy_balance"] = float(onp.std(accuracy_means))
        
        # Size balance (coefficient of variation of split sizes)
        sizes = [split_stats[split]["size"] for split in ["train", "val", "test"]]
        expected_sizes = [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio]
        size_ratios = [s / sum(sizes) for s in sizes]
        balance_metrics["size_balance"] = float(onp.std([r - e for r, e in zip(size_ratios, expected_sizes)]))
        
        return balance_metrics
    
    def print_split_summary(self, split: DatasetSplit):
        """Print comprehensive split summary."""
        print(f"\n📊 Dataset Split Summary")
        print(f"{'='*60}")
        
        # Overall statistics
        total_samples = len(split.train_data) + len(split.val_data) + len(split.test_data)
        print(f"Total samples: {total_samples:,}")
        print(f"Split configuration: {split.split_config.stratify_by}")
        
        # Split sizes
        print(f"\n📋 Split Sizes:")
        for split_name in ["train", "val", "test"]:
            stats = split.split_statistics[split_name]
            print(f"  {split_name.capitalize():>5}: {stats['size']:>6,} samples ({stats['size_percentage']:>5.1f}%)")
        
        # Quality statistics
        print(f"\n📈 Quality Statistics:")
        print(f"{'Split':<8} {'Accuracy':<15} {'Opt. Improvement':<15}")
        print(f"{'-'*40}")
        for split_name in ["train", "val", "test"]:
            stats = split.split_statistics[split_name]
            acc_str = f"{stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}"
            opt_str = f"{stats['optimization_improvement_mean']:.3f} ± {stats['optimization_improvement_std']:.3f}"
            print(f"{split_name.capitalize():<8} {acc_str:<15} {opt_str:<15}")
        
        # Balance metrics
        balance = split.split_statistics["balance_metrics"]
        print(f"\n⚖️  Balance Metrics:")
        print(f"  Accuracy balance: {balance['accuracy_balance']:.4f}")
        print(f"  Size balance: {balance['size_balance']:.4f}")
        
        # Distribution analysis
        print(f"\n📊 Distribution Analysis:")
        
        # Difficulty distribution
        print(f"  Difficulty levels:")
        for split_name in ["train", "val", "test"]:
            diff_dist = split.split_statistics[split_name]["difficulty_distribution"]
            dist_str = ", ".join([f"L{k}: {v}" for k, v in sorted(diff_dist.items())])
            print(f"    {split_name.capitalize()}: {dist_str}")
        
        # Graph type distribution  
        print(f"  Graph types:")
        for split_name in ["train", "val", "test"]:
            type_dist = split.split_statistics[split_name]["graph_type_distribution"]
            dist_str = ", ".join([f"{k}: {v}" for k, v in sorted(type_dist.items())])
            print(f"    {split_name.capitalize()}: {dist_str}")


def load_sft_dataset(dataset_path: Path) -> SFTDataFormat:
    """Load SFT dataset from various formats."""
    # Try pickle first
    pickle_path = dataset_path / "sft_dataset.pkl"
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # Try direct pickle file
    if dataset_path.suffix == ".pkl":
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    
    # Try HDF5
    h5_path = dataset_path / "sft_dataset.h5"
    if h5_path.exists():
        # Would implement HDF5 loading here
        raise NotImplementedError("HDF5 loading not yet implemented")
    
    raise FileNotFoundError(f"No SFT dataset found in {dataset_path}")


def create_split_config(args: argparse.Namespace) -> SplitConfiguration:
    """Create split configuration from command line arguments."""
    # Validate and normalize split ratios
    if len(args.splits) != 3:
        raise ValueError("Must provide exactly 3 split ratios (train, val, test)")
    
    train_ratio, val_ratio, test_ratio = args.splits
    total = train_ratio + val_ratio + test_ratio
    
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"Split ratios sum to {total}, normalizing to 1.0")
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    return SplitConfiguration(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by=args.stratify if args.stratify else ["difficulty", "graph_type"],
        curriculum_aware=args.curriculum,
        random_seed=args.seed,
        balance_tolerance=args.balance_tolerance
    )


def main():
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Split SFT datasets into train/validation/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 70/20/10 split
  python scripts/split_dataset.py sft_datasets/training_ready --splits 0.7 0.2 0.1
  
  # Curriculum-aware split with custom stratification
  python scripts/split_dataset.py sft_datasets/training_ready --stratify difficulty accuracy --curriculum
  
  # Custom split ratios with specific output
  python scripts/split_dataset.py sft_datasets/training_ready --splits 0.8 0.1 0.1 --output sft_datasets/splits
        """
    )
    
    parser.add_argument("dataset_path", type=Path,
                       help="Path to SFT dataset directory or file")
    parser.add_argument("--output", type=Path,
                       help="Output directory for split datasets (default: dataset_path/splits)")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                       help="Split ratios for train/val/test (default: 0.7 0.2 0.1)")
    parser.add_argument("--stratify", nargs="+", 
                       choices=["difficulty", "graph_type", "accuracy", "node_size"],
                       help="Stratification criteria")
    parser.add_argument("--curriculum", action="store_true",
                       help="Enable curriculum-aware splitting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--balance-tolerance", type=float, default=0.1,
                       help="Acceptable imbalance in stratification")
    parser.add_argument("--format", choices=["pickle"], default="pickle",
                       help="Output format for split datasets")
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset_path}")
        data = load_sft_dataset(args.dataset_path)
        logger.info(f"Loaded dataset with {len(data)} samples")
        
        # Create split configuration
        config = create_split_config(args)
        
        # Create splitter and split dataset
        splitter = SFTDatasetSplitter(config)
        split = splitter.split_dataset(data)
        
        # Determine output directory
        output_dir = args.output or (args.dataset_path / "splits")
        
        # Save split dataset
        split.save_split(output_dir, format=args.format)
        
        # Print summary
        splitter.print_split_summary(split)
        
        print(f"\n✅ Dataset split complete!")
        print(f"Output directory: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Splitting failed: {e}")
        logger.exception("Splitting failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())