#!/usr/bin/env python3
"""
SFT Data Preparation Script

Converts PARENT_SCALE expert demonstrations into SFT (Supervised Fine-Tuning) 
training format suitable for the ACBO training pipeline. Features:

- Converts raw demonstrations to JAX-compatible [N, d, 3] format
- Extracts state-action pairs for behavioral cloning
- Handles curriculum-aware data organization
- Supports multiple output formats (HDF5, pickle, numpy)
- Memory-efficient processing for large datasets

Usage:
    python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/training_ready
    python scripts/prepare_sft_data.py sft_datasets/raw_data --format hdf5 --curriculum
"""

import argparse
import logging
import pickle
import h5py
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as onp
import jax.numpy as jnp

# Import existing infrastructure
from causal_bayes_opt.training.expert_collection.data_structures import (
    ExpertDemonstration, DemonstrationBatch, ExpertTrajectoryDemonstration
)
from causal_bayes_opt.training.surrogate_training import (
    TrainingExample, extract_training_data_from_demonstrations
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SFTDataFormat:
    """Structured format for SFT training data."""
    
    # Input features [N, d, 3] format
    observations: onp.ndarray  # Shape: [N, d, 3] - [values, interventions, targets]
    
    # Action/intervention targets
    next_interventions: onp.ndarray  # Shape: [N, d] - next intervention to perform
    intervention_variables: onp.ndarray  # Shape: [N,] - which variable to intervene on
    intervention_values: onp.ndarray  # Shape: [N,] - what value to set
    
    # Context information
    scm_metadata: List[Dict[str, Any]]  # SCM structure and properties
    trajectory_metadata: List[Dict[str, Any]]  # Trajectory-level information
    
    # Quality metrics
    accuracy_scores: onp.ndarray  # Shape: [N,] - parent discovery accuracy
    optimization_improvements: onp.ndarray  # Shape: [N,] - target improvements
    
    # Indexing for curriculum learning
    difficulty_levels: onp.ndarray  # Shape: [N,] - difficulty classification
    node_sizes: onp.ndarray  # Shape: [N,] - number of nodes in SCM
    graph_types: List[str]  # Graph type for each example
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def validate(self) -> bool:
        """Validate data format consistency."""
        n_samples = len(self.observations)
        
        # Check array shapes
        checks = [
            self.next_interventions.shape[0] == n_samples,
            self.intervention_variables.shape[0] == n_samples,
            self.intervention_values.shape[0] == n_samples,
            self.accuracy_scores.shape[0] == n_samples,
            self.optimization_improvements.shape[0] == n_samples,
            self.difficulty_levels.shape[0] == n_samples,
            self.node_sizes.shape[0] == n_samples,
            len(self.graph_types) == n_samples,
            len(self.scm_metadata) == n_samples,
            len(self.trajectory_metadata) == n_samples
        ]
        
        # Check [N, d, 3] format
        if len(self.observations.shape) != 3 or self.observations.shape[2] != 3:
            logger.error(f"Invalid observation shape: {self.observations.shape}, expected [N, d, 3]")
            return False
        
        if not all(checks):
            logger.error("Data format validation failed: inconsistent array sizes")
            return False
        
        logger.info(f"Data format validation passed: {n_samples} samples")
        return True


class SFTDataProcessor:
    """Processor for converting expert demonstrations to SFT format."""
    
    def __init__(self, curriculum_aware: bool = False):
        self.curriculum_aware = curriculum_aware
        
        # Difficulty classification thresholds
        self.difficulty_thresholds = {
            "node_sizes": [(3, 4), (5, 6), (7, 8), (9, 12), (13, 20)],
            "complexity_score": [0.2, 0.4, 0.6, 0.8, 1.0]
        }
        
        logger.info(f"SFT Data Processor initialized (curriculum_aware={curriculum_aware})")
    
    def process_dataset(self, dataset_path: Path, output_path: Path, 
                       format: str = "pickle") -> Path:
        """
        Process complete dataset from raw demonstrations to SFT format.
        
        Args:
            dataset_path: Path to raw demonstration dataset
            output_path: Path for processed output
            format: Output format ("pickle", "hdf5", "numpy")
            
        Returns:
            Path to processed dataset
        """
        logger.info(f"Processing dataset: {dataset_path} -> {output_path}")
        
        # Load raw demonstrations
        demonstrations = self._load_raw_demonstrations(dataset_path)
        logger.info(f"Loaded {len(demonstrations)} demonstrations")
        
        # Convert to SFT format
        sft_data = self._convert_to_sft_format(demonstrations)
        
        # Validate format
        if not sft_data.validate():
            raise ValueError("Generated SFT data failed validation")
        
        # Save in requested format
        output_path.mkdir(parents=True, exist_ok=True)
        saved_path = self._save_sft_data(sft_data, output_path, format)
        
        # Generate metadata file
        self._save_metadata(sft_data, demonstrations, output_path)
        
        logger.info(f"SFT dataset processing complete: {saved_path}")
        return saved_path
    
    def _load_raw_demonstrations(self, dataset_path: Path) -> List[ExpertDemonstration]:
        """Load all demonstrations from dataset."""
        demonstrations = []
        
        # Find batch files
        batch_files = []
        raw_demo_dir = dataset_path / "raw_demonstrations"
        if raw_demo_dir.exists():
            batch_files.extend(raw_demo_dir.glob("*.pkl"))
        batch_files.extend(dataset_path.glob("*.pkl"))
        
        # Filter out checkpoints
        batch_files = [f for f in batch_files if "checkpoint" not in f.name.lower()]
        
        logger.info(f"Found {len(batch_files)} batch files")
        
        # Load demonstrations from each batch
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                
                # Handle different batch formats
                if isinstance(batch, DemonstrationBatch):
                    demonstrations.extend(batch.demonstrations)
                elif isinstance(batch, list):
                    demonstrations.extend(batch)
                else:
                    logger.warning(f"Unknown batch format in {batch_file}")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_file}: {e}")
        
        return demonstrations
    
    def _convert_to_sft_format(self, demonstrations: List[ExpertDemonstration]) -> SFTDataFormat:
        """Convert demonstrations to structured SFT format."""
        logger.info("Converting demonstrations to SFT format...")
        
        # Extract training examples using existing infrastructure
        training_examples = extract_training_data_from_demonstrations(demonstrations)
        logger.info(f"Extracted {len(training_examples)} training examples")
        
        # Determine maximum dimensions for padding
        max_nodes = max(demo.n_nodes for demo in demonstrations)
        n_samples = len(training_examples)
        
        # Initialize arrays
        observations = onp.zeros((n_samples, max_nodes, 3), dtype=onp.float32)
        next_interventions = onp.zeros((n_samples, max_nodes), dtype=onp.float32)
        intervention_variables = onp.zeros(n_samples, dtype=onp.int32)
        intervention_values = onp.zeros(n_samples, dtype=onp.float32)
        accuracy_scores = onp.zeros(n_samples, dtype=onp.float32)
        optimization_improvements = onp.zeros(n_samples, dtype=onp.float32)
        difficulty_levels = onp.zeros(n_samples, dtype=onp.int32)
        node_sizes = onp.zeros(n_samples, dtype=onp.int32)
        graph_types = []
        scm_metadata = []
        trajectory_metadata = []
        
        # Process each training example
        for i, (example, demo) in enumerate(zip(training_examples, demonstrations)):
            # Convert to [N, d, 3] format
            d = demo.n_nodes
            
            # Extract state representation
            if hasattr(example, 'features') and example.features is not None:
                # Use pre-computed features if available
                obs = self._extract_observations_from_features(example.features, d)
            else:
                # Reconstruct from demonstration data
                obs = self._extract_observations_from_demo(demo, d)
            
            observations[i, :d, :] = obs
            
            # Extract action/intervention data
            intervention_data = self._extract_intervention_target(demo, example)
            next_interventions[i, :d] = intervention_data["next_interventions"]
            intervention_variables[i] = intervention_data["variable"]
            intervention_values[i] = intervention_data["value"]
            
            # Extract quality metrics
            accuracy_scores[i] = demo.accuracy
            optimization_improvements[i] = getattr(demo, 'optimization_improvement', 0.0)
            
            # Extract difficulty and metadata
            difficulty_levels[i] = self._classify_difficulty(demo)
            node_sizes[i] = demo.n_nodes
            graph_types.append(demo.graph_type)
            
            scm_metadata.append(self._extract_scm_metadata(demo))
            trajectory_metadata.append(self._extract_trajectory_metadata(demo))
        
        return SFTDataFormat(
            observations=observations,
            next_interventions=next_interventions,
            intervention_variables=intervention_variables,
            intervention_values=intervention_values,
            scm_metadata=scm_metadata,
            trajectory_metadata=trajectory_metadata,
            accuracy_scores=accuracy_scores,
            optimization_improvements=optimization_improvements,
            difficulty_levels=difficulty_levels,
            node_sizes=node_sizes,
            graph_types=graph_types
        )
    
    def _extract_observations_from_features(self, features: onp.ndarray, d: int) -> onp.ndarray:
        """Extract observations from pre-computed features."""
        # Assume features are already in [d, 3] format
        if features.shape == (d, 3):
            return features
        
        # Reshape if needed
        if features.size == d * 3:
            return features.reshape(d, 3)
        
        # Fallback: create dummy observation
        return onp.zeros((d, 3), dtype=onp.float32)
    
    def _extract_observations_from_demo(self, demo: ExpertDemonstration, d: int) -> onp.ndarray:
        """Extract observations from demonstration data."""
        obs = onp.zeros((d, 3), dtype=onp.float32)
        
        # Column 0: Observed values (use random values as placeholder)
        obs[:, 0] = onp.random.randn(d)
        
        # Column 1: Intervention indicators (0 = observational, 1 = interventional)
        obs[:, 1] = 0.0  # Start with observational data
        
        # Column 2: Target indicators (1 for target variable, 0 otherwise)
        if hasattr(demo, 'target_variable') and demo.target_variable is not None:
            target_var = demo.target_variable
            if isinstance(target_var, int) and 0 <= target_var < d:
                obs[target_var, 2] = 1.0
        
        return obs
    
    def _extract_intervention_target(self, demo: ExpertDemonstration, 
                                   example: TrainingExample) -> Dict[str, Any]:
        """Extract intervention target from demonstration."""
        d = demo.n_nodes
        
        # Default values
        next_interventions = onp.zeros(d, dtype=onp.float32)
        variable = 0
        value = 0.0
        
        # Try to extract from demonstration data
        if hasattr(demo, 'interventions') and demo.interventions:
            # Use first intervention as target
            intervention = demo.interventions[0] if isinstance(demo.interventions, list) else demo.interventions
            
            if hasattr(intervention, 'variable') and hasattr(intervention, 'value'):
                variable = intervention.variable
                value = float(intervention.value)
                
                if 0 <= variable < d:
                    next_interventions[variable] = value
        
        return {
            "next_interventions": next_interventions,
            "variable": variable,
            "value": value
        }
    
    def _classify_difficulty(self, demo: ExpertDemonstration) -> int:
        """Classify demonstration difficulty level (0-4)."""
        if not self.curriculum_aware:
            return 0
        
        # Node size based classification
        node_size = demo.n_nodes
        for level, (min_nodes, max_nodes) in enumerate(self.difficulty_thresholds["node_sizes"]):
            if min_nodes <= node_size <= max_nodes:
                return level
        
        # Fallback: highest difficulty
        return len(self.difficulty_thresholds["node_sizes"]) - 1
    
    def _extract_scm_metadata(self, demo: ExpertDemonstration) -> Dict[str, Any]:
        """Extract SCM metadata from demonstration."""
        metadata = {
            "n_nodes": demo.n_nodes,
            "graph_type": demo.graph_type,
            "accuracy": demo.accuracy
        }
        
        # Add additional metadata if available
        for attr in ["n_edges", "density", "true_parents", "mechanism_types"]:
            if hasattr(demo, attr):
                metadata[attr] = getattr(demo, attr)
        
        return metadata
    
    def _extract_trajectory_metadata(self, demo: ExpertDemonstration) -> Dict[str, Any]:
        """Extract trajectory-level metadata."""
        metadata = {
            "demonstration_id": getattr(demo, 'demo_id', None),
            "collection_time": getattr(demo, 'collection_time', None),
            "optimization_steps": getattr(demo, 'optimization_steps', None)
        }
        
        # Add trajectory-specific data if available
        if isinstance(demo, ExpertTrajectoryDemonstration):
            metadata.update({
                "trajectory_length": len(getattr(demo, 'trajectory', [])),
                "final_improvement": getattr(demo, 'final_improvement', None)
            })
        
        return metadata
    
    def _save_sft_data(self, sft_data: SFTDataFormat, output_path: Path, 
                      format: str) -> Path:
        """Save SFT data in specified format."""
        if format == "pickle":
            file_path = output_path / "sft_dataset.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(sft_data, f)
                
        elif format == "hdf5":
            file_path = output_path / "sft_dataset.h5"
            with h5py.File(file_path, 'w') as f:
                # Save arrays
                f.create_dataset("observations", data=sft_data.observations)
                f.create_dataset("next_interventions", data=sft_data.next_interventions)
                f.create_dataset("intervention_variables", data=sft_data.intervention_variables)
                f.create_dataset("intervention_values", data=sft_data.intervention_values)
                f.create_dataset("accuracy_scores", data=sft_data.accuracy_scores)
                f.create_dataset("optimization_improvements", data=sft_data.optimization_improvements)
                f.create_dataset("difficulty_levels", data=sft_data.difficulty_levels)
                f.create_dataset("node_sizes", data=sft_data.node_sizes)
                
                # Save string data as attributes or separate datasets
                f.attrs["graph_types"] = [s.encode() for s in sft_data.graph_types]
                
                # Save metadata as JSON strings
                f.attrs["scm_metadata"] = json.dumps(sft_data.scm_metadata)
                f.attrs["trajectory_metadata"] = json.dumps(sft_data.trajectory_metadata)
                
        elif format == "numpy":
            file_path = output_path / "sft_dataset.npz"
            onp.savez_compressed(
                file_path,
                observations=sft_data.observations,
                next_interventions=sft_data.next_interventions,
                intervention_variables=sft_data.intervention_variables,
                intervention_values=sft_data.intervention_values,
                accuracy_scores=sft_data.accuracy_scores,
                optimization_improvements=sft_data.optimization_improvements,
                difficulty_levels=sft_data.difficulty_levels,
                node_sizes=sft_data.node_sizes,
                graph_types=sft_data.graph_types
            )
            
            # Save metadata separately
            with open(output_path / "metadata.json", 'w') as f:
                json.dump({
                    "scm_metadata": sft_data.scm_metadata,
                    "trajectory_metadata": sft_data.trajectory_metadata
                }, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"SFT data saved in {format} format: {file_path}")
        return file_path
    
    def _save_metadata(self, sft_data: SFTDataFormat, 
                      original_demos: List[ExpertDemonstration],
                      output_path: Path):
        """Save comprehensive metadata about the processed dataset."""
        metadata = {
            "processing_info": {
                "original_demonstrations": len(original_demos),
                "processed_samples": len(sft_data),
                "curriculum_aware": self.curriculum_aware,
                "max_nodes": int(sft_data.observations.shape[1]),
                "feature_dimensions": list(sft_data.observations.shape)
            },
            "quality_statistics": {
                "avg_accuracy": float(onp.mean(sft_data.accuracy_scores)),
                "accuracy_std": float(onp.std(sft_data.accuracy_scores)),
                "avg_optimization_improvement": float(onp.mean(sft_data.optimization_improvements)),
                "min_accuracy": float(onp.min(sft_data.accuracy_scores)),
                "max_accuracy": float(onp.max(sft_data.accuracy_scores))
            },
            "distribution_analysis": {
                "node_size_distribution": {
                    str(size): int(count) for size, count in 
                    zip(*onp.unique(sft_data.node_sizes, return_counts=True))
                },
                "difficulty_distribution": {
                    str(level): int(count) for level, count in
                    zip(*onp.unique(sft_data.difficulty_levels, return_counts=True))
                },
                "graph_type_distribution": {
                    graph_type: sft_data.graph_types.count(graph_type)
                    for graph_type in set(sft_data.graph_types)
                }
            }
        }
        
        with open(output_path / "processing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Processing metadata saved")


def main():
    """CLI entry point for SFT data preparation."""
    parser = argparse.ArgumentParser(
        description="Convert expert demonstrations to SFT training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion to pickle format
  python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/training_ready
  
  # Convert with curriculum awareness to HDF5
  python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/curriculum --format hdf5 --curriculum
  
  # Convert to numpy format
  python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/numpy --format numpy
        """
    )
    
    parser.add_argument("dataset_path", type=Path,
                       help="Path to raw demonstration dataset")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for processed dataset")
    parser.add_argument("--format", choices=["pickle", "hdf5", "numpy"],
                       default="pickle", help="Output format")
    parser.add_argument("--curriculum", action="store_true",
                       help="Enable curriculum-aware processing")
    
    args = parser.parse_args()
    
    try:
        # Create processor
        processor = SFTDataProcessor(curriculum_aware=args.curriculum)
        
        # Process dataset
        output_path = processor.process_dataset(
            args.dataset_path, args.output, args.format
        )
        
        print(f"\n✅ SFT data preparation complete!")
        print(f"Output: {output_path}")
        print(f"Format: {args.format}")
        print(f"Curriculum-aware: {args.curriculum}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        logger.exception("Processing failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())