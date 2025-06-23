#!/usr/bin/env python3
"""
Production SFT Dataset Collection Script

This script provides a robust interface for collecting large-scale SFT datasets
from PARENT_SCALE expert demonstrations. It builds upon the existing 
ExpertDemonstrationCollector infrastructure while adding production features:

- Configurable dataset sizes (small/medium/large)
- Curriculum-aware difficulty progression  
- Resumable collection with checkpointing
- Comprehensive logging and monitoring
- Resource-aware batch sizing

Usage:
    python scripts/collect_sft_dataset.py --size large --difficulty all --output-dir sft_data
    python scripts/collect_sft_dataset.py --resume checkpoints/collection_state.pkl
"""

import argparse
import logging
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as onp

# Import existing infrastructure
from causal_bayes_opt.training.expert_collection.collector import ExpertDemonstrationCollector
from causal_bayes_opt.training.expert_collection.data_structures import DemonstrationBatch
from causal_bayes_opt.training.curriculum import (
    DifficultyLevel, CurriculumManager, create_curriculum_manager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SFTCollectionConfig:
    """Configuration for SFT dataset collection."""
    
    # Dataset sizing
    size: str = "medium"  # small/medium/large
    target_demonstrations: int = field(init=False)
    
    # Curriculum settings
    difficulty_levels: List[str] = field(default_factory=lambda: ["all"])
    progressive_collection: bool = True
    
    # Collection parameters
    batch_size: int = 100
    min_accuracy: float = 0.7
    n_workers: int = 4
    parallel: bool = True
    
    # Output settings
    output_dir: str = "sft_datasets"
    checkpoint_interval: int = 500  # Save checkpoint every N demonstrations
    
    # Resource management
    memory_limit_gb: float = 16.0
    max_batch_size: int = 1000
    
    def __post_init__(self):
        """Set target demonstrations based on size."""
        size_mapping = {
            "small": 1_000,
            "medium": 10_000, 
            "large": 100_000,
            "xlarge": 500_000
        }
        self.target_demonstrations = size_mapping.get(self.size, 10_000)


@dataclass 
class CollectionState:
    """State for resumable collection."""
    
    config: SFTCollectionConfig
    demonstrations_collected: int = 0
    batches_saved: List[str] = field(default_factory=list)
    current_difficulty: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    last_checkpoint: float = field(default_factory=time.time)
    
    def save_checkpoint(self, checkpoint_path: Path):
        """Save collection state for resuming."""
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> 'CollectionState':
        """Load collection state for resuming."""
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return state


class SFTDatasetCollector:
    """Production SFT dataset collector with advanced features."""
    
    def __init__(self, config: SFTCollectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize expert collector
        self.expert_collector = ExpertDemonstrationCollector(
            output_dir=str(self.output_dir / "raw_demonstrations")
        )
        
        # Initialize curriculum manager if using difficulty progression
        if "all" not in config.difficulty_levels:
            self.curriculum_manager = create_curriculum_manager()
        else:
            self.curriculum_manager = None
            
        logger.info(f"SFT Collector initialized - Target: {config.target_demonstrations} demonstrations")
    
    def collect_dataset(self, resume_from: Optional[Path] = None) -> Path:
        """
        Collect complete SFT dataset with optional resuming.
        
        Args:
            resume_from: Path to checkpoint file for resuming
            
        Returns:
            Path to final dataset directory
        """
        # Load or create collection state
        if resume_from and resume_from.exists():
            state = CollectionState.load_checkpoint(resume_from)
            logger.info(f"Resuming collection from {state.demonstrations_collected} demonstrations")
        else:
            state = CollectionState(config=self.config)
            logger.info("Starting fresh collection")
        
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        try:
            # Collect by difficulty levels if specified
            if self.curriculum_manager and self.config.progressive_collection:
                self._collect_by_curriculum(state, checkpoint_dir)
            else:
                self._collect_standard(state, checkpoint_dir)
                
            # Generate final dataset summary
            summary_path = self._generate_dataset_summary(state)
            
            logger.info("🎉 SFT Dataset collection complete!")
            logger.info(f"   Total demonstrations: {state.demonstrations_collected}")
            logger.info(f"   Total time: {time.time() - state.start_time:.1f}s")
            logger.info(f"   Summary: {summary_path}")
            
            return self.output_dir
            
        except KeyboardInterrupt:
            logger.info("Collection interrupted - saving checkpoint...")
            self._save_checkpoint(state, checkpoint_dir / "interrupted_state.pkl")
            raise
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            self._save_checkpoint(state, checkpoint_dir / "error_state.pkl")
            raise
    
    def _collect_by_curriculum(self, state: CollectionState, checkpoint_dir: Path):
        """Collect demonstrations following curriculum progression."""
        difficulty_mapping = {
            "difficulty_1": ([3, 4], ["chain", "fork"]),
            "difficulty_2": ([5, 6], ["chain", "fork", "star"]),
            "difficulty_3": ([7, 8], ["chain", "fork", "star", "collider"]),
            "difficulty_4": ([9, 12], ["chain", "fork", "star", "collider", "random"]),
            "difficulty_5": ([15, 20], ["random"])
        }
        
        demos_per_difficulty = self.config.target_demonstrations // len(self.config.difficulty_levels)
        
        for difficulty in self.config.difficulty_levels:
            if difficulty not in difficulty_mapping:
                logger.warning(f"Unknown difficulty: {difficulty}, skipping")
                continue
                
            node_sizes, graph_types = difficulty_mapping[difficulty]
            state.current_difficulty = difficulty
            
            logger.info(f"Collecting {demos_per_difficulty} demonstrations for {difficulty}")
            logger.info(f"  Node sizes: {node_sizes}, Graph types: {graph_types}")
            
            self._collect_batch_with_params(
                state, checkpoint_dir, demos_per_difficulty,
                node_sizes, graph_types
            )
    
    def _collect_standard(self, state: CollectionState, checkpoint_dir: Path):
        """Collect demonstrations with standard mixed difficulty."""
        node_sizes = [3, 5, 8, 10, 12]
        graph_types = ["chain", "star", "fork", "collider"]
        
        self._collect_batch_with_params(
            state, checkpoint_dir, self.config.target_demonstrations,
            node_sizes, graph_types
        )
    
    def _collect_batch_with_params(
        self, 
        state: CollectionState, 
        checkpoint_dir: Path,
        target_demos: int,
        node_sizes: List[int],
        graph_types: List[str]
    ):
        """Collect batch with specific parameters."""
        demos_needed = target_demos - (state.demonstrations_collected % target_demos)
        
        while demos_needed > 0:
            # Calculate batch size based on memory constraints
            current_batch_size = min(
                self.config.batch_size,
                demos_needed,
                self._calculate_safe_batch_size(node_sizes)
            )
            
            logger.info(f"Collecting batch of {current_batch_size} demonstrations...")
            
            # Collect batch with fallback to serial processing
            batch = None
            parallel_failed = False
            
            if self.config.parallel:
                try:
                    batch = self.expert_collector.collect_demonstration_batch_parallel(
                        n_demonstrations=current_batch_size,
                        node_sizes=node_sizes,
                        graph_types=graph_types,
                        min_accuracy=self.config.min_accuracy,
                        n_workers=self.config.n_workers
                    )
                except Exception as e:
                    # Check if this is a pickle-related error
                    error_msg = str(e)
                    if "pickle" in error_msg.lower() or "local object" in error_msg.lower():
                        logger.warning(f"Parallel processing failed due to serialization issues: {e}")
                        logger.info("Falling back to serial processing...")
                        parallel_failed = True
                    else:
                        # Re-raise non-pickle errors
                        raise
            
            # Use serial processing if parallel is disabled or failed
            if not self.config.parallel or parallel_failed:
                batch = self.expert_collector.collect_demonstration_batch(
                    n_demonstrations=current_batch_size,
                    node_sizes=node_sizes,
                    graph_types=graph_types,
                    min_accuracy=self.config.min_accuracy
                )
                
                if parallel_failed:
                    logger.info("✅ Serial processing completed successfully")
                    # Disable parallel processing for remaining batches to avoid repeated failures
                    self.config.parallel = False
                    logger.info("Parallel processing disabled for remaining batches")
            
            # Save batch
            batch_path = self.expert_collector.save_batch(batch, format="pickle")
            state.batches_saved.append(str(batch_path))
            
            # Update state
            state.demonstrations_collected += len(batch.demonstrations)
            demos_needed -= len(batch.demonstrations)
            
            logger.info(f"Batch complete: {len(batch.demonstrations)} demos (Total: {state.demonstrations_collected})")
            
            # Save checkpoint if needed
            if (time.time() - state.last_checkpoint) > self.config.checkpoint_interval:
                self._save_checkpoint(state, checkpoint_dir / "latest_checkpoint.pkl")
                state.last_checkpoint = time.time()
    
    def _calculate_safe_batch_size(self, node_sizes: List[int]) -> int:
        """Calculate safe batch size based on memory constraints."""
        max_nodes = max(node_sizes)
        # Rough estimate: each demonstration takes ~max_nodes^2 * 8 bytes
        estimated_memory_per_demo = (max_nodes ** 2) * 8 / (1024 ** 3)  # GB
        safe_batch_size = int(self.config.memory_limit_gb / estimated_memory_per_demo)
        return min(safe_batch_size, self.config.max_batch_size)
    
    def _save_checkpoint(self, state: CollectionState, checkpoint_path: Path):
        """Save checkpoint with error handling."""
        try:
            state.save_checkpoint(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _generate_dataset_summary(self, state: CollectionState) -> Path:
        """Generate comprehensive dataset summary."""
        summary = {
            "collection_config": {
                "size": self.config.size,
                "target_demonstrations": self.config.target_demonstrations,
                "difficulty_levels": self.config.difficulty_levels,
                "min_accuracy": self.config.min_accuracy
            },
            "collection_results": {
                "total_demonstrations": state.demonstrations_collected,
                "total_batches": len(state.batches_saved),
                "collection_time_seconds": time.time() - state.start_time,
                "demonstrations_per_second": state.demonstrations_collected / (time.time() - state.start_time)
            },
            "batch_files": state.batches_saved,
            "collection_timestamp": time.time()
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary_path


def create_collection_config(args: argparse.Namespace) -> SFTCollectionConfig:
    """Create collection config from command line arguments."""
    difficulty_levels = args.difficulty if args.difficulty != ["all"] else ["all"]
    
    return SFTCollectionConfig(
        size=args.size,
        difficulty_levels=difficulty_levels,
        progressive_collection=args.progressive,
        batch_size=args.batch_size,
        min_accuracy=args.min_accuracy,
        n_workers=args.workers,
        parallel=not args.serial,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        memory_limit_gb=args.memory_limit
    )


def main():
    """CLI entry point for SFT dataset collection."""
    parser = argparse.ArgumentParser(
        description="Collect SFT datasets for ACBO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect small dataset for testing
  python scripts/collect_sft_dataset.py --size small --output-dir test_data
  
  # Collect large production dataset with curriculum
  python scripts/collect_sft_dataset.py --size large --difficulty difficulty_1 difficulty_2 difficulty_3
  
  # Resume interrupted collection
  python scripts/collect_sft_dataset.py --resume checkpoints/latest_checkpoint.pkl
        """
    )
    
    # Dataset configuration
    parser.add_argument("--size", choices=["small", "medium", "large", "xlarge"], 
                       default="medium", help="Dataset size")
    parser.add_argument("--difficulty", nargs="+", 
                       choices=["difficulty_1", "difficulty_2", "difficulty_3", "difficulty_4", "difficulty_5", "all"],
                       default=["all"], help="Difficulty levels to collect")
    parser.add_argument("--progressive", action="store_true", 
                       help="Use progressive curriculum collection")
    
    # Collection parameters
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Demonstrations per batch")
    parser.add_argument("--min-accuracy", type=float, default=0.7,
                       help="Minimum parent discovery accuracy")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--serial", action="store_true",
                       help="Use serial processing instead of parallel")
    
    # Output and checkpointing
    parser.add_argument("--output-dir", default="sft_datasets",
                       help="Output directory for datasets")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                       help="Checkpoint every N demonstrations")
    parser.add_argument("--resume", type=Path,
                       help="Resume from checkpoint file")
    
    # Resource management  
    parser.add_argument("--memory-limit", type=float, default=16.0,
                       help="Memory limit in GB")
    
    args = parser.parse_args()
    
    try:
        # Create collector
        config = create_collection_config(args)
        collector = SFTDatasetCollector(config)
        
        # Collect dataset
        dataset_path = collector.collect_dataset(resume_from=args.resume)
        
        print(f"\n🎉 SFT Dataset collection successful!")
        print(f"Dataset saved to: {dataset_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())