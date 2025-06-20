#!/usr/bin/env python3
"""
Main Expert Demonstration Collection

Entry point for running expert demonstration collection with
standard configurations. Supports both serial and parallel collection.
"""

import argparse
import logging
from .collector import ExpertDemonstrationCollector


def collect_expert_demonstrations_main(
    n_demonstrations: int = 20,
    parallel: bool = True,
    n_workers: int = 4,
    output_dir: str = "demonstrations"
):
    """
    Main function for collecting expert demonstrations.
    
    Args:
        n_demonstrations: Number of demonstrations to collect
        parallel: Whether to use parallel processing
        n_workers: Number of parallel workers (if parallel=True)
        output_dir: Directory to save demonstrations
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    collector = ExpertDemonstrationCollector(output_dir=output_dir)
    
    # Choose collection method
    if parallel:
        print(f"Using parallel collection with {n_workers} workers")
        batch = collector.collect_demonstration_batch_parallel(
            n_demonstrations=n_demonstrations,
            node_sizes=[3, 5, 8, 10],  # Focus on smaller graphs initially
            graph_types=["chain", "star", "fork"],
            min_accuracy=0.7,
            n_workers=n_workers
        )
    else:
        print("Using serial collection")
        batch = collector.collect_demonstration_batch(
            n_demonstrations=n_demonstrations,
            node_sizes=[3, 5, 8, 10],
            graph_types=["chain", "star", "fork"],
            min_accuracy=0.7
        )
    
    # Save the batch
    saved_path = collector.save_batch(batch, format="pickle")
    
    print(f"\n🎉 Expert demonstration collection complete!")
    print(f"   Total demonstrations collected: {collector.demonstrations_collected}")
    print(f"   Total time spent: {collector.total_time_spent:.1f}s")
    print(f"   Saved to: {saved_path}")
    
    return batch


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for ACBO training")
    parser.add_argument("--n-demonstrations", type=int, default=20,
                        help="Number of demonstrations to collect")
    parser.add_argument("--serial", action="store_true",
                        help="Use serial processing instead of parallel")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default="demonstrations",
                        help="Output directory for demonstrations")
    
    args = parser.parse_args()
    
    collect_expert_demonstrations_main(
        n_demonstrations=args.n_demonstrations,
        parallel=not args.serial,
        n_workers=args.n_workers,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()