#!/usr/bin/env python3
"""
Main Expert Demonstration Collection

Entry point for running expert demonstration collection with
standard configurations.
"""

from .collector import ExpertDemonstrationCollector


def collect_expert_demonstrations_main():
    """Main function for collecting expert demonstrations."""
    collector = ExpertDemonstrationCollector()
    
    # Collect a diverse batch
    batch = collector.collect_demonstration_batch(
        n_demonstrations=20,  # Start with smaller batch for testing
        node_sizes=[3, 5, 8, 10],  # Focus on smaller graphs initially
        graph_types=["chain", "star", "fork"],
        min_accuracy=0.7
    )
    
    # Save the batch
    collector.save_batch(batch, format="pickle")
    
    print(f"\n🎉 Expert demonstration collection complete!")
    print(f"   Total demonstrations collected: {collector.demonstrations_collected}")
    print(f"   Total time spent: {collector.total_time_spent:.1f}s")


if __name__ == "__main__":
    collect_expert_demonstrations_main()