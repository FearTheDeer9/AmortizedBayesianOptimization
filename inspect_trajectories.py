#!/usr/bin/env python3
"""
Inspect collected SFT trajectories and posterior histories.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
from pathlib import Path

def inspect_dataset(dataset_dir: str):
    """Inspect collected SFT dataset."""
    dataset_path = Path(dataset_dir)
    
    print(f"🔍 Inspecting dataset: {dataset_path}")
    
    # Check summary
    summary_file = dataset_path / "collection_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        print(f"\n📊 Collection Summary:")
        config = summary.get('collection_config', {})
        results = summary.get('collection_results', {})
        
        print(f"  Target demonstrations: {config.get('target_demonstrations', 'unknown')}")
        print(f"  Collected demonstrations: {results.get('total_demonstrations', 'unknown')}")
        print(f"  Success rate: {results.get('total_demonstrations', 0) / config.get('target_demonstrations', 1) * 100:.1f}%")
        print(f"  Collection time: {results.get('collection_time_seconds', 0):.1f}s")
        print(f"  Rate: {results.get('demonstrations_per_second', 0):.2f} demos/sec")
    
    # Check batch files
    batch_dir = dataset_path / "batches"
    if batch_dir.exists():
        batch_files = list(batch_dir.glob("batch_*.pkl"))
        print(f"\n📦 Found {len(batch_files)} batch files")
        
        if batch_files:
            # Inspect first batch
            print(f"\n🔬 Inspecting first batch: {batch_files[0].name}")
            with open(batch_files[0], 'rb') as f:
                batch = pickle.load(f)
            
            print(f"  Batch size: {len(batch.demonstrations)}")
            
            if batch.demonstrations:
                demo = batch.demonstrations[0]
                print(f"\n🎯 First demonstration:")
                print(f"    Target: {demo.target_variable}")
                print(f"    Nodes: {demo.n_nodes}")
                print(f"    Graph type: {demo.graph_type}")
                print(f"    Discovered parents: {demo.discovered_parents}")
                print(f"    Accuracy: {demo.accuracy:.3f}")
                print(f"    Confidence: {demo.confidence:.3f}")
                print(f"    Inference time: {demo.inference_time:.1f}s")
                
                # Check posterior history
                posterior_history = demo.parent_posterior.get('posterior_history', [])
                print(f"    Posterior states: {len(posterior_history)}")
                
                if posterior_history:
                    print(f"\n📈 Posterior Evolution:")
                    for i, state in enumerate(posterior_history):
                        posterior = state.get('posterior', {})
                        if posterior:
                            best_parents = max(posterior.keys(), key=lambda x: posterior[x])
                            best_prob = posterior[best_parents]
                            print(f"      State {i}: {best_parents} (p={best_prob:.3f})")
                
                # Check trajectory
                trajectory = demo.parent_posterior.get('trajectory', {})
                if isinstance(trajectory, dict):
                    interventions = trajectory.get('intervention_sequence', [])
                    outcomes = trajectory.get('target_outcomes', [])
                    print(f"\n🛤️  Trajectory:")
                    print(f"      Interventions: {len(interventions)}")
                    if interventions:
                        print(f"      First 3: {interventions[:3]}")
                    if outcomes:
                        print(f"      Outcomes: [{outcomes[0]:.3f}, ..., {outcomes[-1]:.3f}]")
    
    # Check final dataset
    final_dataset = dataset_path / "final_dataset.pkl"
    if final_dataset.exists():
        print(f"\n📁 Final dataset: {final_dataset}")
        print(f"   Size: {final_dataset.stat().st_size / 1024 / 1024:.1f} MB")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect SFT dataset")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_dir).exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return 1
    
    inspect_dataset(args.dataset_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())