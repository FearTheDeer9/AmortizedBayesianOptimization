#!/usr/bin/env python3
"""
Analyze intervention patterns during evaluation.

This script tracks and analyzes:
- Which variables are intervened on
- Intervention sequences and patterns
- Diversity metrics
- Why models get stuck on single variables
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition, create_bc_acquisition,
    create_random_acquisition, create_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm,
    create_sparse_scm
)
from src.causal_bayes_opt.data_structures.scm import get_parents, get_target, get_edges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionTracker:
    """Track intervention patterns during evaluation."""
    
    def __init__(self):
        self.interventions = []
        self.sequences = defaultdict(list)
        
    def track_intervention(self, method, scm_name, step, intervention, outcome):
        """Record an intervention."""
        record = {
            'method': method,
            'scm_name': scm_name,
            'step': step,
            'targets': list(intervention.get('targets', set())),
            'values': intervention.get('values', {}),
            'outcome': outcome
        }
        self.interventions.append(record)
        
        # Track sequence for this method-scm pair
        key = (method, scm_name)
        self.sequences[key].append(record['targets'][0] if record['targets'] else None)
    
    def compute_diversity_metrics(self):
        """Compute intervention diversity metrics."""
        metrics = {}
        
        for (method, scm_name), sequence in self.sequences.items():
            # Filter out None values
            sequence = [s for s in sequence if s is not None]
            
            if not sequence:
                continue
            
            # Count unique interventions
            unique_vars = len(set(sequence))
            total_interventions = len(sequence)
            
            # Compute entropy
            counts = Counter(sequence)
            probs = np.array([count/total_interventions for count in counts.values()])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Compute switching frequency
            switches = sum(1 for i in range(1, len(sequence)) 
                          if sequence[i] != sequence[i-1])
            switch_rate = switches / (len(sequence) - 1) if len(sequence) > 1 else 0
            
            # Store metrics
            key = f"{method}_{scm_name}"
            metrics[key] = {
                'unique_vars': unique_vars,
                'total_interventions': total_interventions,
                'diversity_ratio': unique_vars / total_interventions,
                'entropy': float(entropy),
                'switch_rate': float(switch_rate),
                'most_common': counts.most_common(1)[0] if counts else ('None', 0)
            }
            
        return metrics


def run_tracked_evaluation(method_name, acquisition_fn, scms, config, tracker):
    """Run evaluation while tracking interventions."""
    evaluator = create_universal_evaluator()
    
    for scm_name, scm in scms:
        logger.info(f"Evaluating {method_name} on {scm_name}")
        
        # Monkey-patch the evaluator to track interventions
        original_evaluate = evaluator.evaluate
        
        def tracked_evaluate(*args, **kwargs):
            # Get the original result first
            result = original_evaluate(*args, **kwargs)
            
            # Extract interventions from history
            for step in result.history[1:]:  # Skip initial state
                tracker.track_intervention(
                    method_name, scm_name, step.step,
                    step.intervention, step.outcome_value
                )
            
            return result
        
        evaluator.evaluate = tracked_evaluate
        
        # Run evaluation
        result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            seed=config.get('seed', 42)
        )
        
        # Restore original method
        evaluator.evaluate = original_evaluate


def analyze_stuck_patterns(tracker):
    """Analyze why models get stuck on single variables."""
    stuck_patterns = []
    
    for (method, scm_name), sequence in tracker.sequences.items():
        if not sequence:
            continue
        
        # Check for repetitive patterns
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Check if stuck (more than 50% on same variable)
        counts = Counter(sequence)
        if counts:
            most_common_var, most_common_count = counts.most_common(1)[0]
            stuck_ratio = most_common_count / len(sequence)
            
            if stuck_ratio > 0.5:
                stuck_patterns.append({
                    'method': method,
                    'scm_name': scm_name,
                    'stuck_var': most_common_var,
                    'stuck_ratio': stuck_ratio,
                    'max_consecutive': max_consecutive,
                    'sequence_length': len(sequence)
                })
    
    return stuck_patterns


def plot_intervention_patterns(tracker, output_dir):
    """Create visualizations of intervention patterns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Intervention diversity by method
    diversity_metrics = tracker.compute_diversity_metrics()
    
    methods = defaultdict(list)
    for key, metrics in diversity_metrics.items():
        method = key.split('_')[0]
        methods[method].append(metrics['diversity_ratio'])
    
    plt.figure(figsize=(10, 6))
    method_names = list(methods.keys())
    diversity_values = [np.mean(methods[m]) for m in method_names]
    diversity_stds = [np.std(methods[m]) for m in method_names]
    
    plt.bar(method_names, diversity_values, yerr=diversity_stds, capsize=10)
    plt.ylabel('Diversity Ratio')
    plt.title('Intervention Diversity by Method')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_by_method.png')
    plt.close()
    
    # 2. Intervention sequences heatmap
    # Create a heatmap showing intervention patterns over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    method_data = defaultdict(lambda: defaultdict(list))
    for (method, scm_name), sequence in tracker.sequences.items():
        method_data[method][scm_name] = sequence
    
    for idx, (method, scm_data) in enumerate(method_data.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Create matrix for heatmap
        scm_names = sorted(scm_data.keys())
        max_length = max(len(seq) for seq in scm_data.values()) if scm_data else 0
        
        if max_length > 0:
            # Create variable mapping
            all_vars = set()
            for seq in scm_data.values():
                all_vars.update(s for s in seq if s is not None)
            var_to_idx = {var: i for i, var in enumerate(sorted(all_vars))}
            
            # Create matrix
            matrix = np.full((len(scm_names), max_length), -1)
            for i, scm_name in enumerate(scm_names):
                seq = scm_data[scm_name]
                for j, var in enumerate(seq):
                    if var is not None and var in var_to_idx:
                        matrix[i, j] = var_to_idx[var]
            
            # Plot heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='tab10')
            ax.set_title(f'{method} Intervention Sequences')
            ax.set_xlabel('Step')
            ax.set_ylabel('SCM')
            ax.set_yticks(range(len(scm_names)))
            ax.set_yticklabels(scm_names)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intervention_sequences.png')
    plt.close()
    
    # 3. Stuck patterns analysis
    stuck_patterns = analyze_stuck_patterns(tracker)
    
    if stuck_patterns:
        plt.figure(figsize=(12, 6))
        
        # Group by method
        method_stuck_counts = defaultdict(int)
        for pattern in stuck_patterns:
            method_stuck_counts[pattern['method']] += 1
        
        methods = list(method_stuck_counts.keys())
        counts = [method_stuck_counts[m] for m in methods]
        
        plt.bar(methods, counts)
        plt.ylabel('Number of Stuck Cases')
        plt.title('Stuck Pattern Occurrences by Method')
        plt.tight_layout()
        plt.savefig(output_dir / 'stuck_patterns.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze intervention patterns')
    parser.add_argument('--grpo_checkpoint', type=str, help='Path to GRPO checkpoint')
    parser.add_argument('--bc_checkpoint', type=str, help='Path to BC checkpoint')
    parser.add_argument('--n_interventions', type=int, default=20,
                       help='Number of interventions per evaluation')
    parser.add_argument('--output_dir', type=str, default='intervention_analysis',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Create test SCMs
    scms = [
        ('fork', create_fork_scm(noise_scale=1.0)),
        ('chain_3', create_chain_scm(chain_length=3)),
        ('chain_5', create_chain_scm(chain_length=5)),
        ('collider', create_collider_scm(noise_scale=1.0))
    ]
    
    # Evaluation config
    config = {
        'n_observational': 100,
        'max_interventions': args.n_interventions,
        'n_intervention_samples': 10,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Create tracker
    tracker = InterventionTracker()
    
    # Test each method
    logger.info("Testing Random baseline...")
    random_fn = create_random_acquisition(seed=42)
    run_tracked_evaluation("Random", random_fn, scms, config, tracker)
    
    logger.info("Testing Oracle...")
    for scm_name, scm in scms:
        # Create oracle for this specific SCM
        target = get_target(scm)
        scm_edges = {}
        edges = get_edges(scm)
        for parent, child in edges:
            if child not in scm_edges:
                scm_edges[child] = []
            scm_edges[child].append(parent)
        oracle_fn = create_oracle_acquisition(scm_edges, seed=42)
        run_tracked_evaluation("Oracle", oracle_fn, [(scm_name, scm)], config, tracker)
    
    if args.grpo_checkpoint:
        logger.info("Testing GRPO...")
        grpo_fn = create_grpo_acquisition(Path(args.grpo_checkpoint), seed=42)
        run_tracked_evaluation("GRPO", grpo_fn, scms, config, tracker)
    
    if args.bc_checkpoint:
        logger.info("Testing BC...")
        bc_fn = create_bc_acquisition(Path(args.bc_checkpoint), seed=42)
        run_tracked_evaluation("BC", bc_fn, scms, config, tracker)
    
    # Compute metrics
    diversity_metrics = tracker.compute_diversity_metrics()
    stuck_patterns = analyze_stuck_patterns(tracker)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'diversity_metrics.json', 'w') as f:
        json.dump(diversity_metrics, f, indent=2)
    
    with open(output_dir / 'stuck_patterns.json', 'w') as f:
        json.dump(stuck_patterns, f, indent=2)
    
    # Create plots
    plot_intervention_patterns(tracker, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INTERVENTION PATTERN SUMMARY")
    logger.info("="*60)
    
    for key, metrics in diversity_metrics.items():
        logger.info(f"\n{key}:")
        logger.info(f"  Diversity ratio: {metrics['diversity_ratio']:.3f}")
        logger.info(f"  Entropy: {metrics['entropy']:.3f}")
        logger.info(f"  Switch rate: {metrics['switch_rate']:.3f}")
        logger.info(f"  Most common: {metrics['most_common']}")
    
    if stuck_patterns:
        logger.info("\n" + "="*60)
        logger.info("STUCK PATTERNS DETECTED")
        logger.info("="*60)
        for pattern in stuck_patterns:
            logger.info(f"\n{pattern['method']} on {pattern['scm_name']}:")
            logger.info(f"  Stuck on: {pattern['stuck_var']}")
            logger.info(f"  Ratio: {pattern['stuck_ratio']:.3f}")
            logger.info(f"  Max consecutive: {pattern['max_consecutive']}")


if __name__ == "__main__":
    main()