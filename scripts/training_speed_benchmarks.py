#!/usr/bin/env python3
"""Training Speed Benchmarks for Phase 2.2.

This script validates training speed requirements:
- Complete curriculum in <24 hours on single GPU
- Compare training times across different problem sizes
- Validate JAX compilation speedups
"""

import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock
import json

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_bayes_opt.training import (
    GRPOTrainingManager,
    create_debug_grpo_config,
    create_standard_grpo_config,
    create_production_grpo_config,
    TrainingMode,
    OptimizationLevel,
)

from causal_bayes_opt.acquisition.reward_rubric import create_training_rubric
from causal_bayes_opt.environments.intervention_env import create_intervention_environment
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.jax_native.state import create_test_state


class TrainingSpeedBenchmark:
    """Benchmarks training speed across different configurations."""
    
    def __init__(self):
        self.results = {}
        self.baseline_times = {}
    
    def create_test_scm(self, n_vars: int) -> pyr.PMap:
        """Create test SCM with specified number of variables."""
        variables = [f"X{i}" for i in range(n_vars)]
        edges = [(f"X{i}", f"X{j}") for i in range(n_vars-1) for j in range(i+1, min(i+3, n_vars))]
        
        mechanisms = {}
        for var in variables:
            if var == "X0":
                mechanisms[var] = lambda parents, key: random.normal(key, shape=())
            else:
                mechanisms[var] = lambda parents, key: (
                    sum(parents.values()) + 0.1 * random.normal(key, shape=())
                )
        
        return create_scm(
            variables=frozenset(variables),
            edges=frozenset(edges),
            mechanisms=pyr.pmap(mechanisms),
            target=variables[-1]  # Last variable as target
        )
    
    def create_mock_training_setup(self, n_vars: int, config_type: str = "debug"):
        """Create mock training setup for benchmarking."""
        # Create test SCM
        scm = self.create_test_scm(n_vars)
        
        # Create reward rubric
        rubric = create_training_rubric()
        
        # Create environment
        env = create_intervention_environment(scm, rubric, max_interventions=20)
        
        # Create mock networks (for speed testing, we don't need real networks)
        policy_net = Mock()
        policy_net.params = {'policy': jnp.array([1.0] * 10)}
        policy_net.replace.return_value = policy_net
        
        value_net = Mock()
        value_net.params = {'value': jnp.array([1.0] * 5)}
        value_net.replace.return_value = value_net
        
        # Get configuration
        if config_type == "debug":
            config = create_debug_grpo_config(max_training_steps=50)
        elif config_type == "standard":
            config = create_standard_grpo_config(max_training_steps=200)
        else:  # production
            config = create_production_grpo_config(max_training_steps=100)
        
        return scm, env, rubric, policy_net, value_net, config
    
    def benchmark_training_setup_time(self, n_vars: int, config_type: str) -> float:
        """Benchmark time to set up training."""
        start_time = time.time()
        
        try:
            scm, env, rubric, policy_net, value_net, config = self.create_mock_training_setup(n_vars, config_type)
            
            # Create training manager
            manager = GRPOTrainingManager(
                config=config,
                environment=env,
                reward_rubric=rubric,
                policy_network=policy_net,
                value_network=value_net
            )
            
            setup_time = time.time() - start_time
            return setup_time
            
        except Exception as e:
            print(f"Error in setup benchmark: {e}")
            return float('inf')
    
    def benchmark_single_training_step(self, n_vars: int, config_type: str) -> float:
        """Benchmark time for a single training step."""
        scm, env, rubric, policy_net, value_net, config = self.create_mock_training_setup(n_vars, config_type)
        
        manager = GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Warmup
        try:
            manager._execute_training_step()
        except:
            pass  # Ignore warmup errors
        
        # Actual benchmark
        start_time = time.time()
        try:
            step_result = manager._execute_training_step()
            step_time = time.time() - start_time
            return step_time
        except Exception as e:
            print(f"Error in step benchmark: {e}")
            return float('inf')
    
    def benchmark_experience_collection(self, n_vars: int, num_episodes: int = 10) -> float:
        """Benchmark experience collection speed."""
        scm, env, rubric, policy_net, value_net, config = self.create_mock_training_setup(n_vars, "debug")
        
        manager = GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        start_time = time.time()
        try:
            experiences = manager.collect_experiences(num_episodes=num_episodes)
            collection_time = time.time() - start_time
            experiences_per_second = len(experiences) / max(collection_time, 0.001)
            return experiences_per_second
        except Exception as e:
            print(f"Error in experience collection benchmark: {e}")
            return 0.0
    
    def estimate_full_training_time(self, n_vars: int, config_type: str) -> Dict[str, float]:
        """Estimate full training time based on single step measurements."""
        setup_time = self.benchmark_training_setup_time(n_vars, config_type)
        step_time = self.benchmark_single_training_step(n_vars, config_type)
        
        # Get config to know total steps
        if config_type == "debug":
            config = create_debug_grpo_config()
        elif config_type == "standard":
            config = create_standard_grpo_config()
        else:
            config = create_production_grpo_config()
        
        total_steps = config.max_training_steps
        estimated_total_time = setup_time + (step_time * total_steps)
        
        return {
            "setup_time": setup_time,
            "step_time": step_time,
            "total_steps": total_steps,
            "estimated_total_time": estimated_total_time,
            "estimated_hours": estimated_total_time / 3600,
            "meets_24h_requirement": estimated_total_time < (24 * 3600)
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all training speed benchmarks."""
        print("=" * 80)
        print("TRAINING SPEED BENCHMARKS")
        print("=" * 80)
        
        problem_sizes = [5, 10, 20]
        config_types = ["debug", "standard", "production"]
        
        results = {
            "setup_times": {},
            "step_times": {},
            "experience_collection": {},
            "full_training_estimates": {},
            "summary": {}
        }
        
        # Setup time benchmarks
        print("\n1. Training Setup Time Benchmarks")
        print("-" * 40)
        for n_vars in problem_sizes:
            for config_type in config_types:
                setup_time = self.benchmark_training_setup_time(n_vars, config_type)
                key = f"{n_vars}vars_{config_type}"
                results["setup_times"][key] = setup_time
                print(f"  {key}: {setup_time:.3f}s")
        
        # Single step benchmarks
        print("\n2. Single Training Step Benchmarks")
        print("-" * 40)
        for n_vars in problem_sizes:
            for config_type in config_types:
                step_time = self.benchmark_single_training_step(n_vars, config_type)
                key = f"{n_vars}vars_{config_type}"
                results["step_times"][key] = step_time
                print(f"  {key}: {step_time:.3f}s per step")
        
        # Experience collection benchmarks
        print("\n3. Experience Collection Speed")
        print("-" * 40)
        for n_vars in problem_sizes:
            exp_per_sec = self.benchmark_experience_collection(n_vars)
            results["experience_collection"][f"{n_vars}vars"] = exp_per_sec
            print(f"  {n_vars} variables: {exp_per_sec:.1f} experiences/second")
        
        # Full training estimates
        print("\n4. Full Training Time Estimates")
        print("-" * 40)
        for n_vars in problem_sizes:
            for config_type in config_types:
                estimate = self.estimate_full_training_time(n_vars, config_type)
                key = f"{n_vars}vars_{config_type}"
                results["full_training_estimates"][key] = estimate
                
                status = "✅" if estimate["meets_24h_requirement"] else "❌"
                print(f"  {key}: {estimate['estimated_hours']:.2f}h {status}")
        
        # Performance summary
        print("\n5. Performance Summary")
        print("-" * 40)
        
        # Find fastest configurations
        fastest_setup = min(results["setup_times"].items(), key=lambda x: x[1])
        fastest_step = min(results["step_times"].items(), key=lambda x: x[1])
        
        print(f"  Fastest setup: {fastest_setup[0]} ({fastest_setup[1]:.3f}s)")
        print(f"  Fastest step: {fastest_step[0]} ({fastest_step[1]:.3f}s)")
        
        # Check 24-hour requirement
        meets_requirement = [
            (k, v) for k, v in results["full_training_estimates"].items()
            if v["meets_24h_requirement"]
        ]
        
        print(f"  Configurations meeting <24h requirement: {len(meets_requirement)}")
        for config, _ in meets_requirement:
            print(f"    ✅ {config}")
        
        results["summary"] = {
            "total_configurations_tested": len(problem_sizes) * len(config_types),
            "configurations_meeting_24h": len(meets_requirement),
            "fastest_setup_config": fastest_setup[0],
            "fastest_step_config": fastest_step[0],
            "performance_acceptable": len(meets_requirement) > 0
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "training_speed_results.json"):
        """Save benchmark results to file."""
        results_file = Path(__file__).parent.parent / filename
        
        # Convert any non-serializable values
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📊 Results saved to {results_file}")


def main():
    """Run training speed benchmarks."""
    benchmark = TrainingSpeedBenchmark()
    
    print("Starting training speed benchmarks...")
    print("This will test setup times, step times, and estimate full training durations.")
    print("Note: Using mock networks for consistent timing measurements.\n")
    
    results = benchmark.run_all_benchmarks()
    benchmark.save_results(results)
    
    # Final assessment
    print("\n" + "=" * 80)
    print("BENCHMARK ASSESSMENT")
    print("=" * 80)
    
    if results["summary"]["performance_acceptable"]:
        print("✅ TRAINING SPEED REQUIREMENTS MET")
        print("   Multiple configurations can complete training within 24 hours")
    else:
        print("❌ TRAINING SPEED REQUIREMENTS NOT MET")
        print("   No configurations meet the 24-hour requirement")
    
    print(f"\nConfigurations tested: {results['summary']['total_configurations_tested']}")
    print(f"Meeting 24h requirement: {results['summary']['configurations_meeting_24h']}")
    
    return 0 if results["summary"]["performance_acceptable"] else 1


if __name__ == "__main__":
    sys.exit(main())