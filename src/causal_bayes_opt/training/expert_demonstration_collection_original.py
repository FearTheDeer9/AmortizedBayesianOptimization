#!/usr/bin/env python3
"""
Expert Demonstration Collection for ACBO

Collects expert demonstrations from PARENT_SCALE neural doubly robust method
using validated data scaling requirements. Creates training data for both
surrogate and acquisition models.

Key Features:
1. Uses validated O(d^2.5) data scaling for reliable parent discovery
2. Generates diverse SCM problems for comprehensive training coverage
3. Collects both (data → posterior) pairs and (state → action) sequences
4. Validates demonstration quality before saving
5. Supports batch collection for efficient training data generation
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
from dataclasses import dataclass, field
import time
import json
import pickle
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

# Import our components
from causal_bayes_opt.data_structures.scm import create_scm, get_variables, get_target
from causal_bayes_opt.data_structures.sample import create_sample, get_values
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.environments.sampling import sample_with_intervention
from causal_bayes_opt.interventions.handlers import create_perfect_intervention
from causal_bayes_opt.integration.parent_scale_bridge import (
    create_parent_scale_bridge, calculate_data_requirements, run_parent_discovery, run_full_parent_scale_algorithm
)
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism, sample_from_linear_scm


@dataclass
class ExpertDemonstration:
    """Complete expert demonstration from one PARENT_SCALE run."""
    
    # Problem setup
    scm: pyr.PMap
    target_variable: str
    n_nodes: int
    graph_type: str
    
    # Data and results
    observational_samples: List[pyr.PMap]
    interventional_samples: List[pyr.PMap] 
    discovered_parents: FrozenSet[str]
    confidence: float
    accuracy: float  # Against ground truth
    
    # Training data extracted
    parent_posterior: Dict[str, Any]
    data_requirements: Dict[str, int]
    
    # Performance metrics
    inference_time: float
    total_samples_used: int
    
    # Metadata
    collection_timestamp: float = field(default_factory=time.time)
    validation_passed: bool = True


@dataclass
class ExpertTrajectoryDemonstration:
    """Complete expert trajectory from full PARENT_SCALE CBO algorithm."""
    
    # Problem setup
    scm: pyr.PMap
    target_variable: str
    n_nodes: int
    graph_type: str
    
    # Initial data
    initial_observational_samples: List[pyr.PMap]
    initial_interventional_samples: List[pyr.PMap]
    
    # Complete expert trajectory
    expert_trajectory: Dict[str, Any]  # Full CBO trajectory data
    
    # Algorithm performance
    algorithm_time: float
    total_samples_used: int
    final_optimum: float
    total_improvement: float
    convergence_rate: float
    exploration_efficiency: float
    
    # Configuration
    data_requirements: Dict[str, int]
    algorithm_config: Dict[str, Any]
    
    # Metadata
    collection_timestamp: float = field(default_factory=time.time)
    validation_passed: bool = True


@dataclass 
class DemonstrationBatch:
    """Batch of demonstrations for training."""
    
    demonstrations: List[ExpertDemonstration]
    batch_id: str
    collection_config: Dict[str, Any]
    
    # Summary statistics
    total_demonstrations: int = field(init=False)
    avg_accuracy: float = field(init=False)
    graph_types_covered: List[str] = field(init=False)
    node_sizes_covered: List[int] = field(init=False)
    
    def __post_init__(self):
        self.total_demonstrations = len(self.demonstrations)
        self.avg_accuracy = onp.mean([d.accuracy for d in self.demonstrations])
        self.graph_types_covered = list(set(d.graph_type for d in self.demonstrations))
        self.node_sizes_covered = list(set(d.n_nodes for d in self.demonstrations))


class ExpertDemonstrationCollector:
    """
    Collects expert demonstrations using validated neural doubly robust method.
    
    Generates diverse SCM problems and uses PARENT_SCALE to solve them with
    proper data scaling, creating high-quality training data for ACBO.
    """
    
    def __init__(self, output_dir: str = "demonstrations"):
        create_parent_scale_bridge()  # Validate availability
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Collection statistics
        self.demonstrations_collected = 0
        self.total_time_spent = 0.0
        
    def generate_scm_problems(
        self, 
        n_problems: int = 100,
        node_sizes: List[int] = [3, 5, 8, 10, 15, 20],
        graph_types: List[str] = ["chain", "star", "fork", "collider"],
        key: Optional[jax.Array] = None
    ) -> List[Tuple[pyr.PMap, str]]:
        """
        Generate diverse SCM problems for demonstration collection.
        
        Args:
            n_problems: Number of problems to generate
            node_sizes: List of graph sizes to sample from
            graph_types: List of graph structures to sample from
            key: Random key for reproducible generation
            
        Returns:
            List of (scm, graph_type) tuples
        """
        if key is None:
            key = random.PRNGKey(42)
        
        problems = []
        
        for i in range(n_problems):
            key, subkey = random.split(key)
            
            # Sample problem characteristics
            n_nodes = onp.random.choice(node_sizes)
            graph_type = onp.random.choice(graph_types)
            
            # Generate SCM
            scm = self._generate_scm(n_nodes, graph_type, subkey)
            problems.append((scm, graph_type))
        
        return problems
    
    def _generate_scm(self, n_nodes: int, graph_type: str, key: jax.Array) -> pyr.PMap:
        """Generate a single SCM of specified type and size."""
        variables = [f"X{i}" for i in range(n_nodes)]
        target = variables[-1]  # Last variable is target
        
        # Generate edges based on graph type
        if graph_type == "chain":
            edges = frozenset([(variables[i], variables[i+1]) for i in range(n_nodes-1)])
        elif graph_type == "star":
            # First 3 variables point to target
            edges = frozenset([(variables[i], target) for i in range(min(3, n_nodes-1))])
        elif graph_type == "fork":
            # Root variable affects multiple children
            edges = frozenset([(variables[0], variables[i]) for i in range(1, min(4, n_nodes))])
        elif graph_type == "collider":
            # Multiple variables affect target
            parents = variables[:-1] if n_nodes <= 4 else variables[:3]
            edges = frozenset([(parent, target) for parent in parents])
        else:
            # Random sparse graph
            edges = self._generate_random_edges(variables, key)
        
        # Generate linear mechanisms
        mechanisms = {}
        key, *subkeys = random.split(key, len(variables) + 1)
        
        for i, var in enumerate(variables):
            # Get parents
            parents = [p for p, c in edges if c == var]
            
            if not parents:
                # Root variable
                mean_val = random.normal(subkeys[i]) * 0.5
                mechanisms[var] = create_root_mechanism(mean=float(mean_val), noise_scale=0.1)
            else:
                # Variable with parents
                coefficients = {}
                for parent in parents:
                    coeff = random.normal(subkeys[i]) * 0.8 + 0.5  # Avoid zero coefficients
                    coefficients[parent] = float(coeff)
                
                intercept = random.normal(subkeys[i]) * 0.2
                mechanisms[var] = create_linear_mechanism(
                    parents=parents,
                    coefficients=coefficients,
                    intercept=float(intercept),
                    noise_scale=0.1
                )
        
        return create_scm(
            variables=frozenset(variables),
            edges=edges,
            mechanisms=mechanisms,
            target=target
        )
    
    def _generate_random_edges(self, variables: List[str], key: jax.Array) -> FrozenSet[Tuple[str, str]]:
        """Generate random sparse edges with topological ordering."""
        n_vars = len(variables)
        target_density = 0.3  # Sparse graphs
        max_edges = int(target_density * n_vars * (n_vars - 1) / 2)
        
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if random.uniform(key) < target_density and len(edges) < max_edges:
                    edges.append((variables[i], variables[j]))
                    key, _ = random.split(key)
        
        return frozenset(edges)
    
    def collect_demonstration(
        self, 
        scm: pyr.PMap, 
        graph_type: str,
        min_accuracy: float = 0.7
    ) -> Optional[ExpertDemonstration]:
        """
        Collect a single expert demonstration using PARENT_SCALE.
        
        Args:
            scm: The SCM problem to solve
            graph_type: Type of graph structure 
            min_accuracy: Minimum accuracy required for valid demonstration
            
        Returns:
            ExpertDemonstration if successful, None if failed validation
        """
        n_nodes = len(get_variables(scm))
        target_variable = get_target(scm)
        
        print(f"Collecting demonstration: {n_nodes} nodes, {graph_type} graph, target={target_variable}")
        
        # Calculate data requirements using validated scaling
        data_req = calculate_data_requirements(n_nodes, target_accuracy=0.8)
        
        print(f"  Data requirements: {data_req['total_samples']} samples, {data_req['bootstrap_samples']} bootstraps")
        
        # Generate observational data using linear SCM sampling
        obs_samples = sample_from_linear_scm(scm, n_samples=data_req['observational_samples'])
        
        # Generate interventional data with comprehensive coverage
        int_samples = []
        variables = list(get_variables(scm))
        non_target_vars = [v for v in variables if v != target_variable]
        interventions_per_var = data_req['interventional_samples'] // len(non_target_vars)
        
        for var in non_target_vars:
            for _ in range(interventions_per_var):
                intervention_val = onp.random.normal(0, 1)
                intervention = create_perfect_intervention(
                    targets=frozenset([var]),
                    values={var: intervention_val}
                )
                
                sample_result = sample_with_intervention(scm, intervention, n_samples=1)
                int_samples.extend(sample_result)
        
        # Combine all samples
        all_samples = obs_samples + int_samples
        
        print(f"  Generated {len(all_samples)} samples ({len(obs_samples)} obs + {len(int_samples)} int)")
        
        # Run PARENT_SCALE neural doubly robust
        start_time = time.time()
        
        try:
            results = run_parent_discovery(
                scm=scm,
                samples=all_samples,
                target_variable=target_variable,
                num_bootstraps=data_req['bootstrap_samples']
            )
            
            inference_time = time.time() - start_time
            
            print(f"  Inference completed in {inference_time:.1f}s")
            print(f"  Discovered parents: {results['most_likely_parents']}")
            print(f"  Confidence: {results['confidence']:.3f}")
            
            # Calculate accuracy against ground truth
            true_parents = self._get_true_parents(scm, target_variable)
            discovered_parents = results['most_likely_parents']
            accuracy = self._calculate_accuracy(true_parents, discovered_parents)
            
            print(f"  True parents: {true_parents}")
            print(f"  Accuracy: {accuracy:.3f}")
            
            # Validate demonstration quality
            if accuracy < min_accuracy:
                print(f"  ❌ Accuracy {accuracy:.3f} below threshold {min_accuracy}")
                return None
            
            if results['confidence'] < 0.1:
                print(f"  ❌ Confidence {results['confidence']:.3f} too low")
                return None
            
            # Create demonstration
            demonstration = ExpertDemonstration(
                scm=scm,
                target_variable=target_variable,
                n_nodes=n_nodes,
                graph_type=graph_type,
                observational_samples=obs_samples,
                interventional_samples=int_samples,
                discovered_parents=discovered_parents,
                confidence=results['confidence'],
                accuracy=accuracy,
                parent_posterior=results,
                data_requirements=data_req,
                inference_time=inference_time,
                total_samples_used=len(all_samples)
            )
            
            print(f"  ✅ Demonstration collected successfully")
            return demonstration
            
        except Exception as e:
            print(f"  ❌ Failed to collect demonstration: {e}")
            return None
    
    def collect_full_trajectory_demonstration(
        self,
        scm: pyr.PMap,
        graph_type: str,
        T: int = 5,
        min_optimum_improvement: float = 0.1
    ) -> Optional['ExpertTrajectoryDemonstration']:
        """
        Collect complete expert trajectory using full PARENT_SCALE CBO algorithm.
        
        This provides the complete intervention decision sequence with reasoning,
        suitable for training our ACBO acquisition model.
        
        Args:
            scm: SCM to run algorithm on
            graph_type: Type of graph structure
            T: Number of CBO iterations to run
            min_optimum_improvement: Minimum improvement required for valid trajectory
            
        Returns:
            Complete expert trajectory or None if collection failed
        """
        n_nodes = len(get_variables(scm))
        target_variable = get_target(scm)
        
        print(f"🎯 Collecting full trajectory demonstration:")
        print(f"  Graph: {graph_type} with {n_nodes} nodes")
        print(f"  Target: {target_variable}")
        print(f"  CBO iterations: {T}")
        
        # Calculate data requirements for reliable algorithm performance
        data_req = calculate_data_requirements(n_nodes, target_accuracy=0.8)
        
        print(f"  Data requirements: {data_req['total_samples']} samples")
        
        # Generate initial samples for algorithm
        obs_samples = sample_from_linear_scm(scm, n_samples=data_req['observational_samples'])
        
        # Generate some initial interventional data
        int_samples = []
        variables = list(get_variables(scm))
        non_target_vars = [v for v in variables if v != target_variable]
        
        for _ in range(data_req['interventional_samples']):
            var = onp.random.choice(non_target_vars)
            int_val = onp.random.normal(0, 1)
            intervention = create_perfect_intervention(
                targets=frozenset([var]),
                values={var: int_val}
            )
            int_samples.extend(sample_with_intervention(scm, intervention, n_samples=1))
        
        all_samples = obs_samples + int_samples
        
        print(f"  Generated {len(all_samples)} initial samples")
        
        # Run complete PARENT_SCALE CBO algorithm
        start_time = time.time()
        
        try:
            trajectory = run_full_parent_scale_algorithm(
                scm=scm,
                samples=all_samples,
                target_variable=target_variable,
                T=T,
                nonlinear=True,
                causal_prior=True,
                individual=False,
                use_doubly_robust=True
            )
            
            algorithm_time = time.time() - start_time
            
            print(f"  Algorithm completed in {algorithm_time:.1f}s")
            
            if trajectory.get('status') == 'failed':
                print(f"  ❌ Algorithm failed: {trajectory.get('error')}")
                return None
            
            # Validate trajectory quality
            final_optimum = trajectory['final_optimum']
            initial_optimum = trajectory['global_optimum_trajectory'][0] if trajectory['global_optimum_trajectory'] else final_optimum
            improvement = abs(final_optimum - initial_optimum)
            
            print(f"  Initial optimum: {initial_optimum:.4f}")
            print(f"  Final optimum: {final_optimum:.4f}")
            print(f"  Improvement: {improvement:.4f}")
            
            if improvement < min_optimum_improvement:
                print(f"  ❌ Insufficient improvement {improvement:.4f} < {min_optimum_improvement}")
                return None
            
            convergence_rate = trajectory['convergence_rate']
            exploration_efficiency = trajectory['exploration_efficiency']
            
            print(f"  Convergence rate: {convergence_rate:.3f}")
            print(f"  Exploration efficiency: {exploration_efficiency:.3f}")
            
            # Create trajectory demonstration
            trajectory_demo = ExpertTrajectoryDemonstration(
                scm=scm,
                target_variable=target_variable,
                n_nodes=n_nodes,
                graph_type=graph_type,
                
                # Initial data
                initial_observational_samples=obs_samples,
                initial_interventional_samples=int_samples,
                
                # Complete trajectory
                expert_trajectory=trajectory,
                
                # Algorithm performance
                algorithm_time=algorithm_time,
                total_samples_used=len(all_samples),
                final_optimum=final_optimum,
                total_improvement=improvement,
                convergence_rate=convergence_rate,
                exploration_efficiency=exploration_efficiency,
                
                # Configuration
                data_requirements=data_req,
                algorithm_config=trajectory['config']
            )
            
            print(f"  ✅ Complete trajectory demonstration collected")
            return trajectory_demo
            
        except Exception as e:
            print(f"  ❌ Failed to collect trajectory: {e}")
            return None
    
    def _get_true_parents(self, scm: pyr.PMap, target_variable: str) -> FrozenSet[str]:
        """Extract true parents from SCM structure."""
        edges = scm.get('edges', frozenset())
        return frozenset([parent for parent, child in edges if child == target_variable])
    
    def _calculate_accuracy(self, true_parents: FrozenSet[str], discovered_parents: FrozenSet[str]) -> float:
        """Calculate Jaccard similarity between true and discovered parents."""
        if len(true_parents) == 0 and len(discovered_parents) == 0:
            return 1.0
        
        intersection = len(true_parents.intersection(discovered_parents))
        union = len(true_parents.union(discovered_parents))
        
        return intersection / union if union > 0 else 0.0
    
    def collect_demonstration_batch(
        self,
        n_demonstrations: int = 50,
        node_sizes: List[int] = [3, 5, 8, 10, 15, 20],
        graph_types: List[str] = ["chain", "star", "fork", "collider"],
        min_accuracy: float = 0.7,
        max_failures: int = 20
    ) -> DemonstrationBatch:
        """
        Collect a batch of expert demonstrations.
        
        Args:
            n_demonstrations: Target number of successful demonstrations
            node_sizes: Graph sizes to sample from
            graph_types: Graph structures to sample from
            min_accuracy: Minimum accuracy for valid demonstrations
            max_failures: Maximum failures before giving up
            
        Returns:
            DemonstrationBatch with collected demonstrations
        """
        print(f"Collecting batch of {n_demonstrations} expert demonstrations")
        print(f"Node sizes: {node_sizes}")
        print(f"Graph types: {graph_types}")
        print(f"Min accuracy: {min_accuracy}")
        print()
        
        demonstrations = []
        failures = 0
        total_attempts = 0
        
        # Generate problems
        problems = self.generate_scm_problems(
            n_problems=n_demonstrations * 3,  # Generate extra to handle failures
            node_sizes=node_sizes,
            graph_types=graph_types
        )
        
        start_time = time.time()
        
        for scm, graph_type in problems:
            if len(demonstrations) >= n_demonstrations:
                break
                
            if failures >= max_failures:
                print(f"❌ Reached maximum failures ({max_failures}), stopping collection")
                break
            
            total_attempts += 1
            
            print(f"Attempt {total_attempts}: Target {len(demonstrations) + 1}/{n_demonstrations}")
            
            demonstration = self.collect_demonstration(scm, graph_type, min_accuracy)
            
            if demonstration is not None:
                demonstrations.append(demonstration)
                print(f"  ✅ Success! Collected {len(demonstrations)}/{n_demonstrations}")
            else:
                failures += 1
                print(f"  ❌ Failed (failures: {failures}/{max_failures})")
            
            print()
        
        collection_time = time.time() - start_time
        
        # Create batch
        batch = DemonstrationBatch(
            demonstrations=demonstrations,
            batch_id=f"batch_{int(time.time())}",
            collection_config={
                'n_demonstrations': n_demonstrations,
                'node_sizes': node_sizes,
                'graph_types': graph_types,
                'min_accuracy': min_accuracy,
                'collection_time': collection_time,
                'total_attempts': total_attempts,
                'failures': failures
            }
        )
        
        print(f"📊 Batch Collection Summary:")
        print(f"   Successful demonstrations: {batch.total_demonstrations}")
        print(f"   Average accuracy: {batch.avg_accuracy:.3f}")
        print(f"   Graph types covered: {batch.graph_types_covered}")
        print(f"   Node sizes covered: {batch.node_sizes_covered}")
        print(f"   Total time: {collection_time:.1f}s")
        print(f"   Success rate: {len(demonstrations)}/{total_attempts} ({100*len(demonstrations)/total_attempts:.1f}%)")
        
        # Update collector statistics
        self.demonstrations_collected += len(demonstrations)
        self.total_time_spent += collection_time
        
        return batch
    
    def save_batch(self, batch: DemonstrationBatch, format: str = "pickle") -> str:
        """
        Save demonstration batch to disk.
        
        Args:
            batch: The batch to save
            format: Save format ("pickle" or "json")
            
        Returns:
            Path to saved file
        """
        if format == "pickle":
            filename = f"{batch.batch_id}.pkl"
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(batch, f)
        
        elif format == "json":
            filename = f"{batch.batch_id}.json"
            filepath = self.output_dir / filename
            
            # Convert to JSON-serializable format
            batch_dict = {
                'batch_id': batch.batch_id,
                'collection_config': batch.collection_config,
                'total_demonstrations': batch.total_demonstrations,
                'avg_accuracy': batch.avg_accuracy,
                'graph_types_covered': batch.graph_types_covered,
                'node_sizes_covered': batch.node_sizes_covered,
                'demonstrations': [
                    {
                        'target_variable': d.target_variable,
                        'n_nodes': d.n_nodes,
                        'graph_type': d.graph_type,
                        'discovered_parents': list(d.discovered_parents),
                        'confidence': d.confidence,
                        'accuracy': d.accuracy,
                        'inference_time': d.inference_time,
                        'total_samples_used': d.total_samples_used,
                        'collection_timestamp': d.collection_timestamp
                    }
                    for d in batch.demonstrations
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(batch_dict, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"💾 Batch saved to: {filepath}")
        return str(filepath)
    
    def load_batch(self, filepath: str) -> DemonstrationBatch:
        """Load demonstration batch from disk."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.suffix == '.json':
            # JSON loading would require reconstruction of complex objects
            raise NotImplementedError("JSON loading not yet implemented")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


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