#!/usr/bin/env python3
"""
Module for generating families of related causal tasks (SCMs) based on variations
of a base causal graph.
"""

import random
import copy
import logging # Added for logger usage
from typing import List, Union, Literal, Optional
import numpy as np # Added for noise generation
import traceback # Import traceback for better error logging
import networkx as nx # Import networkx for DAG check

# Assuming CausalGraph and potentially SCM are defined elsewhere
# Adjust imports as necessary based on project structure
from causal_meta.graph.causal_graph import CausalGraph
# from causal_meta.environments.scm import StructuralCausalModel # Might be needed later

# Define supported variation types
VariationType = Literal['edge_weights', 'structure', 'node_function']

class TaskFamilyGenerationError(Exception):
    """Custom exception for errors during task family generation."""
    pass


def generate_task_family(
    base_graph: CausalGraph,
    num_tasks: int,
    variation_type: VariationType = 'edge_weights',
    variation_strength: float = 0.2,
    seed: Optional[int] = None
) -> List[CausalGraph]: # Later this might return SCMs
    """Generates a family of related causal task graphs based on a base DAG.

    This function takes a base causal graph and generates a specified number
    of variant graphs by applying controlled modifications. The type and magnitude
    of these modifications are determined by the `variation_type` and
    `variation_strength` parameters, respectively.

    Supported variation types:
        - 'edge_weights': Modifies the weights of existing edges by adding
          Gaussian noise. The scale of the noise is proportional to the original
          edge weight and the `variation_strength`. The graph structure remains
          identical to the base graph.
        - 'structure': Adds or removes edges while ensuring the resulting graph
          remains a DAG. The number of attempted modifications is proportional
          to the total number of possible edge additions/removals and the
          `variation_strength`.
        - 'node_function': (Not yet implemented) Intended to modify the underlying
          causal mechanisms associated with nodes. Requires integration with
          Structural Causal Models (SCMs).

    Args:
        base_graph: The base `CausalGraph` object. Must be a Directed Acyclic
            Graph (DAG).
        num_tasks: The desired number of variant task graphs in the generated family.
        variation_type: The type of variation to apply. Must be one of
            'edge_weights', 'structure', or 'node_function'. Defaults to
            'edge_weights'.
        variation_strength: A float between 0.0 and 1.0 controlling the magnitude
            of variation. Higher values result in variants that are more
            different from the base graph. Defaults to 0.2.
        seed: An optional integer seed for the random number generators (Python's
            `random` and `numpy.random`) to ensure reproducibility.
            Defaults to None.

    Returns:
        A list containing `num_tasks` `CausalGraph` objects. Each object is a
        variant derived from the `base_graph` according to the specified
        variation type and strength. The graphs include a `task_id` attribute
        indicating their origin (e.g., "BaseTask_var_0").

    Raises:
        TypeError: If `base_graph` is not an instance of `CausalGraph`.
        ValueError: If `base_graph` is not a Directed Acyclic Graph (DAG) (checked
            via `networkx` if available).
        TaskFamilyGenerationError: If `num_tasks` is not a positive integer,
            `variation_type` is invalid, `variation_strength` is outside the
            range [0.0, 1.0], or if an unexpected error occurs during generation
            of a specific variant (the function will log the error and attempt
            to continue generating other variants).

    Example:
        >>> factory = GraphFactory()
        >>> base = factory.create_random_dag(num_nodes=5, edge_probability=0.4, seed=1)
        >>> base.task_id = "MyBaseGraph"
        >>> # Generate 3 variants by changing edge weights
        >>> family_weights = generate_task_family(base, 3, 'edge_weights', 0.5, seed=10)
        >>> len(family_weights)
        3
        >>> # Generate 3 variants by changing structure
        >>> family_structure = generate_task_family(base, 3, 'structure', 0.2, seed=11)
        >>> len(family_structure)
        3
    """
    # --- Parameter Validation ---
    if not isinstance(base_graph, CausalGraph):
        raise TypeError("base_graph must be an instance of CausalGraph.")
    # Add DAG check if available in CausalGraph, otherwise assume it's checked elsewhere
    # Convert to networkx to perform DAG check
    try:
        nx_base = base_graph.to_networkx()
        if not nx.is_directed_acyclic_graph(nx_base):
             raise ValueError("base_graph must be a Directed Acyclic Graph (DAG).")
    except ImportError:
        logger.warning("networkx not found. Cannot perform DAG check on base_graph.")
    except AttributeError:
        logger.warning("base_graph does not have to_networkx method. Cannot perform DAG check.")
    except Exception as e:
        logger.warning(f"Could not perform DAG check on base_graph: {e}")

    if not isinstance(num_tasks, int) or num_tasks <= 0:
        raise TaskFamilyGenerationError("num_tasks must be a positive integer.")

    valid_variation_types = ['edge_weights', 'structure', 'node_function']
    if variation_type not in valid_variation_types:
        raise TaskFamilyGenerationError(
            f"Invalid variation_type '{variation_type}'. "
            f"Must be one of {valid_variation_types}")

    if not isinstance(variation_strength, (float, int)) or not (0.0 <= variation_strength <= 1.0):
        raise TaskFamilyGenerationError(
            "variation_strength must be a float between 0.0 and 1.0.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed) # Seed numpy as well

    task_family = []
    # Get logger instance
    logger = logging.getLogger(__name__)
    logger.info(
        f"Generating task family of size {num_tasks} with variation type "
        f"'{variation_type}' and strength {variation_strength:.2f}"
    )

    # --- Generation Loop ---
    for i in range(num_tasks):
        # Create a deep copy to modify independently
        variant_graph = copy.deepcopy(base_graph)
        # Get base task ID safely, providing a default if it doesn't exist
        base_task_id = getattr(base_graph, 'task_id', 'BaseTask')
        variant_graph.task_id = f"{base_task_id}_var_{i}" # Assign unique ID

        try:
            if variation_type == 'edge_weights':
                # --- Edge Weight Variation Logic --- # 
                # Assuming weights are stored in edge attributes, e.g., 'weight'
                # If no weights exist, initialize them (e.g., to 1.0)
                has_weights = True
                initial_weights = {}
                for u_init, v_init in variant_graph.get_edges():
                    weight = variant_graph.get_edge_attribute(u_init, v_init, 'weight', None)
                    if weight is None:
                        has_weights = False
                        logger.debug(f"Edge ({u_init}, {v_init}) missing 'weight' attribute. Initializing all weights.")
                        break
                    initial_weights[(u_init, v_init)] = weight # Store initial weights
                
                if not has_weights:
                    # Initialize all edge weights if any were missing
                     for u_init, v_init in variant_graph.get_edges():
                        variant_graph.set_edge_attribute(u_init, v_init, 'weight', 1.0)
                        initial_weights[(u_init, v_init)] = 1.0 # Update initial weights map

                # Perturb existing weights
                for u, v in variant_graph.get_edges():
                    original_weight = initial_weights.get((u, v), 1.0) # Use stored/initialized weight
                    # Additive noise scaled by variation strength
                    # Noise scale could be absolute or relative, using relative here
                    noise_scale = abs(original_weight) * variation_strength 
                    # Ensure minimum noise scale if weight is zero or very small
                    noise_scale = max(noise_scale, 0.1 * variation_strength + 1e-6) # Add epsilon for zero case 
                    noise = np.random.normal(loc=0.0, scale=noise_scale)
                    new_weight = original_weight + noise
                    
                    # Optional: Add constraints (e.g., keep weights non-negative)
                    # new_weight = max(0, new_weight)
                    
                    variant_graph.set_edge_attribute(u, v, 'weight', new_weight)
                    logger.debug(f"Edge ({u},{v}): weight {original_weight:.2f} -> {new_weight:.2f}")

            elif variation_type == 'structure':
                # --- Structure Variation Logic --- #
                nodes = list(variant_graph.get_nodes())
                n = len(nodes)
                max_possible_edges = n * (n - 1) // 2 if n > 1 else 0
                current_edges = set(variant_graph.get_edges())
                num_current_edges = len(current_edges)

                # Determine number of modifications based on strength
                # Strength is proportion of *potential* changes to attempt
                # Consider both adding and removing edges
                num_potential_adds = max_possible_edges - num_current_edges
                num_potential_removals = num_current_edges
                # Total possible structural changes
                total_possible_changes = num_potential_adds + num_potential_removals
                
                # Target number of *attempted* modifications
                num_modifications = int(round(total_possible_changes * variation_strength))
                if num_modifications == 0 and variation_strength > 0:
                    num_modifications = 1 # Ensure at least one attempt if strength > 0
                
                logger.debug(f"Attempting {num_modifications} structural modifications.")

                # Get lists of existing edges and potential edges to add
                existing_edges = list(current_edges)
                potential_adds = []
                # Assume nodes are sortable (e.g., integers or strings)
                sorted_nodes = sorted(nodes)
                for i_idx, u in enumerate(sorted_nodes):
                    for v in sorted_nodes[i_idx + 1:]:
                        # Check if adding u -> v is possible (respects potential DAG order)
                        if (u, v) not in current_edges:
                             potential_adds.append((u, v))
                        # Check if adding v -> u is possible
                        if (v, u) not in current_edges:
                            potential_adds.append((v, u))
                            
                modifications_made = 0
                attempts = 0
                max_attempts = num_modifications * 5 # Limit attempts to avoid infinite loops

                while modifications_made < num_modifications and attempts < max_attempts:
                    attempts += 1
                    # Decide whether to attempt add or remove
                    # Probability proportional to available options
                    prob_add = num_potential_adds / total_possible_changes if total_possible_changes > 0 else 0

                    if random.random() < prob_add and potential_adds:
                        # --- Attempt Add --- 
                        idx_to_add = random.randrange(len(potential_adds))
                        u, v = potential_adds.pop(idx_to_add) # Remove from potential to avoid re-attempting
                        num_potential_adds -= 1
                        total_possible_changes -= 1
                        
                        # Check for cycle creation *before* adding
                        # Assumes CausalGraph has a has_path method
                        if not variant_graph.has_path(v, u):
                            variant_graph.add_edge(u, v)
                            modifications_made += 1
                            num_potential_removals += 1 # This edge can now be removed
                            logger.debug(f"Added edge ({u}, {v}). Modifications: {modifications_made}/{num_modifications}")
                        else:
                             logger.debug(f"Skipped adding edge ({u}, {v}) to prevent cycle.")
                             # Add back to potential adds if cycle check fails? No, assume fixed ordering pref.
                             pass 
                             
                    elif existing_edges:
                        # --- Attempt Remove --- 
                        idx_to_remove = random.randrange(len(existing_edges))
                        u, v = existing_edges.pop(idx_to_remove)
                        num_potential_removals -= 1
                        total_possible_changes -= 1
                        
                        variant_graph.remove_edge(u, v)
                        modifications_made += 1
                        num_potential_adds += 1 # This edge could potentially be added back
                        logger.debug(f"Removed edge ({u}, {v}). Modifications: {modifications_made}/{num_modifications}")
                    
                    # Update total possible changes if needed
                    if total_possible_changes <= 0:
                        break # No more possible changes

                if modifications_made < num_modifications:
                    logger.warning(f"Made only {modifications_made} structural modifications after {attempts} attempts (target: {num_modifications}).")

            elif variation_type == 'node_function':
                # Placeholder for node function variation logic (Subtask 2.4)
                # This likely requires operating on an SCM object, not just the graph
                # Modify mechanisms in a variant_scm based on variation_strength
                logger.warning(f"Node function variation ('{variation_type}') is not yet implemented.")
                pass

            task_family.append(variant_graph)

        except Exception as e:
            logger.error(f"Failed to generate variant task {i} for family: {e}")
            # Log the traceback for more detailed debugging info
            logger.debug(traceback.format_exc())
            # Optionally skip failed task or re-raise
            continue # Skip this task and continue

    if len(task_family) != num_tasks:
        logger.warning(f"Generated {len(task_family)} tasks, but {num_tasks} were requested. Some generations may have failed.")

    logger.info(f"Successfully generated task family with {len(task_family)} tasks.")
    return task_family

# Example Usage & Basic Test
if __name__ == "__main__":
    import logging
    import sys # Added for path adjustment
    import os # Added for path adjustment
    # Adjust path assuming script is run from project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    import networkx as nx # Import networkx for testing DAG properties

    # Need GraphFactory to create a base graph
    try:
        from causal_meta.graph.generators.factory import GraphFactory
        logger.info("Running basic test for generate_task_family...")

        # Create a simple base DAG
        base_dag = GraphFactory.create_random_dag(num_nodes=5, edge_probability=0.4, seed=1)
        base_dag.task_id = "BaseDAG"
        
        # Initialize edge weights for the base DAG
        initial_weight = 1.0
        edge_weights_base = {}
        for u, v in base_dag.get_edges():
            base_dag.set_edge_attribute(u, v, 'weight', initial_weight)
            edge_weights_base[(u,v)] = initial_weight
        logger.info(f"Created base DAG with nodes: {base_dag.get_nodes()} and edges: {base_dag.get_edges()}")
        logger.info(f"Initial edge weights: {edge_weights_base}")


        # Test edge weight variation
        variation_strength_edge = 0.5
        family_edge = generate_task_family(base_dag, 
                                         num_tasks=3, 
                                         variation_type='edge_weights', 
                                         variation_strength=variation_strength_edge,
                                         seed=2)
        logger.info(f"Generated family (edge weights): {len(family_edge)} tasks.")
        
        # --- Verification for Edge Weight Variation ---
        assert len(family_edge) == 3
        base_edges = set(base_dag.get_edges())
        total_weight_diff = 0
        num_edges_tested = 0

        for i, variant in enumerate(family_edge):
            logger.info(f"-- Variant {i+1} --")
            variant_edges = set(variant.get_edges())
            # 1. Check structure is identical
            assert base_edges == variant_edges, f"Variant {i} structure differs from base!"
            assert set(base_dag.get_nodes()) == set(variant.get_nodes()), f"Variant {i} nodes differ!"
            
            # 2. Check weights have changed (and log them)
            variant_weights = {}
            changed_count = 0
            for u, v in variant_edges:
                weight_base = base_dag.get_edge_attribute(u, v, 'weight')
                weight_variant = variant.get_edge_attribute(u, v, 'weight')
                variant_weights[(u,v)] = weight_variant
                if not np.isclose(weight_base, weight_variant):
                     changed_count += 1
                     total_weight_diff += abs(weight_base - weight_variant)
                     num_edges_tested +=1
            logger.info(f"Variant {i} weights: {variant_weights}")
            assert changed_count > 0, f"Variant {i} weights did not change from base!"
        
        # 3. Check magnitude of change (approximate check)
        avg_weight_diff = total_weight_diff / num_edges_tested if num_edges_tested > 0 else 0
        logger.info(f"Average absolute weight difference: {avg_weight_diff:.3f}")
        # This check is very approximate, depends heavily on noise distribution and strength
        # assert avg_weight_diff > 0.05 * variation_strength_edge, "Average weight difference seems too small."

        # Test structure variation (placeholder)
        variation_strength_struct = 0.2 # Attempt to modify 20% of possible edges
        family_struct = generate_task_family(base_dag, 
                                             num_tasks=3, 
                                             variation_type='structure', 
                                             variation_strength=variation_strength_struct, 
                                             seed=3)
        logger.info(f"Generated family (structure): {len(family_struct)} tasks.")
        
        # --- Verification for Structure Variation --- 
        assert len(family_struct) == 3
        base_edges = set(base_dag.get_edges())
        num_edges_base = len(base_edges)
        changed_structure_count = 0

        for i, variant in enumerate(family_struct):
            logger.info(f"-- Variant {i+1} (Structure) --")
            # 1. Check node set is identical
            assert set(base_dag.get_nodes()) == set(variant.get_nodes()), f"Variant {i} nodes differ!"
            
            # 2. Check if structure has changed
            variant_edges = set(variant.get_edges())
            num_edges_variant = len(variant_edges)
            logger.info(f"Base edges: {num_edges_base}, Variant edges: {num_edges_variant}")
            logger.info(f"Edges added: {variant_edges - base_edges}")
            logger.info(f"Edges removed: {base_edges - variant_edges}")
            if base_edges != variant_edges:
                changed_structure_count += 1
            
            # 3. Check if it's still a DAG
            # Requires converting to networkx or similar for check
            try:
                nx_variant = nx.DiGraph()
                nx_variant.add_nodes_from(variant.get_nodes())
                nx_variant.add_edges_from(variant.get_edges())
                assert nx.is_directed_acyclic_graph(nx_variant), f"Variant {i} is not a DAG!"
                logger.info(f"Variant {i} confirmed as DAG.")
            except ImportError:
                 logger.warning("Cannot check DAG property without networkx.")
            except AssertionError as e:
                 logger.error(e)
                 raise # Re-raise assertion error

        assert changed_structure_count > 0, "Structure variation did not change structure in any variant!"
        logger.info(f"{changed_structure_count}/{len(family_struct)} variants had structural changes.")

         # Test node function variation (placeholder)
        family_func = generate_task_family(base_dag, num_tasks=3, variation_type='node_function', seed=4)
        logger.info(f"Generated family (node function): {len(family_func)} tasks.")


        # Test invalid inputs
        try:
            generate_task_family(base_dag, num_tasks=3, variation_type='invalid_type')
        except TaskFamilyGenerationError as e:
            logger.info(f"Caught expected error for invalid type: {e}")

        try:
            generate_task_family(base_dag, num_tasks=-1)
        except TaskFamilyGenerationError as e:
             logger.info(f"Caught expected error for invalid num_tasks: {e}")

        logger.info("Basic tests completed.")

    except ImportError:
        logger.error("Could not import GraphFactory. Skipping basic test.")
    except Exception as e:
         logger.exception(f"An unexpected error occurred during testing: {e}")
