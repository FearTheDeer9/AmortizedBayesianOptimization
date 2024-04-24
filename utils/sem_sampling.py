import logging
from collections import OrderedDict
from itertools import combinations, product
from typing import Callable, List

import numpy as np

from utils.graph_utils.graph import GraphStructure


# removed the dynamic component as I am only considering static SEMs
# I also don't think I need this timestep component to it
def sample_from_SEM(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon: dict = None,
    seed: int = None,
    std: float = 0.1,
) -> OrderedDict:
    """
    Function to sample from a SEM, potentially with interventions or initial values.

    Parameters
    ----------
    static_sem : OrderedDict
        SEMs specifying the relationships among variables.
    initial_values : dict, optional
        Initial values of nodes, by default None.
    interventions : dict, optional
        Specifies interventions on variables, by default None.
    epsilon : dict, optional
        Specifies noise for each variable, by default standard Gaussian noise is used.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    OrderedDict
        A sample from the SEM given previously implemented interventions or initial values.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Initialize noise model
    if epsilon is None:
        epsilon = {k: rng.normal(scale=std) for k in static_sem.keys()}
    assert isinstance(epsilon, dict)

    # Initialize sample
    sample = OrderedDict()

    for var, function in static_sem.items():
        # Apply interventions or initial values if specified
        if interventions and var in interventions:
            sample[var] = interventions[var]
        elif initial_values and var in initial_values:
            sample[var] = initial_values[var]
        else:
            # Otherwise, sample from the model using the specified noise
            sample[var] = function(epsilon[var], sample)

    return sample


def sample_from_SEM_hat(
    static_sem: OrderedDict,
    node_parents: Callable,
    initial_values: dict = None,
    interventions: dict = None,
    seed: int = None,
) -> OrderedDict:
    """
    Function to sample from a SEM, considering interventions or initial values.

    Parameters
    ----------
    static_sem : OrderedDict
        SEMs specifying the relationships among variables.
    node_parents : Callable
        Function that returns the parents of the given node.
    initial_values : dict, optional
        Initial values of nodes, by default None.
    interventions : dict, optional
        Specifies interventions on variables, by default None.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    OrderedDict
        A sample from the SEM given previously implemented interventions or initial values.
    """
    if seed:
        np.random.seed(seed)

    sample = OrderedDict()

    for var, function in static_sem.items():
        # Apply interventions or initial values if specified
        if interventions and var in interventions:
            sample[var] = interventions[var]
        elif initial_values and var in initial_values:
            sample[var] = initial_values[var]
        else:
            # Find parents and sample from the model
            parents = node_parents(var, None)
            if parents:
                sample[var] = function(None, parents, sample)
            else:
                # Sample source node marginal
                sample[var] = function(None, var)

    return sample


def sample_model(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    node_parents: Callable = None,
    sample_count: int = 100,
    epsilon: dict = None,
    use_sem_estimate: bool = False,
    seed: int = None,
) -> dict:
    """
    Draws multiple samples from Bayesian Network.

    Per variable the returned array is of the format: n_samples x timesteps in DBN.

    Returns
    -------
    dict
        Dictionary of n_samples per node in graph.
    """
    if seed:
        np.random.seed(seed)

    new_samples = {k: [] for k in static_sem.keys()}
    for i in range(sample_count):
        # This option uses the estimates of the SEMs, estimates found through use of GPs.
        if use_sem_estimate:
            tmp = sample_from_SEM_hat(
                static_sem=static_sem,
                node_parents=node_parents,
                initial_values=initial_values,
                interventions=interventions,
            )
        # This option uses the true SEMs.
        else:
            if epsilon is not None and isinstance(epsilon, list):
                epsilon_term = epsilon[i]
            else:
                epsilon_term = epsilon

            tmp = sample_from_SEM(
                static_sem=static_sem,
                initial_values=initial_values,
                interventions=interventions,
                epsilon=epsilon_term,
            )
        for var in static_sem.keys():
            new_samples[var].append(tmp[var])

    for var in static_sem.keys():
        new_samples[var] = np.vstack(new_samples[var])

    return new_samples


def create_grid_interventions(
    ranges: OrderedDict,
    num_points: int = 10,
    include_full_combination: bool = True,
    get_list_format: bool = False,
) -> List:
    """Create both individual and combined grid interventions for given variable ranges.

    Args:
        ranges (OrderedDict): Ranges for each variable.
        num_points (int): Number of points to sample in each range.

    Returns:
        list: List various interventions in the corresponding ranges
    """
    grids = {
        var: np.linspace(min_val, max_val, num_points)
        for var, (min_val, max_val) in ranges.items()
    }

    interventions = []

    # Individual and combination interventions
    for size in range(1, len(ranges) + 1):
        for variable_subset in combinations(ranges.keys(), size):
            for product_values in product(*(grids[var] for var in variable_subset)):
                interventions.append(dict(zip(variable_subset, product_values)))

    # Optionally remove the full combination if not desired
    if not include_full_combination and len(ranges) > 1:
        interventions = [intv for intv in interventions if len(intv) < len(ranges)]

    new_grid = {}

    # Loop through each dictionary in the original grid
    for entry in interventions:
        keys = tuple(
            sorted(entry.keys())
        )  # Sort keys to ensure consistent order for tuples
        values = tuple(
            entry[key] for key in keys
        )  # Get the corresponding values in tuple form

        if keys in new_grid:
            new_grid[keys].append(values)
        else:
            new_grid[keys] = [values]

    if get_list_format:
        interventions = new_grid.copy()

    return interventions


def draw_interventional_samples(
    interventions: List[dict], exploration_set: List[List[str]], graph: GraphStructure
) -> dict:
    """
    Draw interventional samples from the given list of interventions
    """
    interventional_data = {
        i: {var: [] for var in interventions}
        for i, interventions in enumerate(exploration_set)
    }
    target = graph.target

    # Add a 'target' key for each intervention type
    for i in interventional_data:
        interventional_data[i][target] = []

    for intervention in interventions:
        intervention_keys_sorted = sorted(intervention.keys())
        index = next(
            (
                i
                for i, sublist in enumerate(exploration_set)
                if sorted(sublist) == intervention_keys_sorted
            ),
            None,
        )
        if index is not None:
            sample = sample_model(graph.SEM, sample_count=1, interventions=intervention)
            for var in intervention:
                interventional_data[index][var].append(sample[var][0, 0])

            interventional_data[index][target].append(sample[target][0, 0])

    # Convert lists to numpy arrays for easier manipulation and consistency
    for idx in interventional_data:
        for var in interventional_data[idx]:
            interventional_data[idx][var] = np.array(interventional_data[idx][var])

    return interventional_data
