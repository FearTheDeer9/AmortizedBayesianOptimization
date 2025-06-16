#!/usr/bin/env python3
"""
PARENT_SCALE Helper Functions

Extracted helper functions and utilities from PARENT_SCALE integration.
These are the core utility functions needed for PARENT_SCALE algorithm execution
including custom KDE, optimization utilities, and intervention/sampling functions.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

import numpy as onp
from sklearn.neighbors import KernelDensity
from GPy.models.gp_regression import GPRegression


class MyKDE(KernelDensity):
    """Custom KDE implementation from PARENT_SCALE."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None

    def fit_and_update(self, X):
        self.X = X
        return super().fit(X)

    def predict(self):
        return onp.mean(super().sample(n_samples=500)), onp.var(
            super().sample(n_samples=500)
        )


def safe_optimization(
    gpy_model: GPRegression,
    lower_bound_var: float = 1e-05,
    upper_bound_var: float = 2.0,
    bound_len: int = 20,
) -> GPRegression:
    """Safe optimization function from PARENT_SCALE."""
    if gpy_model.kern.variance[0] < lower_bound_var:
        logging.info("SAFE OPTIMIZATION: Resetting the kernel variance to lower bound")
        gpy_model.kern.variance[0] = lower_bound_var

    if gpy_model.kern.lengthscale[0] > bound_len:
        logging.info("SAFE OPTIMZATION: Resetting kernel lenghtscale")
        gpy_model.kern.lengthscale[0] = 1.0

    if gpy_model.likelihood.variance[0] > upper_bound_var:
        logging.info("SAFE OPTIMIZATION: restting likelihood var to upper bound")
        gpy_model.likelihood.variance[0] = upper_bound_var

    if gpy_model.likelihood.variance[0] < lower_bound_var:
        logging.info("SAFE OPTIMIZATION: resetting likelihood var to lower bound")
        gpy_model.likelihood.variance[0] = lower_bound_var
    return gpy_model


def set_intervention_values(
    variables: Dict, intervention: str, value: float, num_observations: int
):
    """
    Changing the variables that was intervened upon when computing
    the outcome of the new graph
    """
    variables[intervention] = value * onp.ones((num_observations, 1))


def predict_child(
    function: GPRegression, parent_values: Dict[str, onp.ndarray], parents: List
):
    """
    Makes sure that the parent variables are in the correct order for the GP model
    """
    parent_values_cols = onp.hstack([parent_values[val] for val in parents])
    return function.predict(parent_values_cols)[0]


def predict_causal_effect(
    functions: Dict[str, GPRegression],
    variables: Dict[str, onp.ndarray],
    parents_Y: List[str],
    num_observations: int,
    target: str,
):
    """
    Predicts the causal effect after all the variables has been intervened upon
    Returns the mean and the variance of the observation
    """
    input_Y = onp.hstack(
        [variables[parent].reshape(num_observations, -1) for parent in parents_Y]
    )
    gp_Y = functions[target]
    predictions = gp_Y.predict(input_Y)
    return onp.mean(predictions[0]), onp.mean(predictions[1])


def propogate_effects(
    node: str,
    functions: OrderedDict,
    variables: Dict,
    children: Dict,
    parents: Dict,
    observational_samples: Dict,
):
    """Propagate intervention effects through the graph."""
    if node in children:
        for child in children[node]:
            if child not in variables:  # Child has not been intervened upon
                # Get parent values for the child
                parent_values = {
                    p: variables[p] if p in variables else observational_samples[p]
                    for p in parents[child]
                }
                # Calculate new value for the child
                variables[child] = predict_child(
                    functions[child], parent_values, parents[child]
                )
                # Recursively update the child's children
                propogate_effects(
                    child,
                    functions,
                    variables,
                    children,
                    parents,
                    observational_samples,
                )


def sample_from_SEM_original(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon: dict = None,
    seed: int = None,
    std: float = 0.1,
) -> OrderedDict:
    """
    Function to sample from a SEM, potentially with interventions or initial values.
    
    Copied exactly from causal_bayes_opt_old/utils/sem_sampling.py
    """
    if seed is not None:
        rng = onp.random.default_rng(seed)
    else:
        rng = onp.random.default_rng()

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


def sample_model_original(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    node_parents = None,
    sample_count: int = 100,
    epsilon: dict = None,
    use_sem_estimate: bool = False,
    seed: int = None,
    graph = None,
    noiseless: bool = False,
    use_iscm: bool = False,
) -> dict:
    """
    Draws multiple samples from Bayesian Network.
    
    Copied exactly from causal_bayes_opt_old/utils/sem_sampling.py
    """
    if seed:
        onp.random.seed(seed)

    new_samples = {k: [] for k in static_sem.keys()}
    for _ in range(sample_count):
        # This option uses the true SEMs.
        if graph is not None:
            epsilon_term = graph.get_error_distribution(noiseless=noiseless)
        else:
            epsilon_term = epsilon

        tmp = sample_from_SEM_original(
            static_sem=static_sem,
            initial_values=initial_values,
            interventions=interventions,
            epsilon=epsilon_term,
            seed=seed,
        )
        
        for var in static_sem.keys():
            new_samples[var].append(tmp[var])

    for var in static_sem.keys():
        new_samples[var] = onp.vstack(new_samples[var])

    return new_samples


def draw_interventional_samples_sem_original(
    exploration_set,
    graph,
    n_int: int = 2,
    seed: int = None,
    noiseless: bool = True,
    use_iscm: bool = False,
) -> dict:
    """
    Draw interventional samples from the given list of interventions.
    
    Copied exactly from causal_bayes_opt_old/utils/sem_sampling.py
    """
    if seed is not None:
        onp.random.seed(seed=seed)

    interventional_data = {
        tuple(es): {var: [] for var in graph.variables} for es in exploration_set
    }

    interventional_range = graph.get_interventional_range()
    for _ in range(n_int):
        for es in exploration_set:
            intervention = {}
            for var in es:
                intervention_sample = onp.random.uniform(
                    interventional_range[var][0], interventional_range[var][1]
                )
                intervention[var] = intervention_sample

            # Drawing the samples
            sample = sample_model_original(
                graph.SEM,
                sample_count=1,
                interventions=intervention,
                graph=graph,
                noiseless=noiseless,
                use_iscm=use_iscm,
            )
            for var in sample:
                interventional_data[tuple(es)][var].append(sample[var][0, 0])

    for idx in interventional_data:
        for var in interventional_data[idx]:
            interventional_data[idx][var] = onp.array(interventional_data[idx][var])
    return interventional_data


def setup_observational_interventional_original(
    graph,
    n_obs: int = 100,
    n_int: int = 2,
    noiseless: bool = True,
    seed: int = 42,
    use_iscm: bool = False,
):
    """
    Setup the graph based on the structure we are using.
    
    Copied from causal_bayes_opt_old/graphs/data_setup.py
    """
    print(f"Sampling the observational data with original method")

    D_O = sample_model_original(
        graph.SEM, sample_count=n_obs, graph=graph, use_iscm=use_iscm, seed=seed
    )

    exploration_set = graph.get_exploration_set()
    # getting the interventional data in two different formats
    print(f"Sampling the interventional data with original method")
    D_I = draw_interventional_samples_sem_original(
        exploration_set,
        graph,
        n_int=n_int,
        seed=seed,
        noiseless=noiseless,
        use_iscm=use_iscm,
    )

    return D_O, D_I, exploration_set