import logging
import random
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, OrderedDict, Tuple

import numpy as np
import statsmodels.api as sm
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression
from scipy.special import logsumexp

from utils.graph_utils.graph import GraphStructure
from utils.sem_sampling import sample_model
from utils.utils_classes import DoFunctions
from utils.utils_functions import set_up_GP


class MyKDENew(sm.nonparametric.KDEUnivariate):
    def __init__(self, *args):
        super().__init__(*args)

    def sample(self, n_samples=1, random_state=None):
        u = np.random.uniform(0, 1, size=n_samples)
        i = (u * self.endog.shape[0]).astype(np.int64)

        # if self.kernel == 'gaussian':
        return (
            np.atleast_2d(np.random.normal(self.endog[i], self.kernel.h)),
            self.endog[i],
        )


def inverse_cdf(su, W):
    """
    The inverse CDF for a finite distribution
    """
    cumulative_sums = np.cumsum(W)
    return np.searchsorted(cumulative_sums, su)


def update_posterior_interventional(
    graphs: List[GraphStructure],
    posterior: List[float],
    intervened_var: Tuple,
    all_emission_fncs: List[OrderedDict[str, Callable]],
    interventional_samples: Dict[Tuple, Dict[str, np.ndarray]],
    lr=0.05,
):

    interventional_samples = interventional_samples[intervened_var]
    for graph_idx, emission_fncs in enumerate(
        all_emission_fncs
    ):  # as many emission_fncs dicts as graphs

        graph = graphs[graph_idx]
        for var in emission_fncs:
            parents = graph.parents[var]
            xx = np.hstack(
                [interventional_samples[parent].reshape(1, -1) for parent in parents]
            )
            yy = interventional_samples[var].reshape(1, -1)

            if var in intervened_var:
                # Here the truncated assumption comes in. Dont compute posterior
                continue
            # else:
            #     print(var, parents)

            posterior[graph_idx] += lr * log_likelihood(
                emission_fncs[var], xx, yy  # the model
            )

    return posterior


def log_likelihood(model: GPRegression, X_test: np.ndarray, y_test: np.ndarray):
    if y_test is not None:
        log_lik = model.log_predictive_density(x_test=X_test, y_test=y_test)
        res = log_lik.flatten().sum()
    else:
        # Marginal
        log_lik = model.score_samples(X_test)
        res = log_lik.flatten().sum()

    return res


def stratified(W, M):
    # Generate stratified random samples
    su = (np.random.rand(M) + np.arange(M)) / M

    # Ensure su is sorted because inverse CDF requires sorted inputs
    su = np.sort(su)

    # Call inverse_cdf which should be defined to handle this sorted uniform data
    return inverse_cdf(su, W)


def aggregate_mean_function(
    i: int,
    do_functions: List[List[DoFunctions]],
    posterior: np.ndarray,
    x: np.ndarray,
):
    """
    The overall mean of the interventional distribution based on all the graphs, uses the latent
    variable G and the posterior distribution to calculate it
    """
    # calculates E[Y | do(x)] = E[E[Y|(do(x), G]], i.e. it calculates the mean over all the graphs
    unweighted_means = np.hstack(
        [do_functions_es[i].mean_function_do(x) for do_functions_es in do_functions]
    )

    mean = unweighted_means @ posterior
    mean = mean.reshape(-1, 1)
    return mean


def aggregate_var_function(
    i: int,
    do_functions: List[List[DoFunctions]],
    posterior: np.ndarray,
    x: np.ndarray,
):
    """
    The overall variance function for the interventional distribution based on all the graphs,
    introduces the latent variable G in the graph
    """
    # calculates V[Y | do(x)] = E[V[Y|(do(x), G]] + V[E[Y | do(x), G]], i.e., it calculates the variance over all the graphs
    unweighted_means = np.hstack(
        [do_functions_es[i].mean_function_do(x) for do_functions_es in do_functions]
    )

    unweighted_second_moments = np.hstack(
        [do_function_es[i].mean_function_do(x) ** 2 for do_function_es in do_functions]
    )

    unweighted_vars = np.hstack(
        [do_functions_es[i].var_function_do(x) for do_functions_es in do_functions]
    )

    var = (
        unweighted_vars @ posterior
        + unweighted_second_moments @ posterior
        - (unweighted_means @ posterior) ** 2
    )
    var = var.reshape(-1, 1)
    return var


def update_model_mean(i: int):
    pass


def update_posterior_model_aggregate(
    exploration_set: List,
    trial_observed: bool,
    model_list: List[GPyModelWrapper],
    data_x_list: dict,
    data_y_list: dict,
    causal_prior: bool,
    best_variable: int,
    input_space: List,
    do_function_list: List[List[DoFunctions]],
    posterior: np.ndarray,
) -> List[GPyModelWrapper]:
    """
    Update the posterior of the gaussian process if it was intervened on in the previous timestep
    The do_functions is now a list of all variables for all graphs
    """
    # update the Gaussian Processes
    if trial_observed:
        # update all the models if we observed in the previous trial
        # this one uses the computed do functions by fitting the causal graph
        for j in range(len(exploration_set)):
            X = data_x_list[j]
            Y = data_y_list[j].reshape(-1, 1)
            do_function_list[j]
            mean_function = partial(
                aggregate_mean_function, j, do_function_list, posterior
            )
            var_function = partial(
                aggregate_var_function, j, do_function_list, posterior
            )
            model_list[j] = set_up_GP(
                causal_prior, input_space[j], mean_function, var_function, X, Y
            )
    else:
        # only update the model of the set that was intervened upon
        Y = data_y_list[best_variable].reshape(-1, 1)
        X = data_x_list[best_variable]
        mean_function = partial(
            aggregate_mean_function, best_variable, do_function_list, posterior
        )
        var_function = partial(
            aggregate_var_function, best_variable, do_function_list, posterior
        )
        model_list[best_variable] = set_up_GP(
            causal_prior, input_space[best_variable], mean_function, var_function, X, Y
        )

    return model_list


def update_arm_distribution(
    arm_distribution: np.ndarray,
    bo_models: List[GPyModelWrapper],
    inputs: List,
    arm_mapping_n_es: Dict[Tuple, int],
    beta=0.1,
) -> np.ndarray:
    """
    The arm distribution is the probability that the current exploration set is the optimal
    exploration set
    """
    for i in range(len(bo_models)):
        es = arm_mapping_n_es[i]
        inp = inputs[i]
        preds_mean, preds_var = bo_models[i].predict(inp)
        # assuming that the task is to take the minimum
        min_val = np.argmin(preds_mean)
        arm_distribution[i] = preds_mean[min_val] - beta * preds_var[min_val]

    e_x = np.exp(arm_distribution - np.max(arm_distribution))
    return e_x / e_x.sum()


def build_p_y_star(
    exploration_set: List[str],
    bo_models: List[GPyModelWrapper],
    int_grids,
    parameter_int_domain,
    n_samples: int = 200,
) -> Tuple[np.ndarray, List]:
    """
    Building an optimal value for every element in the exploration set
    """
    # sets = bo_models.keys()
    all_ystar = np.empty(shape=(len(exploration_set), n_samples))
    all_xstar = [[] for _ in range(len(exploration_set))]

    for i, es in enumerate(exploration_set):
        emukit_model: GPyModelWrapper = bo_models[i]
        gpy_model: GPRegression = emukit_model.model
        if len(es) > 1:
            inps = parameter_int_domain[es].sample_uniform(point_count=100)
            inps = np.array(inps).resahpe(-1, len(es))
        else:
            inps = parameter_int_domain[tuple(es)]
            inps = np.array(inps).reshape(-1, 1)

        # this is different from the one used in the github code
        samples = gpy_model.posterior_samples_f(inps, size=n_samples).squeeze()
        all_ystar[i, :] = np.min(samples, axis=0).squeeze()
        all_xstar[i] = inps[np.argmin(samples, axis=0), :].squeeze()

    return all_ystar, all_xstar


def sample_global_xystar(
    n_samples_mixture: int, all_ystar: np.ndarray, arm_dist: np.ndarray
):
    """
    This calculate p(y*|D) = sum P(X_I = X*) P(yI*|D)
    It does this by sampling from the conditional distribution
    """
    # Select indexes of mixture components to sample from first
    mixture_idxs = stratified(
        W=arm_dist, M=n_samples_mixture
    )  # lower variance sampling

    local_pystars = []
    # This fits a KDE to each local p(Y*) i.e. p(Y*_(Z) | D), p(Y*_(X) | D)
    # It looks at all the elements in the exploration set
    for mixt_idx in range(all_ystar.shape[0]):
        temp = all_ystar[mixt_idx, :].reshape(-1, 1)
        kde = MyKDENew(temp)
        try:
            kde.fit()
        except RuntimeError:
            kde.fit(bw=0.5)

        local_pystars.append(kde)

    resy = np.empty(mixture_idxs.shape[0])

    unique_mixture_idxs, counts = np.unique(mixture_idxs, return_counts=True)
    running_cumsum = 0

    corresponding_x = []  # TODO

    for j, (mix_id, count) in enumerate(zip(unique_mixture_idxs, counts)):
        if j == 0:
            resy[:count], _ = local_pystars[mix_id].sample(n_samples=count)
        else:
            resy[running_cumsum : running_cumsum + count], _ = local_pystars[
                mix_id
            ].sample(n_samples=count)
        running_cumsum += count

    return resy, corresponding_x


def normalize_log(l):
    return np.exp(l - logsumexp(l))


def get_monte_carlo_expectation(intervention_samples: dict):
    """
    Computes the interventional expectation based on the data
    """
    assert isinstance(intervention_samples, dict)
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = intervention_samples[es].mean(axis=0)

    # Returns the expected value of the intervention via MC sampling
    return new


def optimal_sequence_of_interventions(
    exploration_sets: List,
    interventional_grids: Dict,
    graph: GraphStructure,
    task: str = "min",
) -> Tuple:

    # setting up the noise model
    model_variables = graph.variables
    static_noise_model = {k: 0 for k in model_variables}

    target_variable = graph.target
    SEM = graph.SEM
    assert target_variable is not None
    assert target_variable in model_variables

    best_s_sequence = []
    best_s_values = []
    best_objective_values = []

    optimal_interventions = {setx: [] for setx in exploration_sets}

    y_stars = deepcopy(optimal_interventions)
    all_CE = []
    blank_intervention_blanket = {node: [] for node in graph.nodes}

    CE = {es: [] for es in exploration_sets}

    for s in exploration_sets:

        # Reset blanket so as to not carry over levels from previous exploration set
        intervention_blanket = {}
        for level in interventional_grids[s]:

            # Univariate intervention
            if len(s) == 1:
                intervention_blanket[s[0]] = float(level[0])

            # Multivariate intervention
            else:
                for var, val in zip(s, level):
                    intervention_blanket[var] = val

            intervention_samples = sample_model(
                SEM,
                interventions=intervention_blanket,
                sample_count=10,
                epsilon=static_noise_model,
            )

            out = get_monte_carlo_expectation(intervention_samples)
            CE[s].append((out[target_variable][0]))

    local_target_values = []
    for s in exploration_sets:
        if task == "min":
            idx = np.array(CE[s]).argmin()
        else:
            idx = np.array(CE[s]).argmax()
        local_target_values.append((s, idx, CE[s][idx]))
        y_stars[s] = CE[s][idx]
        optimal_interventions[s] = interventional_grids[s][idx]

    # Find best intervention at time t
    best_s, best_idx, best_objective_value = min(
        local_target_values, key=lambda t: t[2]
    )
    best_s_value = interventional_grids[best_s][best_idx]

    best_s_sequence.append(best_s)
    best_s_values.append(best_s_value)
    best_objective_values.append(best_objective_value)
    all_CE.append(CE)

    return (
        best_s_values,
        best_s_sequence,
        best_objective_values,
        y_stars,
        optimal_interventions,
        all_CE,
    )


def evaluate_acquisition_ceo(graph: GraphStructure, bo_model: GPyModelWrapper):
    """
    This just assumes now that we are using causal entropy search
    """
    pass
