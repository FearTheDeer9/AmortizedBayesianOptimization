import logging
import random
from copy import deepcopy
from typing import Callable, Dict, List, OrderedDict, Tuple

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression

import utils.ceo_utils as ceo_utils
import utils.utils_functions as utils_functions
from utils.graph_utils.graph import GraphStructure
from utils.graph_utils.graph_functions import create_grid_interventions, graph_setup
from utils.graph_utils.toy_graph import ToyGraph
from utils.sem_sampling import (
    change_intervention_list_format,
    draw_interventional_samples_sem,
    sample_model,
)
from utils.utils_acquisitions import evaluate_acquisition_ceo
from utils.utils_classes import DoFunctions, TargetClass

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


class CEO:

    def __init__(
        self,
        graph_type: str = "Toy",
        graphs: List[GraphStructure] = None,
        observational_samples: Dict = None,
        interventional_samples: Dict = None,
        causal_prior: bool = None,
        all_graph_edges: List[List[Tuple[str, str]]] = None,
        cost_num: int = 1,
        n_obs: int = 100,
        n_int: int = 2,
        task: str = "min",
    ):
        if graphs:
            self.graphs = graphs
            self.D_O = observational_samples
            self.D_I = interventional_samples
        else:
            self.graphs: List[GraphStructure] = []
            for edges in all_graph_edges:
                if graph_type == "Toy":
                    graph = ToyGraph()
                    graph.mispecify_graph(edges)
                    self.graphs.append(graph)

            (
                _,
                self.exploration_set,
                self.manipulative_variables,
                self.target,
                self.samples,
                self.observational_samples,
                self.interventional_samples,
            ) = graph_setup(graph_type=graph_type)

        self.causal_prior = causal_prior
        self.cost_num = cost_num
        self.task = task
        self.n_obs = n_obs
        self.n_int = n_int

        self.SEM = self.graphs[0].SEM
        self.variables = self.graphs[0].variables
        graph = self.graphs[0]
        self.interventional_range = graph.get_interventional_range()
        self.intervention_grid = create_grid_interventions(
            self.interventional_range, get_list_format=True
        )

        self.arm_distribution = np.array(
            [1 / len(self.exploration_set)] * len(self.exploration_set)
        )

        self.D_O: Dict[str, np.ndarray] = sample_model(
            self.SEM, sample_count=self.n_obs
        )

        # drawing the interventional samples
        interventional_ranges = graph.get_interventional_range()
        interventions = create_grid_interventions(interventional_ranges)
        self.D_I = draw_interventional_samples_sem(
            interventions, self.exploration_set, graph
        )

        # this just gets the interventional data in a format that is used by another function
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set
        )

        # get the initial posterior model in log form
        self.posterior = np.log(np.asarray([1 / len(self.graphs)] * len(self.graphs)))
        self.all_posteriors = []
        self.all_posteriors.append(ceo_utils.normalize_log(deepcopy(self.posterior)))

        # the cost of intervening on the system
        self.cost_functions = self.graphs[0].get_cost_structure(1)

    def set_values(self, D_O: Dict, D_I: Dict, exploration_set: List[List[str]]):
        logging.info("Using predefined values for the optimization algorithm")
        self.exploration_set = exploration_set
        self.D_O = D_O
        self.D_I = D_I
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set
        )
        print(self.D_I)
        print(self.D_O)
        print(self.interventional_samples)

    def run_algorithm(self, T=30):

        (
            data_x_list,
            data_y_list,
            best_intervention_value,
            current_global_min,
            best_variable,
        ) = utils_functions.define_initial_data_CBO(
            self.interventional_samples,
            self.exploration_set,
            self.manipulative_variables,
            self.target,
        )

        # print(data_x_list)

        # stuff needed at this point, graph, true_objective_value, all_CE

        # get the surrogate model for each of the graphs
        sem_emit_fncs: List[OrderedDict[str, GPRegression]] = []
        do_effects_functions: List[List[DoFunctions]] = []
        for graph in self.graphs:
            graph.fit_samples_to_graph(self.samples, set_priors=False)
            sem_emit_fncs.append(graph.functions)

        all_posteriors = []
        all_posteriors.append(ceo_utils.normalize_log(deepcopy(self.posterior)))

        for es in self.exploration_set:
            self.posterior = ceo_utils.update_posterior_interventional(
                self.graphs, self.posterior, tuple(es), sem_emit_fncs, self.D_I
            )

        for graph in self.graphs:
            # this is the mean and variance for each graph for each element in the exploration set
            do_effects_functions.append(
                utils_functions.update_all_do_functions(
                    graph, self.observational_samples, self.exploration_set
                )
            )

        input_space = [len(es) for es in self.exploration_set]
        causal_prior = True
        # model_list: List[GPyModelWrapper] = [None] * len(exploration_set)
        model_list_overall: List[GPyModelWrapper] = [None] * len(self.exploration_set)
        arm_n_es_mapping = {i: es for i, es in enumerate(self.exploration_set)}
        arm_es_n_mapping = {tuple(es): i for i, es in enumerate(self.exploration_set)}
        target_classes: List[TargetClass] = [
            TargetClass(self.SEM, es, self.variables) for es in self.exploration_set
        ]
        trial_observed = []

        for i in range(T):
            if i == 0:
                # # update the prior of all the Gaussian Processes for each graph
                trial_observed.append(True)
                model_list_overall = ceo_utils.update_posterior_model_aggregate(
                    self.exploration_set,
                    True,
                    model_list_overall,
                    data_x_list,
                    data_y_list,
                    causal_prior,
                    best_variable,
                    input_space,
                    do_effects_functions,
                    all_posteriors[-1],
                )
            else:

                # update the arm distribution, i.e. that each intervention in the exploration set is optimal
                logging.info(f"----------------ITERATION {i}----------------")
                logging.info(
                    f"Current posterior {all_posteriors[-1]}, {all_posteriors[-1].sum()}"
                )
                trial_observed.append(False)

                logging.info(
                    "Updating the models based on the previous observed samples"
                )
                model_list_overall = ceo_utils.update_posterior_model_aggregate(
                    self.exploration_set,
                    trial_observed[i - 1],
                    model_list_overall,
                    data_x_list,
                    data_y_list,
                    causal_prior,
                    best_variable,
                    input_space,
                    do_effects_functions,
                    all_posteriors[-1],
                )

                logging.info("Now setting up the arm distribution")
                # updating the arm distribution
                self.arm_distribution = ceo_utils.update_arm_distribution(
                    self.arm_distribution,
                    model_list_overall,
                    data_x_list,
                    arm_n_es_mapping,
                )
                print(self.arm_distribution)

                # sampling from each exploration set value
                logging.info("Building the py star")
                py_star_samples, p_x_star_samples = ceo_utils.build_p_y_star(
                    self.exploration_set,
                    model_list_overall,
                    self.interventional_range,
                    self.intervention_grid,
                )
                print(py_star_samples)

                # getting the overall sample
                logging.info("Building the global py star")
                samples_global_ystar, samples_global_xstar = (
                    ceo_utils.sample_global_xystar(
                        n_samples_mixture=1000,
                        all_ystar=py_star_samples,
                        arm_dist=self.arm_distribution,
                    )
                )
                print(samples_global_ystar)

                logging.info("Fitting the global KDE estimate")
                kde_global = ceo_utils.MyKDENew(samples_global_ystar)
                try:
                    kde_global.fit()
                except RuntimeError:
                    kde_global.fit(bw=0.5)

                y_acquisition_list = [None] * len(self.exploration_set)
                x_new_list = [None] * len(self.exploration_set)
                logging.info("Starting with the entropy search")
                for s, es in enumerate(self.exploration_set):
                    # figure out the sem_hat and sem_ems_fncs
                    # not sure what to do with inputs and improvements
                    y_acquisition_list[s], x_new_list[s], inputs, improvements = (
                        evaluate_acquisition_ceo(
                            graphs=self.graphs,
                            bo_model=model_list_overall[s],
                            exploration_set=es,
                            cost_functions=self.cost_functions,
                            posterior=all_posteriors[-1],
                            arm_distribution=self.arm_distribution,
                            pystar_samples=py_star_samples,
                            pxstar_samples=p_x_star_samples,
                            samples_global_ystar=samples_global_ystar,
                            samples_global_xstar=samples_global_xstar,
                            kde_globalystar=kde_global,
                            arm_mapping_es_to_num=arm_es_n_mapping,
                            arm_mapping_num_to_es=arm_n_es_mapping,
                            interventional_grid=self.intervention_grid,
                            all_sem_hat=sem_emit_fncs,
                        )
                    )

                logging.info(f"The acquisition is {y_acquisition_list}")
                logging.info(f"The corresponding x value is {x_new_list}")
                logging.debug(f"The inpus are {inputs}")
                logging.debug(f"The improvements are {improvements}")
                # find the optimal intervention, which maximises the acquisition function
                print(y_acquisition_list)
                target_index = np.argmax(np.array(y_acquisition_list))
                var_to_intervene = tuple(self.exploration_set[target_index])
                y_new = (
                    target_classes[target_index]
                    .compute_target(x_new_list[target_index])
                    .reshape(-1)
                )

                data_x_list[target_index] = np.vstack(
                    (data_x_list[target_index], x_new_list[target_index])
                )

                data_y_list[target_index] = np.concatenate(
                    (data_y_list[target_index], y_new)
                )
                # set the new best variable
                best_variable = target_index
                logging.info(
                    f"CEO found {self.exploration_set[best_variable]} as the best variable with value {x_new_list[target_index]} and corresponding y {y_new}"
                )

                # updating the posterior based on the previous interventional data
                all_interventional_data = target_classes[target_index].compute_all(
                    x_new_list[target_index]
                )

                # updating the interventional data
                for var in self.variables:
                    self.D_I[var_to_intervene][var] = np.concatenate(
                        (
                            self.D_I[var_to_intervene][var].reshape(-1),
                            all_interventional_data[var].reshape(-1),
                        )
                    )

                # calculating the posterior data again
                self.posterior = np.zeros(shape=len(self.graphs))
                for es in self.exploration_set:
                    self.posterior = ceo_utils.update_posterior_interventional(
                        self.graphs, self.posterior, tuple(es), sem_emit_fncs, self.D_I
                    )
                all_posteriors.append(ceo_utils.normalize_log(deepcopy(self.posterior)))
