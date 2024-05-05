import logging
import random
from copy import deepcopy
from typing import Callable, Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression

import utils.cbo_functions as cbo_functions
import utils.ceo_utils as ceo_utils
from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_functions import create_grid_interventions, graph_setup
from graphs.toy_graph import ToyGraph
from utils.cbo_classes import DoFunctions, TargetClass
from utils.ceo_acquisitions import evaluate_acquisition_ceo
from utils.sem_sampling import (
    change_intervention_list_format,
    draw_interventional_samples_sem,
    sample_model,
)

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
        self.graph_type = graph_type
        if graphs:
            self.graphs = graphs
            self.D_O = observational_samples
            self.D_I = interventional_samples
        else:
            self.graphs: List[GraphStructure] = []
            for edges in all_graph_edges:
                graph = self.chosen_structure()
                graph.mispecify_graph(edges)
                self.graphs.append(graph)

            # specify all structures in the arguments, no need to do it again
            # self.graphs.append(self.chosen_structure())

            (
                _,
                self.exploration_set,
                self.manipulative_variables,
                self.target,
                _,
                _,
                _,
            ) = graph_setup(graph_type=graph_type)
        logging.info(f"The exploration set in this setup is {self.exploration_set}")

        # create mappings for the exploration set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

        self.causal_prior = causal_prior
        self.cost_num = cost_num
        self.task = task
        self.n_obs = n_obs
        self.n_int = n_int

        self.SEM = self.graphs[0].SEM
        self.variables = self.graphs[0].variables
        self.graph = self.graphs[0]
        self.interventional_range = graph.get_interventional_range()
        self.intervention_grid = create_grid_interventions(
            self.interventional_range, get_list_format=True
        )

        self.arm_distribution = np.array(
            [1 / len(self.exploration_set)] * len(self.exploration_set)
        )

        self.D_O: Dict[str, np.ndarray] = sample_model(
            self.SEM, sample_count=self.n_obs, graph=graph
        )

        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in graph.variables])
        )

        # drawing the interventional samples
        interventional_ranges = graph.get_interventional_range()
        interventions = create_grid_interventions(interventional_ranges)
        self.D_I = draw_interventional_samples_sem(
            interventions, self.exploration_set, graph, n_int=n_int
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
        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in self.variables])
        )
        self.D_I = D_I
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set
        )

        self.arm_distribution = np.array(
            [1 / len(self.exploration_set)] * len(self.exploration_set)
        )

    def chosen_structure(self) -> GraphStructure:
        """
        Setup the graph based on the structure we are using
        """
        assert self.graph_type in ["Toy", "Graph4", "Graph5", "Graph6"]
        if self.graph_type == "Toy":
            graph = ToyGraph()
        elif self.graph_type == "Graph4":
            graph = Graph4Nodes()
        elif self.graph_type == "Graph5":
            graph = Graph5Nodes()
        elif self.graph_type == "Graph6":
            graph = Graph6Nodes()
        return graph

    def fit_samples_to_graphs(self):
        sem_emit_fncs: List[OrderedDict[str, GPRegression]] = []
        for i, graph in enumerate(self.graphs):
            logging.info(f"---Fitting samples for graph {i}---")
            graph.fit_samples_to_graph(self.D_O, set_priors=False)
            sem_emit_fncs.append(graph.functions)
        return sem_emit_fncs

    def calculate_do_statistics(self):
        do_effects_functions: List[List[DoFunctions]] = []
        for i, graph in enumerate(self.graphs):
            # this is the mean and variance for each graph for each element in the exploration set
            logging.info(f"----Computing do function for graph {i}------")
            do_effects_functions.append(
                cbo_functions.update_all_do_functions(
                    graph, self.observational_samples, self.exploration_set
                )
            )
        self.do_effects_functions = do_effects_functions
        return do_effects_functions

    def update_posterior(self):
        """
        Updating the posterior probability, this happens after intervening on the system
        """
        for es in self.exploration_set:
            self.posterior = ceo_utils.update_posterior_interventional(
                self.graphs, self.posterior, tuple(es), self.sem_emit_fncs, self.D_I
            )

    def do_function_graph(self, es: Tuple, size: int = 100, edge_num: int = 0):

        # setting up the plotting stuff
        true_vals = np.zeros(shape=size)
        predictions = np.zeros(shape=size)
        var = np.zeros(shape=size)
        es_num = self.es_to_n_mapping[es]
        interventions = {}

        # getting the number of entries in the exploration set
        intervention_domain = self.graph.get_interventional_range()
        min_intervention, max_intervention = intervention_domain[es[0]]
        intervention_vals = np.linspace(
            start=min_intervention, stop=max_intervention, num=100
        )

        for i in range(1, len(es)):
            min_i, max_i = intervention_domain[es[i]]
            interventions[es[i]] = (min_i + max_i) / 2

        for i, intervention_val in enumerate(intervention_vals):
            interventions[es[0]] = intervention_val
            true_vals[i] = np.mean(
                sample_model(
                    self.graph.SEM,
                    interventions=interventions,
                    sample_count=500,
                    graph=self.graph,
                )["Y"]
            )

            value = np.array([interventions[var] for var in es]).reshape(1, -1)
            predictions[i] = self.do_effects_functions[edge_num][
                es_num
            ].mean_function_do(value)
            var[i] = self.do_effects_functions[edge_num][es_num].var_function_do(value)

        # print(predictions)
        # print(intervention_vals)
        # print(var)
        plt.plot(intervention_vals, true_vals, label="True")
        plt.plot(intervention_vals, predictions, label="Do 1")
        plt.fill_between(
            intervention_vals,
            [p - 1.95 * np.sqrt(e) for p, e in zip(predictions, var)],
            [p + 1.95 * np.sqrt(e) for p, e in zip(predictions, var)],
            color="gray",
            alpha=0.5,
        )
        plt.legend()
        plt.show()

    def run_algorithm(self, T=30):

        (
            data_x_list,
            data_y_list,
            best_intervention_value,
            current_global_min,  # not used in the entropy algorithm
            best_variable,
        ) = cbo_functions.define_initial_data_CBO(
            self.interventional_samples,
            self.exploration_set,
            self.manipulative_variables,
            self.target,
        )

        # get the surrogate model for each of the graphs
        self.sem_emit_fncs: List[OrderedDict[str, GPRegression]] = (
            self.fit_samples_to_graphs()
        )

        # for each graph and each exploration set you have a set of functions
        self.do_effects_functions: List[List[DoFunctions]] = (
            self.calculate_do_statistics()
        )

        self.all_posteriors.append(ceo_utils.normalize_log(deepcopy(self.posterior)))
        logging.info(f"The updated posterior distribution is {self.all_posteriors[-1]}")

        input_space = [len(es) for es in self.exploration_set]
        causal_prior = True
        # model_list: List[GPyModelWrapper] = [None] * len(exploration_set)
        model_list_overall: List[GPyModelWrapper] = [None] * len(self.exploration_set)
        arm_n_es_mapping = {i: es for i, es in enumerate(self.exploration_set)}
        arm_es_n_mapping = {tuple(es): i for i, es in enumerate(self.exploration_set)}
        target_classes: List[TargetClass] = [
            TargetClass(self.SEM, es, self.variables, graph=self.graph)
            for es in self.exploration_set
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
                    self.do_effects_functions,
                    self.all_posteriors[-1],
                )
            else:

                # update the arm distribution, i.e. that each intervention in the exploration set is optimal
                logging.info(f"----------------ITERATION {i}----------------")
                logging.info(
                    f"Current posterior {self.all_posteriors[-1]}, {self.all_posteriors[-1].sum()}"
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
                    self.do_effects_functions,
                    self.all_posteriors[-1],
                )

                logging.info("Now setting up the arm distribution")
                # updating the arm distribution
                self.arm_distribution = ceo_utils.update_arm_distribution(
                    self.arm_distribution,
                    model_list_overall,
                    data_x_list,
                    arm_n_es_mapping,
                )

                # sampling from each exploration set value
                logging.info("Building the py star")
                py_star_samples, p_x_star_samples = ceo_utils.build_p_y_star(
                    self.exploration_set,
                    model_list_overall,
                    self.interventional_range,
                    self.intervention_grid,
                )

                # getting the overall sample
                logging.info("Building the global py star")
                samples_global_ystar, samples_global_xstar = (
                    ceo_utils.sample_global_xystar(
                        n_samples_mixture=1000,
                        all_ystar=py_star_samples,
                        arm_dist=self.arm_distribution,
                    )
                )

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
                            posterior=self.all_posteriors[-1],
                            arm_distribution=self.arm_distribution,
                            pystar_samples=py_star_samples,
                            pxstar_samples=p_x_star_samples,
                            samples_global_ystar=samples_global_ystar,
                            samples_global_xstar=samples_global_xstar,
                            kde_globalystar=kde_global,
                            arm_mapping_es_to_num=arm_es_n_mapping,
                            arm_mapping_num_to_es=arm_n_es_mapping,
                            interventional_grid=self.intervention_grid,
                            all_sem_hat=self.sem_emit_fncs,
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
                # self.posterior = np.zeros(shape=len(self.graphs))
                self.update_posterior()
                self.all_posteriors.append(
                    ceo_utils.normalize_log(deepcopy(self.posterior))
                )
