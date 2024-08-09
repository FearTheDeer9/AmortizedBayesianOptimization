import abc
import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models import GPRegression

from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_functions import create_grid_interventions
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import sample_model


class BASE:
    """
    This is the generic algorithm class where some of the algorithms will have the
    same functions
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_model_dict(self) -> Dict[str, GPyModelWrapper]:
        model_list_dict = {
            key: self.model_list_overall[i]
            for i, key in enumerate(self.exploration_set)
        }
        return model_list_dict

    @abc.abstractmethod
    def get_graph(self) -> GraphStructure:
        return self.graph

    @abc.abstractmethod
    def get_exploration_set(self) -> List[Tuple[str]]:
        return self.exploration_set

    @property
    @abc.abstractmethod
    def graph_type(self) -> str:
        return self._graph_type

    @abc.abstractmethod
    def chosen_structure(self) -> GraphStructure:
        """
        Setup the graph based on the structure we are using
        """
        assert self.graph_type in ["Toy", "Graph4", "Graph5", "Graph6"]
        if self.graph_type == "Toy":
            graph = ToyGraph(noiseless=self.noiseless)
        elif self.graph_type == "Graph4":
            graph = Graph4Nodes()
        elif self.graph_type == "Graph5":
            graph = Graph5Nodes()
        elif self.graph_type == "Graph6":
            graph = Graph6Nodes()
        return graph

    def plot_model_list(
        self,
        model_list: List[GPyModelWrapper],
        es: Tuple[str],
        size: int = 100,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # Setting up the plotting stuff
        true_vals = np.zeros(shape=size)
        predictions = np.zeros(shape=size)
        var = np.zeros(shape=size)
        es_num = self.es_to_n_mapping[es]
        interventions = {}

        # Getting the number of entries in the exploration set
        intervention_domain = self.graph.get_original_interventional_range()
        min_intervention, max_intervention = intervention_domain[es[0]]
        intervention_vals = np.linspace(
            start=min_intervention, stop=max_intervention, num=size
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
                )[self.graph.target]
            )
            value = np.array([interventions[var] for var in es]).reshape(1, -1)
            predictions[i], var[i] = model_list[es_num].model.predict(value)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot the true values and predictions
        ax.plot(
            intervention_vals,
            true_vals,
            label="True",
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        ax.plot(
            intervention_vals,
            predictions,
            label=f"Do {es}",
            color="red",
            linewidth=2,
        )

        # Fill the area between the prediction intervals
        ax.fill_between(
            intervention_vals,
            [p - 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            [p + 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            color="gray",
            alpha=0.3,
        )

        # Add grid, labels, title, and legend
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel(kwargs.get("xlabel", "Intervention Value"), fontsize=12)
        ax.set_ylabel(kwargs.get("ylabel", "Value"), fontsize=12)
        ax.set_title(
            kwargs.get("title", "Model Predictions"), fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)

        return fig, ax

    def quantify_total_uncertainty(self, num_points: int = 100):
        # average it over all the surrogate models
        model_dict = self.get_model_dict()
        exploration_set = self.get_exploration_set()
        graph = self.get_graph()

        self.intervention_grid = create_grid_interventions(
            graph.get_interventional_range(),
            get_list_format=True,
            num_points=num_points,
        )

        if exploration_set is None:
            exploration_set = graph.get_sets()[2]
            exploration_set = [tuple(exploration_set)]
        all_entropies = np.zeros(shape=len(exploration_set))
        for i, es in enumerate(exploration_set):
            grid = self.intervention_grid[es]
            model: GPRegression = model_dict[es].model
            # print(grid)
            values = np.vstack([value for value in grid])
            variance = model.predict(values)[1]
            entropy = 1 / 2 * np.log(2 * variance * np.pi * np.e)
            all_entropies[i] = np.mean(entropy)

        total_uncertainty = {}
        for i in range(len(all_entropies)):
            total_uncertainty[exploration_set[i]] = all_entropies[i]

        total_uncertainty["average"] = np.mean(all_entropies)
        logging.info(f"The total uncertainty is {total_uncertainty}")
        return total_uncertainty
