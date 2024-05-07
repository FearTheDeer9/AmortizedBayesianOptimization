import abc
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import sample_model


class BASE:
    """
    This is the generic algorithm class where some of the algorithms will have the
    same functions
    """

    __metaclass__ = abc.ABCMeta

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
            graph = ToyGraph()
        elif self.graph_type == "Graph4":
            graph = Graph4Nodes()
        elif self.graph_type == "Graph5":
            graph = Graph5Nodes()
        elif self.graph_type == "Graph6":
            graph = Graph6Nodes()
        return graph

    @abc.abstractmethod
    def plot_model_list(
        self, model_list: List[GPyModelWrapper], es: Tuple[str], size: int = 100
    ) -> None:
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
            predictions[i], var[i] = model_list[es_num].model.predict(value)

        plt.plot(intervention_vals, true_vals, label="True")
        plt.plot(intervention_vals, predictions, label="Do 1")
        plt.fill_between(
            intervention_vals,
            [p - 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            [p + 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            color="gray",
            alpha=0.5,
        )
        plt.legend()
        plt.show()
