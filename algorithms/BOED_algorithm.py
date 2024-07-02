import logging
from typing import Dict, List, Tuple

import numpy as np

from algorithms.BASE_algorithm import BASE
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.models.posterior_model import PosteriorModel
from diffcbed.replay_buffer import ReplayBuffer
from diffcbed.strategies.acquisition_strategy import AcquisitionStrategy
from utils.sem_sampling import (
    change_int_data_format_to_mi,
    change_obs_data_format_to_mi,
)


class BOED(BASE):

    def __init__(
        self,
        env: CausalEnvironment,
        posterior_model: PosteriorModel,
        acquisition_strategy: AcquisitionStrategy,
        args: Dict,
        graph_variables: List,
    ):
        # try to get the code the same as for the previous method
        self.env: CausalEnvironment = env
        self.posterior_model: PosteriorModel = posterior_model
        self.acquisition_strategy: AcquisitionStrategy = acquisition_strategy
        self.args = args
        self.graph_variables = graph_variables
        self.buffer: ReplayBuffer = ReplayBuffer(binary=True)

    def set_values(self, D_O: Dict, D_I: Dict):
        self.D_O = change_obs_data_format_to_mi(
            D_O,
            graph_variables=self.graph_variables,
            intervention_node=np.zeros(shape=len(self.graph_variables)),
        )
        self.posterior_model.covariance_matrix = np.cov(self.D_O.samples.T)
        self.D_I = change_int_data_format_to_mi(
            D_I, graph_variables=self.graph_variables
        )
        self.buffer.update(self.D_O)
        for intervention in self.D_I:
            self.buffer.update(intervention)

        self.posterior_model.update(self.buffer.data())

    def run_algorithm(self, T: int = 50):
        # STARTING THE ALGORITHM
        current_cost: List[int] = []
        global_opt: List[float] = []
        current_y: List[float] = []
        # average_uncertainty: List[float] = []
        intervention_set: List[Tuple[str]] = []
        intervention_values: List[Tuple[float]] = []
        # global_opt.append(current_global_min)
        current_cost.append(0.0)
        # cost_functions = self.graph.get_cost_structure(self.cost_num)

        for i in range(T):
            logging.info(f"------------------EXPERIMENT {i}-------------------")
            # just keep it like this for now and don't split it into manipulative and non-manipulative
            valid_interventions = list(range(self.args.num_nodes))
            interventions, _ = self.acquisition_strategy.acquire(valid_interventions, i)

            # assuming a batch size of 1
            intervention_node = interventions["nodes"][0]
            intervention_value = interventions["values"][0]
            intervention_results = self.env.intervene(
                i, 1000, interventions["nodes"][0], interventions["values"][0]
            )
            intervention_results.samples = intervention_results.samples.mean(axis=0)
            self.buffer.update(intervention_results)

            # can change this to args.batch_size, currently assuming a size of 1
            # for k in range(self.args.batch_size):
            #     self.buffer.update(
            #         self.env.intervene(
            #             i, 1, interventions["nodes"][k], interventions["values"][k]
            #         )
            #     )

            current_cost.append(1)
            current_y.append(intervention_results[0, -1])
            global_opt.append(np.min(current_y))
            intervention_set.append(intervention_node)
            intervention_values.append(intervention_value)
            self.posterior_model.update(self.buffer.data())
            logging.info(
                f"Selected node {intervention_node}: {intervention_value} with y = {current_y[-1]}"
            )
            logging.info(f"Current global optimum is {global_opt[-1]}")

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
        )
