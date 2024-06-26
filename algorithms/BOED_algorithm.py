from typing import Dict, List

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
        graph_variables: List
    ):
        # try to get the code the same as for the previous method
        self.env: CausalEnvironment = env
        self.posterior_model: PosteriorModel = posterior_model
        self.acquisition_strategy: AcquisitionStrategy = acquisition_strategy
        self.args = args
        self.graph_variables = graph_variables
        self.buffer: ReplayBuffer = ReplayBuffer(binary=True)

    def set_values(self, D_O: Dict, D_I: Dict):
        self.D_O = change_obs_data_format_to_mi(D_O, graph_variables=self.graph_variables, intervention_node=np.zeros(shape=len(self.graph_variables)))
        self.posterior_model.covariance_matrix = np.cov(self.D_O.samples.T)
        self.D_I = change_int_data_format_to_mi(D_I, graph_variables=self.graph_variables)
        self.buffer.update(self.D_O)
        for intervention in self.D_I:
            self.buffer.update(intervention)

        self.posterior_model.update(self.buffer.data())
        

    def run_algorithm(self, T: int = 50):
        

        for i in range(T):
            # strategy = self.acquisition_strategy(
            #     self.posterior_model, self.acquisition_strategy, self.args
            # )

            # current assume you can intervene on all nodes
            # valid_interventions = self.env.graph.nodes
            valid_interventions = list(range(self.args.num_nodes))
            interventions, _ = self.acquisition_strategy.acquire(valid_interventions, i)

            # can change this to args.batch_size, currently assuming a size of 1
            for k in range(self.args.batch_size):
                self.buffer.update(
                    self.env.intervene(
                        i, 1, interventions["nodes"][k], interventions["values"][k]
                    )
                )

            self.posterior_model.update(self.buffer.data())
