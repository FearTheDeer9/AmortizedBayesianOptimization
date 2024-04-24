from copy import deepcopy
from typing import Union

import numpy as np
from emukit.bayesian_optimization.interfaces import IEntropySearchModel
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel
from scipy.special import logsumexp
from scipy.stats import entropy
from tqdm import tqdm

from utils.sem_sampling import sample_from_SEM_hat


def normalize_log(l):
    return np.exp(l - logsumexp(l))


def sample_global_xystar(
    n_samples_mixture, all_ystar, all_xstar, arm_dist, arm_mapping_n_to_es
):
    # Select indexes of mixture components to sample from first
    mixture_idxs = stratified(
        W=arm_dist, M=n_samples_mixture
    )  # lower variance sampling

    # DEBUG info
    # arms_resampled = [arm_mapping_n_to_es[i] for i in mixture_idxs]

    local_pystars = []
    # This fits a KDE to each local p(Y*) i.e. p(Y*_(Z) | D), p(Y*_(X) | D)
    for mixt_idx in range(all_ystar.shape[0]):
        temp = all_ystar[mixt_idx, :].reshape(-1, 1)
        kde = MyKDENew(temp)
        try:
            kde.fit()
        except RuntimeError:
            kde.fit(bw=0.5)

        local_pystars.append(kde)

        # Plotting
        # temp = temp.flatten(),
        # plt.hist(temp, 100, density=True, facecolor='g', alpha=0.75)
        # grid = np.linspace(np.min(temp) - 2., np.max(temp) + 2., 1000)
        # plt.plot(grid, kde.evaluate(grid))
        # plt.title('histogram p(ystar) local for ' + str(arm_mapping_n_to_es[mixt_idx]))
        # plt.show()

    resy = np.empty(mixture_idxs.shape[0])
    # corresponding_x = [[] for _ in range(mixture_idxs.shape[0])]

    # Long for loop
    # for i, mixture_id in enumerate(mixture_idxs.tolist()):
    #     resy[i], corresponding_x[i] = y_single_sample_from_component(mixture_id , local_pystars)

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

        # temp = convert(temp)
        # corresponding_x = corresponding_x + temp
        running_cumsum += count

    # assert not np.isnan(resy).any() and np.isfinite(resy).all()
    # plt.hist(resy, 100, density=True, facecolor='r', alpha=0.75)
    # plt.title('histogram p(ystar) global')
    # plt.show()
    return resy, corresponding_x


def convert(lst):
    return list(map(lambda el: [el], lst))


def build_pystar(
    arm_mapping, bo_model, int_grids, parameter_int_domain, task, seed_anchor_points
):
    sets = bo_model.keys()
    n_samples = 200  # samples to build local p(y*, x*)
    all_ystar = np.empty((len(sets), n_samples))  # |ES| x n_samples per local dist
    all_xstar = [
        [] for _ in range(len(sets))
    ]  # different dimensions, cannot use numpy array

    for es, i in arm_mapping.items():
        model = bo_model[es]
        # sample from huge grid if len(es) > 1:
        if len(es) > 1:
            np.random.seed(seed_anchor_points)
            inps = parameter_int_domain[es].sample_uniform(point_count=100)
        else:
            inps = int_grids[es]

        sampless = model.posterior_samples(inps, size=n_samples).squeeze()

        if task == "min":
            all_ystar[i, :] = np.min(sampless, axis=0).squeeze()
            all_xstar[i] = inps[np.argmin(sampless, axis=0), :].squeeze()
        else:
            all_ystar[i, :] = np.max(sampless, axis=0).squeeze()
            all_xstar[i] = inps[np.argmax(sampless, axis=0), :].squeeze()

    # assert not np.isnan(all_ystar).any()
    # assert np.isfinite(all_ystar).all()

    return (
        all_ystar,
        all_xstar,
    )  # impossible to define global x star . used only for plotting


def update_pystar_single_model(
    arm_mapping, es, bo_model, inputs, task, all_ystar, all_xstar, space, seed
):
    corresponding_idx = arm_mapping[es]
    n_samples = all_ystar.shape[1]  # samples to build local p(y*, x*)

    sampless = bo_model.posterior_samples(
        inputs, size=n_samples
    )  # less samples to speed up
    sampless = sampless.squeeze()

    # if task == "min":
    # idxs_best = np.argmin(sampless, axis=[0,1])
    # best_values = sampless[idxs_best, :]
    # all_ystar[corresponding_idx, :] = best_values
    # all_xstar[corresponding_idx] = inputs[idxs_best, :]

    all_ystar[corresponding_idx, :] = np.min(
        sampless, axis=0
    )  # NOTE: it is really important all_ystar is the previouss one ! This is an UPDATE move
    # all_xstar[corresponding_idx] = inputs[np.argmin(sampless, axis=0), :].squeeze() # TODO

    # assert not np.isnan(all_ystar).any()  and np.isfinite(all_ystar).all()

    return all_ystar, all_xstar  # used only for plotting so not tracking x for now


def update_arm_dist(
    arm_distribution,
    updated_bo_model,
    inputs,
    temporal_index,
    task,
    arm_mapping_es_to_n,
    beta=0.1,
):
    for es in updated_bo_model[temporal_index].keys():
        corresponding_n = arm_mapping_es_to_n[es]
        inp = inputs[es]
        preds_mean, preds_var = updated_bo_model[temporal_index][es].predict(
            inp
        )  # Predictive mean
        # min or max for this ES
        if task == "min":
            arm_distribution[corresponding_n] = np.min(preds_mean) - beta * np.sqrt(
                preds_var[np.argmin(preds_mean)]
            )  # inefficient! Doing min twice here
        elif task == "max":
            arm_distribution[corresponding_n] = np.max(preds_mean) + beta * np.sqrt(
                preds_var[np.argmax(preds_mean)]
            )
        else:
            continue

    return arm_distribution


def update_arm_dist_single_model(
    arm_distribution,
    es,
    single_updated_bo_model,
    inputs,
    task,
    arm_mapping_es_to_n,
    parameter_int_domain,
    seed_anchor_points,
    beta=0.1,
):
    corresponding_n = arm_mapping_es_to_n[es]
    inps = inputs
    preds_mean, preds_var = single_updated_bo_model.predict(
        inps
    )  # Predictive mean    #

    # TODO: check exactly here to what to do when wanting to maximise. For now only considering minimization
    # if task == "min":
    arm_distribution[corresponding_n] = np.min(preds_mean) - beta * np.sqrt(
        preds_var[np.argmin(preds_mean)]
    )
    # else:
    #     arm_distribution[corresponding_n] = np.max(preds_mean) + beta * np.sqrt(preds_var[np.argmax(preds_mean)])

    return arm_distribution


def to_prob(arm_values, task="min"):
    return (
        softmax(-(1) * np.array(arm_values))
        if task == "min"
        else softmax(np.array(arm_values))
    )


def fake_do_x(
    x,
    node_parents,
    graphs,
    log_graph_post,
    intervened_vars,
    all_sem,
    all_emission_fncs,
):
    # Get a set of all variables
    # all_vars = list(self.all_emission_pairs[0].keys())
    all_vars = list(all_sem[0]().static(0).keys())

    # This will hold the fake intervention
    intervention_blanket = {k: np.array([None]).reshape(-1, 1) for k in all_vars}

    for i, intervened_var in enumerate(intervened_vars):
        intervention_blanket[intervened_var] = np.array(x.reshape(1, -1)[0, i]).reshape(
            -1, 1
        )
    # Better than  MAP
    posterior_to_avg = []
    for idx_graph in range(len(all_sem)):
        sem_hat_map = all_sem[idx_graph]
        # interv_sample = sequential_sample_from_complex_model_hat_new(
        #     static_sem=sem_hat_map().static(moment=0), dynamic_sem=None
        #     , timesteps=1, emission_pairs=all_emission_pairs[idx_graph],
        #     interventions=intervention_blanket)
        interv_sample = sample_from_SEM_hat(
            static_sem=sem_hat_map().static(moment=0),
            dynamic_sem=None,
            timesteps=1,
            node_parents=partial(node_parents, graph=graphs[idx_graph]),
            interventions=intervention_blanket,
        )

        # In theory could/should replace Y with sample from surrogate model
        for var, val in interv_sample.items():
            interv_sample[var] = val.reshape(-1, 1)

        # P(G | D, (x,y) )  . avg over V_y  =  V \ (x,y)
        posterior_to_avg.append(
            update_posterior_interventional(
                graphs=graphs,
                posterior=deepcopy(log_graph_post),
                intervened_var=intervened_vars,
                all_emission_fncs=all_emission_fncs,
                interventional_samples=interv_sample,
                total_timesteps=1,
                it=0,
            )
        )

    posterior_to_avg = np.vstack(posterior_to_avg)
    # Average over intervention outcomes
    return np.average(posterior_to_avg, axis=0, weights=log_graph_post)


class CausalEntropySearch(Acquisition):
    def __init__(
        self,
        all_sem_hat,
        all_emit_fncs,
        graphs,
        node_parents,
        current_posterior,
        es,
        model: Union[IModel, IEntropySearchModel],
        space: ParameterSpace,
        interventional_grid,
        kde,
        es_num_arm_mapping,
        num_es_arm_mapping,
        arm_distr,
        seed,
        task,
        all_xstar,
        all_ystar,
        samples_global_ystar,
        samples_global_xstar,
        do_cdcbo=False,
    ) -> None:
        """
        This is the causal entropy search acquisition funciton, which uses the mutual information
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        self.es = es
        self.model = model
        self.space = space
        self.grid = interventional_grid
        self.pre_kde = kde
        self.es_num_mapping = es_num_arm_mapping
        self.num_es_arm_mapping = es_num_arm_mapping
        self.prev_arm_distr = arm_distr
        self.seed = seed
        self.task = task
        self.init_posterior = current_posterior
        self.node_parents = node_parents
        self.graphs = graphs
        self.all_sem_hat = all_sem_hat
        self.all_emit_fncs = all_emit_fncs
        self.prev_all_ystar = all_ystar
        self.prev_all_xstar = all_xstar
        self.prev_global_samples_ystar = samples_global_ystar
        self.prev_global_samples_xstar = samples_global_xstar
        self.do_cdcbo = do_cdcbo

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the predicted change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we evaluate x.
        :param x: points where the acquisition is evaluated.
        """

        # Make new aquisition points

        grid = self.grid if len(self.es) == 1 else x  # this was wrong

        initial_entropy = self.pre_kde.entropy  # A scalar really
        initial_graph_entropy = entropy(normalize_log(self.init_posterior))
        n_fantasies = 5  # N. of fantasy observations
        n_acquisitions = x.shape[
            0
        ]  # could  choose a subset of them to reduce computation

        n_samples_mixture = self.prev_global_samples_ystar.shape[0]
        new_entropies = np.empty(
            (n_acquisitions,)
        )  # first dimension is n anchor points
        new_entropies_opt = np.empty(
            (n_acquisitions,)
        )  # first dimension is n anchor points
        new_entropies_graph = np.empty(
            (n_acquisitions,)
        )  # first dimension is n anchor points

        # Stores the new samples from the updated p(y* | D, (x,y)).
        new_samples_global_ystar_list = np.empty(
            (n_acquisitions, n_fantasies, n_samples_mixture)
        )  # TODO: check shapes

        # Keeping track of these just because of plotting later
        updated_models_list = [
            [] for _ in range(n_acquisitions)
        ]  # shape will be n_acquisitions x n_fantasies

        const = np.pi**-0.5

        # Approx integral with GQ
        xx, w = np.polynomial.hermite.hermgauss(n_fantasies)

        curr_normalized_graph = normalize_log(self.init_posterior)

        if curr_normalized_graph[0] > 0.90:
            print("graph is found")
            # If you found the graph, optimize
            for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                # Get samples from p(y | D, do(x) )
                if single_x.shape[0] == 1:
                    x_inp = single_x.reshape(-1, 1)
                else:
                    x_inp = single_x.reshape(1, -1)

                m, v = self.model.predict(x_inp)
                m, v = m.squeeze(), v.squeeze()

                # Fantasy samples from sigma points xx
                fantasy_ys = 2**0.5 * np.sqrt(v) * xx + m

                new_entropies_unweighted = np.empty((n_fantasies,))

                for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                    updated_model = deepcopy(self.model)
                    prevx, prevy = updated_model.get_X(), updated_model.get_Y()

                    # squeezed_temp = prevx.squeeze()
                    # if len(squeezed_temp.shape) == 1:
                    #     prevx = prevx.reshape(-1, 1)

                    # if len(self.es) == 2:
                    #     print('hi')

                    tempx = np.concatenate([prevx, x_inp])

                    fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(-1, 1)

                    tempy = np.vstack([prevy, fantasy_y])

                    updated_model.set_XY(tempx, tempy)

                    # Keeping track of them just for plotting ie. debugging reasons
                    updated_models_list[id_acquisition].append(updated_model)

                    # Arm distr gets updated only because model gets updated
                    # start = time.time()#&&
                    new_arm_dist = update_arm_dist_single_model(
                        deepcopy(self.prev_arm_distr),
                        self.es,
                        updated_model,
                        grid,
                        self.task,
                        self.es_num_mapping,
                        self.space,
                        self.seed,
                    )
                    # end = time.time() #&&
                    # print("update_arm_dist_single_model    took: ", end - start) #&&

                    # Use this to build p(y*, x* | D, (x,y) )
                    # start = time.time() #&&
                    pystar_samples, pxstar_samples = update_pystar_single_model(
                        arm_mapping=self.es_num_mapping,
                        es=self.es,
                        bo_model=updated_model,
                        inputs=grid,
                        task="min",
                        all_xstar=self.prev_all_xstar,
                        all_ystar=deepcopy(self.prev_all_ystar),
                        space=self.space,
                        seed=self.seed,
                    )

                    # end = time.time() #&&
                    # print("update_pystar_single_model    took: ", end - start) #&&

                    # start = time.time() #&&
                    new_samples_global_ystar, new_samples_global_xstar = (
                        sample_global_xystar(
                            n_samples_mixture=n_samples_mixture,
                            all_ystar=pystar_samples,
                            all_xstar=pxstar_samples,
                            arm_dist=to_prob(
                                new_arm_dist, self.task  # checked , this works for min
                            ),
                            arm_mapping_n_to_es=self.num_es_arm_mapping,
                        )
                    )

                    # end = time.time() #&&
                    # print("sample_global_xystar    took: ", end - start) #&&

                    # start = time.time() #&&

                    new_kde = MyKDENew(new_samples_global_ystar)
                    try:
                        new_kde.fit()
                    except RuntimeError:
                        new_kde.fit(bw=0.5)

                    new_entropy_ystar = (
                        new_kde.entropy
                    )  # this can be neg. as it's differential entropy

                    new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                    new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = (
                        new_samples_global_ystar
                    )

                    # end = time.time()#&&
                    # print("end    took: ", end - start)#&&

                # GQ average
                new_entropies[id_acquisition] = np.sum(
                    w * const * new_entropies_unweighted
                )

                # Plotting
                # if not len(self.es) > 1:
                #     self.model.model.plot()
                #     plt.title('Pre-fake-intervention model')
                #     # plt.show()
                #     for modell in updated_models_list[0]:
                #         modell.model.plot()
                #         plt.title('Post-fake-intervention model')
                #         plt.show()

                # Remove  when debugging with  batch
            assert new_entropies.shape == (n_acquisitions,) or new_entropies == (
                n_acquisitions,
                1,
            )
            # Represents the improvement in (averaged over fantasy observations!) entropy (it's good if it lowers)
            # It can be negative.
            entropy_changes = initial_entropy - new_entropies

        else:
            print("graph is not found")

            if not self.do_cdcbo:
                # Keep finding graph and optimize JOINTLY
                intervened_vars = [s for s in self.es]
                # Calc updated graph entropy
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    updated_posterior = fake_do_x(
                        x=x_inp,
                        node_parents=self.node_parents,
                        graphs=self.graphs,
                        log_graph_post=deepcopy(self.init_posterior),
                        intervened_vars=intervened_vars,
                        all_emission_fncs=self.all_emit_fncs,
                        all_sem=self.all_sem_hat,
                    )
                    new_entropies_graph[id_acquisition] = entropy(
                        normalize_log(updated_posterior)
                    )

                entropy_changes_graph = initial_graph_entropy - new_entropies_graph

                # Optimization part
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                    # Get samples from p(y | D, do(x) )
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    m, v = self.model.predict(x_inp)
                    m, v = m.squeeze(), v.squeeze()

                    # Fantasy samples from sigma points xx
                    fantasy_ys = 2**0.5 * np.sqrt(v) * xx + m

                    new_entropies_unweighted = np.empty((n_fantasies,))

                    for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                        updated_model = deepcopy(self.model)
                        prevx, prevy = updated_model.get_X(), updated_model.get_Y()

                        # squeezed_temp = prevx.squeeze()
                        # if len(squeezed_temp.shape) == 1:
                        #     prevx = prevx.reshape(-1, 1)

                        # if len(self.es) == 2:
                        #     print('hi')

                        tempx = np.concatenate([prevx, x_inp])

                        fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(
                            -1, 1
                        )

                        tempy = np.vstack([prevy, fantasy_y])

                        updated_model.set_XY(tempx, tempy)

                        # Keeping track of them just for plotting ie. debugging reasons
                        updated_models_list[id_acquisition].append(updated_model)

                        # Arm distr gets updated only because model gets updated
                        # start = time.time()#&&
                        new_arm_dist = update_arm_dist_single_model(
                            deepcopy(self.prev_arm_distr),
                            self.es,
                            updated_model,
                            grid,
                            self.task,
                            self.es_num_mapping,
                            self.space,
                            self.seed,
                        )
                        # end = time.time() #&&
                        # print("update_arm_dist_single_model    took: ", end - start) #&&

                        # Use this to build p(y*, x* | D, (x,y) )
                        # start = time.time() #&&
                        pystar_samples, pxstar_samples = update_pystar_single_model(
                            arm_mapping=self.es_num_mapping,
                            es=self.es,
                            bo_model=updated_model,
                            inputs=grid,
                            task="min",
                            all_xstar=self.prev_all_xstar,
                            all_ystar=deepcopy(self.prev_all_ystar),
                            space=self.space,
                            seed=self.seed,
                        )

                        # end = time.time() #&&
                        # print("update_pystar_single_model    took: ", end - start) #&&

                        # start = time.time() #&&
                        new_samples_global_ystar, new_samples_global_xstar = (
                            sample_global_xystar(
                                n_samples_mixture=n_samples_mixture,
                                all_ystar=pystar_samples,
                                all_xstar=pxstar_samples,
                                arm_dist=to_prob(
                                    new_arm_dist,  # checked , this works for min
                                    self.task,
                                ),
                                arm_mapping_n_to_es=self.num_es_arm_mapping,
                            )
                        )

                        # end = time.time() #&&
                        # print("sample_global_xystar    took: ", end - start) #&&

                        # start = time.time() #&&

                        new_kde = MyKDENew(new_samples_global_ystar)
                        try:
                            new_kde.fit()
                        except RuntimeError:
                            new_kde.fit(bw=0.5)

                        new_entropy_ystar = (
                            new_kde.entropy
                        )  # this can be neg. as it's differential entropy

                        new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                        new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = (
                            new_samples_global_ystar
                        )

                        # end = time.time()#&&
                        # print("end    took: ", end - start)#&&

                    # GQ average
                    new_entropies_opt[id_acquisition] = np.sum(
                        w * const * new_entropies_unweighted
                    )

                    # Plotting
                    # if not len(self.es) > 1:
                    #     self.model.model.plot()
                    #     plt.title('Pre-fake-intervention model')
                    #     # plt.show()
                    #     for modell in updated_models_list[0]:
                    #         modell.model.plot()
                    #         plt.title('Post-fake-intervention model')
                    #         plt.show()

                    # Remove  when debugging with  batch
                    # assert new_entropies.shape == (n_acquisitions,) or new_entropies == (n_acquisitions, 1)
                    # Represents the improvement in (averaged over fantasy observations!) entropy (it's good if it lowers)
                    # It can be negative.

                entropy_changes_opt = initial_entropy - new_entropies_opt

                entropy_changes = entropy_changes_graph + entropy_changes_opt
            else:
                # CD-CBO: only graph !
                # Keep finding graph and optimize jointly
                intervened_vars = [s for s in self.es]
                # Calc updated graph entropy
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    updated_posterior = fake_do_x(
                        x=x_inp,
                        node_parents=self.node_parents,
                        graphs=self.graphs,
                        log_graph_post=deepcopy(self.init_posterior),
                        intervened_vars=intervened_vars,
                        all_emission_fncs=self.all_emit_fncs,
                        all_sem=self.all_sem_hat,
                    )
                    new_entropies[id_acquisition] = entropy(
                        normalize_log(updated_posterior)
                    )

                entropy_changes = initial_graph_entropy - new_entropies

            # end of inner if
        # end of outer if

        # Just in case any are negative, shift all, preserving the total order.
        if np.any(entropy_changes < 0.0):
            smallest = np.absolute(np.min(entropy_changes))
            entropy_changes = entropy_changes + smallest

        # Plotting
        # fig, ax = plt.subplots(1, 1)
        # kwargs = {'levels': np.arange(0, 0.15, 0.01)}
        #
        # sns.kdeplot(self.prev_global_samples_ystar.squeeze(), ax=ax, label='prev global', alpha=0.22, **kwargs)
        # for i, j in zip(range(new_samples_global_ystar_list.shape[0]), range(new_samples_global_ystar_list.shape[1])):
        #     sns.kdeplot(new_samples_global_ystar_list[i, j, :].squeeze(), ax=ax,  shade=True,  alpha=0.22, **kwargs)
        #
        # plt.legend()
        # plt.title("P(y stars) by \"doing\" "+ str(self.es) +  "Best change: " + str(np.max(entropy_changes)))#+ "Acquis. and entropy changes: " + str(x.tolist()) + " " +  str(entropy_changes.tolist()) )
        # plt.show()

        print("Entropy changes for " + str(self.es) + ": ")
        print(str(entropy_changes.tolist()))
        assert entropy_changes.shape[0] == x.shape[0]

        # ax1[0][1].fill_between(inputs[:, 0], (mean - 2. * var)[:, 0], (mean + 2. * var)[:, 0], alpha=0.15)
        # ax1[0][1].plot(
        #     inputs,
        #     mean,
        #     c='b',
        #     label="$do{}$".format(es),
        #     lw=5.,
        # )

        return entropy_changes

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
